"""Read heart rate, RR intervals and accelerometer data from a Polar H10 over Bluetooth.

Uses `bleak` (Bluetooth) and `bleakheart` (decodes the Polar data formats).
Only ONE app can be connected to the H10 at a time -- close the browser Polar panel
(and the Polar Beat app on your phone) before running this.

Commands (from the project root):
    python backend/polar_reader.py --print                       # live values in the terminal
    python backend/polar_reader.py --record recordings/session1  # save hr.csv + acc.csv
"""
import argparse
import asyncio
import csv
import queue
import sys
import threading
import time
from pathlib import Path

from bleak import BleakClient, BleakScanner
from bleakheart import HeartRate, PolarMeasurementData

ACC_SAMPLE_RATE = 50  # Hz. The H10 supports 25, 50, 100 or 200
ACC_RANGE = 8         # +-g. The H10 supports 2, 4 or 8 (8 avoids clipping when moving fast)


class PolarH10:
    """Connects to the strap and calls on_beat(t, hr, rr_ms) and on_acc(t, ax, ay, az).

    t is in seconds (Unix time); acceleration is converted from milli-g to g.
    Reconnects automatically if the strap drops out.
    """

    def __init__(self, on_beat, on_acc, name="Polar H10"):
        self.on_beat, self.on_acc, self.name = on_beat, on_acc, name.lower()
        self.stop = asyncio.Event()

    def _hr_callback(self, frame):
        # bleakheart (unpack=True): ('HR', time_ns, (hr, rr_ms), energy) -- one per heartbeat
        _, t_ns, (hr, rr), _ = frame
        if rr:
            self.on_beat(t_ns / 1e9, float(hr), float(rr))

    def _acc_callback(self, frame):
        # bleakheart: ('ACC', time_ns_of_last_sample, [(x, y, z) in milli-g, ...])
        kind, t_ns, samples = frame
        if kind != "ACC":
            return
        n = len(samples)
        for i, (x, y, z) in enumerate(samples):
            t = t_ns / 1e9 - (n - 1 - i) / ACC_SAMPLE_RATE
            self.on_acc(t, x / 1000.0, y / 1000.0, z / 1000.0)

    async def _session(self):
        print(f"Scanning for {self.name!r} ... (wear the strap so it wakes up)")
        device = await BleakScanner.find_device_by_filter(
            lambda d, adv: bool(d.name) and self.name in d.name.lower(), timeout=20
        )
        if device is None:
            raise RuntimeError("Polar H10 not found. Is it worn, and disconnected from other apps?")
        async with BleakClient(device) as client:
            print(f"Connected to {device.name}")
            hr = HeartRate(client, callback=self._hr_callback, instant_rate=True, unpack=True)
            await hr.start_notify()
            pmd = PolarMeasurementData(client, callback=self._acc_callback)
            code, msg, _ = await pmd.start_streaming(
                "ACC", SAMPLE_RATE=ACC_SAMPLE_RATE, RANGE=ACC_RANGE
            )
            if code != 0:
                print(f"Accelerometer not started: {msg}")
                print("Supported settings:", await pmd.available_settings("ACC"))
            while client.is_connected and not self.stop.is_set():
                await asyncio.sleep(1)

    async def run(self):
        while not self.stop.is_set():
            try:
                await self._session()
            except Exception as exc:  # Bluetooth errors: wait and retry
                print(f"[polar] {exc} -- retrying in 3 s", file=sys.stderr)
            if not self.stop.is_set():
                await asyncio.sleep(3)


def live_events():
    """Yield ('beat', t, hr, rr) and ('acc', t, ax, ay, az) from a live strap, in a background thread."""
    q = queue.Queue()
    polar = PolarH10(
        on_beat=lambda t, hr, rr: q.put(("beat", t, hr, rr)),
        on_acc=lambda t, ax, ay, az: q.put(("acc", t, ax, ay, az)),
    )
    threading.Thread(target=lambda: asyncio.run(polar.run()), daemon=True).start()
    while True:
        yield q.get()


def record(folder):
    """Save a session to <folder>/hr.csv and <folder>/acc.csv (replay it later with polar_coach)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "hr.csv", "w", newline="") as fh, open(folder / "acc.csv", "w", newline="") as fa:
        hw, aw = csv.writer(fh), csv.writer(fa)
        hw.writerow(["t", "hr", "rr_ms"])
        aw.writerow(["t", "ax", "ay", "az"])
        print(f"Recording to {folder}/ -- press Ctrl+C to stop.")
        beats = 0
        last_print = time.time()
        for ev in live_events():
            if ev[0] == "beat":
                hw.writerow([f"{ev[1]:.3f}", f"{ev[2]:.0f}", f"{ev[3]:.0f}"])
                beats += 1
            else:
                aw.writerow([f"{ev[1]:.3f}", f"{ev[2]:.4f}", f"{ev[3]:.4f}", f"{ev[4]:.4f}"])
            if time.time() - last_print > 5:
                fh.flush(), fa.flush()
                print(f"  {beats} beats recorded")
                last_print = time.time()


def print_live():
    last = 0.0
    for ev in live_events():
        if ev[0] == "beat":
            print(f"beat  HR {ev[2]:5.0f} bpm   RR {ev[3]:5.0f} ms")
        elif time.time() - last > 1:  # accelerometer is 50 Hz -- print once a second
            print(f"acc   x {ev[2]:+.2f}  y {ev[3]:+.2f}  z {ev[4]:+.2f} g")
            last = time.time()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--print", action="store_true", help="print live values")
    g.add_argument("--record", metavar="FOLDER", help="save hr.csv and acc.csv into FOLDER")
    args = p.parse_args()
    try:
        record(args.record) if args.record else print_live()
    except KeyboardInterrupt:
        print("\nStopped.")
