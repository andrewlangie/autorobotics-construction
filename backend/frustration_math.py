# backend/frustration_math.py

def calculate_percent_change(current, baseline):
    if baseline is None or current is None or baseline == 0:
        return 0.0
    return ((current - baseline) / baseline) * 100.0

def compute_frustration_percent_changes(current_metrics, baseline_metrics):
    """
    current_metrics and baseline_metrics are dicts with keys:
    'eda', 'scr', 'hr', 'rmssd', 'sdnn'
    """
    return {
        "eda_pct": calculate_percent_change(current_metrics.get("eda"), baseline_metrics.get("eda")),
        "scr_pct": calculate_percent_change(current_metrics.get("scr"), baseline_metrics.get("scr")),
        "hr_pct": calculate_percent_change(current_metrics.get("hr"), baseline_metrics.get("hr")),
        "rmssd_pct": calculate_percent_change(current_metrics.get("rmssd"), baseline_metrics.get("rmssd")),
        "sdnn_pct": calculate_percent_change(current_metrics.get("sdnn"), baseline_metrics.get("sdnn")),
    }