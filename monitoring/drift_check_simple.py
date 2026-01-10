import json
from pathlib import Path

"""
Per il monitoraggio del data drift uso una strategia semplice e interpretabile: 
confronto la distribuzione delle classi e la lunghezza media del testo su una finestra recente rispetto a una baseline iniziale. 
Se lo scostamento supera una soglia predefinita genero un alert. 
Legge monitoring/predictions.jsonl (log delle predizioni) e monitoring/baseline_simple.json (baseline calcolata in precedenza) e verifica la presenza di drift.
"""

LOG_PATH = Path("monitoring/predictions.jsonl")
BASELINE_PATH = Path("monitoring/baseline_simple.json")

WINDOW_SIZE = 50  # ultime N predizioni
THRESH_LABEL = 0.20  # 20 punti percentuali di scostamento
THRESH_LEN = 0.50    # 50% di scostamento sulla lunghezza media


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_last_rows(path: Path, n: int) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing log file: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    lines = lines[-n:] if n > 0 else lines

    rows = []
    for line in lines:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def dist_labels(rows: list[dict]) -> dict:
    counts = {}
    for r in rows:
        lab = r.get("label", "unknown")
        counts[lab] = counts.get(lab, 0) + 1

    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def avg_text_len(rows: list[dict]) -> float:
    vals = [int(r.get("text_len", 0)) for r in rows]
    return sum(vals) / (len(vals) or 1)


def main():
    baseline = load_json(BASELINE_PATH)
    rows = read_last_rows(LOG_PATH, WINDOW_SIZE)

    current_dist = dist_labels(rows)
    current_len = avg_text_len(rows)

    # --- Check A: distribuzione label ---
    alerts = []

    base_dist = baseline["label_dist"]
    for label, base_p in base_dist.items():
        cur_p = current_dist.get(label, 0.0)
        diff = abs(cur_p - base_p)
        if diff >= THRESH_LABEL:
            alerts.append(
                f"Label shift: {label} baseline={base_p:.2f} current={cur_p:.2f} diff={diff:.2f}"
            )

    # --- Check B: lunghezza media ---
    base_len = float(baseline["avg_text_len"])
    if base_len > 0:
        rel_diff = abs(current_len - base_len) / base_len
        if rel_diff >= THRESH_LEN:
            alerts.append(
                f"Text length shift: baseline={base_len:.1f} current={current_len:.1f} rel_diff={rel_diff:.2f}"
            )

    print(f"Window size: {len(rows)}")
    print(f"Current label dist: {current_dist}")
    print(f"Current avg text_len: {current_len:.1f}")

    if alerts:
        print("\nDRIFT/ALERT DETECTED:")
        for a in alerts:
            print("-", a)
        raise SystemExit(1)  # utile per automation dopo
    else:
        print("\nNo drift detected.")
        raise SystemExit(0)


if __name__ == "__main__":
    main()
