#!/usr/bin/env python3
"""
COMP482 Project 1 - Full QA evaluation.
Compares pipeline outputs (result.json / grid.txt) against Appendix-A
ground-truth labels and reports Section-7 metrics, overall and per
condition (C1..C5), plus prepared-vs-unprepared OCR and confusion stats.

Usage:
  python evaluate_all.py --results test_out/results --labels test_photos
  python evaluate_all.py --results devset_results --labels <devset_labels_dir>
  python evaluate_all.py --results test_out/results --labels test_photos --json report.json
"""
import argparse, json, math, os
from pathlib import Path

CELL = 50
def load_json(p): return json.load(open(p, encoding='utf-8'))

def grid_from_txt(p):
    rows = [l.strip() for l in open(p, encoding='utf-8') if l.strip()]
    return [[int(c) for c in r] for r in rows]

def quad_side(gt_corners):
    # mean of the 4 edge lengths of the GT quadrilateral
    e = [math.dist(gt_corners[i], gt_corners[(i+1) % 4]) for i in range(4)]
    return sum(e) / 4.0

def corner_error_pct(pred, gt):
    if pred is None: return None
    d = sum(math.dist(pred[i], gt[i]) for i in range(4)) / 4.0
    return 100.0 * d / quad_side(gt)

def cell_accuracy(pred, gt):
    return sum(pred[r][c] == gt[r][c] for r in range(9) for c in range(9)) / 81.0

def empty_f1(pred, gt):
    # F1 on the binary empty/non-empty decision
    tp = sum((pred[r][c] == 0) == (gt[r][c] == 0) and pred[r][c] == 0
             for r in range(9) for c in range(9))
    fp = sum(pred[r][c] == 0 and gt[r][c] != 0 for r in range(9) for c in range(9))
    fn = sum(pred[r][c] != 0 and gt[r][c] == 0 for r in range(9) for c in range(9))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0

def exact_match(pred, gt):
    return int(all(pred[r][c] == gt[r][c] for r in range(9) for c in range(9)))

def confusion(pred, gt):
    pairs = {}
    for r in range(9):
        for c in range(9):
            if pred[r][c] != gt[r][c] and pred[r][c] and gt[r][c]:
                k = (gt[r][c], pred[r][c])
                pairs[k] = pairs.get(k, 0) + 1
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    res_dir, lab_dir = Path(args.results), Path(args.labels)
    labels = {}
    for f in lab_dir.glob("*.json"):
        d = load_json(f)
        labels[d["image"]] = d  # keyed by image filename

    rows, all_conf = [], {}
    for rf in sorted(res_dir.glob("*_result.json")):
        res = load_json(rf)
        img_name = Path(res.get("image_path", rf.name)).name
        lab = labels.get(img_name)
        if lab is None:
            print(f"[skip] no label for {img_name}"); continue

        name = rf.name[:-len("_result.json")]
        gtxt = res_dir / f"{name}_grid.txt"
        pred = grid_from_txt(gtxt) if gtxt.exists() and res.get("grid") else res.get("grid")
        gt = lab["grid"]

        row = {
            "image": img_name,
            "conditions": lab.get("conditions", []),
            "verdict": res.get("verdict"),
            "runtime_s": res.get("runtime_s"),
            "corner_err_pct": corner_error_pct(res.get("corners"), lab["corners"]),
            "cell_acc": cell_accuracy(pred, gt) if pred else None,
            "empty_f1": empty_f1(pred, gt) if pred else None,
            "exact": exact_match(pred, gt) if pred else 0,
            "robust": int(res.get("runtime_s", 99) < 20 and res.get("verdict") != "error"),
        }
        gu = res.get("grid_unprepared")
        if gu:
            row["unprepared_acc"] = cell_accuracy(gu, gt)
        row["prepared_acc"] = row["cell_acc"]
        for k, v in confusion(pred, gt).items() if pred else {}.items():
            all_conf[k] = all_conf.get(k, 0) + v
        rows.append(row)

    # ---- aggregate ----
    def agg(sub):
        n = len(sub)
        def m(k): 
            v = [r[k] for r in sub if r.get(k) is not None]
            return round(sum(v)/len(v), 4) if v else None
        return {
            "n": n,
            "corner_err_pct": m("corner_err_pct"),
            "cell_acc": m("cell_acc"),
            "empty_f1": m("empty_f1"),
            "exact_match_pct": round(100*sum(r["exact"] for r in sub)/n, 1) if n else None,
            "robustness_pct": round(100*sum(r["robust"] for r in sub)/n, 1) if n else None,
            "mean_runtime_s": m("runtime_s"),
            "prepared_acc": m("prepared_acc"),
            "unprepared_acc": m("unprepared_acc"),
        }

    report = {"overall": agg(rows)}
    for cond in ("C1","C2","C3","C4","C5"):
        sub = [r for r in rows if cond in r["conditions"]]
        report[cond] = agg(sub)

    print(json.dumps(report, indent=2))
    print("\nPer-image:")
    for r in rows:
        print(f"  {r['image']:40s} {str(r['conditions']):10s} "
              f"corner%={r['corner_err_pct']} cell={r['cell_acc']} "
              f"f1={r['empty_f1']} exact={r['exact']} t={r['runtime_s']}s {r['verdict']}")

    if all_conf:
        print("\nTop digit confusions (gt -> pred):")
        for (g,p),cnt in sorted(all_conf.items(), key=lambda x:-x[1])[:10]:
            print(f"  {g} -> {p}: {cnt}")

    if args.json:
        Path(args.json).write_text(json.dumps({"report": report, "rows": rows,
            "confusions": {f"{g}->{p}": c for (g,p),c in all_conf.items()}},
            indent=2), encoding='utf-8')
        print(f"\nsaved: {args.json}")

if __name__ == "__main__":
    main()