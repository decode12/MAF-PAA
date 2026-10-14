#!/usr/bin/env python3
"""Recompute Fig.3 quartiles and Fig.4 case predictions from saved MAF outputs.

The paper's qualitative analysis uses the separately archived five-epoch BBC
checkpoint, not the 20-epoch models in the main/ablation tables. This script can
also analyze new evaluations, but does not relabel them as the published data.

A sample_id is split:NNNNNNNN, using the zero-based row in the ORIGINAL full
split metadata. It is neither a source-filtered row number nor a news ID (news
IDs can occur repeatedly). Equal reliability scores are ordered by sample_id;
quartiles then follow numpy.array_split. The paper plotting script reconstructed
g in float32 using its algebraically equivalent agreement formula. That mode
is retained by default; --gate-source stored uses actual saved forward gates.
No thresholds or models are tuned.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER_CASES = [
    {"role": "success", "record_id": "1462685", "sample_id": "test:00009826"},
    {"role": "failure", "record_id": "1550713", "sample_id": "test:00001220"},
]
PAPER_QUARTILES = [
    {"n": 2343, "fixed": 167, "harmed": 97, "state_accuracy": 74.30644472897994},
    {"n": 2343, "fixed": 9, "harmed": 4, "state_accuracy": 82.62910798122066},
    {"n": 2342, "fixed": 1, "harmed": 0, "state_accuracy": 88.12980358667805},
    {"n": 2342, "fixed": 0, "harmed": 0, "state_accuracy": 92.10076857386849},
]
PAPER_CASE_NUMBERS = {
    "test:00009826": {"base_probability": 0.291015625, "final_probability": 0.7366777062416077,
                       "q_text": 0.5965946316719055, "fusion_gate": 0.5727911591529846},
    "test:00001220": {"base_probability": 0.09130859375, "final_probability": 0.08906779438257217,
                       "q_real": 0.9232373833656311, "cross_head_agreement": 0.9912775158882141},
}


def load_predictions(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) < 4:
        raise ValueError("At least four predictions are required for four quartiles")
    required = {"sample_id", "label", "state", "base_probability", "final_probability",
                "state_prob", "p_text", "p_image", "cross_head_agreement", "fusion_gate",
                "state_coeff", "evidence_logit"}
    seen = set()
    for i, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Prediction {i} lacks fields {sorted(missing)}; rerun evaluation with diagnostics")
        sid = row["sample_id"]
        if not isinstance(sid, str) or not re.fullmatch(r"(?:test|val):\d{8}", sid):
            raise ValueError(f"Invalid original-metadata sample ID: {sid}")
        if sid in seen:
            raise ValueError(f"Duplicate sample ID {sid}; do not use nonunique news IDs")
        seen.add(sid)
        if row["label"] not in (0, 1) or row["state"] not in (0, 1, 2, 3):
            raise ValueError(f"Invalid label/state for {sid}")
        q = np.asarray(row["state_prob"], dtype=float)
        if q.shape != (4,) or not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
            raise ValueError(f"Invalid four-state probabilities for {sid}")
        # BF16 softmax rounding can produce a small sum discrepancy.
        if abs(float(q.sum()) - 1) > .02:
            raise ValueError(f"Four-state probabilities do not sum to one for {sid}")
        for key in ("base_probability", "final_probability", "p_text", "p_image",
                    "cross_head_agreement", "fusion_gate", "state_coeff"):
            if not np.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError(f"Invalid {key} for {sid}")
        if not np.isfinite(row["evidence_logit"]):
            raise ValueError(f"Invalid evidence logit for {sid}")
    return rows


def quartile_statistics(rows, gate_source="paper-reconstructed"):
    q = np.asarray([r["state_prob"] for r in rows], dtype=np.float32)
    pt = np.asarray([r["p_text"] for r in rows], dtype=np.float32)
    pv = np.asarray([r["p_image"] for r in rows], dtype=np.float32)
    # Match the original figure parser's float32 operation order exactly.
    agreement = np.float32(1) - np.float32(.5) * (np.abs(q[:, 1] + q[:, 3] - pt)
                                                 + np.abs(q[:, 2] + q[:, 3] - pv))
    reconstructed = np.clip(agreement * q.max(axis=1), 0, 1)
    saved = np.asarray([r["fusion_gate"] for r in rows])
    selected = reconstructed if gate_source == "paper-reconstructed" else saved
    augmented = [dict(row, analysis_gate=float(value)) for row, value in zip(rows, selected)]
    ordered = sorted(augmented, key=lambda row: (row["analysis_gate"], row["sample_id"]))
    result = []
    for quartile, indices in enumerate(np.array_split(np.arange(len(ordered)), 4), 1):
        group = [ordered[int(i)] for i in indices]
        labels = np.asarray([r["label"] for r in group], dtype=int)
        base = np.asarray([r["base_probability"] for r in group]) >= .5
        final = np.asarray([r["final_probability"] for r in group]) >= .5
        state = np.asarray([1 - r["state_prob"][0] for r in group]) >= .5
        fixed = int(((base != labels) & (final == labels)).sum())
        harmed = int(((base == labels) & (final != labels)).sum())
        count = len(group)
        result.append({
            "quartile": quartile, "n": count,
            "g_min": float(group[0]["analysis_gate"]), "g_max": float(group[-1]["analysis_gate"]),
            "state_accuracy": 100 * float((state == labels).mean()),
            "fixed": fixed, "harmed": harmed,
            "fixed_rate": 100 * fixed / count, "harmed_rate": 100 * harmed / count,
            "net_pp": 100 * (fixed - harmed) / count,
        })
    return result, float(np.max(np.abs(saved - reconstructed)))


def case_predictions(rows, cases):
    by_id = {row["sample_id"]: row for row in rows}
    result = []
    for case in cases:
        sid = case["sample_id"]
        if sid not in by_id:
            result.append({"role": case["role"], "record_id": str(case["record_id"]),
                           "sample_id": sid, "status": "MISSING_FROM_INPUT"})
            continue
        r = by_id[sid]
        news_id = str(r.get("id", r.get("news_id", case["record_id"])))
        if news_id != str(case["record_id"]):
            raise ValueError(f"Case {sid} does not match its news ID; check original-metadata indexing")
        q = r["state_prob"]
        result.append({
            "role": case["role"], "record_id": news_id, "sample_id": sid,
            "status": "FOUND", "label": r["label"], "state": r["state"],
            "base_probability": r["base_probability"], "final_probability": r["final_probability"],
            "base_prediction": int(r["base_probability"] >= .5),
            "final_prediction": int(r["final_probability"] >= .5),
            "q_real": q[0], "q_text": q[1], "q_image": q[2], "q_joint": q[3],
            **{key: r[key] for key in ("p_text", "p_image", "cross_head_agreement",
                                      "fusion_gate", "state_coeff", "evidence_logit")},
        })
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def draw(quartiles, destination, font):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Optional plotting requires matplotlib; install it or omit --plot") from exc
    plt.rcParams.update({"font.family": font, "font.size": 14, "axes.labelsize": 14,
                         "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.7))
    x = np.arange(4)
    labels = [f"Q{r['quartile']}\n[{r['g_min']:.3f},\n{r['g_max']:.3f}]" for r in quartiles]
    for axis in axes:
        axis.set_xlim(-.5, 3.5)
        axis.set_xticks(x, labels)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    accuracy = [r["state_accuracy"] for r in quartiles]
    corrected = [r["fixed_rate"] for r in quartiles]
    harmed = [r["harmed_rate"] for r in quartiles]
    net = [r["net_pp"] for r in quartiles]
    axes[0].plot(x, accuracy, "o-", color="#2878B5", lw=2, ms=6)
    for i, value in enumerate(accuracy):
        axes[0].annotate(f"{value:.2f}%", (i, value), xytext=(0, 8), textcoords="offset points", ha="center")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Accuracy (%)")
    width = .28
    axes[1].bar(x - width / 2, corrected, width, label="Corrected (%)", color="#269D80")
    axes[1].bar(x + width / 2, harmed, width, label="Harmed (%)", color="#E69F47")
    axes[1].plot(x, net, "D-", color="#303030", ms=5, label="Net ACC change (pp)")
    for i, value in enumerate(net):
        axes[1].annotate(f"{value:+.2f}", (i, value), xytext=(0, -18), textcoords="offset points", ha="center", fontsize=12)
    axes[1].axhline(0, color="gray", lw=.8)
    axes[1].set_ylim(min(-1, min(net) - 1), max(max(corrected), max(harmed)) + 2)
    axes[1].set_ylabel("Rate (%) / ΔACC (pp)")
    axes[1].legend(frameon=False, fontsize=12, loc="upper right", handlelength=1.5, labelspacing=.35)
    fig.subplots_adjust(left=.085, right=.985, top=.95, bottom=.30, wspace=.38)
    for axis, title in zip(axes, ["(a) Auxiliary prediction reliability", "(b) Prediction corrections and harms"]):
        axis.text(.5, -.34, title, transform=axis.transAxes, ha="center", va="top", fontsize=14)
    fig.savefig(destination)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "results/predictions/analysis_bbc_legacy.jsonl.gz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/analysis")
    parser.add_argument("--cases-json", type=Path, help="Optional list, or object containing cases with role, record_id and sample_id")
    parser.add_argument("--metadata", type=Path, help="Optional original full split metadata to verify raw sample IDs against news IDs")
    parser.add_argument("--plot", action="store_true", help="Also export a two-panel PDF; requires matplotlib")
    parser.add_argument("--font", default="DejaVu Serif", help="A locally installed font; no proprietary fonts are bundled")
    parser.add_argument("--gate-source", choices=["paper-reconstructed", "stored"], default="paper-reconstructed",
                        help="Match the published float32 parser, or sort using actual saved forward gates")
    parser.add_argument("--verify-paper", action="store_true", help="Fail if quartiles or case numbers differ from the published analysis")
    args = parser.parse_args(argv)
    rows = load_predictions(args.predictions)
    metadata_verified = False
    if args.metadata:
        metadata = json.loads(args.metadata.read_text())
        for row in rows:
            index = int(row["sample_id"].split(":")[1])
            if index >= len(metadata) or str(metadata[index]["id"]) != str(row.get("id", row.get("news_id"))):
                raise ValueError(f"Original metadata does not match {row['sample_id']}")
        metadata_verified = True
    cases = PAPER_CASES
    if args.cases_json:
        cases = json.loads(args.cases_json.read_text())
        if isinstance(cases, dict):
            cases = cases["cases"]
    quartiles, gate_difference = quartile_statistics(rows, args.gate_source)
    case_rows = case_predictions(rows, cases)
    quarter_match = all(
        actual["n"] == expected["n"] and actual["fixed"] == expected["fixed"]
        and actual["harmed"] == expected["harmed"]
        and abs(actual["state_accuracy"] - expected["state_accuracy"]) < 1e-9
        for actual, expected in zip(quartiles, PAPER_QUARTILES)
    )
    by_case = {row["sample_id"]: row for row in case_rows}
    case_match = all(
        sid in by_case and by_case[sid]["status"] == "FOUND"
        and all(abs(by_case[sid][key] - value) < 1e-9 for key, value in expected.items())
        for sid, expected in PAPER_CASE_NUMBERS.items()
    )
    manifest = {
        "input_file": args.predictions.name,
        "input_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "sample_count": len(rows), "threshold": .5,
        "quartiles": "ascending selected gate, then original-metadata sample_id; numpy.array_split",
        "gate_source": args.gate_source,
        "max_stored_vs_reconstructed_gate_difference": gate_difference,
        "metadata_id_verification": "PASS" if metadata_verified else "NOT REQUESTED",
        "paper_quartile_match": "PASS" if quarter_match else "DIFFERENT INPUT RESULTS",
        "paper_case_match": "PASS" if case_match else "DIFFERENT OR MISSING INPUT RESULTS",
        "paper_analysis_checkpoint": "analysis_bbc; older five-epoch BBC checkpoint, distinct from the final 20-epoch main models",
        "interpretation": "Descriptive single-seed analysis; no confidence intervals or model selection",
        "plot_font": args.font if args.plot else None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "quartile_statistics.csv", quartiles)
    write_csv(args.output_dir / "case_predictions.csv", case_rows)
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.plot:
        draw(quartiles, args.output_dir / "reliability_analysis.pdf", args.font)
    print(json.dumps(manifest, indent=2))
    if args.verify_paper and not (quarter_match and case_match):
        raise SystemExit("Prediction input does not reproduce the published analysis; it was not relabeled or tuned")


if __name__ == "__main__":
    main()
