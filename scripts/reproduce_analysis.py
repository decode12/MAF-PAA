#!/usr/bin/env python3
"""Reproduce the current BBC Figures 3/4 from the final main_bbc predictions.

Quartiles use stored forward-pass gates, ascending (fusion_gate, sample_id),
then numpy.array_split. A sample_id indexes the original full split metadata,
not a source-filtered row or a nonunique news ID. All decisions use 0.5.
Use --data-root to recover the Figure 4 images/captions from official data.
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


def quartile_statistics(rows):
    ordered = sorted(rows, key=lambda row: (row["fusion_gate"], row["sample_id"]))
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
            "g_min": float(group[0]["fusion_gate"]), "g_max": float(group[-1]["fusion_gate"]),
            "state_accuracy": 100 * float((state == labels).mean()),
            "fixed": fixed, "harmed": harmed,
            "fixed_rate": 100 * fixed / count, "harmed_rate": 100 * harmed / count,
            "net_pp": 100 * (fixed - harmed) / count,
        })
    return result


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
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
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
    labels = [f"Q{r['quartile']}" for r in quartiles]
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
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def recover_cases(data_root, cases, predictions, destination):
    """Verify exact original/manipulated pairs and export authorized local data."""
    import difflib
    import shutil
    from PIL import Image
    root = data_root.resolve()
    metadata_path = root / 'metadata/test.json'
    expected = json.loads((ROOT / 'datasets/dgm4_assets.json').read_text())
    # Verify the original metadata version before interpreting row identifiers.
    metadata_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    if metadata_hash != expected['files']['metadata/test.json']['sha256']:
        raise ValueError('Test metadata SHA-256 does not match the pinned DGM4 version')
    metadata = json.loads(metadata_path.read_text())
    scores = {r['sample_id']: r for r in predictions}
    result = []
    for case in cases:
        sid = case['sample_id']
        score = scores[sid]
        if score['status'] != 'FOUND':
            raise ValueError(f'Cannot render missing case {sid}')
        manipulated = metadata[int(sid.split(':')[1])]
        original = metadata[int(case['original_sample_id'].split(':')[1])]
        if not (str(manipulated['id']) == str(original['id']) == str(case['record_id'])
                and manipulated['image'] == original['image']
                and original['fake_cls'] == 'orig'
                and manipulated['fake_cls'] in ('text_swap', 'text_attribute')):
            raise ValueError(f'Original/manipulated record pairing failed for {sid}')
        parts = Path(manipulated['image']).parts
        path = root.joinpath(*(parts[1:] if parts[0] == 'DGM4' else parts)).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Image path escapes the dataset directory')
        if hashlib.sha256(path.read_bytes()).hexdigest() != case['image_sha256']:
            raise ValueError(f'Image SHA-256 mismatch for {sid}')
        with Image.open(path) as im:
            if im.size != (case['native_width'], case['native_height']):
                raise ValueError(f'Unexpected native image dimensions for {sid}')
        image_name = f"{case['role']}_{case['record_id']}{path.suffix}"
        shutil.copyfile(path, destination / image_name)
        before, after = original['text'].split(), manipulated['text'].split()
        edits = [{'original': ' '.join(before[i:j]), 'manipulated': ' '.join(after[k:l])}
                 for tag, i, j, k, l in difflib.SequenceMatcher(a=before, b=after).get_opcodes()
                 if tag != 'equal']
        result.append(dict(score, original_sample_id=case['original_sample_id'],
                           image_file=image_name, original_caption=original['text'],
                           manipulated_caption=manipulated['text'], edits=edits))
    (destination / 'cases_with_captions.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def draw_cases(cases, destination, font):
    """Render the same verified examples; paper artwork is manually arranged."""
    import textwrap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({'font.family': font, 'font.size': 12, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(11.5, 6.4))
    for i, case in enumerate(cases):
        left = .04 + i * .50
        fig.text(left + .21, .97, f"({chr(97+i)}) {case['role'].capitalize()} case",
                 ha='center', va='top', fontsize=15, fontweight='bold')
        axis = fig.add_axes([left, .53, .42, .40])
        with Image.open(destination / case['image_file']) as im:
            axis.imshow(im)
        axis.axis('off')
        y = .50
        for label, key in [('Original caption (reference only)', 'original_caption'),
                           ('Manipulated caption (model input)', 'manipulated_caption')]:
            fig.text(left, y, label, fontsize=12, fontweight='bold', va='top')
            lines = textwrap.wrap(case[key], width=53)
            fig.text(left, y-.035, '\n'.join(lines), fontsize=12, va='top', linespacing=1.35)
            y -= .035 + .033 * len(lines) + .02
        final = 'Fake' if case['final_prediction'] else 'Real'
        base = 'Fake' if case['base_prediction'] else 'Real'
        lines = [f"Ground truth: {'Fake' if case['label'] else 'Real'} (Text-only)",
                 f"Global p(Fake): {case['base_probability']:.3f} ({base})",
                 f"MAF p(Fake): {case['final_probability']:.3f} ({final})",
                 f"Text-only probability: {case['q_text']:.3f}   Reliability: {case['fusion_gate']:.3f}"]
        fig.text(left, y, '\n'.join(lines), fontsize=12, va='top', linespacing=1.5)
    for ext in ('pdf', 'png'):
        fig.savefig(destination / f'Figure4.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=ROOT / 'results/predictions/main_bbc.jsonl.gz')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/analysis')
    parser.add_argument('--data-root', type=Path, help='Official DGM4 directory; enables verified case-image/caption export')
    parser.add_argument('--plot', action='store_true', help='Export Figure 3 PDF/PNG; also Figure 4 when --data-root is supplied')
    parser.add_argument('--font', default='DejaVu Serif', help='Use an installed font, e.g. Times New Roman')
    parser.add_argument('--verify-paper', action='store_true', help='Verify current BBC predictions, statistics and case outputs')
    args = parser.parse_args(argv)
    if args.plot:
        from matplotlib.font_manager import findfont, FontProperties
        try:
            findfont(FontProperties(family=args.font), fallback_to_default=False)
        except ValueError:
            parser.error(f"Font {args.font!r} is not installed; omit --font to use DejaVu Serif")
    rows = load_predictions(args.predictions)
    reference = json.loads((ROOT / 'results/case_study_records.json').read_text())
    quartiles = quartile_statistics(rows)
    cases = case_predictions(rows, reference['cases'])
    with (ROOT / 'results/haf_quartiles.csv').open() as stream:
        expected_quartiles = list(csv.DictReader(stream))
    quarter_match = len(quartiles) == len(expected_quartiles) and all(
        abs(actual[key] - float(expected[key])) < 1e-9
        for actual, expected in zip(quartiles, expected_quartiles) for key in actual)
    case_match = all(actual['status'] == 'FOUND' and all(
        actual[key] == expected[key] if key in ('role', 'record_id', 'sample_id')
        else abs(actual[key] - expected[key]) < 1e-9
        for key in ('role', 'record_id', 'sample_id', 'label', 'state', 'base_probability',
                    'final_probability', 'q_real', 'q_text', 'q_image', 'q_joint',
                    'cross_head_agreement', 'fusion_gate'))
        for actual, expected in zip(cases, reference['cases']))
    input_match = None
    if args.verify_paper:
        frozen = load_predictions(ROOT / 'results/predictions/main_bbc.jsonl.gz')
        score_keys = ('sample_id', 'id', 'label', 'state', 'base_probability', 'final_probability',
                      'state_prob', 'p_text', 'p_image', 'cross_head_agreement', 'fusion_gate',
                      'state_coeff', 'evidence_logit')
        input_match = len(rows) == len(frozen) and all(
            all(a[k] == b[k] for k in score_keys) for a, b in zip(rows, frozen))
    if args.verify_paper and not (quarter_match and case_match and input_match):
        raise SystemExit('Input differs from the current main_bbc paper predictions; no outputs were relabeled or tuned')
    manifest = {
        'input_file': args.predictions.name,
        'input_sha256': hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        'sample_count': len(rows), 'threshold': .5,
        'quartiles': 'ascending (stored fusion_gate, sample_id), then numpy.array_split',
        'reference_checkpoint': reference['checkpoint_name'],
        'reference_checkpoint_sha256': reference['checkpoint_sha256'],
        'paper_predictions_match': input_match,
        'paper_quartile_match': quarter_match, 'paper_case_match': case_match,
        'interpretation': 'Descriptive within-model comparison; not a causal HAF ablation',
        'case_data_verification': 'NOT REQUESTED',
        'plot_font': args.font if args.plot else None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'quartile_statistics.csv', quartiles)
    write_csv(args.output_dir / 'case_predictions.csv', cases)
    if args.data_root:
        recovered = recover_cases(args.data_root, reference['cases'], cases, args.output_dir)
        manifest['case_data_verification'] = 'PASS: metadata version, original pair, image SHA-256 and dimensions'
        if args.plot:
            draw_cases(recovered, args.output_dir, args.font)
    if args.plot:
        draw(quartiles, args.output_dir / 'Figure3.pdf', args.font)
    (args.output_dir / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
