#!/usr/bin/env python3
"""Recompute paper MAF metrics from frozen or newly generated predictions."""
import argparse
import csv
import gzip
import json
from pathlib import Path
from metrics import binary_metrics


def read_rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, help='Evaluate one new prediction file')
    parser.add_argument('--output', type=Path, default=Path('outputs/recomputed_results.csv'))
    args = parser.parse_args()
    if args.predictions:
        records = read_rows(args.predictions)
        metrics = binary_metrics([x['label'] for x in records], [x['final_probability'] for x in records])
        print(json.dumps(dict(sample_count=len(records), **metrics), indent=2))
        return
    root = Path(__file__).resolve().parents[1]
    with (root / 'results/reported_results.csv').open() as stream:
        reported = list(csv.DictReader(stream))
    output = []
    cache = {}
    for row in reported:
        run = row['run_id']
        if not run:
            continue
        if run not in cache:
            records = read_rows(root / 'results/predictions' / (run + '.jsonl.gz'))
            cache[run] = (len(records), binary_metrics([x['label'] for x in records], [x['final_probability'] for x in records]))
        n, metrics = cache[run]
        result = dict(table=row['table'], run_id=run, split=row['split'], sample_count=n)
        for label, key in [('ACC', 'acc'), ('AUC', 'auc'), ('EER', 'eer')]:
            observed = 100 * metrics[key]
            expected = float(row[label + '_percent'])
            if abs(observed - expected) > 0.00500000001:
                raise AssertionError(f'{run} {label}: {observed} differs from {expected}')
            result[label + '_percent'] = f'{observed:.2f}'
        result['status'] = 'PASS_SAVED_PREDICTIONS'
        output.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    print(f'PASS: {len(output)} MAF table rows from {len(cache)} complete prediction files; {args.output}')
    print('Baseline rows are recorded/cited results, not recomputed by this command.')


if __name__ == '__main__':
    main()
