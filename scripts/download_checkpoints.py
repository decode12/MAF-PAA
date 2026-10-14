#!/usr/bin/env python3
"""Download only published, checksum-verified MAF model artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='all')
    parser.add_argument('--output', type=Path, default=Path('checkpoints'))
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[1] / 'checkpoints/manifest.json'
    records = json.loads(manifest.read_text())
    records = [r for r in records if args.name == 'all' or r['name'] == args.name]
    if not records:
        parser.error('Unknown checkpoint name')
    args.output.mkdir(parents=True, exist_ok=True)
    for record in records:
        destination = args.output / record['file']
        if destination.exists():
            if sha256(destination) != record['sha256']:
                raise ValueError(f'Checksum mismatch: {destination}')
            print(f'Verified {destination}')
            continue
        if not record.get('url'):
            raise RuntimeError(f"{record['name']}: no public URL has been verified. "
                               'See checkpoints/README.md. Do not substitute another model.')
        temporary = destination.with_suffix('.download')
        urllib.request.urlretrieve(record['url'], temporary)
        if sha256(temporary) != record['sha256']:
            temporary.unlink()
            raise ValueError(f'Download checksum mismatch: {destination.name}')
        temporary.replace(destination)
        print(f'Downloaded and verified {destination}')


if __name__ == '__main__':
    main()
