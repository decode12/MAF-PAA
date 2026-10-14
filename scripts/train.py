#!/usr/bin/env python3
"""Train MAF on train/validation data only; see --help for portable overrides."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maf.engine import main
if __name__ == '__main__':
    sys.argv.insert(1, 'train')
    main()
