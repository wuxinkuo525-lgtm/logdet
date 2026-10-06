#!/usr/bin/env python
"""Verify a completed run and its exported predictions before reusing DONE."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from logodet.dino_run import validate_final_eval, validate_prediction_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--pred-dir', type=Path, required=True)
    ap.add_argument('--union', type=Path, required=True)
    ap.add_argument('--allow-partial', action='store_true')
    args = ap.parse_args()
    receipt = validate_final_eval(args.run_dir)
    ids = {im['id'] for im in json.loads(args.union.read_text(encoding='utf-8'))['images']}
    man = validate_prediction_bundle(args.pred_dir, None if args.allow_partial else ids,
                                     allow_partial=args.allow_partial)
    if man['final_evaluation'] != receipt:
        raise ValueError('Exported predictions refer to an outdated final evaluation')
    print(f'VERIFIED {args.run_dir.name}', flush=True)


if __name__ == '__main__':
    main()
