"""Print the path of every degraded evaluation input, one per line.

Uses the same discovery as evaluate_nii_test.py (dataset.classify_file), so
the baseline enhancers (enhance_swinir.sh / enhance_realesrgan.sh) process
exactly the volumes the flow model is scored on: artifact siblings of raw
stems that have both ground truths, never the clean GT, md target, or d*
denoised volumes.

Usage:
    python list_test_inputs.py [--data_root /vast/tibrahim/jil202/nii_test]
                               [--anatomy brain ...] [--acquisition ACQ ...]
"""

import argparse

from evaluate_nii_test import discover_subjects


def input_paths(data_root, anatomies=None, acquisitions=None):
    """Paths of all degraded inputs, in discover_subjects order."""
    return [i["path"]
            for s in discover_subjects(data_root, anatomies, acquisitions)
            for i in s["inputs"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/vast/tibrahim/jil202/nii_test")
    p.add_argument("--anatomy", nargs="+", default=["brain"])
    p.add_argument("--acquisition", nargs="+", default=None)
    args = p.parse_args()
    for path in input_paths(args.data_root, set(args.anatomy or []),
                            set(args.acquisition or [])):
        print(path)


if __name__ == "__main__":
    main()
