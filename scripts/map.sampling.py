import sys
import os
import argparse
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Select max line per batch based on probability.")
    parser.add_argument(
        "--th", type=int, default=1024,
        help="Number of lines per batch (default: 1024)"
    )
    args = parser.parse_args()

    th = args.th
    results = {}

    for i, line in enumerate(sys.stdin):

        line, prob = line.split('\t')
        results[float(prob.strip())] = line.strip()

        if (i + 1) % th == 0:
            print(results[max(results)])
            results.clear()  # reset dictionary for next batch

    # 端数の処理（最後のバッチ）
    if results:
        print(results[max(results)])


if __name__ == '__main__':
    main()
