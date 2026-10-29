"""Regenerate the default split of IJCNN1 (ijcnn1-32) and compare it with the tracked partition.

    python -m tools.default_partition            (from the repository root, PYTHONPATH=.; a few seconds)
    python -m tools.default_partition --write    rewrite data/ijcnn1-32-partition.json from the regeneration

Without --write nothing is written: the exit status is 0 if the regenerated partition equals the tracked one (same SHA-256),
1 if not. The datasets and figures use the tracked partition, never a regeneration: the construction sorts the rows by a
floating-point margin, so another CPU or BLAS can order it differently (the tracked partition is the one every default-split
result used).
"""

import sys

from adafed import data, generate


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    result = generate.regenerate_default(data.RAW)
    print(f"regenerated: SHA-256 {result['sha256']}")
    print(f"tracked:     SHA-256 {result['tracked_sha256']}")
    print("equal" if result["equal"] else "DIFFERENT")
    if "--write" in argv:
        generate.DEFAULT_PARTITION.write_text(
            generate.default_partition_text(result["partition"], result["lambda_max"], result["reference_kkt"]))
        print(f"wrote {generate.DEFAULT_PARTITION}")
    return 0 if result["equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
