"""Regenerate the k-means split of IJCNN1 (or of a synthetic dataset) and compare it with the tracked partition.

    python -m tools.kmeans_partition            (from the repository root, PYTHONPATH=.; about two minutes)
    python -m tools.kmeans_partition --write    rewrite data/ijcnn1-32-kmeans-partition.json from the regeneration
    python -m tools.kmeans_partition --dataset synthetic-kappa10 [--write]
                                                the same for data/synthetic-kappa10-kmeans-partition.json (seconds)

Without --write nothing is written: the exit status is 0 if the regenerated partition equals the tracked one (same SHA-256),
1 if not (or if none is tracked yet). The datasets and figures use the tracked partition, never a regeneration; a different
result on another machine (BLAS and SIMD differ in the last bits) does not change them.
"""

import sys

from adafed import data, generate


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    base = argv[argv.index("--dataset") + 1] if "--dataset" in argv else generate.DIRICHLET_BASE
    path = generate.KMEANS_PARTITION if base == generate.DIRICHLET_BASE else generate.kmeans_partition_path(base)
    result = generate.regenerate_kmeans(data.RAW, base=base)
    zs = [r["zeta2"] for r in result["info"]["restarts"]]
    print(f"{generate.kmeans_name(base)} regenerated: SHA-256 {result['sha256']}, restart {result['info']['best']} of "
          f"{len(zs)} kept, zeta^2 at x* {max(zs):.10g} (worst restart {min(zs):.10g})")
    print(f"tracked:     SHA-256 {result['tracked_sha256']}")
    print("equal" if result["equal"] else "DIFFERENT")
    if "--write" in argv:
        path.write_text(generate.kmeans_partition_text(result["partition"], result["info"], base))
        print(f"wrote {path}")
    return 0 if result["equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
