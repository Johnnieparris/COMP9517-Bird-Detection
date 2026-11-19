"""Build the 15 distorted dataset roots for the robustness study (Member E, Study A).

    python scripts/make_distorted_roots.py               # all 15 variants, seed 0
    python scripts/make_distorted_roots.py --only noise_s1 blur_s3

Each root data/distorted/<type>_s<level>/ holds distorted val and test images (for
tuning mitigations and for reporting, respectively) and the clean train split
(hard-linked), which M2 retrains on. Labels are never changed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.dataset import DEFAULT_ROOT  # noqa: E402
from src.robustness.distortions import DISTORTIONS, apply_distortion, rng_for, variant_name  # noqa: E402
from src.robustness.roots import DISTORTED_DIR, build_root  # noqa: E402


def all_variants():
    return [(name, s) for name in DISTORTIONS for s in (1, 2, 3)]


def build_variant(name, severity, seed):
    tag = variant_name(name, severity)

    def transform(image, boxes, image_id):
        return apply_distortion(image, boxes, name, severity, rng_for(seed, image_id, tag))

    dst = DISTORTED_DIR / tag
    if dst.exists():
        return f"exists, skipped: {dst.name}"
    build_root(DEFAULT_ROOT, dst, {"train": None, "val": transform, "test": transform},
               info={"distortion": name, "severity": severity,
                     "parameter": DISTORTIONS[name][severity - 1], "seed": seed})
    return f"built {dst.name}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="*", help="variant names, e.g. noise_s1")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    jobs = [v for v in all_variants() if not args.only or variant_name(*v) in args.only]
    with ProcessPoolExecutor(args.workers) as pool:
        for msg in pool.map(build_variant, *zip(*jobs), [args.seed] * len(jobs)):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
