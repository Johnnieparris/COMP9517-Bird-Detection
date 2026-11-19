"""Build preprocessed roots for the classical-method mitigation (Member E, Study A).

    python scripts/make_preprocessed_roots.py

Reads the setting chosen on val (results/robustness_preprocess_choice.json) and writes:
    data/preprocessed/clean/        all three clean splits preprocessed (M2 retrains on train)
    data/preprocessed/<variant>/    the distorted TEST split preprocessed; train and val are
                                    hard-linked from data/preprocessed/clean (val there is only
                                    read for the clean-val threshold, never for a variant)
"""
from concurrent.futures import ProcessPoolExecutor
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.dataset import DEFAULT_ROOT  # noqa: E402
from src.robustness.distortions import DISTORTIONS, variant_name  # noqa: E402
from src.robustness.preprocess import preprocess  # noqa: E402
from src.robustness.roots import DISTORTED_DIR, build_root  # noqa: E402

OUT = REPO / "data/preprocessed"
CHOICE = json.loads((REPO / "results/robustness_preprocess_choice.json").read_text())


def transform(image, boxes, image_id):
    return preprocess(image, CHOICE["setting"])


def build_variant(variant):
    dst = OUT / variant
    if dst.exists():
        return f"exists, skipped: {variant}"
    build_root(DISTORTED_DIR / variant, dst, {"train": None, "val": None, "test": transform},
               split_sources={"train": OUT / "clean", "val": OUT / "clean"},
               info={"preprocess": CHOICE["chosen"], "setting": CHOICE["setting"], "variant": variant})
    return f"built {variant}"


def main():
    if CHOICE["chosen"] == "none":
        print("Tuning chose 'none': no preprocessed roots needed (report this as a finding).")
        return
    if not (OUT / "clean").exists():
        build_root(DEFAULT_ROOT, OUT / "clean", {"train": transform, "val": transform, "test": transform},
                   info={"preprocess": CHOICE["chosen"], "setting": CHOICE["setting"], "variant": "clean"})
        print("built clean")
    variants = [variant_name(n, s) for n in DISTORTIONS for s in (1, 2, 3)]
    with ProcessPoolExecutor(6) as pool:
        for msg in pool.map(build_variant, variants):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
