"""Nested training subsets for the data-efficiency study (Member E, Study B)."""
import numpy as np


def nested_subsets(ids, fractions=(0.5, 0.25, 0.1), seed=0):
    """One seeded permutation of ids; fraction f takes its first round(f * n) ids.

    Smaller subsets are therefore contained in larger ones (10% in 25% in 50%), so the
    data-efficiency curve changes only the amount of data, not which images are seen.
    """
    order = list(np.random.default_rng(seed).permutation(sorted(ids)))
    return {f: [str(i) for i in order[:int(round(f * len(order)))]] for f in fractions}
