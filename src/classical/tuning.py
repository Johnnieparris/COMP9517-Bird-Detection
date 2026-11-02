"""Random search helpers for Method 1. Tuning runs use val or train, never test."""
import dataclasses
from concurrent.futures import ProcessPoolExecutor

from src.classical.pipeline import detect
from src.classical.run import read_image


def sample_config(base, space, rng):
    """Copy base with each field in space set to a uniform choice from its list."""
    return dataclasses.replace(base, **{key: rng.choice(values) for key, values in space.items()})


def _detect_rows(job):
    record, cfg = job
    dets = detect(read_image(record.path), cfg)
    return [{"image_id": record.image_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score}
            for x1, y1, x2, y2, score in dets.tolist()]


def detect_parallel(records, cfg, workers):
    """Same rows, in the same order, as run_split. Too contended to use for timing."""
    with ProcessPoolExecutor(workers) as pool:
        results = pool.map(_detect_rows, [(record, cfg) for record in records], chunksize=8)
        return [row for rows in results for row in rows]
