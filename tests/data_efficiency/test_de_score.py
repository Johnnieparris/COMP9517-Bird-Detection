import json

from src.data_efficiency.score import run_summary


def _write(run_dir, status="completed"):
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({
        "status": status, "best_epoch": 2, "completed_epochs": 3, "dataset": {"train_images": 137}}))
    (run_dir / "history.json").write_text(json.dumps([
        {"epoch": e, "elapsed_s": 10.0 * e, "scores": {"mAP50": m}}
        for e, m in ((1, 0.5), (2, 0.7), (3, 0.6))]))


def test_run_summary_reads_run_dir(tmp_path):
    _write(tmp_path / "e_de_10")
    assert run_summary(tmp_path / "e_de_10") == {
        "run": "de_10", "status": "completed", "train_images": 137, "epochs": 3, "best_epoch": 2,
        "val_mAP50": 0.7, "train_time_s": 30.0}


def test_run_summary_marks_unfinished_run(tmp_path):
    _write(tmp_path / "e_ln_drop20", status="training")
    assert run_summary(tmp_path / "e_ln_drop20")["status"] == "training"
