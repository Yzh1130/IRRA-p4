"""Small JSON run record with no random operations."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import subprocess

from tools.p4.deg_retrieval import resolve_deg_rng_seed


def _now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


class RunRecord:
    def __init__(self, args):
        self.path = Path(args.output_dir) / "run_record.json"
        try:
            repository = Path(__file__).resolve().parents[3]
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                             cwd=repository,
                                             text=True).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                                cwd=repository, text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            commit = None
            dirty = None
        train_seed = int(getattr(args, "train_seed", 1))
        self.data = {"git_commit": commit, "git_worktree_dirty": dirty,
                     "args": vars(args).copy(),
                     "train_seed": train_seed,
                     "deg_rng_seed": resolve_deg_rng_seed(train_seed, getattr(args, "deg_rng_seed", None)),
                     "deg_lambda": float(getattr(args, "deg_lambda", 1.0)),
                     "loss_names": str(getattr(args, "loss_names", "")),
                     "deg_retrieval_loss": str(getattr(args, "deg_retrieval_loss", "none")),
                     "started_at": _now(), "ended_at": None,
                     "status": "running", "epochs": [], "steps": []}
        self.write()

    def write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False,
                                        default=str) + "\n", encoding="utf-8")

    def add_epoch(self, row):
        self.data["epochs"].append(row)
        self.write()

    def add_step(self, row):
        self.data["steps"].append(row)

    def finish(self, status="completed"):
        self.data["status"] = status
        self.data["ended_at"] = _now()
        self.write()
