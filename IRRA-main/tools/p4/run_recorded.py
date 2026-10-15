"""Record immutable CPU protocol scripts without changing their source bytes.

Example: python -B tools/p4/run_recorded.py --script calib_protocol
  --record_file OUT/tool_run_record.json -- --feats_dir FEATS ... --out_dir OUT
The record directory is created after the child finishes, so exist_ok=False in
calib_protocol is respected. This wrapper never overwrites its own records.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = {"calib_protocol": "analysis/calib_protocol.py", "pack_feats": "pack_feats.py",
           "verify_joint_loss": "analysis/verify_joint_loss.py"}


def effective_parameters(script, arguments):
    """Read literal argparse declarations; do not import or execute the script."""
    parser = argparse.ArgumentParser(add_help=False)
    types = {"int": int, "float": float, "str": str}
    tree = ast.parse(Path(script).read_bytes())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        names = [ast.literal_eval(value) for value in node.args]
        options = {}
        for keyword in node.keywords:
            if keyword.arg == "type":
                options["type"] = types[keyword.value.id]
            else:
                options[keyword.arg] = ast.literal_eval(keyword.value)
        parser.add_argument(*names, **options)
    if "--help" in arguments or "-h" in arguments:
        return {"help_only": True, "defaults": vars(parser.parse_args([], namespace=argparse.Namespace()))} if not any(
            action.required for action in parser._actions) else {"help_only": True}
    return vars(parser.parse_args(arguments))


def run_recorded(script_name, record_file, arguments):
    script = Path(__file__).resolve().parent / SCRIPTS[script_name]
    record_file = Path(record_file).resolve()
    stdout_file = record_file.with_name(record_file.stem + "_stdout.txt")
    stderr_file = record_file.with_name(record_file.stem + "_stderr.txt")
    for path in (record_file, stdout_file, stderr_file):
        if path.exists():
            raise FileExistsError(f"Record output already exists: {path}")
    parameters = effective_parameters(script, arguments)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=CODE_ROOT, capture_output=True, text=True, check=True).stdout
    seeds = {key: value for key, value in parameters.items() if "seed" in key}
    if script_name == "calib_protocol":
        seeds.update(bootstrap_seed=2026, rank_seed_formula="init_seed + 97*split_index + fold", m4_seed_formula="assign_seed + 40000")
    elif script_name == "verify_joint_loss":
        seeds.update(generator_seed=20261015, unit_vector_encoder_seed=7)
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8")
    command = [sys.executable, "-B", str(script), *arguments]
    result = subprocess.run(command, capture_output=True, env=env)
    elapsed = time.perf_counter() - start
    record_file.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation and an early existence check protect earlier records.
    with stdout_file.open("xb") as handle:
        handle.write(result.stdout)
    with stderr_file.open("xb") as handle:
        handle.write(result.stderr)
    record = {"git_commit": commit, "git_status_porcelain": dirty, "script": str(script),
              "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
              "argv": command, "parameters": parameters, "seeds": seeds,
              "started_at_utc": started, "elapsed_seconds": elapsed,
              "peak_cuda_memory_bytes": 0, "device": "cpu",
              "memory_note": "These three immutable scripts use CPU tensors; CUDA is hidden in this child only",
              "returncode": result.returncode, "stdout": str(stdout_file), "stderr": str(stderr_file)}
    with record_file.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    sys.stdout.write(result.stdout.decode("utf-8", errors="replace"))
    sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))
    return result.returncode


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", choices=tuple(SCRIPTS), required=True)
    parser.add_argument("--record_file", required=True, help="New record JSON; its parent is created after execution")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Script arguments after --")
    return parser


if __name__ == "__main__":
    cli = build_parser().parse_args()
    arguments = cli.arguments[1:] if cli.arguments[:1] == ["--"] else cli.arguments
    raise SystemExit(run_recorded(cli.script, cli.record_file, arguments))
