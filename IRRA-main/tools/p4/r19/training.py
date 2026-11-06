"""R19 MLM-once runtime recipe derivation; this module never trains a model.

Only a small path-free override specification is committed. The driver supplies
its original server-side base recipe and writes derived recipes under R19 OUT.
"""
import argparse
from collections.abc import Mapping
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[3]
OVERRIDE_PATH = CODE / "configs/p4/r19_mlm1_override.yaml"
NAME_PREFIX = "base_mlm1"
COMMON_ALLOWED = frozenset({"name", "output_dir", "train_seed", "deg_rng_seed", "mlm_single_count"})
RGBNT_ALLOWED = frozenset({"dataset_name", "root_dir", "dev_split_file"})
DATASETS = ("RGBT-PEDES", "RGBNT201-PEDES")


def config_differences(base, derived):
    """Report additions, removals and unequal values without hiding missing keys."""
    return {
        key: {"base": base.get(key), "derived": derived.get(key),
              "missing_base": key not in base, "missing_derived": key not in derived}
        for key in sorted(set(base) | set(derived))
        if key not in base or key not in derived or base[key] != derived[key]
    }


def validate_derivation(base, derived, dataset, smoke=False, historical_root_dir=None):
    if dataset not in DATASETS:
        raise ValueError("Unsupported R19 dataset: " + str(dataset))
    if not isinstance(smoke, bool):
        raise ValueError("smoke must be a boolean")
    allowed = set(COMMON_ALLOWED)
    if dataset == "RGBNT201-PEDES":
        allowed.update(RGBNT_ALLOWED)
    elif historical_root_dir is not None:
        if str(derived.get("root_dir")) != str(historical_root_dir):
            raise ValueError("RGBT root_dir must equal the supplied historical root_dir")
        allowed.add("root_dir")
    if smoke:
        allowed.add("num_epoch")
    changes = config_differences(base, derived)
    forbidden = set(changes) - allowed
    if forbidden:
        raise ValueError("Recipe changed outside the R19 whitelist: " + ", ".join(sorted(forbidden)))
    if derived.get("dataset_name") != dataset:
        raise ValueError("Derived dataset_name does not match the requested dataset")
    if derived.get("mlm_single_count") is not True:
        raise ValueError("The R19 recipe must set mlm_single_count: true")
    if smoke and derived.get("num_epoch") != 2:
        raise ValueError("A smoke recipe must run exactly two epochs")
    return {"allowed_keys": sorted(allowed), "differences": changes,
            "forbidden_differences": {}, "ok": True, "smoke": smoke}


def derive_config(base, dataset, seed, output_dir, root_dir=None,
                  dev_split_file=None, smoke=False, historical_configs=None):
    """Copy a full historical recipe and alter only the authorized R19 keys.

    RGBNT201's dataset/root/split overrides are the same three dataset keys used
    by the R18 L1 command. The RGBT split remains inherited; its data root
    may be replaced only with the common value authenticated by all
    three historical configs. With no historical context the original pure
    helper contract is retained: it can only inherit the original base path.
    The production derive CLI always requires historical context for RGBT.
    Only the separately named smoke recipe may
    change num_epoch. No random source is read or written.
    """
    if not isinstance(base, Mapping):
        raise TypeError("base must be a mapping")
    if dataset not in DATASETS:
        raise ValueError("Unsupported R19 dataset: " + str(dataset))
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(smoke, bool):
        raise ValueError("smoke must be a boolean")
    if not str(output_dir).strip():
        raise ValueError("output_dir must be nonempty")
    derived = copy.deepcopy(dict(base))
    history = load_historical_configs(historical_configs) if historical_configs is not None else None
    if dataset == "RGBT-PEDES":
        if base.get("dataset_name") != dataset:
            raise ValueError("RGBT must use its unchanged RGBT-PEDES base recipe")
        if history is not None:
            if root_dir is None or str(root_dir) != history["historical_root_dir"]:
                raise ValueError("RGBT requires its supplied historical root_dir")
            derived["root_dir"] = str(root_dir)
        for key, value in (("root_dir", root_dir if history is None else None),
                           ("dev_split_file", dev_split_file)):
            if value is not None and str(value) != str(base.get(key, "")):
                raise ValueError("RGBT " + key + " must be inherited from its base recipe")
    else:
        derived["dataset_name"] = dataset
        for key, value in (("root_dir", root_dir), ("dev_split_file", dev_split_file)):
            value = derived.get(key) if value is None else value
            if not str(value or "").strip():
                raise ValueError("RGBNT201 requires " + key)
            derived[key] = str(value)
    stem = NAME_PREFIX + ("_rgbnt" if dataset == "RGBNT201-PEDES" else "")
    derived["name"] = stem + "_s" + str(seed) + ("_smoke" if smoke else "")
    derived["output_dir"] = str(output_dir)
    derived["train_seed"] = seed
    # Same default formula as tools.p4.deg_retrieval.resolve_deg_rng_seed.
    # The clean recipe owns no degraded-branch RNG; the seed is provenance.
    derived["deg_rng_seed"] = seed * 1000 + 7919
    derived["mlm_single_count"] = True
    if smoke:
        derived["num_epoch"] = 2
    validate_derivation(base, derived, dataset, smoke,
                        history["historical_root_dir"] if history and dataset == "RGBT-PEDES" else None)
    return derived


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_historical_configs(paths):
    """Authenticate three distinct saved configs and their common data root.

    Entries may be paths or {path, sha256} records. When a caller supplies a
    digest it is verified, never trusted as a substitute for reading the file.
    Selection of successful checkpoint runs is the driver's responsibility.
    """
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    from utils.config import load_config_mapping
    entries = list(paths)
    if len(entries) != 3:
        raise ValueError("Exactly three historical configs are required")
    records, roots = [], []
    for entry in entries:
        path = Path(entry["path"] if isinstance(entry, Mapping) else entry).resolve()
        digest = _sha256(path)
        if isinstance(entry, Mapping) and entry.get("sha256") != digest:
            raise ValueError("Historical config SHA256 mismatch: " + str(path))
        config = load_config_mapping(path)
        root = config.get("root_dir")
        if not isinstance(root, str) or not root.strip():
            raise ValueError("Historical config has no valid root_dir: " + str(path))
        roots.append(root)
        records.append({"path": str(path), "sha256": digest})
    if len({item["path"] for item in records}) != 3:
        raise ValueError("Historical configs must be three distinct files")
    if len(set(roots)) != 1:
        raise ValueError("The three historical root_dir values disagree")
    return {"historical_root_dir": roots[0], "historical_configs": records}


def _write_check(path, record):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(_json_value(record), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    return record


def _resolved_recipe_seeds(config, seed):
    """Resolve only train.py's two seed assignments; do not fill other keys."""
    result = copy.deepcopy(dict(config))
    train_seed = result.get("train_seed")
    train_seed = seed if train_seed is None else int(train_seed)
    result["train_seed"] = train_seed
    # train.py uses TRAIN_SEED (+ rank 0), then resolve_deg_rng_seed.
    # Kept as the exact integer formula to avoid importing torch for a check.
    value = result.get("deg_rng_seed")
    result["deg_rng_seed"] = train_seed * 1000 + 7919 if value is None else int(value)
    return result


def compare_recipe_history(expected_path, historical_path, seed, output_path=None, smoke=False):
    """Compare every runtime field before training and retain failed checks."""
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    from utils.config import load_config_mapping
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(smoke, bool):
        raise ValueError("smoke must be a boolean")
    expected_path, historical_path = Path(expected_path), Path(historical_path)
    expected_raw = load_config_mapping(expected_path)
    historical_raw = load_config_mapping(historical_path)
    expected = _resolved_recipe_seeds(expected_raw, seed)
    historical = _resolved_recipe_seeds(historical_raw, seed)
    allowed = {"name", "output_dir", "config_file", "mlm_single_count"}
    if smoke:
        allowed.add("num_epoch")
    differences = config_differences(historical, expected)
    forbidden = {key: value for key, value in differences.items() if key not in allowed}
    seed_ok = expected["train_seed"] == historical["train_seed"] == seed
    record = {"expected": str(expected_path.resolve()), "expected_sha256": _sha256(expected_path),
              "historical": str(historical_path.resolve()), "historical_sha256": _sha256(historical_path),
              "seed": seed, "smoke": smoke, "allowed_keys": sorted(allowed),
              "differences": differences, "forbidden_differences": forbidden,
              "resolved_seeds": {"expected": {key: expected[key] for key in ("train_seed", "deg_rng_seed")},
                                 "historical": {key: historical[key] for key in ("train_seed", "deg_rng_seed")}},
              "seed_matches_requested": seed_ok, "ok": not forbidden and seed_ok,
              "training_ran": False}
    _write_check(output_path, record)
    if not record["ok"]:
        raise ValueError("Recipe differs from historical config: " +
                         (", ".join(sorted(forbidden)) if forbidden else "train_seed does not match requested seed"))
    return record


def data_fingerprint(dataset, root_dir, output_path=None):
    """Read only annotation bytes and directory entries; never decode images.

    RGBT resolution executes only its two path resolver methods extracted from
    the actual loader, avoiding package imports that could construct datasets.
    """
    import ast
    import os.path as op
    if dataset not in DATASETS:
        raise ValueError("Unsupported R19 dataset: " + str(dataset))
    root = Path(root_dir)
    if dataset == "RGBT-PEDES":
        source = CODE / "datasets/rgbtpedes.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RGBTPEDES")
        names = {"_resolve_data_dir", "_resolve_anno_path"}
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in methods} != names:
            raise ValueError("RGBT path resolvers are missing")
        namespace = {"op": op}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
        directory = Path(namespace["_resolve_data_dir"](None, str(root / "RGBT-PEDES")))
        annotation = Path(namespace["_resolve_anno_path"](None, str(directory)))
    else:
        source = CODE / "datasets/rgbnt201pedes.py"
        directory = root / "RGBNT201"
        annotation = directory / "captionform0_8B.json"
    rgb, thermal = directory / "RGB", directory / "T"
    missing = [str(path) for path in (annotation, rgb, thermal)
               if not (path.is_file() if path == annotation else path.is_dir())]
    counts = {"RGB": sum(path.is_file() for path in rgb.rglob("*")) if rgb.is_dir() else 0,
              "T": sum(path.is_file() for path in thermal.rglob("*")) if thermal.is_dir() else 0}
    expected_counts = {"RGB": 4723, "T": 4723} if dataset == "RGBT-PEDES" else None
    counts_ok = counts == expected_counts if expected_counts is not None else all(counts.values())
    record = {"dataset": dataset, "root_dir": str(root), "data_dir": str(directory),
              "annotation_path": str(annotation),
              "annotation_sha256": _sha256(annotation) if annotation.is_file() else None,
              "rgb_dir": str(rgb), "t_dir": str(thermal), "counts": counts,
              "expected_counts": expected_counts, "missing": missing,
              "resolver": str(source), "resolver_sha256": _sha256(source),
              "ok": not missing and bool(counts_ok), "images_decoded": False, "training_ran": False}
    _write_check(output_path, record)
    if not record["ok"]:
        raise ValueError("Data fingerprint failed: " + ("missing " + ", ".join(missing) if missing else
                         "RGB/T file counts " + str(counts) + "; expected " + str(expected_counts)))
    return record


def _json_value(value):
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_derived_config(base_path, output_path, dataset, seed, output_dir,
                         root_dir=None, dev_split_file=None, smoke=False,
                         override_path=OVERRIDE_PATH, historical_configs=None):
    """Write a new YAML plus its SHA/difference provenance; never overwrite."""
    import yaml
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    from utils.config import load_config_mapping
    started = time.perf_counter()
    base_path, output_path = Path(base_path), Path(output_path)
    override_path = Path(override_path)
    provenance = output_path.with_suffix(".derivation.json")
    if output_path.exists() or provenance.exists():
        raise FileExistsError("Refusing to overwrite an R19 recipe or its derivation")
    override = yaml.safe_load(override_path.read_text(encoding="utf-8"))
    expected = {"mlm_single_count": True, "name_prefix": NAME_PREFIX,
                "rgbnt201_l1_override_keys": ["dataset_name", "root_dir", "dev_split_file"]}
    if override != expected:
        raise ValueError("The path-free R19 override specification differs from the frozen contract")
    base = load_config_mapping(base_path)
    history = load_historical_configs(historical_configs) if historical_configs is not None else None
    historical_root = history["historical_root_dir"] if history and dataset == "RGBT-PEDES" else None
    derived = derive_config(base, dataset, seed, output_dir, root_dir, dev_split_file, smoke,
                            historical_configs=historical_configs)
    validation = validate_derivation(base, derived, dataset, smoke, historical_root)
    class RecipeDumper(yaml.SafeDumper):
        pass

    # Historical recipes may contain !!python/tuple (not arbitrary objects).
    # Preserve those supported tuple values rather than silently turn them into
    # lists, which would alter fields outside the whitelist after reloading.
    RecipeDumper.add_representer(tuple, lambda dumper, value: dumper.represent_sequence(
        "tag:yaml.org,2002:python/tuple", value))
    body = yaml.dump(derived, Dumper=RecipeDumper, allow_unicode=True, sort_keys=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(body)
    reloaded = load_config_mapping(output_path)
    if config_differences(derived, reloaded):
        raise ValueError("The generated YAML changed recipe values or types on reload")
    validate_derivation(base, reloaded, dataset, smoke, historical_root)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE.parent,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    record = {"base": str(base_path.resolve()), "base_sha256": _sha256(base_path),
              "generated": str(output_path.resolve()), "generated_sha256": _sha256(output_path),
              "override": str(override_path.resolve()), "override_sha256": _sha256(override_path),
              "git_commit": commit, "script_sha256": _sha256(__file__),
              "parameters": {"dataset": dataset, "seed": seed, "output_dir": str(output_dir),
                             "root_dir": root_dir, "dev_split_file": dev_split_file,
                             "smoke": smoke},
              "validation": validation, "elapsed_seconds": time.perf_counter() - started,
              "peak_memory_bytes": 0, "peak_memory_note": "CPU recipe generation; no CUDA allocation"}
    if history is not None:
        record.update(history)
    with provenance.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(_json_value(record), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return record


def runtime_health_base(config_path, output_path, parse_args=None):
    """Materialize train.py's parser-normalized namespace for strict health.

    Parsing adds supported defaults and replaces config_file with the supplied
    recipe path. The only additions outside the parser are train.py's known
    single-process metadata before save_train_configs. There is no dataset,
    model, optimizer, random seeding or training call. This expected namespace
    is not a new recipe and does not enlarge the derivation whitelist.
    """
    import yaml
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    from utils.config import load_config_mapping
    started = time.perf_counter()
    config_path, output_path = Path(config_path), Path(output_path)
    provenance = output_path.with_suffix(".runtime.json")
    if output_path.exists() or provenance.exists():
        raise FileExistsError("Refusing to overwrite a runtime health base")
    recipe = load_config_mapping(config_path)
    if recipe.get("mlm_single_count") is not True:
        raise ValueError("Expected an MLM-once derived recipe")
    if parse_args is None:
        from utils.options import get_args
        parse_args = get_args
    saved_argv = sys.argv
    try:
        sys.argv = ["train.py", "--config_file", str(config_path)]
        namespace = vars(parse_args()).copy()
    finally:
        sys.argv = saved_argv
    # train.py:57-65 assigns these before writing configs.yaml. The R19 driver
    # uses one process and TRAIN_SEED equal to the derived recipe's seed.
    seed = int(recipe["train_seed"])
    namespace["train_seed"] = seed
    namespace["deg_rng_seed"] = (seed * 1000 + 7919 if namespace.get("deg_rng_seed") is None
                                  else int(namespace["deg_rng_seed"]))
    namespace["distributed"] = False
    class RuntimeDumper(yaml.SafeDumper):
        pass
    RuntimeDumper.add_representer(tuple, lambda dumper, value: dumper.represent_sequence(
        "tag:yaml.org,2002:python/tuple", value))
    body = yaml.dump(namespace, Dumper=RuntimeDumper, allow_unicode=True, sort_keys=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(body)
    if config_differences(namespace, load_config_mapping(output_path)):
        raise ValueError("Runtime health YAML changed values or types on reload")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE.parent,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    record = {"recipe": str(config_path.resolve()), "recipe_sha256": _sha256(config_path),
              "expected": str(output_path.resolve()), "expected_sha256": _sha256(output_path),
              "script_sha256": _sha256(__file__), "parser_sha256": _sha256(CODE / "utils/options.py"),
              "fields": len(namespace), "parser_and_pre_save_differences": config_differences(recipe, namespace),
              "training_ran": False, "single_process": True, "seed": seed,
              "git_commit": commit, "parameters": {"config": str(config_path), "output": str(output_path)},
              "elapsed_seconds": time.perf_counter() - started, "peak_memory_bytes": 0,
              "peak_memory_note": "CPU argument parsing; no CUDA allocation"}
    with provenance.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(_json_value(record), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    derive = sub.add_parser("derive", help="derive an MLM-once recipe without running training")
    derive.add_argument("--base", required=True)
    derive.add_argument("--dataset", choices=DATASETS, required=True)
    derive.add_argument("--seed", type=int, required=True)
    derive.add_argument("--output_dir", "--output-dir", dest="output_dir", required=True)
    derive.add_argument("--root_dir", "--root-dir", dest="root_dir")
    derive.add_argument("--dev_split_file", "--dev-split-file", dest="dev_split_file")
    derive.add_argument("--output", required=True)
    derive.add_argument("--override", default=str(OVERRIDE_PATH))
    derive.add_argument("--historical_configs", nargs=3,
                        help="three successful saved configs authenticating the RGBT data root")
    derive.add_argument("--smoke", action="store_true")
    expected = sub.add_parser("runtime-health-base", help="parse the derived recipe for strict health; no training")
    expected.add_argument("--config", required=True)
    expected.add_argument("--output", required=True)
    compare = sub.add_parser("compare-history", help="check expected runtime recipe against its saved historical run")
    compare.add_argument("--expected", required=True)
    compare.add_argument("--historical", required=True)
    compare.add_argument("--seed", type=int, required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--smoke", action="store_true")
    fingerprint = sub.add_parser("data-fingerprint", help="check annotation hash and RGB/T file counts without loading images")
    fingerprint.add_argument("--dataset", choices=DATASETS, required=True)
    fingerprint.add_argument("--root_dir", required=True)
    fingerprint.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == "runtime-health-base":
            record = runtime_health_base(args.config, args.output)
        elif args.operation == "compare-history":
            record = compare_recipe_history(args.expected, args.historical, args.seed, args.output, args.smoke)
        elif args.operation == "data-fingerprint":
            record = data_fingerprint(args.dataset, args.root_dir, args.output)
        else:
            if args.dataset == "RGBT-PEDES" and (args.historical_configs is None or args.root_dir is None):
                raise ValueError("RGBT derive requires --root_dir and --historical_configs from successful runs")
            record = write_derived_config(args.base, args.output, args.dataset, args.seed,
                                          args.output_dir, args.root_dir, args.dev_split_file,
                                          args.smoke, args.override, args.historical_configs)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
