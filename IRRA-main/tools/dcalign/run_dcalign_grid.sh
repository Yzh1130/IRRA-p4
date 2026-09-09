#!/usr/bin/env bash
# Stage 4 DCAlign grid. The 11 protocol scenarios use last.pth. best.pth is
# retained and evaluated only once on clean for comparison with R1=58.47.
# This runner never deletes DCAlign training artifacts or checkpoints.

set -uo pipefail
IFS=$'\n\t'
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DCALIGN_ROOT="${DCALIGN_ROOT:-/root/autodl-tmp/code/RGBT-PEDE}"
BASE_CONFIG="${BASE_CONFIG:-}"
OUT="${OUT:-/root/autodl-tmp/DCALIGN_GRID}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-25}"
VALIDATE_ONLY=0

usage() {
    printf '%s\n' \
        "Usage: $0 --base-config PATH [--dcalign-root PATH] [--out PATH]" \
        "          [--python PATH] [--validate-only]" \
        "" \
        "Runs DC-add and DC-dyn with seeds 101..105. Each run evaluates" \
        "11 scenarios on last.pth and clean once on best.pth." \
        "--validate-only performs startup/parity/CLI/command checks and exits" \
        "without training, evaluation, or shutdown."
}

while (($#)); do
    case "$1" in
        --base-config) (($# >= 2)) || exit 2; BASE_CONFIG="$2"; shift 2 ;;
        --dcalign-root) (($# >= 2)) || exit 2; DCALIGN_ROOT="$2"; shift 2 ;;
        --out) (($# >= 2)) || exit 2; OUT="$2"; shift 2 ;;
        --python) (($# >= 2)) || exit 2; PYTHON_BIN="$2"; shift 2 ;;
        --validate-only) VALIDATE_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'error: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$OUT" && -n "$DCALIGN_ROOT" ]] || exit 2
OUT="$(realpath -m "$OUT")"
DCALIGN_ROOT="$(realpath -m "$DCALIGN_ROOT")"
[[ -z "$BASE_CONFIG" ]] || BASE_CONFIG="$(realpath -m "$BASE_CONFIG")"
mkdir -p "$OUT"/{runs,sim,fragments,marks,logs} || exit 2

RUNS_DIR="$OUT/runs"
SIM_DIR="$OUT/sim"
FRAGMENTS_DIR="$OUT/fragments"
MARKS_DIR="$OUT/marks"
LOGS_DIR="$OUT/logs"
PROGRESS_TSV="$OUT/progress.tsv"
METRICS_TSV="$OUT/metrics.tsv"
FAILED_TSV="$OUT/failed.tsv"
WARNINGS_TSV="$OUT/warnings.tsv"
DRIVER_LOG="$LOGS_DIR/driver.log"
exec > >(tee -a "$DRIVER_LOG") 2>&1

now() { date '+%Y-%m-%dT%H:%M:%S%z'; }
log() { printf '[%s] %s\n' "$(now)" "$*"; }
sanitize_tsv() {
    local value="$1"
    value="${value//$'\t'/ }"; value="${value//$'\r'/ }"; value="${value//$'\n'/ }"
    printf '%s' "$value"
}
ensure_tsv_header() {
    local path="$1" header="$2"
    [[ -s "$path" ]] || printf '%b\n' "$header" > "$path"
}
ensure_tsv_header "$PROGRESS_TSV" 'timestamp\tgroup\tseed\tstatus\telapsed_seconds\tremaining_gb\trun_dir'
ensure_tsv_header "$METRICS_TSV" 'group\tseed\tscenario\tmode\tstrength\tkernel\tdseed\tsplit\tR1\tR5\tR10\tmAP\tmINP'
ensure_tsv_header "$FAILED_TSV" 'timestamp\tgroup\tseed\tstage\texit_code\treason'
ensure_tsv_header "$WARNINGS_TSV" 'timestamp\tgroup\tseed\tcheckpoint\tobserved_R1\treference_R1\tabsolute_difference\treason'

append_failed() {
    local group="$1" seed="$2" stage="$3" exit_code="$4" reason
    reason="$(sanitize_tsv "$5")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(now)" "$group" "$seed" \
        "$stage" "$exit_code" "$reason" >> "$FAILED_TSV"
    log "[FAILED] group=$group seed=$seed stage=$stage exit=$exit_code reason=$reason"
}
remaining_kb() { df -Pk "$OUT" | awk 'NR == 2 {print $4}'; }
remaining_gb() {
    local kb; kb="$(remaining_kb)" || return 1
    awk -v kb="$kb" 'BEGIN {printf "%.2f", kb / 1024 / 1024}'
}
has_minimum_disk() {
    local kb required; kb="$(remaining_kb)" || return 1
    required=$((MIN_FREE_GB * 1024 * 1024))
    [[ "$kb" =~ ^[0-9]+$ ]] && ((kb >= required))
}
compute_irra_runtime_kit_sha() {
    local relative path digest manifest=""
    for relative in "${IRRA_RUNTIME_KIT_FILES[@]}"; do
        path="$PROJECT_ROOT/$relative"
        [[ -f "$path" ]] || return 1
        digest="$(sha256sum "$path" | awk '{print $1}')" || return 1
        [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || return 1
        manifest+="${relative}=${digest}"$'\n'
    done
    printf '%s' "$manifest" | sha256sum | awk '{print $1}'
}
assert_runtime_inputs_unchanged() {
    local actual_base_sha actual_kit_sha
    actual_base_sha="$(sha256sum "$BASE_CONFIG" | awk '{print $1}')" || return 1
    actual_kit_sha="$(compute_irra_runtime_kit_sha)" || return 1
    [[ "$actual_base_sha" == "$BASE_CONFIG_SHA256" \
        && "$actual_kit_sha" == "$IRRA_KIT_SHA256" ]]
}
on_signal() {
    append_failed GLOBAL - signal 130 "interrupted; shutdown cancelled"
    log "Interrupted; checkpoints and partial outputs retained."
    exit 130
}
trap on_signal INT TERM
GLOBAL_START="$(date +%s)"

ARM_NAMES=(DC-add DC-dyn)
SEEDS=(101 102 103 104 105)
EXPECTED_DCALIGN_HEAD="1ca105f668fb71352d888a6c1f37e0b382461d0d"
EXPECTED_DCALIGN_DIRTY_PATHS=(
    datasets/bases.py
    model/build.py
    processor/processor.py
    test.py
    train.py
    utils/metrics.py
    utils/options.py
)
IRRA_RUNTIME_KIT_FILES=(
    tools/dcalign/run_dcalign_grid.sh
    tools/dcalign/eval_dcalign_degraded.py
    tools/dcalign/eval_degradation.py
    tools/dcalign/check_degradation_parity.py
    utils/metrics.py
    utils/eval_degradation.py
)
CORE_SCENARIOS=(
    'clean|none|1.0|7|2026'
    'rgb_missing|rgb_missing|1.0|7|2026'
    'ir_missing|ir_missing|1.0|7|2026'
    'rgb_noise_005|rgb_noise|0.05|7|2026'
    'rgb_noise_010|rgb_noise|0.10|7|2026'
    'ir_noise_005|ir_noise|0.05|7|2026'
    'ir_noise_010|ir_noise|0.10|7|2026'
    'rgb_blur_k7|rgb_blur|1.0|7|2026'
    'ir_blur_k7|ir_blur|1.0|7|2026'
    'both_noise_005|both_noise|0.05|7|2026'
    'both_blur_k7|both_blur|1.0|7|2026'
)
# Array audit: ARM_NAMES, SEEDS, EXPECTED_DCALIGN_DIRTY_PATHS,
# IRRA_RUNTIME_KIT_FILES, CORE_SCENARIOS,
# required_external_files, dcalign_dirty_paths, untracked_python_files,
# seen_seeds, seen_scenarios, command, matches, train_command, eval_command,
# score_sources, score_targets, last_eval_command, best_eval_command,
# pending_score_sources, pending_score_targets and publish_arguments are not
# Bash special variables. BASH_SOURCE/PIPESTATUS are read only; IFS is
# intentional; GROUPS is never used.

[[ ${#ARM_NAMES[@]} -eq 2 && "${ARM_NAMES[0]}" == DC-add \
    && "${ARM_NAMES[1]}" == DC-dyn ]] || exit 2
[[ ${#SEEDS[@]} -eq 5 && ${#CORE_SCENARIOS[@]} -eq 11 ]] || exit 2
declare -A seen_seeds=() seen_scenarios=()
for seed in "${SEEDS[@]}"; do
    [[ "$seed" =~ ^[1-9][0-9]*$ && -z "${seen_seeds[$seed]:-}" ]] || exit 2
    seen_seeds["$seed"]=1
done
for scenario_spec in "${CORE_SCENARIOS[@]}"; do
    IFS='|' read -r scenario mode strength kernel dseed <<< "$scenario_spec"
    [[ -n "$scenario" && -n "$mode" && -n "$strength" && -n "$kernel" \
        && -n "$dseed" && -z "${seen_scenarios[$scenario]:-}" ]] || exit 2
    seen_scenarios["$scenario"]=1
done
[[ "$MIN_FREE_GB" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$AUTO_SHUTDOWN" == 0 || "$AUTO_SHUTDOWN" == 1 ]] || exit 2
[[ -f "$BASE_CONFIG" ]] || {
    append_failed GLOBAL - startup 2 "existing --base-config is required"; exit 2;
}
BASE_CONFIG_SHA256="$(sha256sum "$BASE_CONFIG" | awk '{print $1}')" || exit 2
[[ "$BASE_CONFIG_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
"$PYTHON_BIN" - "$BASE_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8") as handle:
    data = yaml.load(handle, Loader=yaml.FullLoader)
if not isinstance(data, dict):
    raise SystemExit(f"saved config is not a mapping: {path}")
if bool(data.get("dcalign_dynamic_fusion", False)):
    raise SystemExit(
        "base config must not pre-enable dcalign_dynamic_fusion; "
        "only the DC-dyn command may enable it"
    )
if bool(data.get("resume", False)):
    raise SystemExit("base config must set resume=false for independent grid seeds")
if "training" in data and not bool(data["training"]):
    raise SystemExit("base config must not disable training")
print("[BASE_CONFIG_ASSERT] dcalign_dynamic_fusion=false status=PASS")
PY
rc=$?
((rc == 0)) || {
    append_failed GLOBAL - base_config_assert "$rc" \
        "base config does not satisfy the additive-fusion baseline"
    exit "$rc"
}

required_external_files=(train.py model/build.py model/feature_fusion.py utils/options.py)
for rel in "${required_external_files[@]}"; do
    [[ -f "$DCALIGN_ROOT/$rel" ]] || {
        append_failed GLOBAL - dcalign_root_assert 2 "missing $DCALIGN_ROOT/$rel"; exit 2;
    }
done
for rel in eval_dcalign_degraded.py check_degradation_parity.py enable_dynamic_fusion.patch; do
    [[ -f "$SCRIPT_DIR/$rel" ]] || {
        append_failed GLOBAL - kit_assert 2 "missing $SCRIPT_DIR/$rel"; exit 2;
    }
done
IRRA_KIT_SHA256="$(compute_irra_runtime_kit_sha)" || {
    append_failed GLOBAL - runtime_kit_assert 2 \
        "could not hash the pinned IRRA runtime-kit files"; exit 2;
}
[[ "$IRRA_KIT_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
log "[IRRA_RUNTIME_KIT_STATE] manifest_sha256=$IRRA_KIT_SHA256 files=${#IRRA_RUNTIME_KIT_FILES[@]} base_config_sha256=$BASE_CONFIG_SHA256"
git -C "$DCALIGN_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    append_failed GLOBAL - source_state_assert 2 "$DCALIGN_ROOT is not a Git worktree"; exit 2;
}
DCALIGN_HEAD="$(git -C "$DCALIGN_ROOT" rev-parse HEAD)" || exit 2
[[ "$DCALIGN_HEAD" == "$EXPECTED_DCALIGN_HEAD" ]] || {
    append_failed GLOBAL - source_state_assert 2 \
        "DCAlign HEAD mismatch expected=$EXPECTED_DCALIGN_HEAD actual=$DCALIGN_HEAD"
    exit 2
}
declare -a dcalign_dirty_paths=() untracked_python_files=()
mapfile -d '' -t dcalign_dirty_paths < <(
    if git -C "$DCALIGN_ROOT" diff --name-only -z HEAD --; then
        printf '__GIT_STATUS_OK__\0'
    else
        printf '__GIT_STATUS_FAIL__\0'
    fi
)
(( ${#dcalign_dirty_paths[@]} >= 1 )) || exit 2
dcalign_dirty_status="${dcalign_dirty_paths[-1]}"
unset 'dcalign_dirty_paths[-1]'
[[ "$dcalign_dirty_status" == __GIT_STATUS_OK__ ]] || {
    append_failed GLOBAL - source_state_assert 2 \
        "git diff --name-only failed for DCAlign"; exit 2;
}
[[ "$(printf '%s\n' "${dcalign_dirty_paths[@]}")" \
    == "$(printf '%s\n' "${EXPECTED_DCALIGN_DIRTY_PATHS[@]}")" ]] || {
    append_failed GLOBAL - source_state_assert 2 \
        "DCAlign tracked dirty paths differ from the pinned seven-file adaptation"
    printf 'expected dirty paths:\n%s\nactual dirty paths:\n%s\n' \
        "$(printf '%s\n' "${EXPECTED_DCALIGN_DIRTY_PATHS[@]}")" \
        "$(printf '%s\n' "${dcalign_dirty_paths[@]}")" >&2
    exit 2
}
mapfile -d '' -t untracked_python_files < <(
    if git -C "$DCALIGN_ROOT" ls-files --others --exclude-standard -z -- '*.py'; then
        printf '__GIT_STATUS_OK__\0'
    else
        printf '__GIT_STATUS_FAIL__\0'
    fi
)
(( ${#untracked_python_files[@]} >= 1 )) || exit 2
untracked_python_status="${untracked_python_files[-1]}"
unset 'untracked_python_files[-1]'
[[ "$untracked_python_status" == __GIT_STATUS_OK__ ]] || {
    append_failed GLOBAL - source_state_assert 2 \
        "git ls-files failed for DCAlign"; exit 2;
}
((${#untracked_python_files[@]} == 0)) || {
    append_failed GLOBAL - source_state_assert 2 \
        "untracked Python files could alter DCAlign imports"
    printf 'untracked Python: %s\n' "${untracked_python_files[@]}" >&2
    exit 2
}
DCALIGN_DIFF_SHA256="$(
    git -C "$DCALIGN_ROOT" diff --binary --full-index --no-ext-diff HEAD -- \
        | sha256sum | awk '{print $1}'
)" || exit 2
[[ "$DCALIGN_DIFF_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
log "[DCALIGN_SOURCE_STATE] head=$DCALIGN_HEAD tracked_diff_sha256=$DCALIGN_DIFF_SHA256 dirty_paths=${#dcalign_dirty_paths[@]}"
grep -Fq dcalign_dynamic_fusion "$DCALIGN_ROOT/model/build.py" || {
    append_failed GLOBAL - patch_assert 2 "model/build.py patch marker absent"; exit 2;
}
grep -Fq dcalign_dynamic_fusion "$DCALIGN_ROOT/utils/options.py" || {
    append_failed GLOBAL - patch_assert 2 "utils/options.py patch marker absent"; exit 2;
}
grep -Eq "_modules\\[['\"]fuse['\"]\\]" "$DCALIGN_ROOT/model/build.py" || {
    append_failed GLOBAL - patch_assert 2 "self._modules['fuse'] path absent"; exit 2;
}

# Public commit 1ca105f uses fixed seed=1, forces CUDA device 1, and has no
# config-file option. Require the server adaptations instead of guessing.
grep -Fq TRAIN_SEED "$DCALIGN_ROOT/train.py" || {
    append_failed GLOBAL - seed_source_assert 2 "train.py does not consume TRAIN_SEED"; exit 2;
}
"$PYTHON_BIN" - "$DCALIGN_ROOT/train.py" <<'PY'
import ast, sys
from pathlib import Path
tree = ast.parse(Path(sys.argv[1]).read_text(encoding="utf-8"))
forced = []

def dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None

def target_names(node):
    if isinstance(node, (ast.Tuple, ast.List)):
        return [name for item in node.elts for name in target_names(item)]
    name = dotted_name(node)
    return [name] if name else []

def depends_on_train_seed(node, tainted):
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and child.value == "TRAIN_SEED":
            return True
        name = dotted_name(child)
        if name in tainted:
            return True
    return False

def calls_in(node):
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and (dotted_name(child.func) or "").split(".")[-1] == "set_seed"
    ]

bound_seed_calls = []
unbound_seed_calls = []

def check_calls(node, tainted):
    for call in calls_in(node):
        arguments = list(call.args) + [kw.value for kw in call.keywords]
        if arguments and any(depends_on_train_seed(arg, tainted) for arg in arguments):
            bound_seed_calls.append(call.lineno)
        else:
            unbound_seed_calls.append(call.lineno)

def assign_targets(targets, value, tainted):
    names = [name for target in targets for name in target_names(target)]
    is_tainted = depends_on_train_seed(value, tainted)
    for name in names:
        if is_tainted:
            tainted.add(name)
        else:
            # Ordered kill is essential: ``seed = getenv(...); seed = 1`` must
            # not make a later ``set_seed(seed)`` look environment-bound.
            tainted.discard(name)

def scan_statements(statements, initial_taint):
    tainted = set(initial_taint)
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(statement, ast.Assign):
            check_calls(statement.value, tainted)
            assign_targets(statement.targets, statement.value, tainted)
        elif isinstance(statement, ast.AnnAssign):
            if statement.value is not None:
                check_calls(statement.value, tainted)
                assign_targets([statement.target], statement.value, tainted)
            else:
                for name in target_names(statement.target):
                    tainted.discard(name)
        elif isinstance(statement, ast.AugAssign):
            check_calls(statement.value, tainted)
            names = target_names(statement.target)
            remains_tainted = any(name in tainted for name in names) \
                or depends_on_train_seed(statement.value, tainted)
            for name in names:
                if remains_tainted:
                    tainted.add(name)
                else:
                    tainted.discard(name)
        elif isinstance(statement, ast.If):
            check_calls(statement.test, tainted)
            body_state = scan_statements(statement.body, tainted)
            else_state = scan_statements(statement.orelse, tainted) \
                if statement.orelse else set(tainted)
            tainted = body_state & else_state
        elif isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
            if isinstance(statement, (ast.For, ast.AsyncFor)):
                check_calls(statement.iter, tainted)
                loop_state = set(tainted)
                for name in target_names(statement.target):
                    loop_state.discard(name)
            else:
                check_calls(statement.test, tainted)
                loop_state = set(tainted)
            body_state = scan_statements(statement.body, loop_state)
            else_state = scan_statements(statement.orelse, tainted) \
                if statement.orelse else set(tainted)
            # A loop may execute zero times, so only facts true before and
            # after every represented path survive.
            tainted &= body_state & else_state
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            for item in statement.items:
                check_calls(item.context_expr, tainted)
                if item.optional_vars is not None:
                    for name in target_names(item.optional_vars):
                        tainted.discard(name)
            tainted = scan_statements(statement.body, tainted)
        elif isinstance(statement, ast.Try):
            body_state = scan_statements(statement.body, tainted)
            branch_states = [body_state]
            branch_states.extend(
                scan_statements(handler.body, tainted)
                for handler in statement.handlers
            )
            if statement.orelse:
                branch_states.append(scan_statements(statement.orelse, body_state))
            merged = set.intersection(*branch_states) if branch_states else set(tainted)
            tainted = scan_statements(statement.finalbody, merged)
        else:
            check_calls(statement, tainted)
    return tainted

# Analyse executable statements in each lexical scope independently.  This is
# deliberately ordered and scope-aware: assignments in another function or a
# later overwritten value cannot taint a real set_seed call.
scan_statements(tree.body, set())
for scope in ast.walk(tree):
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        scan_statements(scope.body, set())
if not bound_seed_calls and not unbound_seed_calls:
    raise SystemExit("train.py has no active set_seed(...) call")
if unbound_seed_calls:
    raise SystemExit(
        "set_seed call is not execution-order-bound to TRAIN_SEED at lines "
        f"{sorted(set(unbound_seed_calls))}"
    )
for node in ast.walk(tree):
    if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        continue
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    for target in targets:
        if not isinstance(target, ast.Subscript):
            continue
        value = target.value
        slice_node = target.slice
        if hasattr(ast, "Index") and isinstance(slice_node, ast.Index):
            slice_node = slice_node.value
        if (isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name)
                and value.value.id == "os" and value.attr == "environ"
                and isinstance(slice_node, ast.Constant)
                and slice_node.value == "CUDA_VISIBLE_DEVICES"):
            forced.append(node.lineno)
if forced:
    raise SystemExit(f"active CUDA_VISIBLE_DEVICES assignment at lines {forced}")
print(
    f"[TRAIN_SOURCE_ASSERT] TRAIN_SEED_bound_set_seed_lines="
    f"{sorted(set(bound_seed_calls))}; "
    "no forced CUDA device"
)
PY
rc=$?; ((rc == 0)) || {
    append_failed GLOBAL - train_source_assert "$rc" "source adaptation failed"; exit "$rc";
}

TRAIN_HELP_LOG="$LOGS_DIR/dcalign_train_help.txt"
(cd "$DCALIGN_ROOT" && "$PYTHON_BIN" train.py --help) > "$TRAIN_HELP_LOG" 2>&1
rc=$?; ((rc == 0)) || {
    append_failed GLOBAL - train_cli_assert "$rc" "train.py --help failed"; exit "$rc";
}
for option in --config_file --name --output_dir --dcalign_dynamic_fusion; do
    grep -Fq -- "$option" "$TRAIN_HELP_LOG" || {
        append_failed GLOBAL - train_cli_assert 2 "missing CLI $option"; exit 2;
    }
done
EVAL_HELP_LOG="$LOGS_DIR/eval_dcalign_degraded_help.txt"
"$PYTHON_BIN" "$SCRIPT_DIR/eval_dcalign_degraded.py" --help > "$EVAL_HELP_LOG" 2>&1
rc=$?; ((rc == 0)) || {
    append_failed GLOBAL - eval_cli_assert "$rc" "evaluator --help failed"; exit "$rc";
}
for option in --dcalign-root --config-file --checkpoint --degrade-mode \
    --degrade-strength --degrade-kernel --degrade-seed --scenario-name \
    --group-name --seed --dump-sim --metrics-out --expected-dcalign-head \
    --expected-dcalign-diff-sha256 --expected-irra-kit-sha256 \
    --protocol-base-config --expected-base-config-sha256; do
    grep -Fq -- "$option" "$EVAL_HELP_LOG" || {
        append_failed GLOBAL - eval_cli_assert 2 "missing evaluator CLI $option"; exit 2;
    }
done
(cd "$PROJECT_ROOT" && "$PYTHON_BIN" "$SCRIPT_DIR/check_degradation_parity.py")
rc=$?; ((rc == 0)) || {
    append_failed GLOBAL - degradation_parity "$rc" "parity failed"; exit "$rc";
}

option_count() {
    local option="$1" argument count=0; shift
    for argument in "$@"; do
        [[ "$argument" == "$option" || "$argument" == "$option="* ]] \
            && count=$((count + 1))
    done
    printf '%s' "$count"
}
option_value() {
    local option="$1"; shift; local -a command=("$@"); local index argument
    for ((index=0; index<${#command[@]}; index++)); do
        argument="${command[$index]}"
        if [[ "$argument" == "$option" ]]; then
            ((index + 1 < ${#command[@]})) || return 2
            printf '%s' "${command[$((index + 1))]}"; return 0
        elif [[ "$argument" == "$option="* ]]; then
            printf '%s' "${argument#*=}"; return 0
        fi
    done
    return 1
}
assert_train_command() {
    local group="$1" seed="$2"; shift 2; local -a command=("$@")
    local option argument seed_count=0 dynamic_count train_script_count=0
    for option in --config_file --name --output_dir; do
        [[ "$(option_count "$option" "${command[@]}")" == 1 ]] || return 1
    done
    for argument in "${command[@]}"; do
        [[ "$argument" == "TRAIN_SEED=$seed" ]] && seed_count=$((seed_count + 1))
        [[ "$argument" == train.py ]] && train_script_count=$((train_script_count + 1))
        [[ "$argument" != --resume && "$argument" != --resume=* \
            && "$argument" != --resume_ckpt_file \
            && "$argument" != --resume_ckpt_file=* ]] || return 1
    done
    ((seed_count == 1 && train_script_count == 1)) || return 1
    dynamic_count="$(option_count --dcalign_dynamic_fusion "${command[@]}")"
    [[ "$group" == DC-add && "$dynamic_count" == 0 \
        || "$group" == DC-dyn && "$dynamic_count" == 1 ]] || return 1
    [[ "$(option_value --name "${command[@]}")" == "${group}_seed${seed}" \
        && "$(option_value --config_file "${command[@]}")" == "$BASE_CONFIG" \
        && "$(option_value --output_dir "${command[@]}")" \
            == "$RUNS_DIR/${group}_seed${seed}/"* ]]
}
assert_eval_command() {
    local ckpt="$1" group="$2" seed="$3" scenario="$4" mode="$5"
    local strength="$6" kernel="$7" dseed="$8"; shift 8
    local -a command=("$@"); local option argument eval_script_count=0
    for option in --dcalign-root --config-file --checkpoint --degrade-mode \
        --degrade-strength --degrade-kernel --degrade-seed --scenario-name \
        --group-name --seed --dump-sim --metrics-out --expected-dcalign-head \
        --expected-dcalign-diff-sha256 --expected-irra-kit-sha256 \
        --protocol-base-config --expected-base-config-sha256; do
        [[ "$(option_count "$option" "${command[@]}")" == 1 ]] || return 1
    done
    for argument in "${command[@]}"; do
        [[ "$argument" == "$SCRIPT_DIR/eval_dcalign_degraded.py" ]] \
            && eval_script_count=$((eval_script_count + 1))
        [[ "$argument" != test.py && "$argument" != *"/test.py" ]] || return 1
    done
    ((eval_script_count == 1)) || return 1
    [[ "$(option_value --dcalign-root "${command[@]}")" == "$DCALIGN_ROOT" \
        && "$(option_value --expected-dcalign-head "${command[@]}")" == "$DCALIGN_HEAD" \
        && "$(option_value --expected-dcalign-diff-sha256 "${command[@]}")" == "$DCALIGN_DIFF_SHA256" \
        && "$(option_value --expected-irra-kit-sha256 "${command[@]}")" == "$IRRA_KIT_SHA256" \
        && "$(option_value --protocol-base-config "${command[@]}")" == "$BASE_CONFIG" \
        && "$(option_value --expected-base-config-sha256 "${command[@]}")" == "$BASE_CONFIG_SHA256" \
        && "$(option_value --checkpoint "${command[@]}")" == "$ckpt" \
        && "$(option_value --group-name "${command[@]}")" == "$group" \
        && "$(option_value --seed "${command[@]}")" == "$seed" \
        && "$(option_value --scenario-name "${command[@]}")" == "$scenario" \
        && "$(option_value --degrade-mode "${command[@]}")" == "$mode" \
        && "$(option_value --degrade-strength "${command[@]}")" == "$strength" \
        && "$(option_value --degrade-kernel "${command[@]}")" == "$kernel" \
        && "$(option_value --degrade-seed "${command[@]}")" == "$dseed" ]] || return 1
    if [[ "$ckpt" == *"/best.pth" ]]; then
        [[ "$group" == *-best && "$scenario" == clean && "$mode" == none ]]
    else
        [[ "$ckpt" == *"/last.pth" && "$group" != *-best ]]
    fi
}

validate_train_log() {
    local train_log="$1"
    "$PYTHON_BIN" - "$train_log" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
expectations = {
    "train": (1266, 3250, 5497),
    "test": (556, 1473, 2490),
}
for split, values in expectations.items():
    pattern = re.compile(
        rf"(?im)^.*\b{split}\b.*{values[0]}.*{values[1]}.*{values[2]}.*$"
    )
    if pattern.search(text) is None:
        raise SystemExit(
            f"training log does not prove {split} statistics "
            f"{values[0]}/{values[1]}/{values[2]}"
        )
print(
    "[DATASET_ASSERT] "
    "train=1266/3250/5497 test=556/1473/2490 status=PASS"
)
PY
}

find_single_file() {
    local search_root="$1" filename="$2"
    local -a matches=()
    mapfile -d '' -t matches < <(
        find "$search_root" -type f -name "$filename" -print0
    )
    if ((${#matches[@]} != 1)); then
        printf 'expected exactly one %s below %s, found %s\n' \
            "$filename" "$search_root" "${#matches[@]}" >&2
        return 1
    fi
    printf '%s' "${matches[0]}"
}

validate_saved_config() {
    local config_file="$1" group="$2" seed="$3"
    "$PYTHON_BIN" - "$config_file" "$group" "$seed" <<'PY'
import sys
from pathlib import Path
import yaml

path = Path(sys.argv[1])
group = sys.argv[2]
seed = int(sys.argv[3])
with path.open("r", encoding="utf-8") as handle:
    data = yaml.load(handle, Loader=yaml.FullLoader)
if not isinstance(data, dict):
    raise SystemExit(f"saved config is not a mapping: {path}")
expected_dynamic = group == "DC-dyn"
actual_dynamic = bool(data.get("dcalign_dynamic_fusion", False))
if actual_dynamic != expected_dynamic:
    raise SystemExit(
        "saved dynamic-fusion value mismatch: "
        f"expected={expected_dynamic}, actual={actual_dynamic}"
    )
if bool(data.get("resume", False)):
    raise SystemExit("saved config unexpectedly enables resume")
if "training" in data and not bool(data["training"]):
    raise SystemExit("saved config unexpectedly disables training")
if "seed" in data and int(data["seed"]) != seed:
    raise SystemExit(
        f"saved seed mismatch: expected={seed}, actual={data['seed']}"
    )
print(
    f"[CONFIG_ASSERT] group={group} seed={seed} "
    f"dcalign_dynamic_fusion={actual_dynamic} status=PASS"
)
PY
}

validate_checkpoint() {
    local checkpoint="$1"
    "$PYTHON_BIN" - "$checkpoint" <<'PY'
import sys
from collections.abc import Mapping
from pathlib import Path
import torch

path = Path(sys.argv[1])
try:
    payload = torch.load(path, map_location="cpu", weights_only=True)
except TypeError:
    payload = torch.load(path, map_location="cpu")
if not isinstance(payload, Mapping) or "model" not in payload:
    raise SystemExit(
        f"checkpoint top level must be a mapping containing model: {path}"
    )
if not isinstance(payload["model"], Mapping):
    raise SystemExit(f"checkpoint model entry is not a state dict: {path}")
print(f"[CHECKPOINT_ASSERT] path={path} top_level_model=true")
PY
}

validate_eval_artifacts() {
    local fragment="$1" score_file="$2" group="$3" seed="$4"
    local scenario="$5" mode="$6" strength="$7" kernel="$8" dseed="$9"
    local expected_head="${10}" expected_diff_sha="${11}" expected_kit_sha="${12}"
    local expected_base_sha="${13}"
    "$PYTHON_BIN" - "$fragment" "$score_file" "$group" "$seed" \
        "$scenario" "$mode" "$strength" "$kernel" "$dseed" \
        "$expected_head" "$expected_diff_sha" "$expected_kit_sha" \
        "$expected_base_sha" <<'PY'
import csv
import math
import sys
from collections.abc import Mapping
from pathlib import Path
import torch

(
    fragment_raw,
    score_raw,
    group,
    seed,
    scenario,
    mode,
    strength,
    kernel,
    dseed,
    expected_head,
    expected_diff_sha,
    expected_kit_sha,
    expected_base_sha,
) = sys.argv[1:]
fragment = Path(fragment_raw)
score_file = Path(score_raw)
header = [
    "group", "seed", "scenario", "mode", "strength", "kernel", "dseed",
    "split", "R1", "R5", "R10", "mAP", "mINP",
]
with fragment.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    rows = list(reader)
    if reader.fieldnames != header:
        raise SystemExit(
            f"metrics fragment header mismatch: {reader.fieldnames!r}"
        )
if len(rows) != 1:
    raise SystemExit(f"metrics fragment must have one row, got {len(rows)}")
row = rows[0]
expected = {
    "group": group,
    "seed": seed,
    "scenario": scenario,
    "mode": mode,
    "kernel": kernel,
    "dseed": dseed,
    "split": "test",
}
for key, value in expected.items():
    if row[key] != value:
        raise SystemExit(
            f"metrics fragment {key} mismatch: expected={value}, actual={row[key]}"
        )
if not math.isclose(float(row["strength"]), float(strength), rel_tol=0, abs_tol=1e-12):
    raise SystemExit(
        f"metrics fragment strength mismatch: expected={strength}, "
        f"actual={row['strength']}"
    )
for key in ("R1", "R5", "R10", "mAP", "mINP"):
    value = float(row[key])
    if not math.isfinite(value) or not (0.0 <= value <= 100.0):
        raise SystemExit(f"invalid metric {key}={value}")

try:
    payload = torch.load(score_file, map_location="cpu", weights_only=True)
except TypeError:
    payload = torch.load(score_file, map_location="cpu")
if not isinstance(payload, Mapping):
    raise SystemExit(f"score payload is not a mapping: {score_file}")
required = {
    "sim", "txt_pids", "img_pids", "txt_indices", "img_indices",
    "metrics", "metadata", "order_hash",
}
if not required.issubset(payload):
    raise SystemExit(
        f"score payload missing keys: {sorted(required - set(payload))}"
    )
if tuple(payload["sim"].shape) != (2490, 1473):
    raise SystemExit(
        f"score shape mismatch: expected=(2490, 1473), "
        f"actual={tuple(payload['sim'].shape)}"
    )
if not bool(torch.isfinite(payload["sim"]).all().item()):
    raise SystemExit(f"score contains NaN or infinity: {score_file}")
txt_unique = torch.unique(payload["txt_pids"])
img_unique = torch.unique(payload["img_pids"])
if txt_unique.numel() != 556 or img_unique.numel() != 556 \
        or not torch.equal(txt_unique, img_unique):
    raise SystemExit(
        "score query/gallery pid coverage mismatch: "
        f"txt_unique={txt_unique.numel()}, img_unique={img_unique.numel()}"
    )
metadata = payload["metadata"]
if not isinstance(metadata, Mapping):
    raise SystemExit("score metadata is not a mapping")
metadata_expected = {
    "scenario": scenario,
    "mode": mode,
    "kernel": int(kernel),
    "dseed": int(dseed),
    "group": group,
    "seed": int(seed),
    "source": "dcalign",
}
for key, value in metadata_expected.items():
    if metadata.get(key) != value:
        raise SystemExit(
            f"score metadata {key} mismatch: "
            f"expected={value!r}, actual={metadata.get(key)!r}"
        )
if not math.isclose(
    float(metadata.get("strength")), float(strength), rel_tol=0, abs_tol=1e-12
):
    raise SystemExit("score metadata strength mismatch")
source_state = metadata.get("dcalign_source_state")
if not isinstance(source_state, Mapping):
    raise SystemExit("score metadata lacks DCAlign source state")
expected_dirty_paths = [
    "datasets/bases.py",
    "model/build.py",
    "processor/processor.py",
    "test.py",
    "train.py",
    "utils/metrics.py",
    "utils/options.py",
]
if source_state.get("tracked_dirty_paths") != expected_dirty_paths:
    raise SystemExit(
        "score DCAlign dirty path set mismatch: "
        f"actual={source_state.get('tracked_dirty_paths')}"
    )
if source_state.get("head") != expected_head:
    raise SystemExit(
        "score DCAlign HEAD mismatch: "
        f"expected={expected_head}, actual={source_state.get('head')}"
    )
if source_state.get("tracked_diff_sha256") != expected_diff_sha:
    raise SystemExit(
        "score DCAlign diff SHA mismatch: "
        f"expected={expected_diff_sha}, "
        f"actual={source_state.get('tracked_diff_sha256')}"
    )
kit_state = metadata.get("irra_runtime_kit_state")
if not isinstance(kit_state, Mapping):
    raise SystemExit("score metadata lacks IRRA runtime-kit state")
if kit_state.get("manifest_sha256") != expected_kit_sha:
    raise SystemExit(
        "score IRRA runtime-kit SHA mismatch: "
        f"expected={expected_kit_sha}, "
        f"actual={kit_state.get('manifest_sha256')}"
    )
if metadata.get("protocol_base_config_sha256") != expected_base_sha:
    raise SystemExit(
        "score protocol base-config SHA mismatch: "
        f"expected={expected_base_sha}, "
        f"actual={metadata.get('protocol_base_config_sha256')}"
    )
print(
    f"[EVAL_ARTIFACT_ASSERT] group={group} seed={seed} "
    f"scenario={scenario} shape=[2490,1473] status=PASS"
)
PY
}

run_eval_scenario() {
    local checkpoint="$1" group="$2" seed="$3" scenario="$4" mode="$5"
    local strength="$6" kernel="$7" dseed="$8" config_file="$9"
    local score_file="${10}" fragment="${11}" eval_log="${12}"
    if ! assert_runtime_inputs_unchanged; then
        append_failed "$group" "$seed" "runtime_input_drift:$scenario" 1 \
            "base config or IRRA runtime kit changed after startup"
        return 1
    fi
    if ! has_minimum_disk; then
        append_failed "$group" "$seed" "disk_guard:$scenario" 1 \
            "free disk is below MIN_FREE_GB=$MIN_FREE_GB"
        return 1
    fi
    local -a eval_command=(
        "$PYTHON_BIN" -u "$SCRIPT_DIR/eval_dcalign_degraded.py"
        --dcalign-root "$DCALIGN_ROOT"
        --expected-dcalign-head "$DCALIGN_HEAD"
        --expected-dcalign-diff-sha256 "$DCALIGN_DIFF_SHA256"
        --expected-irra-kit-sha256 "$IRRA_KIT_SHA256"
        --protocol-base-config "$BASE_CONFIG"
        --expected-base-config-sha256 "$BASE_CONFIG_SHA256"
        --config-file "$config_file"
        --checkpoint "$checkpoint"
        --degrade-mode "$mode"
        --degrade-strength "$strength"
        --degrade-kernel "$kernel"
        --degrade-seed "$dseed"
        --scenario-name "$scenario"
        --group-name "$group"
        --seed "$seed"
        --dump-sim "$score_file"
        --metrics-out "$fragment"
    )
    if ! assert_eval_command "$checkpoint" "$group" "$seed" "$scenario" \
        "$mode" "$strength" "$kernel" "$dseed" "${eval_command[@]}"; then
        append_failed "$group" "$seed" "eval_command_assert:$scenario" 2 \
            "evaluation command violates last/best or one-scenario protocol"
        return 1
    fi
    log "[EVAL_START] group=$group seed=$seed scenario=$scenario mode=$mode checkpoint=$(basename "$checkpoint")"
    "${eval_command[@]}" 2>&1 | tee -a "$eval_log"
    local eval_rc="${PIPESTATUS[0]}"
    if ((eval_rc != 0)); then
        append_failed "$group" "$seed" "eval:$scenario" "$eval_rc" \
            "evaluation process failed; checkpoint retained"
        return 1
    fi
    if ! validate_eval_artifacts "$fragment" "$score_file" "$group" "$seed" \
        "$scenario" "$mode" "$strength" "$kernel" "$dseed" \
        "$DCALIGN_HEAD" "$DCALIGN_DIFF_SHA256" "$IRRA_KIT_SHA256" \
        "$BASE_CONFIG_SHA256"; then
        append_failed "$group" "$seed" "eval_artifacts:$scenario" 1 \
            "metrics fragment or score dump validation failed"
        return 1
    fi
}

commit_unit_metrics() {
    local unit_metrics="$1" group="$2" seed="$3"
    "$PYTHON_BIN" - "$METRICS_TSV" "$unit_metrics" "$group" "$seed" <<'PY'
import csv
import math
import os
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
unit_path = Path(sys.argv[2])
group = sys.argv[3]
seed = sys.argv[4]
header = [
    "group", "seed", "scenario", "mode", "strength", "kernel", "dseed",
    "split", "R1", "R5", "R10", "mAP", "mINP",
]
expected_scenarios = {
    "clean", "rgb_missing", "ir_missing", "rgb_noise_005",
    "rgb_noise_010", "ir_noise_005", "ir_noise_010", "rgb_blur_k7",
    "ir_blur_k7", "both_noise_005", "both_blur_k7",
}
with unit_path.open("r", encoding="utf-8", newline="") as handle:
    unit_reader = csv.DictReader(handle, delimiter="\t")
    unit_rows = list(unit_reader)
    if unit_reader.fieldnames != header:
        raise SystemExit("unit metrics header mismatch")
if len(unit_rows) != 12:
    raise SystemExit(f"unit metrics must contain 12 rows, got {len(unit_rows)}")
last_rows = [
    row for row in unit_rows if row["group"] == group and row["seed"] == seed
]
best_rows = [
    row for row in unit_rows
    if row["group"] == f"{group}-best" and row["seed"] == seed
]
if len(last_rows) != 11 or {row["scenario"] for row in last_rows} != expected_scenarios:
    raise SystemExit("last.pth metrics do not contain the exact 11 scenarios")
if len(best_rows) != 1 or best_rows[0]["scenario"] != "clean" \
        or best_rows[0]["mode"] != "none":
    raise SystemExit("best.pth metrics must contain only clean/none")
for row in unit_rows:
    if row["split"] != "test":
        raise SystemExit("all DCAlign metrics must use split=test")
    for key in ("R1", "R5", "R10", "mAP", "mINP"):
        value = float(row[key])
        if not math.isfinite(value) or not (0.0 <= value <= 100.0):
            raise SystemExit(f"invalid {key}={value}")

with metrics_path.open("r", encoding="utf-8", newline="") as handle:
    global_reader = csv.DictReader(handle, delimiter="\t")
    existing = list(global_reader)
    if global_reader.fieldnames != header:
        raise SystemExit("global metrics.tsv header mismatch")
target_groups = {group, f"{group}-best"}
existing = [
    row for row in existing
    if not (row["group"] in target_groups and row["seed"] == seed)
]
existing.extend(unit_rows)
temporary = metrics_path.with_name(f".{metrics_path.name}.{os.getpid()}.tmp")
with temporary.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle, fieldnames=header, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(existing)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, metrics_path)
print(
    f"[METRICS_COMMIT] group={group} seed={seed} rows=12 status=PASS"
)
PY
}

commit_reproduction_warning() {
    local best_fragment="$1" group="$2" seed="$3" checkpoint="$4"
    "$PYTHON_BIN" - "$WARNINGS_TSV" "$best_fragment" "$group" "$seed" \
        "$checkpoint" "$(now)" <<'PY'
import csv
import os
import sys
from pathlib import Path

warnings_path = Path(sys.argv[1])
fragment_path = Path(sys.argv[2])
group, seed, checkpoint, timestamp = sys.argv[3:]
reference = 58.47
header = [
    "timestamp", "group", "seed", "checkpoint", "observed_R1",
    "reference_R1", "absolute_difference", "reason",
]
with fragment_path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if len(rows) != 1:
    raise SystemExit("best metrics fragment must contain exactly one row")
observed = float(rows[0]["R1"])
difference = abs(observed - reference)
with warnings_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    warnings = list(reader)
    if reader.fieldnames != header:
        raise SystemExit("warnings.tsv header mismatch")
warnings = [
    row for row in warnings
    if not (row["group"] == f"{group}-best" and row["seed"] == seed)
]
if difference > 5.0:
    warnings.append(
        {
            "timestamp": timestamp,
            "group": f"{group}-best",
            "seed": seed,
            "checkpoint": checkpoint,
            "observed_R1": f"{observed:.10g}",
            "reference_R1": f"{reference:.2f}",
            "absolute_difference": f"{difference:.10g}",
            "reason": "clean best.pth R1 differs from published 58.47 by >5 points",
        }
    )
temporary = warnings_path.with_name(f".{warnings_path.name}.{os.getpid()}.tmp")
with temporary.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle, fieldnames=header, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(warnings)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, warnings_path)
status = "WARN" if difference > 5.0 else "PASS"
print(
    f"[LITERATURE_R1_CHECK] group={group}-best seed={seed} "
    f"observed={observed:.6f} reference={reference:.2f} "
    f"difference={difference:.6f} status={status}"
)
PY
}

publish_scores() {
    local attempt_root="$1"
    shift
    local -a score_sources=()
    local -a score_targets=()
    local half=$(( $# / 2 )) index source target source_real target_real
    (($# > 0 && $# % 2 == 0)) || return 2
    score_sources=("${@:1:half}")
    score_targets=("${@:half+1:half}")
    for ((index=0; index<half; index++)); do
        source="${score_sources[$index]}"
        target="${score_targets[$index]}"
        source_real="$(realpath -m "$source")"
        target_real="$(realpath -m "$target")"
        [[ "$source_real" == "$attempt_root/"* && "$target_real" == "$SIM_DIR/"* \
            && -f "$source_real" ]] || {
            printf 'unsafe score publish source=%s target=%s\n' \
                "$source_real" "$target_real" >&2
            return 1
        }
        mkdir -p "$(dirname -- "$target_real")" || return 1
        mv -f -- "$source_real" "$target_real" || return 1
    done
}

compute_score_manifest_sha() {
    local group="$1" seed="$2" scenario_spec scenario mode strength kernel dseed
    local score_file digest manifest=""
    for scenario_spec in "${CORE_SCENARIOS[@]}"; do
        IFS='|' read -r scenario mode strength kernel dseed <<< "$scenario_spec"
        score_file="$SIM_DIR/${group}_seed${seed}/last/${scenario}.pt"
        [[ -f "$score_file" ]] || return 1
        digest="$(sha256sum "$score_file" | awk '{print $1}')" || return 1
        manifest+="last/${scenario}.pt=${digest}"$'\n'
    done
    score_file="$SIM_DIR/${group}_seed${seed}/best/clean.pt"
    [[ -f "$score_file" ]] || return 1
    digest="$(sha256sum "$score_file" | awk '{print $1}')" || return 1
    manifest+="best/clean.pt=${digest}"$'\n'
    printf '%s' "$manifest" | sha256sum | awk '{print $1}'
}

verify_completed_unit() {
    local mark="$1" group="$2" seed="$3"
    local score_manifest_sha
    score_manifest_sha="$(compute_score_manifest_sha "$group" "$seed")" || {
        printf 'could not compute score manifest for group=%s seed=%s\n' \
            "$group" "$seed" >&2
        return 1
    }
    "$PYTHON_BIN" - "$mark" "$METRICS_TSV" "$SIM_DIR" "$group" "$seed" \
        "$score_manifest_sha" "$DCALIGN_HEAD" "$DCALIGN_DIFF_SHA256" \
        "$BASE_CONFIG_SHA256" "$IRRA_KIT_SHA256" <<'PY'
import csv
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

mark_path = Path(sys.argv[1])
metrics_path = Path(sys.argv[2])
sim_root = Path(sys.argv[3])
group = sys.argv[4]
seed = sys.argv[5]
actual_score_manifest_sha = sys.argv[6]
expected_dcalign_head = sys.argv[7]
expected_dcalign_diff_sha = sys.argv[8]
expected_base_config_sha = sys.argv[9]
expected_irra_kit_sha = sys.argv[10]
values = {}
for line in mark_path.read_text(encoding="utf-8").splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        values[key] = value
for key in (
    "last_checkpoint",
    "best_checkpoint",
    "last_sha256",
    "best_sha256",
    "score_manifest_sha256",
    "dcalign_head",
    "dcalign_diff_sha256",
    "base_config_sha256",
    "irra_kit_sha256",
):
    if not values.get(key):
        raise SystemExit(f"completion marker lacks {key}")
expected_marker = {
    "group": group,
    "seed": seed,
    "metrics_rows": "12",
    "score_files": "12",
    "dcalign_head": expected_dcalign_head,
    "dcalign_diff_sha256": expected_dcalign_diff_sha,
    "base_config_sha256": expected_base_config_sha,
    "irra_kit_sha256": expected_irra_kit_sha,
}
for key, expected in expected_marker.items():
    if values.get(key) != expected:
        raise SystemExit(
            f"completion marker {key} mismatch: "
            f"expected={expected!r}, actual={values.get(key)!r}"
        )
if values["score_manifest_sha256"] != actual_score_manifest_sha:
    raise SystemExit(
        "completion marker score manifest SHA mismatch: "
        f"expected={values['score_manifest_sha256']}, "
        f"actual={actual_score_manifest_sha}"
    )

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for path_key, sha_key in (
    ("last_checkpoint", "last_sha256"),
    ("best_checkpoint", "best_sha256"),
):
    checkpoint = Path(values[path_key])
    if not checkpoint.is_file():
        raise SystemExit(f"completion marker checkpoint no longer exists: {checkpoint}")
    actual_sha = sha256_file(checkpoint)
    if actual_sha != values[sha_key]:
        raise SystemExit(
            f"completion marker checkpoint SHA mismatch for {checkpoint}: "
            f"expected={values[sha_key]}, actual={actual_sha}"
        )

metrics_header = [
    "group", "seed", "scenario", "mode", "strength", "kernel", "dseed",
    "split", "R1", "R5", "R10", "mAP", "mINP",
]
with metrics_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    rows = list(reader)
    if reader.fieldnames != metrics_header:
        raise SystemExit("completion marker metrics.tsv header mismatch")
last_rows = [
    row for row in rows if row["group"] == group and row["seed"] == seed
]
best_rows = [
    row for row in rows
    if row["group"] == f"{group}-best" and row["seed"] == seed
]
if len(last_rows) != 11 or len(best_rows) != 1:
    raise SystemExit("completion marker metrics are incomplete")

core_specs = (
    ("clean", "none", 1.0, 7, 2026),
    ("rgb_missing", "rgb_missing", 1.0, 7, 2026),
    ("ir_missing", "ir_missing", 1.0, 7, 2026),
    ("rgb_noise_005", "rgb_noise", 0.05, 7, 2026),
    ("rgb_noise_010", "rgb_noise", 0.10, 7, 2026),
    ("ir_noise_005", "ir_noise", 0.05, 7, 2026),
    ("ir_noise_010", "ir_noise", 0.10, 7, 2026),
    ("rgb_blur_k7", "rgb_blur", 1.0, 7, 2026),
    ("ir_blur_k7", "ir_blur", 1.0, 7, 2026),
    ("both_noise_005", "both_noise", 0.05, 7, 2026),
    ("both_blur_k7", "both_blur", 1.0, 7, 2026),
)
spec_by_scenario = {item[0]: item for item in core_specs}
if {row["scenario"] for row in last_rows} != set(spec_by_scenario):
    raise SystemExit("completion marker last metrics scenario set is incorrect")
if best_rows[0]["scenario"] != "clean" or best_rows[0]["mode"] != "none":
    raise SystemExit("completion marker best metrics must contain only clean/none")

def validate_metric_row(row, expected_group, spec):
    scenario, mode, strength, kernel, dseed = spec
    expected = {
        "group": expected_group,
        "seed": seed,
        "scenario": scenario,
        "mode": mode,
        "kernel": str(kernel),
        "dseed": str(dseed),
        "split": "test",
    }
    for key, expected_value in expected.items():
        if row[key] != expected_value:
            raise SystemExit(
                f"completion metrics {key} mismatch for {scenario}: "
                f"expected={expected_value!r}, actual={row[key]!r}"
            )
    if not math.isclose(float(row["strength"]), strength, rel_tol=0, abs_tol=1e-12):
        raise SystemExit(f"completion metrics strength mismatch for {scenario}")
    for metric in ("R1", "R5", "R10", "mAP", "mINP"):
        value = float(row[metric])
        if not math.isfinite(value) or not (0.0 <= value <= 100.0):
            raise SystemExit(f"completion metrics {metric} invalid for {scenario}")

last_by_scenario = {row["scenario"]: row for row in last_rows}
for scenario, spec in spec_by_scenario.items():
    validate_metric_row(last_by_scenario[scenario], group, spec)
validate_metric_row(
    best_rows[0], f"{group}-best", ("clean", "none", 1.0, 7, 2026)
)

last_dir = sim_root / f"{group}_seed{seed}" / "last"
best_file = sim_root / f"{group}_seed{seed}" / "best" / "clean.pt"
expected_last_files = {
    last_dir / f"{scenario}.pt" for scenario in spec_by_scenario
}
actual_last_files = set(last_dir.glob("*.pt"))
if actual_last_files != expected_last_files or not best_file.is_file():
    raise SystemExit("completion marker score dumps are incomplete")

row_by_key = {
    (row["group"], row["scenario"]): row for row in last_rows + best_rows
}

def stable_order_hash(payload):
    encoded = json.dumps(
        {
            "txt_pids": payload["txt_pids"].view(-1).tolist(),
            "img_pids": payload["img_pids"].view(-1).tolist(),
            "score_shape": list(payload["sim"].shape),
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]

def validate_score(path, expected_group, spec, expected_checkpoint):
    scenario, mode, strength, kernel, dseed = spec
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    required = {
        "sim", "txt_pids", "img_pids", "txt_indices", "img_indices",
        "metrics", "metadata", "order_hash",
    }
    if not isinstance(payload, Mapping) or not required.issubset(payload):
        raise SystemExit(f"resume score payload is invalid: {path}")
    sim = payload["sim"]
    if not torch.is_tensor(sim) or tuple(sim.shape) != (2490, 1473):
        raise SystemExit(f"resume score shape mismatch: {path}")
    if sim.device.type != "cpu" or sim.dtype != torch.float32:
        raise SystemExit(f"resume score dtype/device mismatch: {path}")
    if payload["txt_pids"].numel() != 2490 or payload["img_pids"].numel() != 1473:
        raise SystemExit(f"resume score pid lengths mismatch: {path}")
    txt_unique = torch.unique(payload["txt_pids"])
    img_unique = torch.unique(payload["img_pids"])
    if txt_unique.numel() != 556 or img_unique.numel() != 556 \
            or not torch.equal(txt_unique, img_unique):
        raise SystemExit(f"resume score pid coverage mismatch: {path}")
    if not torch.equal(payload["txt_indices"], torch.arange(2490)) \
            or not torch.equal(payload["img_indices"], torch.arange(1473)):
        raise SystemExit(f"resume score canonical indices mismatch: {path}")
    expected_hash = stable_order_hash(payload)
    if payload["order_hash"] != expected_hash:
        raise SystemExit(f"resume score order_hash mismatch: {path}")
    metadata = payload["metadata"]
    metadata_expected = {
        "scenario": scenario,
        "mode": mode,
        "strength": float(strength),
        "kernel": int(kernel),
        "dseed": int(dseed),
        "group": expected_group,
        "seed": int(seed),
        "source": "dcalign",
        "order_hash": expected_hash,
    }
    if not isinstance(metadata, Mapping):
        raise SystemExit(f"resume score metadata is invalid: {path}")
    for key, expected in metadata_expected.items():
        actual = metadata.get(key)
        if key == "strength":
            if not math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12):
                raise SystemExit(f"resume score metadata {key} mismatch: {path}")
        elif actual != expected:
            raise SystemExit(f"resume score metadata {key} mismatch: {path}")
    source_state = metadata.get("dcalign_source_state")
    if not isinstance(source_state, Mapping):
        raise SystemExit(f"resume score lacks DCAlign source state: {path}")
    expected_dirty_paths = [
        "datasets/bases.py",
        "model/build.py",
        "processor/processor.py",
        "test.py",
        "train.py",
        "utils/metrics.py",
        "utils/options.py",
    ]
    if source_state.get("tracked_dirty_paths") != expected_dirty_paths:
        raise SystemExit(f"resume score DCAlign dirty path set mismatch: {path}")
    if source_state.get("head") != expected_dcalign_head:
        raise SystemExit(f"resume score DCAlign HEAD mismatch: {path}")
    if source_state.get("tracked_diff_sha256") != expected_dcalign_diff_sha:
        raise SystemExit(f"resume score DCAlign diff SHA mismatch: {path}")
    kit_state = metadata.get("irra_runtime_kit_state")
    if not isinstance(kit_state, Mapping):
        raise SystemExit(f"resume score lacks IRRA runtime-kit state: {path}")
    if kit_state.get("manifest_sha256") != expected_irra_kit_sha:
        raise SystemExit(f"resume score IRRA runtime-kit SHA mismatch: {path}")
    if metadata.get("protocol_base_config_sha256") != expected_base_config_sha:
        raise SystemExit(f"resume score base-config SHA mismatch: {path}")
    recorded_checkpoint = metadata.get("test_ckpt_file")
    if not recorded_checkpoint or Path(recorded_checkpoint).resolve() \
            != Path(expected_checkpoint).resolve():
        raise SystemExit(
            f"resume score checkpoint binding mismatch for {path}: "
            f"expected={Path(expected_checkpoint).resolve()}, "
            f"actual={recorded_checkpoint}"
        )
    metrics = payload["metrics"]
    row = row_by_key[(expected_group, scenario)]
    for metric in ("R1", "R5", "R10", "mAP", "mINP"):
        if metric not in metrics or not math.isclose(
            float(metrics[metric]), float(row[metric]), rel_tol=1e-9, abs_tol=1e-8
        ):
            raise SystemExit(f"resume score metric {metric} mismatch: {path}")

for spec in core_specs:
    validate_score(
        last_dir / f"{spec[0]}.pt", group, spec, values["last_checkpoint"]
    )
validate_score(
    best_file,
    f"{group}-best",
    ("clean", "none", 1.0, 7, 2026),
    values["best_checkpoint"],
)
print(f"[RESUME_ASSERT] group={group} seed={seed} status=PASS")
PY
}

if ((VALIDATE_ONLY)); then
    for group in "${ARM_NAMES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_key="${group}_seed${seed}"
            plan_root="$RUNS_DIR/$run_key/validate_only"
            train_command=(
                env "TRAIN_SEED=$seed"
                "$PYTHON_BIN" -u train.py
                --config_file "$BASE_CONFIG"
                --name "$run_key"
                --output_dir "$plan_root"
            )
            [[ "$group" != DC-dyn ]] || train_command+=(--dcalign_dynamic_fusion)
            assert_train_command "$group" "$seed" "${train_command[@]}" || {
                append_failed "$group" "$seed" validate_train_command 2 \
                    "generated train command failed assertion"
                exit 2
            }
            last_plan="$plan_root/last.pth"
            best_plan="$plan_root/best.pth"
            last_eval_command=(
                "$PYTHON_BIN" -u "$SCRIPT_DIR/eval_dcalign_degraded.py"
                --dcalign-root "$DCALIGN_ROOT"
                --expected-dcalign-head "$DCALIGN_HEAD"
                --expected-dcalign-diff-sha256 "$DCALIGN_DIFF_SHA256"
                --expected-irra-kit-sha256 "$IRRA_KIT_SHA256"
                --protocol-base-config "$BASE_CONFIG"
                --expected-base-config-sha256 "$BASE_CONFIG_SHA256"
                --config-file "$BASE_CONFIG"
                --checkpoint "$last_plan" --degrade-mode none
                --degrade-strength 1.0 --degrade-kernel 7
                --degrade-seed 2026 --scenario-name clean
                --group-name "$group" --seed "$seed"
                --dump-sim "$plan_root/clean.pt"
                --metrics-out "$plan_root/clean.tsv"
            )
            best_eval_command=(
                "$PYTHON_BIN" -u "$SCRIPT_DIR/eval_dcalign_degraded.py"
                --dcalign-root "$DCALIGN_ROOT"
                --expected-dcalign-head "$DCALIGN_HEAD"
                --expected-dcalign-diff-sha256 "$DCALIGN_DIFF_SHA256"
                --expected-irra-kit-sha256 "$IRRA_KIT_SHA256"
                --protocol-base-config "$BASE_CONFIG"
                --expected-base-config-sha256 "$BASE_CONFIG_SHA256"
                --config-file "$BASE_CONFIG"
                --checkpoint "$best_plan" --degrade-mode none
                --degrade-strength 1.0 --degrade-kernel 7
                --degrade-seed 2026 --scenario-name clean
                --group-name "${group}-best" --seed "$seed"
                --dump-sim "$plan_root/best_clean.pt"
                --metrics-out "$plan_root/best_clean.tsv"
            )
            assert_eval_command "$last_plan" "$group" "$seed" clean none \
                1.0 7 2026 "${last_eval_command[@]}" || exit 2
            assert_eval_command "$best_plan" "${group}-best" "$seed" clean none \
                1.0 7 2026 "${best_eval_command[@]}" || exit 2
            log "[VALIDATE_ONLY_PLAN] group=$group seed=$seed train=1 last_scenarios=11 best_clean=1"
        done
    done
    log "[VALIDATE_ONLY] startup, parity, CLI and command assertions passed; no training/evaluation/shutdown"
    exit 0
fi

run_one() {
    local group="$1" seed="$2" run_key="${1}_seed${2}"
    local mark="$MARKS_DIR/${run_key}.done"
    if [[ -f "$mark" ]]; then
        if verify_completed_unit "$mark" "$group" "$seed"; then
            log "[SKIP_DONE] group=$group seed=$seed"
            return 0
        fi
        append_failed "$group" "$seed" resume_assert 1 \
            "completion marker exists but artifacts are incomplete; starting a new attempt"
        local stale_mark="${mark}.invalid.$(date '+%Y%m%d_%H%M%S').$$"
        if ! mv -- "$mark" "$stale_mark"; then
            append_failed "$group" "$seed" marker_quarantine 1 \
                "invalid completion marker could not be quarantined: $mark"
            return 1
        fi
        log "[MARKER_QUARANTINE] group=$group seed=$seed path=$stale_mark"
    fi
    if ! has_minimum_disk; then
        append_failed "$group" "$seed" disk_guard 1 \
            "free disk is below MIN_FREE_GB=$MIN_FREE_GB"
        return 1
    fi
    if ! assert_runtime_inputs_unchanged; then
        append_failed "$group" "$seed" runtime_input_drift 1 \
            "base config or IRRA runtime kit changed after startup"
        return 1
    fi

    local timestamp attempt_root train_log train_rc config_file run_start
    local last_checkpoint best_checkpoint
    timestamp="$(date '+%Y%m%d_%H%M%S')_$$"
    run_start="$(date +%s)"
    attempt_root="$RUNS_DIR/$run_key/attempt_$timestamp"
    train_log="$LOGS_DIR/${run_key}_${timestamp}_train.log"
    mkdir -p "$attempt_root" || {
        append_failed "$group" "$seed" attempt_prepare 1 \
            "could not create $attempt_root"
        return 1
    }
    printf '%s\t%s\t%s\trunning\t0\t%s\t%s\n' \
        "$(now)" "$group" "$seed" "$(remaining_gb 2>/dev/null || printf unknown)" \
        "$attempt_root" >> "$PROGRESS_TSV"

    local -a train_command=(
        env "TRAIN_SEED=$seed"
        "$PYTHON_BIN" -u train.py
        --config_file "$BASE_CONFIG"
        --name "$run_key"
        --output_dir "$attempt_root"
    )
    [[ "$group" != DC-dyn ]] || train_command+=(--dcalign_dynamic_fusion)
    if ! assert_train_command "$group" "$seed" "${train_command[@]}"; then
        append_failed "$group" "$seed" train_command_assert 2 \
            "generated train command violates seed/config/dynamic-fusion contract"
        return 1
    fi
    log "[RUN_START] group=$group seed=$seed attempt=$attempt_root"
    (
        cd "$DCALIGN_ROOT" || exit 2
        "${train_command[@]}"
    ) 2>&1 | tee -a "$train_log"
    train_rc="${PIPESTATUS[0]}"
    if ((train_rc != 0)); then
        append_failed "$group" "$seed" train "$train_rc" \
            "DCAlign training failed; all artifacts retained"
        return 1
    fi
    if ! validate_train_log "$train_log"; then
        append_failed "$group" "$seed" dataset_stats_assert 1 \
            "training log lacks exact train/test dataset statistics"
        return 1
    fi
    config_file="$(find_single_file "$attempt_root" configs.yaml)" || {
        append_failed "$group" "$seed" config_discovery 1 \
            "expected exactly one saved configs.yaml"
        return 1
    }
    last_checkpoint="$(find_single_file "$attempt_root" last.pth)" || {
        append_failed "$group" "$seed" last_checkpoint_discovery 1 \
            "expected exactly one last.pth"
        return 1
    }
    best_checkpoint="$(find_single_file "$attempt_root" best.pth)" || {
        append_failed "$group" "$seed" best_checkpoint_discovery 1 \
            "expected exactly one best.pth"
        return 1
    }
    validate_saved_config "$config_file" "$group" "$seed" || {
        append_failed "$group" "$seed" saved_config_assert 1 \
            "saved config does not prove seed/fusion semantics"
        return 1
    }
    validate_checkpoint "$last_checkpoint" || {
        append_failed "$group" "$seed" last_checkpoint_assert 1 \
            "last.pth lacks a top-level model mapping"
        return 1
    }
    validate_checkpoint "$best_checkpoint" || {
        append_failed "$group" "$seed" best_checkpoint_assert 1 \
            "best.pth lacks a top-level model mapping"
        return 1
    }

    local score_work="$attempt_root/eval_scores"
    local fragment_work="$attempt_root/eval_fragments"
    local unit_metrics="$attempt_root/unit_metrics.tsv"
    mkdir -p "$score_work/last" "$score_work/best" "$fragment_work" || return 1
    printf '%b\n' 'group\tseed\tscenario\tmode\tstrength\tkernel\tdseed\tsplit\tR1\tR5\tR10\tmAP\tmINP' > "$unit_metrics"
    local -a pending_score_sources=()
    local -a pending_score_targets=()
    local scenario_spec scenario mode strength kernel dseed
    local score_file fragment eval_log
    for scenario_spec in "${CORE_SCENARIOS[@]}"; do
        IFS='|' read -r scenario mode strength kernel dseed <<< "$scenario_spec"
        score_file="$score_work/last/${scenario}.pt"
        fragment="$fragment_work/last_${scenario}.tsv"
        eval_log="$LOGS_DIR/${run_key}_${timestamp}_last_${scenario}.log"
        if ! run_eval_scenario "$last_checkpoint" "$group" "$seed" "$scenario" \
            "$mode" "$strength" "$kernel" "$dseed" "$config_file" \
            "$score_file" "$fragment" "$eval_log"; then
            return 1
        fi
        tail -n 1 "$fragment" >> "$unit_metrics" || return 1
        pending_score_sources+=("$score_file")
        pending_score_targets+=("$SIM_DIR/$run_key/last/${scenario}.pt")
    done

    local best_fragment="$fragment_work/best_clean.tsv"
    local best_score="$score_work/best/clean.pt"
    local best_log="$LOGS_DIR/${run_key}_${timestamp}_best_clean.log"
    if ! run_eval_scenario "$best_checkpoint" "${group}-best" "$seed" clean \
        none 1.0 7 2026 "$config_file" "$best_score" "$best_fragment" \
        "$best_log"; then
        return 1
    fi
    tail -n 1 "$best_fragment" >> "$unit_metrics" || return 1
    pending_score_sources+=("$best_score")
    pending_score_targets+=("$SIM_DIR/$run_key/best/clean.pt")

    local -a publish_arguments=(
        "${pending_score_sources[@]}" "${pending_score_targets[@]}"
    )
    if ! publish_scores "$attempt_root" "${publish_arguments[@]}"; then
        append_failed "$group" "$seed" score_publish 1 \
            "atomic score publication failed; checkpoints retained"
        return 1
    fi
    if ! commit_unit_metrics "$unit_metrics" "$group" "$seed"; then
        append_failed "$group" "$seed" metrics_commit 1 \
            "12-row metrics commit failed; checkpoints retained"
        return 1
    fi
    if ! commit_reproduction_warning "$best_fragment" "$group" "$seed" \
        "$best_checkpoint"; then
        append_failed "$group" "$seed" warning_commit 1 \
            "literature warning update failed; checkpoints retained"
        return 1
    fi

    local last_sha best_sha score_manifest_sha elapsed disk_gb mark_tmp
    last_sha="$(sha256sum "$last_checkpoint" | awk '{print $1}')" || return 1
    best_sha="$(sha256sum "$best_checkpoint" | awk '{print $1}')" || return 1
    score_manifest_sha="$(compute_score_manifest_sha "$group" "$seed")" || {
        append_failed "$group" "$seed" score_manifest 1 \
            "could not hash the 12 published score dumps"
        return 1
    }
    elapsed="$(($(date +%s) - run_start))"
    disk_gb="$(remaining_gb 2>/dev/null || printf unknown)"
    mark_tmp="${mark}.tmp.$$"
    if ! printf 'group=%s\nseed=%s\ncompleted_at=%s\nmetrics_rows=12\nscore_files=12\nlast_checkpoint=%s\nbest_checkpoint=%s\nlast_sha256=%s\nbest_sha256=%s\nscore_manifest_sha256=%s\ndcalign_head=%s\ndcalign_diff_sha256=%s\nbase_config_sha256=%s\nirra_kit_sha256=%s\nattempt_root=%s\n' \
        "$group" "$seed" "$(now)" "$last_checkpoint" "$best_checkpoint" \
        "$last_sha" "$best_sha" "$score_manifest_sha" "$DCALIGN_HEAD" \
        "$DCALIGN_DIFF_SHA256" "$BASE_CONFIG_SHA256" "$IRRA_KIT_SHA256" \
        "$attempt_root" \
        > "$mark_tmp"; then
        append_failed "$group" "$seed" marker_prepare 1 \
            "could not prepare completion marker"
        return 1
    fi
    if ! mv -f -- "$mark_tmp" "$mark"; then
        append_failed "$group" "$seed" marker_commit 1 \
            "could not publish completion marker"
        return 1
    fi
    printf '%s\t%s\t%s\tcompleted\t%s\t%s\t%s\n' \
        "$(now)" "$group" "$seed" "$elapsed" "$disk_gb" "$attempt_root" \
        >> "$PROGRESS_TSV"
    log "[RUN_DONE] group=$group seed=$seed metrics_rows=12 scores=12 checkpoints_retained=last,best"
}

for group in "${ARM_NAMES[@]}"; do
    first_seed_failed=0
    for seed in "${SEEDS[@]}"; do
        if ((first_seed_failed)); then
            append_failed "$group" "$seed" skipped_after_first_seed 1 \
                "first seed ${SEEDS[0]} failed for this arm; remaining seeds skipped"
            continue
        fi
        if ! run_one "$group" "$seed"; then
            [[ "$seed" != "${SEEDS[0]}" ]] || first_seed_failed=1
        fi
    done
done

success_count=0
for group in "${ARM_NAMES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        mark="$MARKS_DIR/${group}_seed${seed}.done"
        if [[ -f "$mark" ]] \
            && verify_completed_unit "$mark" "$group" "$seed" >/dev/null 2>&1; then
            success_count=$((success_count + 1))
        fi
    done
done
total_runs=$((${#ARM_NAMES[@]} * ${#SEEDS[@]}))
failed_count=$((total_runs - success_count))
grid_exit_code=0
((failed_count == 0)) || grid_exit_code=1
total_elapsed="$(($(date +%s) - GLOBAL_START))"
disk_gb="$(remaining_gb 2>/dev/null || printf unknown)"
log "[GRID_SUMMARY] total_runs=$total_runs successful_runs=$success_count failed_runs=$failed_count total_elapsed_seconds=$total_elapsed remaining_disk_gb=$disk_gb"

if [[ "$AUTO_SHUTDOWN" != 1 ]]; then
    log "[SHUTDOWN] AUTO_SHUTDOWN=$AUTO_SHUTDOWN; automatic shutdown disabled"
    exit "$grid_exit_code"
fi
log "[SHUTDOWN] automatic shutdown scheduled after a 60-second grace period"
for remaining in $(seq 60 -1 1); do
    if [[ -e "$OUT/STOP" ]]; then
        log "[SHUTDOWN] cancelled because $OUT/STOP exists"
        exit "$grid_exit_code"
    fi
    if ((remaining == 60 || remaining % 10 == 0 || remaining <= 5)); then
        log "[SHUTDOWN] countdown=${remaining}s; create $OUT/STOP to cancel"
    fi
    sleep 1
done
if [[ -e "$OUT/STOP" ]]; then
    log "[SHUTDOWN] cancelled because $OUT/STOP exists"
    exit "$grid_exit_code"
fi
log "[SHUTDOWN] grace period complete; executing shutdown -h now"
shutdown -h now
shutdown_rc=$?
if ((shutdown_rc != 0)); then
    log "[SHUTDOWN] command failed exit=$shutdown_rc"
    exit "$shutdown_rc"
fi
exit "$grid_exit_code"
