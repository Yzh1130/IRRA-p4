"""Paired PID-cluster bootstrap of per_query.csv (B minus A, percentage points).

Resample identities with replacement; all queries and scenarios of a sampled
identity travel together. Scenario means retain the original text-query weighting;
the degraded aggregate averages scenario metrics without including clean.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import random


def load_records(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"query_id", "pid", "caption", "scenario", "best_pos_rank", "AP", "n_pos"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Missing per_query columns in {path}: {sorted(required - set(reader.fieldnames or []))}")
        records = []
        for row in reader:
            parsed = dict(row)
            parsed.update(pid=int(row["pid"]), best_pos_rank=int(row["best_pos_rank"]),
                          AP=float(row["AP"]), n_pos=int(row["n_pos"]))
            if parsed["best_pos_rank"] < 1 or parsed["n_pos"] < 1 or not math.isfinite(parsed["AP"]) or not 0 <= parsed["AP"] <= 1:
                raise ValueError(f"Invalid retrieval row in {path}: {row['query_id']} / {row['scenario']}")
            records.append(parsed)
    if not records:
        raise ValueError(f"Empty per_query CSV: {path}")
    return records


def align_records(records_a, records_b):
    def indexed(records):
        result = {}
        for row in records:
            key = (str(row["scenario"]), str(row["query_id"]))
            if key in result:
                raise ValueError(f"Duplicate scenario/query pair: {key}")
            result[key] = row
        return result

    a, b = indexed(records_a), indexed(records_b)
    if set(a) != set(b):
        raise ValueError("Paired CSVs have different scenario/query keys")
    query_identities = {}
    scenario_queries = {}
    for key in sorted(a):
        for field in ("pid", "caption", "n_pos"):
            if a[key][field] != b[key][field]:
                raise ValueError(f"Paired CSV {field} mismatch at {key}")
        scenario, query_id = key
        identity = (int(a[key]["pid"]), str(a[key]["caption"]), int(a[key]["n_pos"]))
        if query_id in query_identities and query_identities[query_id] != identity:
            raise ValueError(f"Query identity/caption/n_pos changes across scenarios: {query_id}")
        query_identities[query_id] = identity
        scenario_queries.setdefault(scenario, set()).add(query_id)
    query_sets = list(scenario_queries.values())
    if any(keys != query_sets[0] for keys in query_sets[1:]):
        raise ValueError("Queries differ across scenarios within a paired CSV")
    return a, b


def _percentile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def paired_bootstrap(records_a, records_b, n_resamples=1000, seed=2026, scenarios=None):
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    a, b = align_records(records_a, records_b)
    available = sorted({key[0] for key in a})
    chosen = available if scenarios is None else list(scenarios)
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(available):
        raise ValueError("Invalid requested bootstrap scenarios")
    pids = sorted({int(a[key]["pid"]) for key in a if key[0] in chosen})
    # Preaggregate per PID/scenario. Repeated sampled identities repeat all of
    # their text rows, instead of incorrectly giving every PID equal query weight.
    stats = {scenario: {} for scenario in chosen}
    for key, row_a in a.items():
        scenario = key[0]
        if scenario not in stats:
            continue
        row_b, pid = b[key], int(row_a["pid"])
        current = stats[scenario].setdefault(pid, [0, 0.0, 0.0])
        current[0] += 1
        current[1] += float(row_b["best_pos_rank"] == 1) - float(row_a["best_pos_rank"] == 1)
        current[2] += float(row_b["AP"]) - float(row_a["AP"])

    def differences(sampled_pids):
        result = {}
        for scenario in chosen:
            entries = [stats[scenario][pid] for pid in sampled_pids]
            count = sum(entry[0] for entry in entries)
            result[scenario] = (sum(entry[1] for entry in entries) / count * 100,
                                sum(entry[2] for entry in entries) / count * 100)
        return result

    original = differences(pids)
    degraded = [scenario for scenario in chosen if scenario != "clean"]
    aggregate_names = {"mean_all": chosen}
    if degraded:
        aggregate_names["mean_degraded"] = degraded
    draws = {scenario: [[], []] for scenario in chosen}
    draws.update({name: [[], []] for name in aggregate_names})
    rng = random.Random(seed)
    for _ in range(n_resamples):
        sampled = [pids[rng.randrange(len(pids))] for _ in pids]
        values = differences(sampled)
        for scenario in chosen:
            for index in (0, 1):
                draws[scenario][index].append(values[scenario][index])
        for name, members in aggregate_names.items():
            for index in (0, 1):
                draws[name][index].append(sum(values[s][index] for s in members) / len(members))

    def summarize(name, estimates):
        return {metric: {
            "difference": estimates[index],
            "ci95": [_percentile(draws[name][index], 0.025), _percentile(draws[name][index], 0.975)],
        } for index, metric in enumerate(("R1", "mAP"))}

    aggregate = {name: summarize(name, tuple(sum(original[s][index] for s in members) / len(members)
                                            for index in (0, 1)))
                 for name, members in aggregate_names.items()}
    return {
        "direction": "B minus A", "units": "percentage_points", "cluster": "pid",
        "seed": seed, "n_resamples": n_resamples, "n_pids": len(pids),
        "n_queries": len({key[1] for key in a if key[0] in chosen}),
        "scenario_weighting": "unweighted mean of scenario query means",
        "query_weighting": "each caption equally weighted; resample PID clusters",
        "per_scenario": {scenario: summarize(scenario, original[scenario]) for scenario in chosen},
        **aggregate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="Reference per_query.csv")
    parser.add_argument("--b", required=True, help="Comparison per_query.csv")
    parser.add_argument("--output", required=True, help="Output JSON")
    parser.add_argument("--n_resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scenarios", nargs="+", default=None)
    cli = parser.parse_args()
    result = paired_bootstrap(load_records(cli.a), load_records(cli.b), cli.n_resamples, cli.seed, cli.scenarios)
    result["inputs"] = {"A": str(Path(cli.a).resolve()), "B": str(Path(cli.b).resolve())}
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
