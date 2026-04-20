"""
Build a high-quality synthetic caption bank using an LLM for paraphrasing.

Why this exists
---------------
v2/v3/v4 use rule-based regex rewriting and end up either dropping >90% of
captions (-> bank_matches near zero) or producing paraphrases that are
lexically too close to the original to act as a meaningful augmentation.

v5 takes a different approach:
  - For each unique train image, take ONE canonical source caption.
  - Ask an LLM to produce K attribute-preserving paraphrases that VARY
    wording but keep all visual attributes (color, clothing, hair,
    accessory, action, objects).
  - Emit a bank with the same format v2/v3/v4 use:
      { "<basename(img_path)>": ["cap1", "cap2", ...] }
    i.e. check_synth_caption_bank.py will accept it as-is.

Transport
---------
We call an OpenAI-compatible Chat Completions endpoint. This works with:
  * vLLM   -> python -m vllm.entrypoints.openai.api_server --model <hf_id>
  * SGLang -> python -m sglang.launch_server ...
  * Ollama -> `ollama serve` exposes /v1 on 11434 since 0.1.28
  * Any hosted provider that speaks OpenAI's /v1/chat/completions schema

Resume
------
The run writes a temporary shard (output_file + ".shard.jsonl") as it goes.
On restart we reload the shard and only call the LLM for keys we have not
finished yet. This matters because at ~5500 keys with ~3s/key you get
~5 hours of wall time, and you don't want a dropped SSH session to destroy
that.

Recommended defaults for RGBT-PEDES + Qwen2.5-7B-Instruct on a 4090:
  --captions_per_image 4
  --temperature 0.7
  --concurrency 8      (OK on vLLM; lower if the server is small)
"""
import argparse
import json
import os
import os.path as op
import re
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Match the sys.path convention used by v2/v3/v4.
PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), ".."))
TOOL_ROOT = op.abspath(op.dirname(__file__))
for path in (PROJECT_ROOT, TOOL_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

# Reuse v2's canonical source picker so we start from the same caption that
# the rule-based path starts from. Keeps behaviour comparable.
from build_large_synth_bank_v2 import (
    COLOR_WORDS,
    DATASET_FACTORY,
    _choose_canonical_source,
)


# ----------------------------------------------------------------------------
# Attribute preservation check
# ----------------------------------------------------------------------------
#
# We do a light-weight check to detect LLM outputs that have dropped obvious
# attributes (e.g. the original said "red jacket" and the paraphrase said
# just "shirt"). This catches the common failure mode where the LLM
# hallucinates a simpler description.
#
# We deliberately keep the check cheap and permissive: if the rewrite keeps
# at least MIN_ATTR_KEEP_RATIO of the original attribute tokens, we accept.

CLOTHING_ANCHOR_WORDS = {
    "shirt", "tshirt", "t-shirt", "top", "coat", "jacket", "hoodie",
    "jumper", "sweater", "sleeves", "sleeve",
    "pants", "trousers", "jeans", "shorts", "skirt", "dress",
    "shoes", "sneakers", "boots",
    "bag", "backpack", "shoulder bag", "handbag", "purse",
    "hat", "cap", "glasses", "sunglasses",
    "hair",
}

ACTION_ANCHOR_WORDS = {
    "walking", "walks", "walk", "standing", "stands", "stand",
    "riding", "carrying", "carries", "holding", "holds", "running", "runs",
}

MIN_ATTR_KEEP_RATIO = 0.75  # paraphrase must preserve >=75% of anchor tokens
MIN_WORD_COUNT = 8
MAX_WORD_COUNT = 40
BANNED_PRONOUNS = {"he", "she", "his", "her", "their", "hers", "him"}


def _token_set(text, vocab):
    toks = set(re.findall(r"[a-z-]+", str(text).lower()))
    return toks & vocab


def _color_set(text):
    return set(re.findall(
        r"\b(?:" + "|".join(sorted(COLOR_WORDS, key=len, reverse=True)) + r")\b",
        str(text).lower(),
    ))


def _word_count(text):
    return len(re.findall(r"[a-z-]+", str(text).lower()))


def _has_banned_pronoun(text):
    words = set(re.findall(r"[a-z]+", str(text).lower()))
    return bool(words & BANNED_PRONOUNS)


def attribute_preservation_report(original, rewrite):
    """Return (ok: bool, reason: str). Cheap lexical check only."""
    wc = _word_count(rewrite)
    if wc < MIN_WORD_COUNT:
        return False, "too_short"
    if wc > MAX_WORD_COUNT:
        return False, "too_long"
    if _has_banned_pronoun(rewrite):
        return False, "pronoun_residual"

    vocab = CLOTHING_ANCHOR_WORDS | ACTION_ANCHOR_WORDS
    orig_anchors = _token_set(original, vocab)
    rew_anchors = _token_set(rewrite, vocab)
    if orig_anchors:
        keep_ratio = len(orig_anchors & rew_anchors) / float(len(orig_anchors))
        if keep_ratio < MIN_ATTR_KEEP_RATIO:
            return False, f"anchor_loss_keep={keep_ratio:.2f}"

    orig_colors = _color_set(original)
    rew_colors = _color_set(rewrite)
    # We allow dropping at most one color word (LLM might merge "black and
    # white" into "monochrome"). More than that is suspicious.
    missing_colors = orig_colors - rew_colors
    if len(missing_colors) >= 2:
        return False, f"color_loss={sorted(missing_colors)}"

    extra_colors = rew_colors - orig_colors
    if extra_colors:
        return False, f"color_hallucination={sorted(extra_colors)}"

    return True, "ok"


# ----------------------------------------------------------------------------
# LLM prompt
# ----------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You paraphrase pedestrian descriptions for a text-to-image retrieval "
    "dataset. Your job: produce multiple lexically diverse paraphrases that "
    "preserve EVERY visible attribute of the original description without "
    "adding or removing any."
)

# Few-shot pair picked to cover typical RGBT-PEDES content: clothing colour,
# hair, accessory, action. Works well with Qwen2.5-Instruct and Llama-3.1-Instruct.
USER_PROMPT_TEMPLATE = """Rewrite the description below into {K} paraphrases.

STRICT RULES:
1. Preserve EVERY visual attribute: colors, clothing items, hair description, bags/accessories, objects carried, actions.
2. Do NOT add new attributes or details not in the original.
3. Do NOT drop any attribute mentioned in the original.
4. Do NOT use pronouns (he/she/his/her/their). Use neutral phrasing ("a person", "a pedestrian", "a figure").
5. Vary word order and sentence structure across the {K} paraphrases.
6. Each paraphrase: 10-30 words, ends with a period.

Output: a single JSON array of exactly {K} strings. No prose, no markdown, no explanation.

--- EXAMPLES ---

Original: "The woman is wearing a white jacket and blue jeans. She has long black hair and carries a brown backpack."

Output:
[
  "A person with long black hair wearing a white jacket and blue jeans, carrying a brown backpack.",
  "A pedestrian in a white jacket and blue jeans, long black hair, brown backpack on the shoulder.",
  "A figure wearing a white jacket paired with blue jeans, long black hair, carrying a brown backpack.",
  "A person carries a brown backpack, dressed in a white jacket and blue jeans, with long black hair."
]

--- YOUR TURN ---

Original: "{caption}"

Output:"""


def build_prompt(caption, K):
    return USER_PROMPT_TEMPLATE.format(caption=caption.strip(), K=K)


# ----------------------------------------------------------------------------
# OpenAI-compatible client (thin, no SDK dependency)
# ----------------------------------------------------------------------------

def _post_chat_completion(
    base_url, api_key, model, messages, temperature, max_tokens, timeout
):
    """Minimal POST to /v1/chat/completions with one retry on 5xx."""
    import urllib.request
    import urllib.error

    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # JSON-only response; vLLM / SGLang respect this flag; Ollama ignores
        # it silently which is fine because our prompt already demands JSON.
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")

    last_err = None
    for attempt in range(2):
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code < 500:
                raise
            time.sleep(1.0 + attempt)
        except Exception as e:
            last_err = e
            time.sleep(1.0 + attempt)
    raise last_err


def _strip_json_fence(text):
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_llm_json_array(raw):
    """Tolerate both {"paraphrases": [...]} and bare [...] shapes."""
    cleaned = _strip_json_fence(raw)
    try:
        obj = json.loads(cleaned)
    except Exception:
        # Model sometimes trails explanation after the array. Grab the first
        # well-formed JSON array.
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON array found in: {cleaned[:200]!r}")
        obj = json.loads(match.group(0))

    if isinstance(obj, dict):
        for key in ("paraphrases", "output", "results", "data"):
            if key in obj and isinstance(obj[key], list):
                obj = obj[key]
                break
        else:
            raise ValueError(f"unexpected JSON shape: {list(obj.keys())}")

    if not isinstance(obj, list):
        raise ValueError(f"expected list, got {type(obj).__name__}")
    return [str(x).strip() for x in obj if str(x).strip()]


def generate_one(client_cfg, source_caption, K):
    """Return (list_of_kept_paraphrases, list_of_drop_reasons). Never raises on
    model failure; returns empty kept list if the call or parse fails."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(source_caption, K)},
    ]
    try:
        raw = _post_chat_completion(
            base_url=client_cfg["base_url"],
            api_key=client_cfg["api_key"],
            model=client_cfg["model"],
            messages=messages,
            temperature=client_cfg["temperature"],
            max_tokens=client_cfg["max_tokens"],
            timeout=client_cfg["timeout"],
        )
    except Exception as e:
        return [], [f"llm_error={type(e).__name__}"]

    try:
        paraphrases = _parse_llm_json_array(raw)
    except Exception as e:
        return [], [f"json_parse_error={type(e).__name__}"]

    kept, dropped = [], []
    for p in paraphrases:
        ok, reason = attribute_preservation_report(source_caption, p)
        if ok:
            kept.append(p)
        else:
            dropped.append(reason)
    return kept, dropped


# ----------------------------------------------------------------------------
# Train set collection (one canonical source per unique image)
# ----------------------------------------------------------------------------

def _extract_img_path_from_train_sample(sample):
    if len(sample) in (4, 5):
        return sample[2]
    raise ValueError(
        f"Unsupported train sample format. Expected len=4 or 5, got {len(sample)}."
    )


def _collect_train_jobs(train_samples, max_train_keys=-1):
    """Returns list of (key, source_caption) pairs, one per unique image."""
    groups = OrderedDict()
    for sample in train_samples:
        image_id = sample[1]
        if image_id not in groups:
            if max_train_keys > 0 and len(groups) >= max_train_keys:
                continue
            img_path = _extract_img_path_from_train_sample(sample)
            groups[image_id] = {
                "key": op.basename(str(img_path)),
                "captions": [],
            }
        groups[image_id]["captions"].append(sample[-1])

    jobs = []
    for g in groups.values():
        source = _choose_canonical_source(g["captions"])
        jobs.append((g["key"], source))
    return jobs


# ----------------------------------------------------------------------------
# Resume support: shard.jsonl is our checkpoint
# ----------------------------------------------------------------------------

def _load_shard(shard_path):
    done = {}
    if not op.isfile(shard_path):
        return done
    with open(shard_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "key" in rec and "captions" in rec:
                done[rec["key"]] = rec
    return done


def _append_shard(shard_path, record):
    with open(shard_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an LLM-paraphrased synthetic caption bank (v5)."
    )
    parser.add_argument("--dataset_name", default="RGBT-PEDES",
                        choices=list(DATASET_FACTORY.keys()))
    parser.add_argument("--root_dir", default="./data")
    parser.add_argument("--output_file", default="./synth_caption_bank_v5.json")
    parser.add_argument("--captions_per_image", type=int, default=4,
                        help="K paraphrases requested per image")
    parser.add_argument("--min_kept_per_image", type=int, default=1,
                        help="images with fewer accepted paraphrases than "
                             "this are excluded from the final bank")

    parser.add_argument("--max_train_keys", type=int, default=-1,
                        help="debug: limit number of train keys processed; "
                             "-1 means all")

    # LLM endpoint
    parser.add_argument("--base_url", type=str,
                        default=os.environ.get("OPENAI_BASE_URL",
                                               "http://127.0.0.1:8000/v1"),
                        help="OpenAI-compatible base url (e.g. vLLM at "
                             "http://127.0.0.1:8000/v1 or "
                             "https://api.openai.com/v1)")
    parser.add_argument("--api_key", type=str,
                        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                        help="API key. vLLM / Ollama accept any non-empty "
                             "string.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=600)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--concurrency", type=int, default=8)

    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Collect jobs
    dataset = DATASET_FACTORY[args.dataset_name](root=args.root_dir, verbose=False)
    jobs = _collect_train_jobs(dataset.train, max_train_keys=args.max_train_keys)
    print(f"[SYNTH-V5] dataset_name={args.dataset_name}")
    print(f"[SYNTH-V5] caption_file={dataset.anno_path}")
    print(f"[SYNTH-V5] total_train_keys={len(jobs)}")
    print(f"[SYNTH-V5] captions_per_image={args.captions_per_image}")
    print(f"[SYNTH-V5] min_kept_per_image={args.min_kept_per_image}")
    print(f"[SYNTH-V5] model={args.model}")
    print(f"[SYNTH-V5] base_url={args.base_url}")
    print(f"[SYNTH-V5] concurrency={args.concurrency}")

    # 2. Load resume shard
    output_file = op.abspath(args.output_file)
    shard_path = output_file + ".shard.jsonl"
    output_dir = op.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    already_done = _load_shard(shard_path)
    pending = [(k, s) for (k, s) in jobs if k not in already_done]
    print(f"[SYNTH-V5] resume: already_done={len(already_done)} "
          f"pending={len(pending)} shard={shard_path}")

    # 3. Fire LLM calls in parallel; stream results to shard for resumability
    client_cfg = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
    }

    num_done = len(already_done)
    num_llm_errors = 0
    num_parse_errors = 0
    num_kept_total = 0
    num_dropped_total = 0
    drop_reason_counts = {}
    t0 = time.time()

    def _worker(key, source):
        kept, dropped = generate_one(client_cfg, source, args.captions_per_image)
        return key, source, kept, dropped

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futures = [ex.submit(_worker, k, s) for (k, s) in pending]
        for fut in as_completed(futures):
            key, source, kept, dropped = fut.result()
            record = {
                "key": key,
                "source": source,
                "captions": kept,
                "dropped_reasons": dropped,
            }
            _append_shard(shard_path, record)
            already_done[key] = record

            num_kept_total += len(kept)
            num_dropped_total += len(dropped)
            for r in dropped:
                drop_reason_counts[r] = drop_reason_counts.get(r, 0) + 1
                if r.startswith("llm_error"):
                    num_llm_errors += 1
                elif r.startswith("json_parse_error"):
                    num_parse_errors += 1

            num_done += 1
            if num_done % args.log_every == 0 or num_done == len(jobs):
                elapsed = time.time() - t0
                rate = (num_done - len(already_done) + len(pending)) / max(elapsed, 1e-6)
                print(
                    f"[SYNTH-V5] progress={num_done}/{len(jobs)} "
                    f"kept={num_kept_total} dropped={num_dropped_total} "
                    f"llm_err={num_llm_errors} parse_err={num_parse_errors} "
                    f"rate={rate:.1f}/s",
                    flush=True,
                )

    # 4. Materialize final bank (apply min_kept_per_image filter)
    bank = OrderedDict()
    excluded_low_keep = 0
    for key, _source in jobs:
        rec = already_done.get(key)
        if not rec:
            excluded_low_keep += 1
            continue
        caps = rec["captions"]
        if len(caps) < args.min_kept_per_image:
            excluded_low_keep += 1
            continue
        bank[key] = caps

    with open(output_file, "w") as f:
        json.dump(bank, f, indent=2, ensure_ascii=False)

    # 5. Summary
    total_captions_in_bank = sum(len(v) for v in bank.values())
    avg = total_captions_in_bank / float(len(bank)) if bank else 0.0
    print(f"[SYNTH-V5] output_file={output_file}")
    print(f"[SYNTH-V5] bank_total_keys={len(bank)}")
    print(f"[SYNTH-V5] bank_total_captions={total_captions_in_bank}")
    print(f"[SYNTH-V5] avg_captions_per_key={avg:.3f}")
    print(f"[SYNTH-V5] excluded_low_keep_keys={excluded_low_keep}")
    print(f"[SYNTH-V5] num_llm_errors={num_llm_errors}")
    print(f"[SYNTH-V5] num_parse_errors={num_parse_errors}")
    print("[SYNTH-V5] drop_reason_counts_top10:")
    for r, c in sorted(drop_reason_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  - {r}: {c}")
    print(
        "[SYNTH-V5] NEXT STEPS:\n"
        f"  1) python check_synth_caption_bank.py --dataset_name {args.dataset_name} "
        f"--root_dir {args.root_dir} --synth_caption_bank_file {output_file}\n"
        "  2) train with --use_synth_caption_bank "
        f"--synth_caption_bank_file {output_file} --synth_caption_keep_per_image 2"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
