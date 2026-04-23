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
BANNED_PRONOUNS = {
    "he", "she", "his", "her", "their", "hers", "him",
    "they", "them", "theirs",
}
PARSER_MODE = "tolerant_v5_1"
PARSER_FAILURE_REASONS = {"no_json_found", "invalid_json", "wrong_schema"}


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


def attribute_preservation_report(
    original, rewrite, pronoun_reason="pronoun_residual"
):
    """Return (ok: bool, reason: str). Cheap lexical check only."""
    wc = _word_count(rewrite)
    if wc < MIN_WORD_COUNT:
        return False, "too_short"
    if wc > MAX_WORD_COUNT:
        return False, "too_long"
    if _has_banned_pronoun(rewrite):
        return False, pronoun_reason

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
    "Rewrite pedestrian descriptions. Preserve visible attributes only. "
    "Return valid JSON only."
)

# Keep the prompt short and concrete for small instruct models. The example is
# generated with exactly K items to reduce "copied example length" failures.
USER_PROMPT_TEMPLATE = """Output valid JSON only:
{{"captions": ["sentence 1", "sentence 2"]}}

Task: rewrite Original into exactly {K} attribute-preserving captions.

Hard rules:
- JSON object only; one field named "captions"; no markdown, code block, or explanation.
- "captions" must be an array of exactly {K} complete English sentences.
- Do not output a bare array or text before/after the JSON object.
- Do not use pronouns: he, she, his, her, hers, him, they, them, their, theirs.
- Do not invent new visual attributes; do not drop clothing, colors, actions, hair, bags, or carried objects.
- Use neutral nouns such as "a person", "a pedestrian", or "a figure".

Good JSON example:
{example}

Original: "{caption}"
JSON:"""


def _build_format_example(K):
    example_pool = [
        "A person in a white jacket and blue jeans carries a brown backpack.",
        "A pedestrian carries a brown backpack while wearing a white jacket and blue jeans.",
        "A figure with long black hair wears a white jacket, blue jeans, and a brown backpack.",
        "A person wearing a white jacket and blue jeans walks with a brown backpack.",
    ]
    count = max(1, int(K))
    captions = [example_pool[i % len(example_pool)] for i in range(count)]
    return json.dumps({"captions": captions}, ensure_ascii=False)


def build_prompt(caption, K):
    return USER_PROMPT_TEMPLATE.format(
        caption=caption.strip(),
        K=K,
        example=_build_format_example(K),
    )


# ----------------------------------------------------------------------------
# OpenAI-compatible client (thin, no SDK dependency)
# ----------------------------------------------------------------------------

def _post_chat_completion(
    base_url, api_key, model, messages, temperature, max_tokens, timeout,
    response_mode,
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
    }
    if response_mode == "strict_json_object":
        # JSON-only response; vLLM / SGLang respect this flag; Ollama ignores
        # it silently which is fine because our prompt already demands JSON.
        payload["response_format"] = {"type": "json_object"}
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


class LLMParseError(ValueError):
    def __init__(self, reason, detail=""):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


def _extract_first_json_value(raw):
    """Find the first decodable JSON object/array inside a noisy response."""
    cleaned = _strip_json_fence(raw)
    starts = [i for i, ch in enumerate(cleaned) if ch in "[{"]
    if not starts:
        raise LLMParseError("no_json_found", cleaned[:200])

    decoder = json.JSONDecoder()
    for start in starts:
        try:
            obj, _end = decoder.raw_decode(cleaned[start:])
            return obj
        except json.JSONDecodeError:
            continue
    raise LLMParseError("invalid_json", cleaned[:200])


def _list_is_all_strings(value):
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def _select_caption_list(obj):
    """Accept {"captions": [...]}, a bare array, or one unique list[str] value."""
    if isinstance(obj, list):
        return obj

    if not isinstance(obj, dict):
        raise LLMParseError("wrong_schema", f"type={type(obj).__name__}")

    if isinstance(obj.get("captions"), list):
        return obj["captions"]

    candidates = [value for value in obj.values() if _list_is_all_strings(value)]
    if len(candidates) == 1:
        return candidates[0]

    keys = ",".join(str(k) for k in obj.keys())
    raise LLMParseError("wrong_schema", f"keys={keys}")


def _clean_caption_text(text):
    text = str(text).strip()
    text = re.sub(r"^\s*(?:[-*]\s*|\d+\s*[\.\)\:-]\s*)", "", text)
    text = text.strip().strip("\"'")
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _parse_llm_captions(raw, expected_count):
    """Return (captions, parser_reasons) without throwing for partial issues."""
    obj = _extract_first_json_value(raw)
    values = _select_caption_list(obj)

    captions = []
    reasons = []
    for item in values:
        if not isinstance(item, str):
            reasons.append("non_string_item")
            continue
        cleaned = _clean_caption_text(item)
        if not cleaned:
            reasons.append("empty_after_clean")
            continue
        captions.append(cleaned)

    if len(captions) > expected_count:
        captions = captions[:expected_count]
    if not captions and "empty_after_clean" not in reasons:
        reasons.append("empty_after_clean")
    if len(captions) < expected_count and "too_few_items" not in reasons:
        reasons.append("too_few_items")

    return captions, reasons


_SUBJECT_PRONOUN_REWRITES = [
    (re.compile(r"\b(?:he|she)\s+is\s+", re.IGNORECASE), "a person is "),
    (re.compile(r"\b(?:he|she)\s+has\s+", re.IGNORECASE), "a person has "),
    (re.compile(r"\b(?:he|she)\s+wears\s+", re.IGNORECASE), "a person wears "),
    (re.compile(r"\b(?:he|she)\s+carries\s+", re.IGNORECASE), "a person carries "),
    (re.compile(r"\b(?:he|she)\s+holds\s+", re.IGNORECASE), "a person holds "),
    (re.compile(r"\b(?:he|she)\s+walks\s+", re.IGNORECASE), "a person walks "),
    (re.compile(r"\b(?:he|she)\s+stands\s+", re.IGNORECASE), "a person stands "),
    (re.compile(r"\b(?:he|she)\s+runs\s+", re.IGNORECASE), "a person runs "),
    (re.compile(r"\b(?:he|she)\s+rides\s+", re.IGNORECASE), "a person rides "),
    (re.compile(r"\bthey\s+are\s+", re.IGNORECASE), "a person is "),
    (re.compile(r"\bthey\s+have\s+", re.IGNORECASE), "a person has "),
    (re.compile(r"\bthey\s+wear\s+", re.IGNORECASE), "a person wears "),
    (re.compile(r"\bthey\s+carry\s+", re.IGNORECASE), "a person carries "),
    (re.compile(r"\bthey\s+hold\s+", re.IGNORECASE), "a person holds "),
    (re.compile(r"\bthey\s+walk\s+", re.IGNORECASE), "a person walks "),
    (re.compile(r"\bthey\s+stand\s+", re.IGNORECASE), "a person stands "),
    (re.compile(r"\bthey\s+run\s+", re.IGNORECASE), "a person runs "),
    (re.compile(r"\bthey\s+ride\s+", re.IGNORECASE), "a person rides "),
]

_POSSESSIVE_ATTR_NOUNS = {
    "shirt", "tshirt", "top", "coat", "jacket", "hoodie", "jumper", "sweater",
    "sleeves", "sleeve", "pants", "trousers", "jeans", "shorts", "skirt",
    "dress", "shoes", "sneakers", "boots", "bag", "backpack", "handbag",
    "purse", "hat", "cap", "glasses", "sunglasses", "hair",
}
_PLURAL_OR_MASS_ATTR_NOUNS = {
    "sleeves", "pants", "trousers", "jeans", "shorts", "shoes", "sneakers",
    "boots", "glasses", "sunglasses", "hair",
}
_POSSESSIVE_ATTR_RE = re.compile(
    r"\b(?:his|her|their)\s+"
    r"((?:[a-z-]+\s+){0,4}(?:"
    + "|".join(sorted(_POSSESSIVE_ATTR_NOUNS, key=len, reverse=True))
    + r"))\b",
    re.IGNORECASE,
)


def _article_for_phrase(phrase):
    first = phrase.split()[0].lower()
    return "an" if first[:1] in "aeiou" else "a"


def _rewrite_possessive_attr(match):
    phrase = re.sub(r"\s+", " ", match.group(1).strip())
    noun = phrase.lower().split()[-1]
    if noun in _PLURAL_OR_MASS_ATTR_NOUNS:
        return phrase
    return f"{_article_for_phrase(phrase)} {phrase}"


def _rewrite_banned_pronouns(text):
    rewritten = str(text)
    for pattern, replacement in _SUBJECT_PRONOUN_REWRITES:
        rewritten = pattern.sub(replacement, rewritten)
    rewritten = _POSSESSIVE_ATTR_RE.sub(_rewrite_possessive_attr, rewritten)
    return _clean_caption_text(rewritten)


def generate_one(client_cfg, source_caption, K, pronoun_rewrite_before_drop=False):
    """Return kept captions, true drop reasons, side events, and debug metadata."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(source_caption, K)},
    ]
    meta = {"parse_error": None, "pronoun_fixes": []}
    try:
        raw = _post_chat_completion(
            base_url=client_cfg["base_url"],
            api_key=client_cfg["api_key"],
            model=client_cfg["model"],
            messages=messages,
            temperature=client_cfg["temperature"],
            max_tokens=client_cfg["max_tokens"],
            timeout=client_cfg["timeout"],
            response_mode=client_cfg["response_mode"],
        )
    except Exception as e:
        return [], [f"llm_error={type(e).__name__}"], [], meta

    try:
        paraphrases, parse_reasons = _parse_llm_captions(raw, K)
    except LLMParseError as e:
        meta["parse_error"] = {
            "reason": e.reason,
            "detail": e.detail,
            "raw": str(raw),
        }
        return [], [e.reason], [], meta

    kept, dropped, events = [], list(parse_reasons), []
    for p in paraphrases:
        candidate = p
        pronoun_fixed = False
        if pronoun_rewrite_before_drop and _has_banned_pronoun(candidate):
            events.append("pronoun_residual_before_rewrite")
            rewritten = _rewrite_banned_pronouns(candidate)
            if rewritten != candidate:
                candidate = rewritten
            pronoun_fixed = not _has_banned_pronoun(candidate)

        pronoun_reason = (
            "pronoun_residual_after_rewrite"
            if pronoun_rewrite_before_drop
            else "pronoun_residual"
        )
        ok, reason = attribute_preservation_report(
            source_caption, candidate, pronoun_reason=pronoun_reason
        )
        if ok:
            kept.append(candidate)
            if pronoun_fixed:
                events.append("pronoun_fixed_by_rewrite")
                meta["pronoun_fixes"].append({
                    "before": p,
                    "after": candidate,
                })
        else:
            dropped.append(reason)
    return kept, dropped, events, meta


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
    with open(shard_path, "r", encoding="utf-8") as f:
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


def _append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_shard(shard_path, record):
    _append_jsonl(shard_path, record)


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
    parser.add_argument(
        "--llm_response_mode",
        type=str,
        default="strict_json_object",
        choices=["strict_json_object", "loose_json_text"],
        help="strict_json_object sends response_format={type: json_object}; "
             "loose_json_text relies on prompt + tolerant parser only",
    )
    parser.add_argument(
        "--pronoun_rewrite_before_drop",
        default=False,
        action="store_true",
        help="try conservative pronoun removal before final drop checks",
    )
    parser.add_argument(
        "--debug_sample_limit",
        type=int,
        default=5,
        help="max parse/pronoun debug samples written per run",
    )

    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.captions_per_image < 1:
        raise ValueError("captions_per_image must be >= 1")
    if args.debug_sample_limit < 0:
        raise ValueError("debug_sample_limit must be >= 0")

    # 1. Collect jobs
    dataset = DATASET_FACTORY[args.dataset_name](root=args.root_dir, verbose=False)
    jobs = _collect_train_jobs(dataset.train, max_train_keys=args.max_train_keys)
    print("[SYNTH-V5] version=v5.1")
    print(f"[SYNTH-V5] dataset_name={args.dataset_name}")
    print(f"[SYNTH-V5] caption_file={dataset.anno_path}")
    print(f"[SYNTH-V5] total_train_keys={len(jobs)}")
    print(f"[SYNTH-V5] captions_per_image={args.captions_per_image}")
    print(f"[SYNTH-V5] min_kept_per_image={args.min_kept_per_image}")
    print(f"[SYNTH-V5] model={args.model}")
    print(f"[SYNTH-V5] base_url={args.base_url}")
    print(f"[SYNTH-V5] concurrency={args.concurrency}")
    print(f"[SYNTH-V5] parser_mode={PARSER_MODE}")
    print(f"[SYNTH-V5] llm_response_mode={args.llm_response_mode}")
    print(
        "[SYNTH-V5] pronoun_rewrite_before_drop="
        f"{args.pronoun_rewrite_before_drop}"
    )

    # 2. Load resume shard
    output_file = op.abspath(args.output_file)
    shard_path = output_file + ".shard.jsonl"
    output_dir = op.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    parse_error_sample_path = op.join(output_dir, "parse_error_samples.jsonl")
    pronoun_fix_sample_path = op.join(output_dir, "pronoun_fix_samples.jsonl")
    if args.debug_sample_limit > 0:
        with open(parse_error_sample_path, "w", encoding="utf-8"):
            pass
        with open(pronoun_fix_sample_path, "w", encoding="utf-8"):
            pass

    already_done = _load_shard(shard_path)
    pending = [(k, s) for (k, s) in jobs if k not in already_done]
    print(f"[SYNTH-V5] resume: already_done={len(already_done)} "
          f"pending={len(pending)} shard={shard_path}")
    if args.debug_sample_limit > 0:
        print(f"[SYNTH-V5] parse_error_samples={parse_error_sample_path}")
        print(f"[SYNTH-V5] pronoun_fix_samples={pronoun_fix_sample_path}")

    # 3. Fire LLM calls in parallel; stream results to shard for resumability
    client_cfg = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "response_mode": args.llm_response_mode,
    }

    num_done = len(already_done)
    num_llm_errors = 0
    num_parse_errors = 0
    num_kept_total = 0
    num_dropped_total = 0
    drop_reason_counts = {}
    event_reason_counts = {}
    parse_error_samples_written = 0
    pronoun_fix_samples_written = 0
    t0 = time.time()

    def _worker(key, source):
        kept, dropped, events, meta = generate_one(
            client_cfg,
            source,
            args.captions_per_image,
            pronoun_rewrite_before_drop=args.pronoun_rewrite_before_drop,
        )
        return key, source, kept, dropped, events, meta

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futures = [ex.submit(_worker, k, s) for (k, s) in pending]
        for fut in as_completed(futures):
            key, source, kept, dropped, events, meta = fut.result()
            record = {
                "key": key,
                "source": source,
                "captions": kept,
                "dropped_reasons": dropped,
                "event_reasons": events,
            }
            _append_shard(shard_path, record)
            already_done[key] = record

            num_kept_total += len(kept)
            num_dropped_total += len(dropped)
            for r in dropped:
                drop_reason_counts[r] = drop_reason_counts.get(r, 0) + 1
                if r.startswith("llm_error"):
                    num_llm_errors += 1
                elif r in PARSER_FAILURE_REASONS:
                    num_parse_errors += 1
            for r in events:
                event_reason_counts[r] = event_reason_counts.get(r, 0) + 1

            if (
                args.debug_sample_limit > 0
                and meta.get("parse_error")
                and parse_error_samples_written < args.debug_sample_limit
            ):
                sample = {
                    "key": key,
                    "source": source,
                    **meta["parse_error"],
                }
                _append_jsonl(parse_error_sample_path, sample)
                parse_error_samples_written += 1

            for fix in meta.get("pronoun_fixes", []):
                if (
                    args.debug_sample_limit <= 0
                    or pronoun_fix_samples_written >= args.debug_sample_limit
                ):
                    break
                _append_jsonl(pronoun_fix_sample_path, {
                    "key": key,
                    "source": source,
                    **fix,
                })
                pronoun_fix_samples_written += 1

            num_done += 1
            if num_done % args.log_every == 0 or num_done == len(jobs):
                elapsed = time.time() - t0
                rate = (num_done - len(already_done) + len(pending)) / max(elapsed, 1e-6)
                print(
                    f"[SYNTH-V5] progress={num_done}/{len(jobs)} "
                    f"kept={num_kept_total} dropped={num_dropped_total} "
                    f"llm_err={num_llm_errors} parse_err={num_parse_errors} "
                    f"pronoun_fixed={event_reason_counts.get('pronoun_fixed_by_rewrite', 0)} "
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

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(bank, f, indent=2, ensure_ascii=False)

    # 5. Summary
    total_captions_in_bank = sum(len(v) for v in bank.values())
    avg = total_captions_in_bank / float(len(bank)) if bank else 0.0
    total_records_with_captions = sum(
        len(rec.get("captions", [])) for rec in already_done.values()
    )
    avg_kept_per_processed_key = (
        total_records_with_captions / float(len(already_done))
        if already_done else 0.0
    )
    print(f"[SYNTH-V5] output_file={output_file}")
    print(f"[SYNTH-V5] bank_total_keys={len(bank)}")
    print(f"[SYNTH-V5] bank_total_captions={total_captions_in_bank}")
    print(f"[SYNTH-V5] avg_captions_per_key={avg:.3f}")
    print(
        "[SYNTH-V5] avg_kept_per_processed_key="
        f"{avg_kept_per_processed_key:.3f}"
    )
    print(f"[SYNTH-V5] excluded_low_keep_keys={excluded_low_keep}")
    print(f"[SYNTH-V5] num_llm_errors={num_llm_errors}")
    print(f"[SYNTH-V5] num_parse_errors={num_parse_errors}")
    print(f"[SYNTH-V5] parse_error_samples_written={parse_error_samples_written}")
    print(f"[SYNTH-V5] pronoun_fix_samples_written={pronoun_fix_samples_written}")
    print("[SYNTH-V5] drop_reason_counts_top10:")
    for r, c in sorted(drop_reason_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  - {r}: {c}")
    print("[SYNTH-V5] event_reason_counts_top10:")
    for r, c in sorted(event_reason_counts.items(), key=lambda x: -x[1])[:10]:
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
