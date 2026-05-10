"""
Build a GPT/SynthCap synthetic caption bank with an OpenAI-compatible API.

This is the v5 LLM path standardized for Yunwu/OpenAI-compatible chat
completions while preserving the existing IRRA bank contract:

    { "<basename(rgb_path)>": ["synthetic caption 1", ...] }

The script reuses the existing dataset loaders and canonical caption picker
from build_large_synth_bank_v2.py, writes a JSON object bank for training, and
keeps a .shard.jsonl file for resumable generation.
"""
import argparse
import json
import os
import os.path as op
import re
import sys
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed


PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), ".."))
TOOL_ROOT = op.abspath(op.dirname(__file__))
for path in (PROJECT_ROOT, TOOL_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

# Keep the same dataset/key conventions as the older synthetic bank builders.
from build_large_synth_bank_v2 import (  # noqa: E402
    ACCESSORY_TERMS,
    COLOR_WORDS,
    DATASET_FACTORY,
    GENDER_WORDS,
    _choose_canonical_source,
)


SYSTEM_PROMPT = (
    "You are generating synthetic training captions for a text-to-image person "
    "retrieval dataset.\n"
    "Rewrite the given person description without adding unsupported visual "
    "attributes.\n"
    "Do not invent colors, clothing, accessories, gender, age, race, emotion, "
    "profession, brand, identity, or background.\n"
    "Use simple, concrete English.\n"
    "Output only valid JSON."
)

USER_PROMPT_TEMPLATE = """Given the original person caption below, generate concise rewritten captions for training.

Rules:
- Preserve the original meaning.
- Do not add any detail absent from the original caption.
- If a color is not mentioned, do not invent one.
- If gender is not mentioned, do not infer it.
- If bag, backpack, hat, umbrella, or carried item is not mentioned, do not add it.
- Avoid phrases such as "the image shows", "appears to", "possibly", "maybe", "as an AI".
- Each caption must be one sentence.
- Each caption should be 8 to 25 words.

Output exactly this JSON schema:
{{
  "captions": [
    "...",
    "...",
    "..."
  ]
}}

The three captions should be:
1. a natural complete rewrite;
2. a clothing/color-focused rewrite using only attributes in the original;
3. a structure/silhouette-friendly rewrite using only attributes in the original.

Original caption:
"{caption}"
"""

PARSER_FAILURE_REASONS = {"no_json_found", "invalid_json", "wrong_schema"}
BANNED_PHRASES = (
    "the image shows",
    "appears to",
    "possibly",
    "maybe",
    "as an ai",
)
MIN_WORD_COUNT = 8
MAX_WORD_COUNT = 25
MAX_FAILED_DETAIL_CHARS = 800


class LLMRequestError(RuntimeError):
    def __init__(self, message, status=None, detail=""):
        super().__init__(message)
        self.status = status
        self.detail = detail


class LLMParseError(ValueError):
    def __init__(self, reason, detail=""):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


def _default_api_base():
    return (
        os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://yunwu.ai/v1"
    )


def _mask_api_key(api_key):
    value = str(api_key or "")
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return value[:2] + "***"
    return value[:4] + "..." + value[-4:]


def _build_prompt(caption):
    escaped = str(caption).strip().replace('"', '\\"')
    return USER_PROMPT_TEMPLATE.format(caption=escaped)


def _word_count(text):
    return len(re.findall(r"[a-z0-9-]+", str(text).lower()))


def _normalize_caption(text):
    text = str(text).strip()
    text = re.sub(r"^\s*(?:[-*]\s*|\d+\s*[\.\)\:-]\s*)", "", text)
    text = text.strip().strip("\"'")
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _dedupe_captions(captions, limit):
    seen = set()
    kept = []
    for caption in captions:
        clean = _normalize_caption(caption)
        if not clean:
            continue
        key = re.sub(r"\s+", " ", clean.lower()).strip(" .!?")
        if key in seen:
            continue
        seen.add(key)
        kept.append(clean)
        if limit > 0 and len(kept) >= limit:
            break
    return kept


def _term_set(text, terms):
    lowered = str(text).lower()
    tokens = set(re.findall(r"[a-z-]+", lowered))
    found = set()
    for term in terms:
        term_l = str(term).lower()
        if " " in term_l:
            if re.search(r"\b" + re.escape(term_l) + r"\b", lowered):
                found.add(term_l)
        elif term_l in tokens:
            found.add(term_l)
    return found


def _color_set(text):
    return _term_set(text, COLOR_WORDS)


def _gender_set(text):
    return _term_set(text, GENDER_WORDS)


def _accessory_set(text):
    tracked = set(ACCESSORY_TERMS) | {
        "bag",
        "backpack",
        "hat",
        "cap",
        "umbrella",
        "handbag",
        "purse",
        "phone",
        "item",
    }
    return _term_set(text, tracked)


def attribute_preservation_report(original, rewrite):
    """Cheap guardrail for common hallucinations after the prompt does the work."""
    original = str(original)
    rewrite = _normalize_caption(rewrite)
    lowered = rewrite.lower()

    wc = _word_count(rewrite)
    if wc < MIN_WORD_COUNT:
        return False, "too_short"
    if wc > MAX_WORD_COUNT:
        return False, "too_long"
    if any(phrase in lowered for phrase in BANNED_PHRASES):
        return False, "banned_phrase"

    extra_colors = _color_set(rewrite) - _color_set(original)
    if extra_colors:
        return False, f"color_hallucination={sorted(extra_colors)}"

    orig_gender = _gender_set(original)
    rewrite_gender = _gender_set(rewrite)
    if rewrite_gender and not orig_gender:
        return False, f"gender_hallucination={sorted(rewrite_gender)}"

    extra_accessories = _accessory_set(rewrite) - _accessory_set(original)
    if extra_accessories:
        return False, f"accessory_hallucination={sorted(extra_accessories)}"

    return True, "ok"


def _strip_json_fence(text):
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_first_json_value(raw):
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


def _select_caption_list(obj):
    if isinstance(obj, dict) and isinstance(obj.get("captions"), list):
        return obj["captions"]
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        candidates = [
            value for value in obj.values()
            if isinstance(value, list) and all(isinstance(x, str) for x in value)
        ]
        if len(candidates) == 1:
            return candidates[0]
        raise LLMParseError("wrong_schema", "keys=" + ",".join(map(str, obj.keys())))
    raise LLMParseError("wrong_schema", f"type={type(obj).__name__}")


def _parse_llm_captions(raw, expected_count):
    obj = _extract_first_json_value(raw)
    values = _select_caption_list(obj)
    captions = []
    reasons = []
    for value in values:
        if not isinstance(value, str):
            reasons.append("non_string_item")
            continue
        caption = _normalize_caption(value)
        if not caption:
            reasons.append("empty_after_clean")
            continue
        captions.append(caption)
    if len(captions) > expected_count:
        captions = captions[:expected_count]
    if len(captions) < expected_count:
        reasons.append("too_few_items")
    return captions, reasons


def _http_error_detail(error):
    try:
        detail = error.read().decode("utf-8", errors="replace")
    except Exception:
        detail = str(error)
    return detail[:MAX_FAILED_DETAIL_CHARS]


def _response_format_unsupported(status, detail):
    lowered = str(detail).lower()
    return (
        int(status or 0) in {400, 422}
        and "response_format" in lowered
        and (
            "unsupported" in lowered
            or "not support" in lowered
            or "invalid" in lowered
            or "unrecognized" in lowered
        )
    )


def _post_chat_completion(
    api_base,
    api_key,
    model,
    messages,
    temperature,
    max_tokens,
    timeout,
    max_retries,
    retry_sleep,
    use_response_format=True,
):
    """POST to /chat/completions, with a one-time response_format fallback."""
    url = api_base.rstrip("/") + "/chat/completions"
    attempts_left = max(1, int(max_retries) + 1)
    response_format_enabled = bool(use_response_format)
    fallback_used = False
    last_error = None

    while attempts_left > 0:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if response_format_enabled:
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Authorization", f"Bearer {api_key or ''}")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"], fallback_used
        except urllib.error.HTTPError as error:
            detail = _http_error_detail(error)
            if (
                response_format_enabled
                and not fallback_used
                and _response_format_unsupported(error.code, detail)
            ):
                response_format_enabled = False
                fallback_used = True
                continue

            last_error = LLMRequestError(
                f"HTTP {error.code} from chat/completions",
                status=error.code,
                detail=detail,
            )
            retryable = error.code >= 500 or error.code in {408, 409, 425, 429}
        except Exception as error:
            last_error = LLMRequestError(
                f"{type(error).__name__} while calling chat/completions",
                detail=str(error)[:MAX_FAILED_DETAIL_CHARS],
            )
            retryable = True

        attempts_left -= 1
        if attempts_left <= 0 or not retryable:
            break
        time.sleep(float(retry_sleep))

    raise last_error or LLMRequestError("unknown chat/completions failure")


def _build_messages(caption):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(caption)},
    ]


def generate_one(client_cfg, source_caption, captions_per_image):
    messages = _build_messages(source_caption)
    meta = {"parse_error": None, "request_error": None, "fallback_used": False}
    try:
        raw, fallback_used = _post_chat_completion(
            api_base=client_cfg["api_base"],
            api_key=client_cfg["api_key"],
            model=client_cfg["model"],
            messages=messages,
            temperature=client_cfg["temperature"],
            max_tokens=client_cfg["max_tokens"],
            timeout=client_cfg["timeout"],
            max_retries=client_cfg["max_retries"],
            retry_sleep=client_cfg["retry_sleep"],
            use_response_format=client_cfg["use_response_format"],
        )
        meta["fallback_used"] = bool(fallback_used)
    except LLMRequestError as error:
        meta["request_error"] = {
            "status": error.status,
            "message": str(error),
            "detail": error.detail,
        }
        return [], [f"llm_error={type(error).__name__}"], [], meta

    try:
        candidates, parse_reasons = _parse_llm_captions(raw, captions_per_image)
    except LLMParseError as error:
        meta["parse_error"] = {
            "reason": error.reason,
            "detail": error.detail,
            "raw": str(raw)[:MAX_FAILED_DETAIL_CHARS],
        }
        return [], [error.reason], [], meta

    kept, dropped = [], list(parse_reasons)
    for candidate in candidates:
        ok, reason = attribute_preservation_report(source_caption, candidate)
        if ok:
            kept.append(candidate)
        else:
            dropped.append(reason)
    return kept, dropped, [], meta


def _extract_img_path_from_train_sample(sample):
    if len(sample) in (4, 5):
        return sample[2]
    raise ValueError(
        f"Unsupported train sample format. Expected len=4 or 5, got {len(sample)}."
    )


def _collect_train_jobs(train_samples, max_train_keys=-1, source_captions_per_image=0):
    """Return jobs as (stable_image_key, [source captions])."""
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
        caption = str(sample[-1]).strip()
        if caption:
            groups[image_id]["captions"].append(caption)

    jobs = []
    for group in groups.values():
        captions = _dedupe_captions(group["captions"], limit=0)
        if not captions:
            continue
        canonical = _choose_canonical_source(captions)
        ordered = [canonical] + [c for c in captions if c != canonical]
        if source_captions_per_image > 0:
            ordered = ordered[:source_captions_per_image]
        jobs.append((group["key"], ordered))
    return jobs


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
                record = json.loads(line)
            except Exception:
                continue
            if isinstance(record, dict) and "key" in record and "captions" in record:
                done[str(record["key"])] = record
    return done


def _load_existing_bank(output_file):
    done = {}
    if not op.isfile(output_file):
        return done
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            bank = json.load(f)
    except Exception:
        return done
    if not isinstance(bank, dict):
        return done
    for key, captions in bank.items():
        if isinstance(captions, list):
            done[str(key)] = {
                "key": str(key),
                "source": "",
                "sources": [],
                "captions": _dedupe_captions(captions, limit=0),
                "dropped_reasons": [],
                "event_reasons": [],
            }
    return done


def _append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_max_train_keys(args):
    if int(args.limit) > 0:
        return int(args.limit)
    if int(args.max_train_keys) > 0:
        return int(args.max_train_keys)
    return -1


def _run_test_api(args):
    if args.provider != "openai_compatible":
        raise ValueError(f"Unsupported provider: {args.provider}")
    messages = [
        {"role": "system", "content": "Output only valid JSON."},
        {"role": "user", "content": 'Return only this JSON: {"ok": true}'},
    ]
    print("[SYNTH-V5] test_api=True")
    print(f"[SYNTH-V5] provider={args.provider}")
    print(f"[SYNTH-V5] api_base={args.api_base}")
    print(f"[SYNTH-V5] model={args.model}")
    print(f"[SYNTH-V5] api_key={_mask_api_key(args.api_key)}")
    raw, fallback_used = _post_chat_completion(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        use_response_format=True,
    )
    obj = _extract_first_json_value(raw)
    print(f"[SYNTH-V5] response_format_fallback_used={fallback_used}")
    print(f"[SYNTH-V5] test_api_response={json.dumps(obj, ensure_ascii=False)}")
    if not (isinstance(obj, dict) and obj.get("ok") is True):
        raise RuntimeError("test_api response did not contain {'ok': true}")
    return 0


def _run_dry_run(args, jobs):
    sample_count = int(args.limit) if int(args.limit) > 0 else min(3, len(jobs))
    print("[SYNTH-V5] dry_run=True")
    print(f"[SYNTH-V5] provider={args.provider}")
    print(f"[SYNTH-V5] api_base={args.api_base}")
    print(f"[SYNTH-V5] model={args.model}")
    print(f"[SYNTH-V5] api_key={_mask_api_key(args.api_key)}")
    print(f"[SYNTH-V5] dry_run_samples={sample_count}")
    for idx, (key, sources) in enumerate(jobs[:sample_count], start=1):
        source = sources[0] if sources else ""
        print(f"[SYNTH-V5][DRY_RUN] sample={idx} key={key}")
        print(f"[SYNTH-V5][DRY_RUN] source={source}")
        print("[SYNTH-V5][DRY_RUN] system_prompt:")
        print(SYSTEM_PROMPT)
        print("[SYNTH-V5][DRY_RUN] user_prompt:")
        print(_build_prompt(source))
    print("[SYNTH-V5] dry_run complete; no API calls were made.")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an LLM synthetic caption bank (v5, OpenAI-compatible)."
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai_compatible",
        choices=["openai_compatible"],
    )
    parser.add_argument("--dataset_name", default="RGBT-PEDES",
                        choices=list(DATASET_FACTORY.keys()))
    parser.add_argument("--root_dir", default="datasets/TextReID")
    parser.add_argument(
        "--output_file",
        default="pack/synthcap_bank/synthcap_train_yunwu_gpt55_v5.json",
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="0 means all train image keys")
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--dry_run", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--test_api", default=False, action="store_true")

    # Legacy-compatible generation controls.
    parser.add_argument("--captions_per_image", type=int, default=3,
                        help="captions requested per source caption")
    parser.add_argument("--max_captions_per_image", type=int, default=6,
                        help="final deduplicated cap per image")
    parser.add_argument("--source_captions_per_image", type=int, default=0,
                        help="0 means use all original captions for an image")
    parser.add_argument("--min_kept_per_image", type=int, default=1)
    parser.add_argument("--max_train_keys", type=int, default=-1,
                        help="legacy alias for --limit; -1 means all")

    # OpenAI-compatible endpoint. --base_url is kept as a legacy alias.
    parser.add_argument("--api_base", "--base_url", dest="api_base",
                        type=str, default=_default_api_base())
    parser.add_argument("--api_key", type=str,
                        default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", type=str,
                        default=os.environ.get("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--llm_response_mode",
        type=str,
        default="strict_json_object",
        choices=["strict_json_object", "loose_json_text"],
        help="legacy: strict_json_object sends response_format when supported",
    )
    parser.add_argument("--pronoun_rewrite_before_drop",
                        default=False, action="store_true",
                        help="accepted for old command compatibility")
    parser.add_argument("--debug_sample_limit", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.provider != "openai_compatible":
        raise ValueError(f"Unsupported provider: {args.provider}")
    if args.captions_per_image < 1:
        raise ValueError("captions_per_image must be >= 1")
    if args.max_captions_per_image < 1:
        raise ValueError("max_captions_per_image must be >= 1")
    if args.min_kept_per_image < 0:
        raise ValueError("min_kept_per_image must be >= 0")
    if args.concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    if args.test_api:
        return _run_test_api(args)

    max_train_keys = _resolve_max_train_keys(args)
    dataset = DATASET_FACTORY[args.dataset_name](root=args.root_dir, verbose=False)
    jobs = _collect_train_jobs(
        dataset.train,
        max_train_keys=max_train_keys,
        source_captions_per_image=int(args.source_captions_per_image),
    )

    print("[SYNTH-V5] version=v5.2_yunwu_openai_compatible")
    print(f"[SYNTH-V5] dataset_name={args.dataset_name}")
    print(f"[SYNTH-V5] root_dir={args.root_dir}")
    print(f"[SYNTH-V5] caption_file={dataset.anno_path}")
    print(f"[SYNTH-V5] total_train_keys={len(jobs)}")
    print(f"[SYNTH-V5] limit={args.limit}")
    print(f"[SYNTH-V5] captions_per_image={args.captions_per_image}")
    print(f"[SYNTH-V5] max_captions_per_image={args.max_captions_per_image}")
    print(f"[SYNTH-V5] min_kept_per_image={args.min_kept_per_image}")
    print(f"[SYNTH-V5] provider={args.provider}")
    print(f"[SYNTH-V5] api_base={args.api_base}")
    print(f"[SYNTH-V5] model={args.model}")
    print(f"[SYNTH-V5] api_key={_mask_api_key(args.api_key)}")
    print(f"[SYNTH-V5] concurrency={args.concurrency}")
    print(f"[SYNTH-V5] dry_run={args.dry_run}")
    print(f"[SYNTH-V5] debug={args.debug}")

    if args.dry_run:
        return _run_dry_run(args, jobs)

    output_file = op.abspath(args.output_file)
    output_dir = op.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    shard_path = output_file + ".shard.jsonl"
    failed_path = output_file + ".failed.jsonl"
    parse_error_sample_path = op.join(output_dir, "parse_error_samples.jsonl")

    use_legacy_implicit_resume = op.isfile(shard_path)
    resume_enabled = bool(args.resume or use_legacy_implicit_resume)
    already_done = {}
    if resume_enabled:
        already_done.update(_load_existing_bank(output_file))
        already_done.update(_load_shard(shard_path))
    else:
        with open(shard_path, "w", encoding="utf-8"):
            pass
        with open(failed_path, "w", encoding="utf-8"):
            pass

    if args.debug_sample_limit > 0:
        with open(parse_error_sample_path, "w", encoding="utf-8"):
            pass

    pending = [(key, sources) for key, sources in jobs if key not in already_done]
    print(
        f"[SYNTH-V5] resume={resume_enabled} already_done={len(already_done)} "
        f"pending={len(pending)}"
    )
    print(f"[SYNTH-V5] output_file={output_file}")
    print(f"[SYNTH-V5] shard_file={shard_path}")
    print(f"[SYNTH-V5] failed_file={failed_path}")

    client_cfg = {
        "api_base": args.api_base,
        "api_key": args.api_key,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "retry_sleep": args.retry_sleep,
        "use_response_format": args.llm_response_mode == "strict_json_object",
    }

    num_started_done = len(already_done)
    num_done = num_started_done
    num_llm_errors = 0
    num_parse_errors = 0
    num_kept_total = 0
    num_dropped_total = 0
    num_response_format_fallbacks = 0
    parse_error_samples_written = 0
    drop_reason_counts = {}
    t0 = time.time()

    def _worker(key, sources):
        all_kept = []
        all_dropped = []
        all_events = []
        metas = []
        for source in sources:
            kept, dropped, events, meta = generate_one(
                client_cfg,
                source,
                args.captions_per_image,
            )
            metas.append({"source": source, **meta})
            all_kept.extend(kept)
            all_dropped.extend(dropped)
            all_events.extend(events)
            all_kept = _dedupe_captions(
                all_kept,
                limit=int(args.max_captions_per_image),
            )
            if len(all_kept) >= int(args.max_captions_per_image):
                break
        return key, sources, all_kept, all_dropped, all_events, metas

    with ThreadPoolExecutor(max_workers=int(args.concurrency)) as executor:
        futures = [executor.submit(_worker, key, sources) for key, sources in pending]
        for future in as_completed(futures):
            key, sources, kept, dropped, events, metas = future.result()
            record = {
                "key": key,
                "source": sources[0] if sources else "",
                "sources": sources,
                "captions": kept,
                "dropped_reasons": dropped,
                "event_reasons": events,
            }
            _append_jsonl(shard_path, record)
            already_done[key] = record

            request_errors = [m for m in metas if m.get("request_error")]
            parse_errors = [m for m in metas if m.get("parse_error")]
            if request_errors or parse_errors:
                fail_record = {
                    "key": key,
                    "sources": sources,
                    "request_errors": request_errors,
                    "parse_errors": parse_errors,
                    "dropped_reasons": dropped,
                }
                _append_jsonl(failed_path, fail_record)

            for meta in metas:
                if meta.get("fallback_used"):
                    num_response_format_fallbacks += 1
                if (
                    args.debug_sample_limit > 0
                    and meta.get("parse_error")
                    and parse_error_samples_written < args.debug_sample_limit
                ):
                    _append_jsonl(parse_error_sample_path, {
                        "key": key,
                        "source": meta.get("source", ""),
                        **meta["parse_error"],
                    })
                    parse_error_samples_written += 1

            num_kept_total += len(kept)
            num_dropped_total += len(dropped)
            for reason in dropped:
                drop_reason_counts[reason] = drop_reason_counts.get(reason, 0) + 1
                if str(reason).startswith("llm_error"):
                    num_llm_errors += 1
                elif reason in PARSER_FAILURE_REASONS:
                    num_parse_errors += 1

            num_done += 1
            processed_now = max(1, num_done - num_started_done)
            if num_done % int(args.log_every) == 0 or num_done == len(jobs):
                elapsed = time.time() - t0
                rate = processed_now / max(elapsed, 1e-6)
                print(
                    f"[SYNTH-V5] progress={num_done}/{len(jobs)} "
                    f"kept={num_kept_total} dropped={num_dropped_total} "
                    f"llm_err={num_llm_errors} parse_err={num_parse_errors} "
                    f"response_format_fallbacks={num_response_format_fallbacks} "
                    f"rate={rate:.3f}/s",
                    flush=True,
                )

    bank = OrderedDict()
    excluded_low_keep = 0
    for key, _sources in jobs:
        record = already_done.get(key)
        if not record:
            excluded_low_keep += 1
            continue
        captions = _dedupe_captions(
            record.get("captions", []),
            limit=int(args.max_captions_per_image),
        )
        if len(captions) < int(args.min_kept_per_image):
            excluded_low_keep += 1
            continue
        bank[key] = captions

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(bank, f, indent=2, ensure_ascii=False)

    total_captions = sum(len(captions) for captions in bank.values())
    avg = total_captions / float(len(bank)) if bank else 0.0
    print(f"[SYNTH-V5] bank_total_keys={len(bank)}")
    print(f"[SYNTH-V5] bank_total_captions={total_captions}")
    print(f"[SYNTH-V5] avg_captions_per_key={avg:.3f}")
    print(f"[SYNTH-V5] excluded_low_keep_keys={excluded_low_keep}")
    print(f"[SYNTH-V5] num_llm_errors={num_llm_errors}")
    print(f"[SYNTH-V5] num_parse_errors={num_parse_errors}")
    print(f"[SYNTH-V5] response_format_fallbacks={num_response_format_fallbacks}")
    print("[SYNTH-V5] drop_reason_counts_top10:")
    for reason, count in sorted(drop_reason_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  - {reason}: {count}")
    print(
        "[SYNTH-V5] NEXT STEPS:\n"
        f"  1) python tools/check_synth_caption_bank.py "
        f"--dataset_name {args.dataset_name} --root_dir {args.root_dir} "
        f"--synth_caption_bank_file {output_file}\n"
        "  2) train with --use_synth_caption_bank "
        f"--synth_caption_bank_file {output_file} --synth_caption_keep_per_image 2"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
