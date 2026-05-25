"""
Build SynthCap-v6 controlled synthetic caption banks for RGBT-PEDES.

Outputs:
  1) metadata bank with quality/source metadata
  2) flat bank compatible with the existing train-time append path:
       {"1_1.jpg": ["caption ..."]}
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
from datetime import datetime, timezone


VERSION = "synthcap_v6_controlled"
DATASET_NAME = "RGBT-PEDES"
SPLIT = "train"
KEY_TYPE = "rgb_basename"
DEFAULT_ROOT_DIR = "datasets/TextReID"
DEFAULT_META_OUTPUT = "pack/synthcap_bank/synthcap_train_v6_controlled_meta.json"
DEFAULT_FLAT_OUTPUT = "pack/synthcap_bank/synthcap_train_v6_controlled_flat.json"
MAX_ERROR_DETAIL_CHARS = 1000
RAW_RESPONSE_LOG_CHARS = 800

SYSTEM_PROMPT = (
    "You generate conservative synthetic training captions for text-to-image "
    "person retrieval. Rewrite only facts clearly stated in the source caption. "
    "Preserve visible facts exactly and do not infer, invent, omit, or alter "
    "attributes. Output strict JSON only."
)

USER_PROMPT_TEMPLATE = """Rewrite the source person caption into one safe synthetic caption.

Rules:
- Rewrite only the original caption; preserve visible facts, do not infer.
- Do not add facts that are not explicitly stated in the original caption.
- Do not change any color words. Gray must stay gray, green must stay green, white must stay white.
- Do not change clothing item types. A hoodie cannot become clothing. Trousers cannot become jeans unless the source says jeans.
- Do not add body build words such as thin, slim, heavy, medium, average, or well-proportioned unless the source explicitly contains them.
- Do not add viewpoint words such as from behind, back, front, or side unless the source explicitly contains them.
- Do not add or change bags, backpacks, handbags, purses, shoes, sneakers, clothing, actions, hair, gender, or accessories.
- Do not omit major clothing facts if the source contains upper clothing, lower clothing, shoes, or a bag.
- Preserve visible appearance, clothing, carried items, and posture/action when stated.
- Do not use uncertain wording such as maybe, probably, possibly, appears to, or seems.
- Do not mention thermal, infrared, RGB, modality, dataset, camera, photo, or image.
- Use natural, concise English.
- Target length: 12 to 35 English words.
- If a safe rewrite is not possible, return pass=false.

Return exactly one JSON object:
{{"caption":"...","pass":true,"reason":""}}

Source caption:
"{caption}"
"""

FORBIDDEN_PHRASES = (
    "maybe",
    "probably",
    "possibly",
    "appears",
    "appears to",
    "seems",
    "seems to",
    "the image",
    "the photo",
    "the picture",
    "this image",
    "this photo",
    "as an ai",
    "thermal",
    "infrared",
    "rgb",
    "modality",
    "dataset",
)

FIRST_PERSON_PATTERNS = (
    r"\bi\b",
    r"\bme\b",
    r"\bmy\b",
    r"\bmine\b",
    r"\bwe\b",
    r"\bus\b",
    r"\bour\b",
    r"\bours\b",
)

COLOR_WORDS = {
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "gray",
    "grey",
    "brown",
    "orange",
    "pink",
    "purple",
    "beige",
    "khaki",
    "cyan",
    "gold",
    "golden",
    "silver",
    "blond",
    "blonde",
    "dark",
    "light",
}

CLOTHING_WORDS = {
    "shirt",
    "tshirt",
    "t-shirt",
    "clothing",
    "clothes",
    "top",
    "coat",
    "jacket",
    "hoodie",
    "sweater",
    "jumper",
    "dress",
    "skirt",
    "pants",
    "trousers",
    "jeans",
    "shorts",
    "shoes",
    "sneakers",
    "boots",
    "hat",
    "cap",
    "uniform",
}

BAG_TERMS = {
    "bag",
    "backpack",
    "handbag",
    "purse",
    "satchel",
    "school bag",
    "shoulder bag",
    "crossbody bag",
    "schoolbag",
    "briefcase",
}

SHOE_TERMS = {"shoes", "sneakers", "boots", "trainers", "footwear"}
BODY_BUILD_TERMS = {
    "thin",
    "slim",
    "skinny",
    "heavy",
    "fat",
    "obese",
    "stout",
    "medium",
    "average",
    "well-proportioned",
    "well proportioned",
    "medium build",
    "average build",
    "heavy build",
    "slim build",
    "thin build",
}
VIEWPOINT_TERMS = {
    "back",
    "behind",
    "from behind",
    "back view",
    "rear",
    "front",
    "from front",
    "front view",
    "side",
    "side view",
    "profile",
}
PHONE_TERMS = {"phone", "mobile phone", "cell phone", "cellphone", "telephone"}
HELD_ITEM_TERMS = PHONE_TERMS | BAG_TERMS | {"umbrella", "book", "paper", "bottle"}
ACTION_VARIANTS = {
    "walking": {"walk", "walks", "walking", "walked"},
    "standing": {"stand", "stands", "standing", "stood"},
    "holding": {"hold", "holds", "holding", "held"},
    "carrying": {"carry", "carries", "carrying", "carried", "take", "takes", "taking"},
    "using": {"use", "uses", "using", "used", "texting"},
    "riding": {"ride", "rides", "riding", "rode"},
}
GENDER_WORDS = {"man", "woman", "boy", "girl", "male", "female", "lady"}
MALE_WORDS = {"man", "boy", "male"}
FEMALE_WORDS = {"woman", "girl", "female", "lady"}
ACTION_WORDS = {"walking", "standing", "holding", "carrying", "using", "riding"}


class ApiError(RuntimeError):
    def __init__(self, message, status=None, detail=""):
        super().__init__(message)
        self.status = status
        self.detail = detail


class ParseError(ValueError):
    pass


def _utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _mask_api_key(api_key):
    value = str(api_key or "")
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return value[:2] + "***"
    return value[:4] + "..." + value[-4:]


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, obj):
    parent = op.dirname(op.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _append_jsonl(path, record):
    parent = op.dirname(op.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _truncate(value, limit=RAW_RESPONSE_LOG_CHARS):
    value = str(value or "")
    return value[: int(limit)]


def resolve_caption_json(root_dir, caption_json):
    if caption_json:
        path = op.abspath(caption_json)
        if op.isfile(path):
            return path
        raise FileNotFoundError(f"caption_json does not exist: {path}")

    candidates = [
        op.join(root_dir, "RGBT-PEDES", "Caption.json"),
        op.join(root_dir, "RGBT-PEDES", "caption.json"),
        op.join(root_dir, "RGBT-PEDES", "RGBT-PEDES-Final", "Caption.json"),
        op.join(root_dir, "RGBT-PEDES", "RGBT-PEDES-Final", "caption.json"),
        op.join(root_dir, "RGBT-PEDES", "RGBT-PEDES", "Caption.json"),
        op.join(root_dir, "RGBT-PEDES", "RGBT-PEDES", "caption.json"),
    ]
    for path in candidates:
        if op.isfile(path):
            return op.abspath(path)
    raise FileNotFoundError(
        "Could not find RGBT-PEDES Caption.json. Tried: "
        + ", ".join(op.abspath(path) for path in candidates)
    )


def _basename_from_file_path(file_path):
    return op.basename(str(file_path).replace("\\", "/"))


def _word_count(text):
    return len(re.findall(r"[A-Za-z0-9-]+", str(text)))


def _normalize_text(text):
    text = str(text or "").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _canonical_for_exact_compare(text):
    text = _normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _term_set(text, terms):
    lowered = str(text or "").lower()
    found = set()
    for term in sorted(terms, key=len, reverse=True):
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", lowered):
            found.add(term.lower())
    return found


def _color_set(text):
    found = _term_set(text, COLOR_WORDS)
    if "grey" in found:
        found.remove("grey")
        found.add("gray")
    if "blonde" in found:
        found.remove("blonde")
        found.add("blond")
    return found


def _bag_set(text):
    found = _term_set(text, BAG_TERMS)
    if any(term in found for term in found if term != "bag"):
        found.add("bag")
    return found


def _required_bag_set(text):
    return _term_set(text, BAG_TERMS)


def _shoe_set(text):
    return _term_set(text, SHOE_TERMS)


def _clothing_set(text):
    return _term_set(text, CLOTHING_WORDS)


def _body_build_set(text):
    return _term_set(text, BODY_BUILD_TERMS)


def _viewpoint_set(text):
    return _term_set(text, VIEWPOINT_TERMS)


def _action_set(text):
    lowered = str(text or "").lower()
    found = set()
    for canonical, variants in ACTION_VARIANTS.items():
        if any(re.search(r"\b" + re.escape(term) + r"\b", lowered) for term in variants):
            found.add(canonical)
    return found


def _source_allows_hold_or_carry(source_caption):
    return bool(
        _action_set(source_caption) & {"holding", "carrying", "using"}
        or _term_set(source_caption, HELD_ITEM_TERMS)
    )


def _gender_set(text):
    return _term_set(text, GENDER_WORDS)


def _has_gender_conflict(source_gender, synth_gender):
    if not source_gender:
        return bool(synth_gender)
    if source_gender & MALE_WORDS and synth_gender & FEMALE_WORDS:
        return True
    if source_gender & FEMALE_WORDS and synth_gender & MALE_WORDS:
        return True
    return False


def _richness_score(caption):
    caption = str(caption or "")
    return (
        3.0 * len(_color_set(caption))
        + 3.0 * len(_term_set(caption, CLOTHING_WORDS))
        + 3.0 * len(_bag_set(caption))
        + 1.0 * len(_gender_set(caption))
        + 1.0 * len(_term_set(caption, ACTION_WORDS))
        + min(_word_count(caption), 60) * 0.05
    )


def choose_source_caption(captions, rule):
    valid = [
        (idx, _normalize_text(caption))
        for idx, caption in enumerate(captions or [])
        if isinstance(caption, str) and _normalize_text(caption)
    ]
    if not valid:
        return -1, ""
    rule = str(rule).lower()
    if rule == "first":
        return valid[0]
    if rule == "longest":
        return max(valid, key=lambda item: (_word_count(item[1]), len(item[1]), -item[0]))
    if rule == "richest":
        return max(
            valid,
            key=lambda item: (
                _richness_score(item[1]),
                _word_count(item[1]),
                len(item[1]),
                -item[0],
            ),
        )
    raise ValueError(f"Unsupported source_rule: {rule}")


def collect_train_jobs(caption_json_path, source_rule, limit):
    annos = _read_json(caption_json_path)
    if not isinstance(annos, list):
        raise ValueError(f"Caption file must contain a JSON list: {caption_json_path}")

    jobs = []
    seen_keys = set()
    for anno in annos:
        if not isinstance(anno, dict):
            continue
        if str(anno.get("split", "train")).lower() != SPLIT:
            continue
        file_path = str(anno.get("file_path", anno.get("img_path", "")))
        if not file_path:
            continue
        key = _basename_from_file_path(file_path)
        if key in seen_keys:
            continue
        captions = anno.get("captions", [])
        source_idx, source_caption = choose_source_caption(captions, source_rule)
        if source_idx < 0 or not source_caption:
            continue
        jobs.append(
            {
                "key": key,
                "file_path": file_path,
                "pid": anno.get("id", None),
                "source_caption": source_caption,
                "source_caption_index": source_idx,
                "num_source_captions": len(captions) if isinstance(captions, list) else 0,
            }
        )
        seen_keys.add(key)
        if int(limit) > 0 and len(jobs) >= int(limit):
            break
    return jobs


def build_user_prompt(source_caption):
    escaped = str(source_caption).strip().replace('"', '\\"')
    return USER_PROMPT_TEMPLATE.format(caption=escaped)


def _http_error_detail(error):
    try:
        detail = error.read().decode("utf-8", errors="replace")
    except Exception:
        detail = str(error)
    return detail[:MAX_ERROR_DETAIL_CHARS]


def _response_format_unsupported(status, detail):
    lowered = str(detail or "").lower()
    return (
        int(status or 0) in {400, 422}
        and "response_format" in lowered
        and ("unsupported" in lowered or "invalid" in lowered or "not support" in lowered)
    )


def post_chat_completion(
    api_base,
    api_key,
    model,
    messages,
    timeout,
    max_retries,
    sleep_seconds,
    max_tokens,
    temperature,
    use_response_format=True,
):
    url = str(api_base).rstrip("/") + "/chat/completions"
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

        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
        )
        request.add_header("Authorization", f"Bearer {api_key or ''}")
        request.add_header("Content-Type", "application/json")
        request.add_header("Accept", "application/json")

        try:
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                data = json.loads(response.read().decode("utf-8"))
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
            last_error = ApiError(
                f"HTTP {error.code} from chat/completions",
                status=error.code,
                detail=detail,
            )
            retryable = error.code >= 500 or error.code in {408, 409, 425, 429}
        except Exception as error:
            last_error = ApiError(
                f"{type(error).__name__} while calling chat/completions",
                detail=str(error)[:MAX_ERROR_DETAIL_CHARS],
            )
            retryable = True

        attempts_left -= 1
        if attempts_left <= 0 or not retryable:
            break
        if float(sleep_seconds) > 0:
            time.sleep(float(sleep_seconds))

    raise last_error or ApiError("unknown chat/completions failure")


def _strip_json_fence(text):
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_first_json_value(raw):
    cleaned = _strip_json_fence(raw)
    starts = [idx for idx, ch in enumerate(cleaned) if ch in "[{"]
    decoder = json.JSONDecoder()
    for start in starts:
        try:
            obj, _end = decoder.raw_decode(cleaned[start:])
            return obj
        except json.JSONDecodeError:
            continue
    raise ParseError("no valid JSON object found")


def parse_llm_object(raw):
    obj = _extract_first_json_value(raw)
    if not isinstance(obj, dict):
        raise ParseError(f"expected JSON object, got {type(obj).__name__}")
    if "caption" not in obj or "pass" not in obj:
        raise ParseError("expected keys: caption, pass, reason")
    if not isinstance(obj.get("pass"), bool):
        raise ParseError("key 'pass' must be boolean")
    caption = _normalize_text(obj.get("caption", ""))
    reason = _normalize_text(obj.get("reason", ""))
    return {
        "caption": caption,
        "pass": bool(obj.get("pass")),
        "reason": reason,
    }


def local_filter(source_caption, synth_caption):
    source_caption = _normalize_text(source_caption)
    synth_caption = _normalize_text(synth_caption)
    if not synth_caption:
        return False, "empty_caption"

    wc = _word_count(synth_caption)
    if wc < 8:
        return False, "too_short"
    if wc > 45:
        return False, "too_long"

    if _canonical_for_exact_compare(source_caption) == _canonical_for_exact_compare(synth_caption):
        return False, "same_as_source"

    lowered = synth_caption.lower()
    for phrase in FORBIDDEN_PHRASES:
        if re.search(r"\b" + re.escape(phrase) + r"\b", lowered):
            return False, "forbidden_phrase:" + phrase

    for pattern in FIRST_PERSON_PATTERNS:
        if re.search(pattern, lowered):
            return False, "first_person"

    source_colors = _color_set(source_caption)
    synth_colors = _color_set(synth_caption)
    new_colors = synth_colors - source_colors
    if new_colors:
        return False, "new_color:" + ",".join(sorted(new_colors))
    deleted_colors = source_colors - synth_colors
    if deleted_colors:
        return False, "deleted_color:" + ",".join(sorted(deleted_colors))

    source_bags = _bag_set(source_caption)
    synth_bags = _bag_set(synth_caption)
    new_bags = synth_bags - source_bags
    if new_bags:
        return False, "new_bag:" + ",".join(sorted(new_bags))
    required_source_bags = _required_bag_set(source_caption)
    if required_source_bags:
        required_synth_bags = _required_bag_set(synth_caption)
        missing_bags = required_source_bags - required_synth_bags
        if missing_bags:
            return False, "deleted_bag:" + ",".join(sorted(missing_bags))

    source_shoes = _shoe_set(source_caption)
    synth_shoes = _shoe_set(synth_caption)
    new_shoes = synth_shoes - source_shoes
    if new_shoes:
        return False, "new_shoe:" + ",".join(sorted(new_shoes))
    missing_shoes = source_shoes - synth_shoes
    if missing_shoes:
        return False, "deleted_shoe:" + ",".join(sorted(missing_shoes))

    source_clothing = _clothing_set(source_caption)
    synth_clothing = _clothing_set(synth_caption)
    new_clothing = synth_clothing - source_clothing
    if new_clothing:
        return False, "new_clothing:" + ",".join(sorted(new_clothing))
    deleted_clothing = source_clothing - synth_clothing
    if deleted_clothing:
        return False, "deleted_clothing:" + ",".join(sorted(deleted_clothing))

    source_body = _body_build_set(source_caption)
    synth_body = _body_build_set(synth_caption)
    new_body = synth_body - source_body
    if new_body:
        return False, "new_body:" + ",".join(sorted(new_body))

    source_views = _viewpoint_set(source_caption)
    synth_views = _viewpoint_set(synth_caption)
    new_views = synth_views - source_views
    if new_views:
        return False, "new_view:" + ",".join(sorted(new_views))

    source_actions = _action_set(source_caption)
    synth_actions = _action_set(synth_caption)
    new_actions = synth_actions - source_actions
    allowed_new_actions = set()
    if _source_allows_hold_or_carry(source_caption):
        allowed_new_actions.update({"holding", "carrying", "using"})
    new_actions = new_actions - allowed_new_actions
    if new_actions:
        return False, "new_action:" + ",".join(sorted(new_actions))

    source_gender = _gender_set(source_caption)
    synth_gender = _gender_set(synth_caption)
    if _has_gender_conflict(source_gender, synth_gender):
        if not source_gender:
            return False, "new_gender:" + ",".join(sorted(synth_gender))
        return False, "gender_conflict"

    return True, "pass"


def make_failure_record(job, reason, raw="", llm_obj=None, request_error=None):
    return {
        "version": VERSION,
        "key": job["key"],
        "file_path": job["file_path"],
        "pid": job["pid"],
        "source_caption": job["source_caption"],
        "source_caption_index": job["source_caption_index"],
        "caption": "" if llm_obj is None else llm_obj.get("caption", ""),
        "pass": False,
        "filter": reason,
        "reason": reason,
        "llm_reason": "" if llm_obj is None else llm_obj.get("reason", ""),
        "request_error": request_error,
        "raw": _truncate(raw),
        "created_at": _utc_now_iso(),
    }


def generate_one(job, cfg):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(job["source_caption"])},
    ]
    try:
        raw, fallback_used = post_chat_completion(
            api_base=cfg["api_base"],
            api_key=cfg["api_key"],
            model=cfg["model"],
            messages=messages,
            timeout=cfg["timeout"],
            max_retries=cfg["max_retries"],
            sleep_seconds=cfg["sleep"],
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            use_response_format=cfg["use_response_format"],
        )
    except ApiError as error:
        return make_failure_record(
            job,
            "api_error",
            request_error={
                "status": error.status,
                "message": str(error),
                "detail": error.detail,
            },
        )

    try:
        llm_obj = parse_llm_object(raw)
    except ParseError as error:
        return make_failure_record(job, "parse_error:" + str(error), raw=raw)

    if not llm_obj["pass"]:
        reason = llm_obj["reason"] or "llm_pass_false"
        return make_failure_record(job, "llm_pass_false:" + reason, raw=raw, llm_obj=llm_obj)

    ok, filter_reason = local_filter(job["source_caption"], llm_obj["caption"])
    if not ok:
        return make_failure_record(job, filter_reason, raw=raw, llm_obj=llm_obj)

    return {
        "version": VERSION,
        "key": job["key"],
        "file_path": job["file_path"],
        "pid": job["pid"],
        "source_caption": job["source_caption"],
        "source_caption_index": job["source_caption_index"],
        "caption": llm_obj["caption"],
        "pass": True,
        "filter": "pass",
        "reason": "",
        "llm_reason": llm_obj["reason"],
        "response_format_fallback_used": bool(fallback_used),
        "created_at": _utc_now_iso(),
    }


def load_shard_records(shard_file):
    records = OrderedDict()
    if not op.isfile(shard_file):
        return records
    with open(shard_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            key = str(record.get("key", ""))
            if key:
                records[key] = record
    return records


def build_banks_from_records(jobs, records, args, model_name):
    items = OrderedDict()
    flat = OrderedDict()
    for job in jobs:
        record = records.get(job["key"])
        if not record or not record.get("pass"):
            continue
        caption = _normalize_text(record.get("caption", ""))
        if not caption:
            continue
        entry = {
            "caption": caption,
            "source_caption": record.get("source_caption", job["source_caption"]),
            "source_caption_index": int(
                record.get("source_caption_index", job["source_caption_index"])
            ),
            "weight": float(args.weight),
            "quality": {
                "pass": True,
                "filter": "pass",
                "reason": record.get("reason", ""),
            },
        }
        items[job["key"]] = [entry]
        flat[job["key"]] = [caption]

    metadata = OrderedDict(
        [
            ("version", VERSION),
            ("dataset", DATASET_NAME),
            ("split", SPLIT),
            ("key_type", KEY_TYPE),
            ("source_rule", args.source_rule),
            ("created_at", _utc_now_iso()),
            ("model", model_name),
            ("caption_json", op.abspath(args.caption_json_resolved)),
            ("weight", float(args.weight)),
            ("items", items),
        ]
    )
    return metadata, flat


def run_test_api(args, cfg):
    messages = [
        {"role": "system", "content": "Output strict JSON only."},
        {"role": "user", "content": 'Return exactly {"ok":true}.'},
    ]
    print("[SYNTH-V6] test_api=True")
    print(f"[SYNTH-V6] api_base={cfg['api_base']}")
    print(f"[SYNTH-V6] model={cfg['model']}")
    print(f"[SYNTH-V6] api_key={_mask_api_key(cfg['api_key'])}")
    raw, fallback_used = post_chat_completion(
        api_base=cfg["api_base"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        messages=messages,
        timeout=args.timeout,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep,
        max_tokens=64,
        temperature=0.0,
        use_response_format=True,
    )
    obj = _extract_first_json_value(raw)
    print(f"[SYNTH-V6] response_format_fallback_used={fallback_used}")
    print(f"[SYNTH-V6] test_api_response={json.dumps(obj, ensure_ascii=False)}")
    if not (isinstance(obj, dict) and obj.get("ok") is True):
        raise RuntimeError("test_api response did not contain {'ok': true}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build SynthCap-v6 controlled caption banks for RGBT-PEDES."
    )
    parser.add_argument("--root_dir", default=DEFAULT_ROOT_DIR)
    parser.add_argument(
        "--caption_json",
        default="",
        help="Caption.json path. Empty means auto-resolve under --root_dir.",
    )
    parser.add_argument("--output_meta", default=DEFAULT_META_OUTPUT)
    parser.add_argument("--output_flat", default=DEFAULT_FLAT_OUTPUT)
    parser.add_argument("--shard_file", default="")
    parser.add_argument("--failed_file", default="")
    parser.add_argument("--source_rule", default="richest", choices=["richest", "longest", "first"])
    parser.add_argument("--weight", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=0, help="0 means all train images")
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--test_api", default=False, action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=160)
    parser.add_argument(
        "--llm_response_mode",
        default="strict_json_object",
        choices=["strict_json_object", "loose_json_text"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.caption_json_resolved = resolve_caption_json(args.root_dir, args.caption_json)
    args.shard_file = args.shard_file or (args.output_meta + ".shard.jsonl")
    args.failed_file = args.failed_file or (args.output_meta + ".failed.jsonl")

    api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL") or ""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL", "")
    cfg = {
        "api_base": api_base,
        "api_key": api_key,
        "model": model,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "sleep": args.sleep,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "use_response_format": args.llm_response_mode == "strict_json_object",
    }

    if not api_base or not api_key or not model:
        print(
            "[SYNTH-V6] error: OPENAI_API_BASE, OPENAI_API_KEY, and OPENAI_MODEL "
            "must be set in the environment."
        )
        print(f"[SYNTH-V6] api_base={api_base or '<empty>'}")
        print(f"[SYNTH-V6] model={model or '<empty>'}")
        print(f"[SYNTH-V6] api_key={_mask_api_key(api_key)}")
        return 2

    if args.test_api:
        return run_test_api(args, cfg)

    jobs = collect_train_jobs(args.caption_json_resolved, args.source_rule, args.limit)
    print("[SYNTH-V6] version=" + VERSION)
    print("[SYNTH-V6] dataset=RGBT-PEDES split=train")
    print(f"[SYNTH-V6] caption_json={args.caption_json_resolved}")
    print(f"[SYNTH-V6] total_jobs={len(jobs)}")
    print(f"[SYNTH-V6] source_rule={args.source_rule}")
    print(f"[SYNTH-V6] output_meta={op.abspath(args.output_meta)}")
    print(f"[SYNTH-V6] output_flat={op.abspath(args.output_flat)}")
    print(f"[SYNTH-V6] shard_file={op.abspath(args.shard_file)}")
    print(f"[SYNTH-V6] failed_file={op.abspath(args.failed_file)}")
    print(f"[SYNTH-V6] api_base={api_base}")
    print(f"[SYNTH-V6] model={model}")
    print(f"[SYNTH-V6] api_key={_mask_api_key(api_key)}")

    if args.resume:
        records = load_shard_records(args.shard_file)
    else:
        records = OrderedDict()
        for path in (args.shard_file, args.failed_file):
            parent = op.dirname(op.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8"):
                pass

    pending = [job for job in jobs if job["key"] not in records]
    print(f"[SYNTH-V6] resume={bool(args.resume)} already_done={len(records)} pending={len(pending)}")

    started = time.time()
    passed = sum(1 for record in records.values() if record.get("pass"))
    failed = sum(1 for record in records.values() if not record.get("pass"))
    for idx, job in enumerate(pending, start=1):
        record = generate_one(job, cfg)
        records[job["key"]] = record
        _append_jsonl(args.shard_file, record)
        if record.get("pass"):
            passed += 1
        else:
            failed += 1
            _append_jsonl(args.failed_file, record)

        done_now = len(records)
        if idx == 1 or idx % 20 == 0 or idx == len(pending):
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[SYNTH-V6] progress={done_now}/{len(jobs)} "
                f"passed={passed} failed={failed} rate={idx / elapsed:.3f}/s",
                flush=True,
            )
        if args.sleep > 0 and idx < len(pending):
            time.sleep(float(args.sleep))

    metadata, flat = build_banks_from_records(jobs, records, args, model)
    _write_json(args.output_meta, metadata)
    _write_json(args.output_flat, flat)

    print(f"[SYNTH-V6] wrote_metadata_items={len(metadata['items'])}")
    print(f"[SYNTH-V6] wrote_flat_keys={len(flat)}")
    print(f"[SYNTH-V6] failed_records={failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
