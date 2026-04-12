import argparse
import json
import os
import os.path as op
import re
import sys
from collections import OrderedDict

PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.cuhkpedes import CUHKPEDES
from datasets.icfgpedes import ICFGPEDES
from datasets.rstpreid import RSTPReid
from datasets.rgbtpedes import RGBTPEDES


DATASET_FACTORY = {
    "CUHK-PEDES": CUHKPEDES,
    "ICFG-PEDES": ICFGPEDES,
    "RSTPReid": RSTPReid,
    "RGBT-PEDES": RGBTPEDES,
}

AGE_BODY_WORDS = {
    "young",
    "old",
    "slim",
    "thin",
    "obese",
    "fat",
    "moderate",
    "medium",
    "rather",
    "relatively",
}

GENDER_WORDS = {
    "man",
    "woman",
    "boy",
    "girl",
    "male",
    "female",
    "men",
    "women",
    "boys",
    "girls",
}

PRONOUN_WORDS = {"he", "she", "his", "her", "their"}

COLOR_WORDS = {
    "black",
    "white",
    "gray",
    "grey",
    "blue",
    "red",
    "green",
    "yellow",
    "pink",
    "purple",
    "orange",
    "brown",
    "beige",
    "khaki",
    "cyan",
    "dark",
    "light",
}

CLOTHING_TERMS = {
    "shirt",
    "tshirt",
    "t-shirt",
    "coat",
    "jacket",
    "hoodie",
    "jumper",
    "sweater",
    "top",
    "sleeve",
    "sleeves",
    "pants",
    "trousers",
    "jeans",
    "shorts",
    "skirt",
    "dress",
    "shoes",
    "sneakers",
    "boots",
    "clothes",
    "clothing",
}

ACCESSORY_TERMS = {
    "bag",
    "backpack",
    "shoulder bag",
    "phone",
    "mobile phone",
    "glasses",
    "hat",
    "cap",
    "umbrella",
}

STRUCTURE_TERMS = {
    "hair",
    "stripe",
    "stripes",
    "sleeve",
    "sleeves",
    "shoulder",
}

ACTION_TERMS = {
    "walking",
    "standing",
    "holding",
    "carrying",
    "using",
    "wearing",
}

STRUCTURAL_ITEMS = {
    "top",
    "shirt",
    "jacket",
    "hoodie",
    "coat",
    "pants",
    "jeans",
    "skirt",
    "shorts",
    "shoes",
    "backpack",
    "shoulder bag",
    "glasses",
    "hat",
}

WEARABLE_ITEMS = {
    "top",
    "shirt",
    "jacket",
    "hoodie",
    "coat",
    "pants",
    "jeans",
    "skirt",
    "shorts",
    "shoes",
}

BAG_ITEMS = {"bag", "backpack", "shoulder bag"}
WITH_ITEMS = {"glasses", "hat"}

CANONICAL_ITEM_MAP = {
    "trouser": "pants",
    "trousers": "pants",
    "pant": "pants",
    "sneaker": "shoes",
    "sneakers": "shoes",
    "boot": "shoes",
    "boots": "shoes",
    "casual shoes": "shoes",
    "sport shoes": "shoes",
    "sports shoes": "shoes",
    "shoe": "shoes",
    "jumper": "top",
    "sweater": "top",
    "clothes": "top",
    "clothing": "top",
    "cap": "hat",
    "bike": "bicycle",
}

PLURAL_ITEMS = {
    "pants",
    "jeans",
    "shorts",
    "shoes",
    "glasses",
}

PLURAL_NOUNS = {
    "shoes",
    "sneakers",
    "boots",
    "trousers",
    "pants",
    "jeans",
    "shorts",
    "glasses",
    "sleeves",
}

JUNK_PHRASES = (
    "of medium",
    "she has",
    "he has",
    "rather",
    "relatively",
    "in his left hand",
    "in his right hand",
    "in her left hand",
    "in her right hand",
    "in their left hand",
    "in their right hand",
)


def _word_pattern(words):
    escaped = [re.escape(word) for word in sorted(words, key=len, reverse=True)]
    return r"\b(?:" + "|".join(escaped) + r")\b"


AGE_BODY_RE = re.compile(_word_pattern(AGE_BODY_WORDS))
GENDER_RE = re.compile(_word_pattern(GENDER_WORDS))
PRONOUN_RE = re.compile(_word_pattern(PRONOUN_WORDS))
COLOR_RE = re.compile(_word_pattern(COLOR_WORDS))
PLURAL_ARTICLE_RE = re.compile(r"\b(?:a|an)\s+(" + "|".join(PLURAL_NOUNS) + r")\b")
COLOR_TOKEN_PATTERN = r"(?:black|white|gray|grey|blue|red|green|yellow|pink|purple|orange|brown|beige|khaki|cyan|dark|light)"
COLOR_SEQ_PATTERN = rf"(?:(?:{COLOR_TOKEN_PATTERN})\s+){{1,2}}"
ITEM_PATTERN = (
    r"shoulder bag|casual shoes|sports shoes|sport shoes|"
    r"backpack|t-shirt|tshirt|shirt|jacket|hoodie|coat|"
    r"trousers|trouser|pants|pant|jeans|skirt|shorts|"
    r"sneakers|sneaker|boots|boot|shoes|shoe|jumper|sweater|"
    r"clothes|clothing|glasses|hat|cap|bag|top"
)


def _normalize_raw(text):
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1. \2", text)
    text = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"[\"'`()\[\]{}]", " ", text)
    text = text.replace("/", " ")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_spaces(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+-\s+", "-", text)
    return text.strip(" ,.;:-")


def _has_term(text, terms):
    return any(re.search(r"\b" + re.escape(term) + r"\b", text) for term in terms)


def _split_clauses(text):
    text = _normalize_raw(text)
    return [part.strip() for part in re.split(r"[,.;:!?]+", text) if part.strip()]


def _score_source_caption(caption):
    text = _normalize_raw(caption)
    tokens = re.findall(r"[a-z-]+", text)
    score = len(tokens)

    score += 6 * sum(1 for token in tokens if token in PRONOUN_WORDS)
    score += 5 * sum(1 for token in tokens if token in AGE_BODY_WORDS)
    score += 2 * sum(1 for token in tokens if token in GENDER_WORDS)
    score += 6 * len(re.findall(r"\bof\b", text))
    score += 4 * text.count("..")

    for phrase in JUNK_PHRASES:
        if phrase in text:
            score += 10

    cleaned = _rewrite_caption(caption, drop_colors=False, max_words=32)
    cleaned_tokens = re.findall(r"[a-z-]+", cleaned.lower())
    if len(cleaned_tokens) < 5:
        score += 30
    if not _has_term(text, CLOTHING_TERMS | ACCESSORY_TERMS):
        score += 12
    return score


def _choose_canonical_source(captions):
    valid_captions = [caption.strip() for caption in captions if isinstance(caption, str) and caption.strip()]
    if not valid_captions:
        return ""
    scored = enumerate(valid_captions)
    return min(scored, key=lambda item: (_score_source_caption(item[1]), item[0]))[1]


def _preclean_clause(clause):
    clause = _normalize_spaces(clause)
    replacements = (
        (r"\b(?:his|her|their|the)\s+upper\s+body\s+(?:is|was|are|were)\b", "wearing"),
        (r"\b(?:his|her|their|the)\s+lower\s+body\s+(?:is|was|are|were)\b", "wearing"),
        (r"\b(?:his|her|their|the)\s+shoes\s+(?:are|were)\b", "wearing"),
        (r"\b(?:his|her|their|the)\s+(trousers|pants|jeans|shoes|sneakers|boots)\s+(?:are|were|is|was)\s+", r"\1 "),
        (r"\bthe\s+(coat|shirt|jacket|hoodie|top|clothes|clothing)\s+(?:is|was|are|were)\s+", ""),
        (r"\b(?:he|she)\s+has\b", "with"),
        (r"\b(?:he|she)\s+(?:wears|wear|wore)\b", "wearing"),
        (r"\b(?:he|she)\s+(?:holds|held)\b", "holding"),
        (r"\b(?:he|she)\s+(?:carries|carried)\b", "carrying"),
        (r"\b(?:he|she)\s+(?:is|was|are|were)\s+", ""),
        (r"\b(?:this is|it is|it was)\s+(?:a|an|the)?\s*", ""),
        (r"\bwears\b|\bwore\b", "wearing"),
        (r"\bholds\b|\bheld\b", "holding"),
        (r"\bcarries\b|\bcarried\b", "carrying"),
        (r"\bwalked\b|\bwalks\b", "walking"),
        (r"\ba\s+pair\s+of\s+", ""),
        (r"\bon both sides of the sleeves?\b", "on the sleeves"),
        (r"\bas a base\b", ""),
        (r"\bunderneath\b|\bbelow\b", ""),
        (r"\bbowing\s+(?:his|her|their|the)?\s*head\s+to\s+play\s+with\s+(?:his|her|their|the)?\s*mobile\s+phone\b", "using a mobile phone"),
        (r"\bplay(?:ing)?\s+with\s+(?:his|her|their|the)?\s*mobile\s+phone\b", "using a mobile phone"),
        (r"\b(?:in|on)\s+(?:his|her|their)\s+feet\b", ""),
        (r"\bin\s+(?:his|her|their)?\s*(?:left|right)?\s*hand(?:s)?\b", ""),
    )
    for pattern, replacement in replacements:
        clause = re.sub(pattern, replacement, clause)
    clause = re.sub(r"^(?:and|also)\s+", "", clause)
    return _normalize_spaces(clause)


def _remove_noise_words(clause, drop_colors):
    clause = re.sub(r"\bof\s+medium\b", "", clause)
    clause = GENDER_RE.sub("person", clause)
    clause = AGE_BODY_RE.sub("", clause)
    clause = re.sub(r"\b(?:a|an|the)\s+(?=person\b)", "", clause)
    clause = re.sub(r"\bperson\s+has\b", "with", clause)
    clause = re.sub(r"\bperson\s+(?:is|was|are|were)\s+(?:wearing|wears|wear)\b", "wearing", clause)
    clause = re.sub(r"\bperson\s+(?:is|was|are|were)\s+", "", clause)
    clause = re.sub(r"\bperson\s+(?:wearing|wears|wear)\b", "wearing", clause)
    if drop_colors:
        clause = COLOR_RE.sub("", clause)
    clause = PRONOUN_RE.sub("", clause)
    clause = re.sub(r"\bof\b", "", clause)
    clause = re.sub(r"\bbody\b", "", clause)
    clause = re.sub(r"\bperson\s+person\b", "person", clause)
    clause = PLURAL_ARTICLE_RE.sub(r"\1", clause)
    return _normalize_spaces(clause)


def _prefix_clause(clause):
    if not clause or clause == "person":
        return ""

    clause = re.sub(r"^(?:a|an|the)?\s*person\s+(?=with|wearing|holding|carrying|walking|standing|using)", "", clause)
    clause = re.sub(r"^(?:a|an|the)?\s*person\s*$", "", clause)
    clause = _normalize_spaces(clause)
    if not clause:
        return ""

    if clause.startswith(("with ", "wearing ", "holding ", "carrying ", "walking", "standing", "using ")):
        return clause
    if _has_term(clause, CLOTHING_TERMS):
        return "wearing " + clause
    if _has_term(clause, {"bag", "backpack", "shoulder bag"}):
        return "carrying " + clause
    if _has_term(clause, {"phone", "mobile phone", "umbrella"}):
        return "holding " + clause
    if _has_term(clause, ACCESSORY_TERMS | STRUCTURE_TERMS):
        return "with " + clause
    return ""


def _clean_clause(raw_clause, drop_colors):
    clause = _preclean_clause(raw_clause)
    clause = _remove_noise_words(clause, drop_colors=drop_colors)
    clause = _prefix_clause(clause)
    return _normalize_spaces(clause)


def _dedupe_clauses(clauses):
    result = []
    seen = set()
    wearing_run = False
    for clause in clauses:
        if not clause or clause in seen:
            continue
        seen.add(clause)

        if clause.startswith("wearing "):
            if wearing_run:
                clause = clause[len("wearing ") :]
            wearing_run = True
        elif clause.startswith(("with ", "holding ", "carrying ", "walking", "standing", "using ")):
            wearing_run = False

        result.append(clause)
    return result


def _limit_words(clauses, max_words):
    selected = []
    for clause in clauses:
        candidate = selected + [clause]
        candidate_text = "A person " + ", ".join(candidate)
        if len(re.findall(r"[a-z-]+", candidate_text.lower())) <= max_words:
            selected = candidate
        elif not selected:
            selected = [clause]
            break
        else:
            break
    return selected


def _rewrite_caption(source, drop_colors, max_words):
    clauses = [_clean_clause(clause, drop_colors=drop_colors) for clause in _split_clauses(source)]
    clauses = _dedupe_clauses(clauses)
    clauses = _limit_words(clauses, max_words=max_words)
    if not clauses:
        clauses = ["with visible clothing"]

    caption = "A person " + ", ".join(clauses)
    caption = _normalize_spaces(caption)
    caption = caption[:1].upper() + caption[1:]
    if not caption.endswith("."):
        caption += "."
    return caption


def _prepare_extraction_text(source):
    text = _normalize_raw(source)
    replacements = (
        (r"\bin\s+(?:his|her|their|the)?\s*(?:left|right)?\s*hand(?:s)?\b", " "),
        (r"\bon\s+(?:his|her|their)\s+feet\b", " "),
        (r"\b(?:his|her|their|the)\s+upper\s+body\s+(?:is|was|are|were)\b", "wearing"),
        (r"\b(?:his|her|their|the)\s+lower\s+body\s+(?:is|was|are|were)\b", "wearing"),
        (r"\b(?:his|her|their|the)\s+shoes\s+(?:are|were|is|was)\b", "wearing"),
        (r"\b(?:he|she)\s+has\b", "with"),
        (r"\b(?:he|she)\s+(?:wears|wear|wore)\b", "wearing"),
        (r"\b(?:he|she)\s+(?:is|was|are|were)\b", " "),
        (r"\bthis is\b|\bit is\b|\bit was\b", " "),
        (r"\ba\s+pair\s+of\b", " "),
        (r"\bunderneath\b|\bbelow\b|\bas a base\b|\bon top\b", " "),
        (r"\bmobile\s+phone\b|\bcell\s+phone\b", "phone"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    text = PRONOUN_RE.sub(" ", text)
    text = AGE_BODY_RE.sub(" ", text)
    text = GENDER_RE.sub(" person ", text)
    text = re.sub(r"\bof\s+medium\b", " ", text)
    text = re.sub(r"\bof\b", " ", text)
    return _normalize_spaces(text)


def _canonicalize_item(item):
    item = _normalize_spaces(item)
    item = CANONICAL_ITEM_MAP.get(item, item)
    if item == "tshirt":
        item = "shirt"
    if item == "t-shirt":
        item = "shirt"
    return item


def _normalize_color(color):
    color = _normalize_spaces(color or "")
    if color == "":
        return ""
    parts = [part for part in color.split() if part in COLOR_WORDS]
    if not parts:
        return ""
    if len(parts) > 2:
        parts = parts[-2:]
    return " ".join(parts)


def _dedupe_records(records, key_fields):
    deduped = []
    seen = set()
    for record in records:
        key = tuple(record.get(field, "") for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _extract_hair_records(text):
    records = []
    patterns = (
        rf"\b(?P<len>long|short)\s+(?P<color>{COLOR_SEQ_PATTERN})?hair\b",
        rf"\b(?P<color>{COLOR_SEQ_PATTERN})(?P<len>long|short)\s+hair\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            records.append(
                {
                    "pos": match.start(),
                    "length": match.group("len"),
                    "color": _normalize_color(match.groupdict().get("color", "")),
                }
            )
    records.sort(key=lambda item: item["pos"])
    return _dedupe_records(records, ("length", "color"))


def _extract_item_records(text):
    records = []

    sleeve_pattern = rf"\b(?P<color>{COLOR_SEQ_PATTERN})?(?:long|short)\s+sleeves?\b"
    for match in re.finditer(sleeve_pattern, text):
        records.append(
            {
                "pos": match.start(),
                "item": "top",
                "color": _normalize_color(match.groupdict().get("color", "")),
            }
        )

    item_pattern = (
        rf"\b(?P<color>{COLOR_SEQ_PATTERN})?"
        rf"(?:(?:long|short)\s+sleeved?\s+)?"
        rf"(?:(?:sport|sports|casual)\s+)?"
        rf"(?P<item>{ITEM_PATTERN})\b"
    )
    for match in re.finditer(item_pattern, text):
        item = _canonicalize_item(match.group("item"))
        if item not in STRUCTURAL_ITEMS and item != "bag":
            continue
        records.append(
            {
                "pos": match.start(),
                "item": item,
                "color": _normalize_color(match.groupdict().get("color", "")),
            }
        )

    records.sort(key=lambda item: item["pos"])
    return _dedupe_records(records, ("item", "color"))


def _extract_action_records(text):
    actions = []
    if re.search(r"\b(?:walk|walks|walking|walked)\b", text):
        actions.append("walking")
    if re.search(r"\b(?:stand|stands|standing|stood)\b", text):
        actions.append("standing")
    if re.search(r"\briding\s+(?:a\s+)?(?:bicycle|bike)\b", text):
        actions.append("riding a bicycle")
    return actions


def _extract_structured_parts(source):
    text = _prepare_extraction_text(source)
    items = _extract_item_records(text)
    return {
        "hair": _extract_hair_records(text),
        "wearing": [item for item in items if item["item"] in WEARABLE_ITEMS],
        "bags": [item for item in items if item["item"] in BAG_ITEMS],
        "with_items": [item for item in items if item["item"] in WITH_ITEMS],
        "actions": _extract_action_records(text),
    }


def _render_item(record, keep_colors):
    item = record["item"]
    color = record.get("color", "") if keep_colors else ""
    phrase = f"{color} {item}".strip()
    if item in PLURAL_ITEMS:
        return phrase
    return f"a {phrase}"


def _join_items(items):
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _render_hair(record, keep_colors):
    color = record.get("color", "") if keep_colors else ""
    return _normalize_spaces(f"{record['length']} {color} hair")


def _cleanup_caption(caption, subject):
    caption = _normalize_spaces(caption)
    caption = re.sub(r"\bA person\s+of\b", "A person", caption)
    caption = re.sub(r"\bA figure\s+person\b", "A figure", caption)
    caption = re.sub(r"\bA figure\s+(?:he|she|his|her|their)\b", "A figure", caption, flags=re.IGNORECASE)
    caption = re.sub(r"\bA person\s+(?:he|she|his|her|their)\b", "A person", caption, flags=re.IGNORECASE)
    caption = re.sub(r"\bwearing\s+a\s+dress\s+in\s+(?:his|her|their)\s+(?:left|right)?\s*hand\b", "wearing a dress", caption)
    caption = PRONOUN_RE.sub("", caption.lower()).strip()
    if caption.startswith(subject.lower()):
        caption = subject + caption[len(subject):]
    else:
        caption = subject + " " + caption
    caption = _normalize_spaces(caption)
    caption = re.sub(r"\s+,", ",", caption)
    caption = re.sub(r",\s*,", ",", caption)
    caption = caption.strip(" ,.;:")
    if not caption.endswith("."):
        caption += "."
    return caption


def _render_structured_caption(source, subject, keep_colors, max_wearing, strong=False):
    parts = _extract_structured_parts(source)
    phrases = []

    hair_records = parts["hair"]
    if hair_records and not strong:
        phrases.append("with " + _render_hair(hair_records[0], keep_colors=keep_colors))

    wearing_records = parts["wearing"]
    if not keep_colors:
        wearing_records = _dedupe_records(wearing_records, ("item",))
    wearing_items = [
        _render_item(record, keep_colors=keep_colors)
        for record in wearing_records[:max_wearing]
    ]
    if wearing_items:
        phrases.append("wearing " + _join_items(wearing_items))

    bag_records = parts["bags"]
    if not keep_colors:
        bag_records = _dedupe_records(bag_records, ("item",))
    if bag_records:
        bag = bag_records[0] if strong else bag_records[:1][0]
        phrases.append("carrying " + _render_item(bag, keep_colors=keep_colors))

    with_records = parts["with_items"]
    if not keep_colors:
        with_records = _dedupe_records(with_records, ("item",))
    for record in with_records[:1 if strong else 2]:
        phrases.append("with " + _render_item(record, keep_colors=keep_colors))

    for action in parts["actions"]:
        if action not in phrases:
            phrases.append(action)
        if strong:
            break

    if not phrases:
        fallback = "wearing a top" if subject == "A person" else "wearing a top"
        phrases.append(fallback)

    max_words = 30 if keep_colors else 20
    if strong:
        max_words = 14

    selected = []
    for phrase in phrases:
        candidate = selected + [phrase]
        candidate_text = subject + " " + ", ".join(candidate)
        if len(re.findall(r"[a-z-]+", candidate_text.lower())) <= max_words:
            selected = candidate
        elif not selected:
            selected = [phrase]
            break
        else:
            break

    caption = subject + " " + ", ".join(selected)
    return _cleanup_caption(caption, subject=subject)


def _caption_body(caption, subject):
    caption = _normalize_spaces(caption).rstrip(".").lower()
    subject = subject.lower()
    if caption.startswith(subject):
        return caption[len(subject):].strip()
    return caption


def build_normal_caption(source):
    return _render_structured_caption(
        source,
        subject="A person",
        keep_colors=True,
        max_wearing=4,
        strong=False,
    )


def build_thermal_caption(source):
    thermal = _render_structured_caption(
        source,
        subject="A figure",
        keep_colors=False,
        max_wearing=3,
        strong=False,
    )
    normal = build_normal_caption(source)
    if thermal == normal or _caption_body(thermal, "A figure") == _caption_body(normal, "A person"):
        thermal = _render_structured_caption(
            source,
            subject="A figure",
            keep_colors=False,
            max_wearing=2,
            strong=True,
        )
    return thermal



def _extract_img_path_from_train_sample(sample):
    if len(sample) in (4, 5):
        return sample[2]
    raise ValueError(
        "Unsupported train sample format for synthetic captions. "
        f"Expected len=4 or len=5, got len={len(sample)}."
    )


def _collect_train_groups(train_samples):
    groups = OrderedDict()
    for sample in train_samples:
        image_id = sample[1]
        if image_id not in groups:
            img_path = _extract_img_path_from_train_sample(sample)
            groups[image_id] = {
                "key": op.basename(str(img_path)),
                "captions": [],
            }
        groups[image_id]["captions"].append(sample[-1])
    return groups


def build_bank(train_samples):
    groups = _collect_train_groups(train_samples)
    bank = OrderedDict()
    sources = OrderedDict()
    for group in groups.values():
        key = group["key"]
        source = _choose_canonical_source(group["captions"])
        sources[key] = source
        bank[key] = [
            build_normal_caption(source),
            build_thermal_caption(source),
        ]
    return bank, sources


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a cleaner v2 synthetic caption bank for the train split."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="RGBT-PEDES",
        choices=list(DATASET_FACTORY.keys()),
        help="dataset name used to build train split",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./data",
        help="dataset root dir (same as training --root_dir)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./synth_caption_bank_large_v2.json",
        help="output json path for the synthetic caption bank",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = DATASET_FACTORY[args.dataset_name](root=args.root_dir, verbose=False)
    bank, sources = build_bank(dataset.train)

    output_file = op.abspath(args.output_file)
    output_dir = op.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(bank, f, indent=4, separators=(",", ": "))

    print(f"[SYNTH-V2] dataset_name={args.dataset_name}")
    print(f"[SYNTH-V2] root_dir={args.root_dir}")
    print(f"[SYNTH-V2] caption_file={dataset.anno_path}")
    print(f"[SYNTH-V2] output_file={output_file}")
    print("[SYNTH-V2] split=train")
    print("[SYNTH-V2] key_rule=basename(img_path)")
    print(f"[SYNTH-V2] total_train_keys={len(bank)}")
    print("[SYNTH-V2] first_5_examples:")
    for idx, (key, captions) in enumerate(bank.items()):
        if idx >= 5:
            break
        print(f"  - {key}")
        print(f"    original canonical source: {sources[key]}")
        print(f"    rewritten normal: {captions[0]}")
        print(f"    rewritten thermal: {captions[1]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
