import re
from pprint import pprint

import torch


GENDER_CLASSES = ("unknown", "male", "female")
COLOR_CLASSES = (
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "gray",
    "brown",
    "pink",
    "purple",
    "orange",
)

ATTR_NAMES = (
    "gender",
    "upper_color",
    "lower_color",
    "shoe_color",
    "bag_color",
    "bag_present",
    "hair_length",
)

COLOR_TO_ID = {name: idx for idx, name in enumerate(COLOR_CLASSES)}
COLOR_ALIASES = {
    "black": "black",
    "white": "white",
    "red": "red",
    "blue": "blue",
    "green": "green",
    "yellow": "yellow",
    "gray": "gray",
    "grey": "gray",
    "brown": "brown",
    "pink": "pink",
    "purple": "purple",
    "orange": "orange",
}

MALE_WORDS = {"man", "men", "male", "boy", "gentleman"}
FEMALE_WORDS = {"woman", "women", "female", "girl", "lady"}
UPPER_WORDS = {
    "shirt",
    "tshirt",
    "t-shirt",
    "jacket",
    "coat",
    "hoodie",
    "sweater",
    "top",
    "blouse",
}
LOWER_WORDS = {"pants", "jeans", "trousers", "shorts", "skirt"}
SHOE_WORDS = {"shoe", "shoes", "sneaker", "sneakers", "boot", "boots"}
BAG_WORDS = {"bag", "backpack", "handbag", "purse"}

SLOT_WORDS = {
    "upper_color": UPPER_WORDS,
    "lower_color": LOWER_WORDS,
    "shoe_color": SHOE_WORDS,
    "bag_color": BAG_WORDS,
}

TOKEN_RE = re.compile(r"[a-z]+(?:-[a-z]+)*|\d+", re.IGNORECASE)


def _as_caption_list(raw_captions):
    if raw_captions is None:
        return []
    if isinstance(raw_captions, str):
        return [raw_captions]
    return ["" if item is None else str(item) for item in list(raw_captions)]


def _tokenize(text):
    return [token.lower() for token in TOKEN_RE.findall(str(text).lower())]


def _slot_token_key(token):
    token = str(token).lower()
    if token == "tee":
        return "tshirt"
    return token


def _color_key(token):
    return COLOR_ALIASES.get(str(token).lower(), None)


def _parse_gender(tokens):
    has_male = any(token in MALE_WORDS for token in tokens)
    has_female = any(token in FEMALE_WORDS for token in tokens)
    if has_male and has_female:
        return 0, False, True
    if has_male:
        return 1, True, False
    if has_female:
        return 2, True, False
    return 0, False, False


def _parse_slot_color(tokens, slot_words, window=4):
    color_hits = set()
    slot_positions = [
        idx for idx, token in enumerate(tokens) if _slot_token_key(token) in slot_words
    ]
    if not slot_positions:
        return 0, False, False

    color_positions = [
        (idx, _color_key(token))
        for idx, token in enumerate(tokens)
        if _color_key(token) is not None
    ]
    for color_idx, color_name in color_positions:
        if any(abs(color_idx - slot_idx) <= int(window) for slot_idx in slot_positions):
            color_hits.add(color_name)

    if len(color_hits) == 1:
        return COLOR_TO_ID[next(iter(color_hits))], True, False
    if len(color_hits) > 1:
        return 0, False, True
    return 0, False, False


def _parse_bag_present(tokens):
    has_bag = any(_slot_token_key(token) in BAG_WORDS for token in tokens)
    if has_bag:
        return 1, True
    return 0, False


def _parse_hair_length(tokens):
    has_short = False
    has_long = False
    for idx, token in enumerate(tokens):
        if token not in {"short", "long"}:
            continue
        lookahead = tokens[idx + 1 : idx + 4]
        if "hair" not in lookahead:
            continue
        if token == "short":
            has_short = True
        else:
            has_long = True
    if has_short and has_long:
        return 0, False, True
    if has_short:
        return 0, True, False
    if has_long:
        return 1, True, False
    return 0, False, False


def _new_long(size, device):
    return torch.zeros(size, dtype=torch.long, device=device)


def _new_bool(size, device):
    return torch.zeros(size, dtype=torch.bool, device=device)


def parse_basic_attributes(raw_captions, device=None):
    captions = _as_caption_list(raw_captions)
    batch_size = len(captions)

    targets = {
        "gender": _new_long(batch_size, device),
        "upper_color": _new_long(batch_size, device),
        "lower_color": _new_long(batch_size, device),
        "shoe_color": _new_long(batch_size, device),
        "bag_color": _new_long(batch_size, device),
        "bag_present": _new_long(batch_size, device),
        "hair_length": _new_long(batch_size, device),
    }
    masks = {name: _new_bool(batch_size, device) for name in ATTR_NAMES}
    stats = {
        "batch_size": batch_size,
        "gender_conflicts": 0,
        "upper_color_conflicts": 0,
        "lower_color_conflicts": 0,
        "shoe_color_conflicts": 0,
        "bag_color_conflicts": 0,
        "hair_length_conflicts": 0,
    }

    for row_idx, caption in enumerate(captions):
        tokens = _tokenize(caption)

        gender_target, gender_mask, gender_conflict = _parse_gender(tokens)
        targets["gender"][row_idx] = gender_target
        masks["gender"][row_idx] = gender_mask
        stats["gender_conflicts"] += int(gender_conflict)

        for attr_name, slot_words in SLOT_WORDS.items():
            target, mask, conflict = _parse_slot_color(tokens, slot_words)
            targets[attr_name][row_idx] = target
            masks[attr_name][row_idx] = mask
            stats[f"{attr_name}_conflicts"] += int(conflict)

        bag_target, bag_mask = _parse_bag_present(tokens)
        targets["bag_present"][row_idx] = bag_target
        masks["bag_present"][row_idx] = bag_mask

        hair_target, hair_mask, hair_conflict = _parse_hair_length(tokens)
        targets["hair_length"][row_idx] = hair_target
        masks["hair_length"][row_idx] = hair_mask
        stats["hair_length_conflicts"] += int(hair_conflict)

    for attr_name in ATTR_NAMES:
        valid_count = int(masks[attr_name].sum().item())
        stats[f"valid_{attr_name}"] = valid_count
    stats["valid_total"] = int(
        sum(stats[f"valid_{attr_name}"] for attr_name in ATTR_NAMES)
    )
    stats["valid_avg_per_sample"] = (
        float(stats["valid_total"]) / float(batch_size) if batch_size > 0 else 0.0
    )

    return {
        "targets": targets,
        "masks": masks,
        "stats": stats,
    }


def _tensor_dict_to_lists(tensor_dict):
    return {key: value.detach().cpu().tolist() for key, value in tensor_dict.items()}


def _demo():
    examples = [
        "A man wearing a black jacket, blue jeans, white shoes and a gray backpack.",
        "The woman has long black hair, a red blouse, brown skirt, and no visible bag.",
        "A boy and girl in green tops with purple shoes.",
        "Person with short hair carrying a purse.",
    ]
    parsed = parse_basic_attributes(examples)
    pprint(
        {
            "captions": examples,
            "targets": _tensor_dict_to_lists(parsed["targets"]),
            "masks": _tensor_dict_to_lists(parsed["masks"]),
            "stats": parsed["stats"],
        }
    )


if __name__ == "__main__":
    _demo()
