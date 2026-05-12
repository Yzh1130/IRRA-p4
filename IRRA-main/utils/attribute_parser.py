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
    "tops",
    "blouse",
}
LOWER_WORDS = {"pants", "jeans", "trousers", "shorts", "skirt"}
SHOE_WORDS = {"shoe", "shoes", "sneaker", "sneakers", "boot", "boots"}
BAG_WORDS = {"bag", "backpack", "handbag", "purse"}
BAG_NEGATION_WORDS = {"no", "not", "without", "none"}

SLOT_WORDS = {
    "upper_color": UPPER_WORDS,
    "lower_color": LOWER_WORDS,
    "shoe_color": SHOE_WORDS,
    "bag_color": BAG_WORDS,
}
ALL_SLOT_WORDS = set().union(*SLOT_WORDS.values())

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


def _is_slot_token(token):
    return _slot_token_key(token) in ALL_SLOT_WORDS


def _left_slot_colors(tokens, slot_idx, max_distance=3):
    colors = []
    for distance in range(1, max_distance + 1):
        idx = slot_idx - distance
        if idx < 0:
            break
        if _is_slot_token(tokens[idx]):
            break
        color = _color_key(tokens[idx])
        if color is not None:
            colors.append(color)
    return colors


def _right_slot_colors(tokens, slot_idx, max_distance=2):
    colors = []
    for distance in range(1, max_distance + 1):
        idx = slot_idx + distance
        if idx >= len(tokens):
            break
        if _is_slot_token(tokens[idx]):
            break
        color = _color_key(tokens[idx])
        if color is None:
            continue
        next_idx = idx + 1
        if next_idx < len(tokens) and _is_slot_token(tokens[next_idx]):
            continue
        colors.append(color)
    return colors


def _single_slot_color(tokens, slot_idx):
    left_colors = _left_slot_colors(tokens, slot_idx)
    if left_colors:
        unique_left = set(left_colors)
        if len(unique_left) == 1:
            return left_colors[0], False
        return None, True

    right_colors = _right_slot_colors(tokens, slot_idx)
    if right_colors:
        unique_right = set(right_colors)
        if len(unique_right) == 1:
            return right_colors[0], False
        return None, True

    return None, False


def _parse_slot_color(tokens, slot_words):
    color_hits = []
    has_conflict = False
    for idx, token in enumerate(tokens):
        if _slot_token_key(token) not in slot_words:
            continue
        color_name, conflict = _single_slot_color(tokens, idx)
        has_conflict = has_conflict or conflict
        if color_name is not None:
            color_hits.append(color_name)

    unique_hits = set(color_hits)
    if len(unique_hits) == 1 and not has_conflict:
        return COLOR_TO_ID[color_hits[0]], True, False
    if len(unique_hits) > 1 or has_conflict:
        return 0, False, True
    return 0, False, False


def _parse_bag_present(tokens):
    bag_positions = [
        idx for idx, token in enumerate(tokens) if _slot_token_key(token) in BAG_WORDS
    ]
    if not bag_positions:
        return 0, False

    positive_bags = []
    negated_bags = []
    for idx in bag_positions:
        prev_tokens = tokens[max(0, idx - 3) : idx]
        is_negated = any(token in BAG_NEGATION_WORDS for token in prev_tokens)
        if is_negated:
            negated_bags.append(idx)
        else:
            positive_bags.append(idx)

    if positive_bags:
        return 1, True
    if negated_bags:
        return 0, True
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


def _label_for_demo(attr_name, target):
    if attr_name == "gender":
        return GENDER_CLASSES[target]
    if attr_name.endswith("_color"):
        return COLOR_CLASSES[target]
    if attr_name == "bag_present":
        return "has_bag" if target == 1 else "no_bag"
    if attr_name == "hair_length":
        return ("short", "long")[target]
    return str(target)


def _decoded_demo(parsed, captions):
    decoded = []
    for row_idx, caption in enumerate(captions):
        attrs = {}
        for attr_name in ATTR_NAMES:
            mask = bool(parsed["masks"][attr_name][row_idx].item())
            target = int(parsed["targets"][attr_name][row_idx].item())
            attrs[attr_name] = _label_for_demo(attr_name, target) if mask else "MASKED"
        decoded.append({"caption": caption, "attrs": attrs})
    return decoded


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
            "decoded": _decoded_demo(parsed, examples),
            "targets": _tensor_dict_to_lists(parsed["targets"]),
            "masks": _tensor_dict_to_lists(parsed["masks"]),
            "stats": parsed["stats"],
        }
    )


if __name__ == "__main__":
    _demo()
