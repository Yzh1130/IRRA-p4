"""Rule-based semantic text views for USPC.

This module provides lightweight semantic view construction for
Uncertainty-guided Semantic Prototype Calibration (USPC). It is intentionally
simple and standalone: it does not implement MGANet-style global-local
alignment, and it is not wired into training by itself.
"""

import re
from typing import Any, Dict, Iterable, List


COLOR_WORDS = {
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "pink",
    "purple",
    "gray",
    "grey",
    "brown",
    "dark",
    "light",
    "navy",
    "khaki",
    "beige",
    "colorful",
}


OBJECT_WORDS = {
    "shirt",
    "tshirt",
    "t-shirt",
    "coat",
    "jacket",
    "hoodie",
    "sweater",
    "pants",
    "trousers",
    "jeans",
    "shorts",
    "skirt",
    "dress",
    "shoes",
    "sneaker",
    "sneakers",
    "hat",
    "cap",
    "bag",
    "backpack",
    "handbag",
    "umbrella",
    "clothes",
    "clothing",
    "top",
    "bottom",
}


_TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*|\d+(?:\.\d+)?|[^\w\s]", re.UNICODE)
_CONTENT_TOKEN_RE = re.compile(r"^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$")
_NO_SPACE_BEFORE = {".", ",", ";", ":", "!", "?", "%", ")", "]", "}"}
_NO_SPACE_AFTER = {"(", "[", "{"}


def _as_caption_list(captions: Any) -> List[Any]:
    if captions is None:
        return []
    if isinstance(captions, str):
        return [captions]
    try:
        return list(captions)
    except TypeError:
        return [captions]


def _safe_text(caption: Any) -> str:
    if caption is None:
        return ""
    if isinstance(caption, str):
        return caption
    try:
        return str(caption)
    except Exception:
        return ""


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text)


def _is_content_token(token: str) -> bool:
    return bool(_CONTENT_TOKEN_RE.match(token))


def _token_key(token: str) -> str:
    return token.lower()


def _detokenize(tokens: Iterable[str]) -> str:
    pieces: List[str] = []
    for token in tokens:
        if not token:
            continue
        if not pieces:
            pieces.append(token)
        elif token in _NO_SPACE_BEFORE:
            pieces[-1] = pieces[-1] + token
        elif pieces[-1] in _NO_SPACE_AFTER:
            pieces[-1] = pieces[-1] + token
        else:
            pieces.append(" " + token)
    return "".join(pieces).strip()


def _build_color_view(text: str, window: int) -> str:
    tokens = _tokenize(text)
    if not tokens:
        return text

    keep_indices = set()
    color_positions = [
        idx for idx, token in enumerate(tokens) if _token_key(token) in COLOR_WORDS
    ]
    safe_window = max(0, int(window))

    for idx in color_positions:
        start = max(0, idx - safe_window)
        end = min(len(tokens), idx + safe_window + 1)
        keep_indices.update(range(start, end))

    for idx, token in enumerate(tokens):
        if _token_key(token) in OBJECT_WORDS:
            keep_indices.add(idx)

    if not keep_indices:
        return text

    selected = [token for idx, token in enumerate(tokens) if idx in keep_indices]
    color_view = _detokenize(selected)
    return color_view if color_view else text


def _build_struct_view(text: str) -> str:
    tokens = _tokenize(text)
    if not tokens:
        return text

    kept = [token for token in tokens if _token_key(token) not in COLOR_WORDS]
    content_count = sum(1 for token in kept if _is_content_token(token))
    if content_count < 3:
        return text

    struct_view = _detokenize(kept)
    return struct_view if struct_view else text


def build_uspc_text_views(captions, mode: str = "rule", window: int = 2) -> Dict[str, List[str]]:
    """Build full/color/structure text views for USPC.

    Args:
        captions: List-like caption inputs. ``None`` elements become empty
            strings; other non-string inputs are converted with ``str``.
        mode: Currently only ``"rule"`` is supported.
        window: Number of tokens to keep on each side of every color token in
            the color view.

    Returns:
        A dict with ``full``, ``color``, and ``struct`` lists. Each list has the
        same length as the normalized input caption list.
    """
    if mode != "rule":
        raise ValueError(f"Unsupported USPC text view mode: {mode!r}. Expected 'rule'.")

    raw_captions = _as_caption_list(captions)
    full = [_safe_text(caption) for caption in raw_captions]
    color = [_build_color_view(text, window=window) for text in full]
    struct = [_build_struct_view(text) for text in full]
    return {
        "full": full,
        "color": color,
        "struct": struct,
    }


def _demo() -> None:
    examples = [
        "A man in a black jacket and blue jeans carrying a backpack.",
        "The woman wears a red dress, white shoes, and a small handbag.",
        "Person with umbrella walking beside another pedestrian.",
    ]
    views = build_uspc_text_views(examples)
    for raw, full, color, struct in zip(
        examples, views["full"], views["color"], views["struct"]
    ):
        print(f"raw:    {raw}")
        print(f"full:   {full}")
        print(f"color:  {color}")
        print(f"struct: {struct}")
        print()


if __name__ == "__main__":
    _demo()
