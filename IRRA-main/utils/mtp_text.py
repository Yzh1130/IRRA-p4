import re


_RGB_PREFIX = "Visible-view description: "
_THERMAL_PREFIX = "Thermal-view description: "
_COLOR_WORD_PATTERN = re.compile(
    r"\b(?:red|blue|green|yellow|black|white|gray|grey|brown|purple|pink|orange)\b",
    re.IGNORECASE,
)


def _normalize_whitespace(text):
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def _process_thermal_caption(raw_caption, color_drop_mode):
    normalized = build_full_text(raw_caption)
    if color_drop_mode == "mask":
        processed = _COLOR_WORD_PATTERN.sub("[COLOR]", normalized)
    elif color_drop_mode == "delete":
        processed = _COLOR_WORD_PATTERN.sub("", normalized)
    else:
        raise ValueError(
            "Unsupported color_drop_mode={}, expected one of ['delete', 'mask'].".format(
                color_drop_mode
            )
        )
    return _normalize_whitespace(processed)


def build_full_text(raw_caption):
    return _normalize_whitespace(raw_caption)


def build_rgb_text(raw_caption):
    full_text = build_full_text(raw_caption)
    if not full_text:
        return _RGB_PREFIX.strip()
    return "{}{}".format(_RGB_PREFIX, full_text)


def build_thermal_text(raw_caption, color_drop_mode):
    processed = _process_thermal_caption(raw_caption, color_drop_mode)
    if not processed:
        return _THERMAL_PREFIX.strip()
    return "{}{}".format(_THERMAL_PREFIX, processed)

