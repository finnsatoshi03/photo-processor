"""Output size presets.

MUST mirror rms-avisha-web-system/src/components/printing/print-config.ts
(300 DPI, mm -> px via 25.4, rounded). See ../CONTRACT.md "Size presets".
"""

DPI = 300

# preset id -> (width_px, height_px)
SIZE_PRESETS: dict[str, tuple[int, int]] = {
    "1x1": (300, 300),        # 1 x 1 in
    "2x2": (600, 600),        # 2 x 2 in
    "passport": (413, 531),   # 35 x 45 mm
    "visa": (600, 600),       # 2 x 2 in (51 x 51 mm)
}

MIN_DIM_PX = 64
MAX_DIM_PX = 4000


def resolve_size(
    size_preset: str | None, width_px: int | None, height_px: int | None
) -> tuple[int, int]:
    """Contract: preset wins over explicit pixels; explicit px clamped-checked."""
    if size_preset is not None and size_preset != "":
        if size_preset not in SIZE_PRESETS:
            raise ValueError(
                f"Unknown size_preset {size_preset!r}; expected one of "
                + ", ".join(sorted(SIZE_PRESETS))
            )
        return SIZE_PRESETS[size_preset]
    if width_px is None or height_px is None:
        raise ValueError("Provide size_preset or both width_px and height_px")
    for name, v in (("width_px", width_px), ("height_px", height_px)):
        if not (MIN_DIM_PX <= v <= MAX_DIM_PX):
            raise ValueError(f"{name} must be in [{MIN_DIM_PX}, {MAX_DIM_PX}]")
    return width_px, height_px
