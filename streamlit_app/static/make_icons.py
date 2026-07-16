"""
Generate BioFusion PWA icons with PIL (no external SVG tooling needed).

Draws a simple, recognizable "lungs" mark in white on the brand sapphire, and
exports the icon sizes a PWA + iOS need. A maskable variant adds safe-zone
padding so Android's adaptive-icon crop never clips the mark.
"""
from pathlib import Path

from PIL import Image, ImageDraw

SAPPHIRE = (0, 102, 204)      # aqryl primary blue #0066CC
SAPPHIRE_DK = (0, 61, 122)    # #003D7A (aqryl active)
WHITE = (255, 255, 255)

OUT = Path(__file__).parent / "icons"
OUT.mkdir(exist_ok=True)


def draw_lungs(size: int, pad_frac: float) -> Image.Image:
    """Render the icon at `size`px. `pad_frac` = fraction of margin (maskable)."""
    S = 512  # draw large, then downsample for crisp edges
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Rounded-square background with a subtle vertical shade.
    r = int(S * 0.22)
    d.rounded_rectangle([0, 0, S, S], radius=r, fill=SAPPHIRE)
    # a darker foot for depth
    d.rounded_rectangle([0, int(S * 0.55), S, S], radius=r, fill=SAPPHIRE_DK)
    d.rounded_rectangle([0, 0, S, int(S * 0.6)], radius=r, fill=SAPPHIRE)

    # Lungs mark (white). Trachea + two lobes drawn from primitives.
    cx = S / 2
    # trachea
    d.rounded_rectangle([cx - S*0.028, S*0.26, cx + S*0.028, S*0.46],
                        radius=int(S*0.02), fill=WHITE)
    d.ellipse([cx - S*0.06, S*0.23, cx + S*0.06, S*0.31], fill=WHITE)
    # bronchi stems
    d.line([(cx, S*0.42), (S*0.36, S*0.5)], fill=WHITE, width=int(S*0.03))
    d.line([(cx, S*0.42), (S*0.64, S*0.5)], fill=WHITE, width=int(S*0.03))
    # left lobe
    d.pieslice([S*0.16, S*0.40, S*0.50, S*0.80], start=70, end=280, fill=WHITE)
    d.rectangle([S*0.30, S*0.46, S*0.44, S*0.74], fill=WHITE)
    # right lobe (mirror)
    d.pieslice([S*0.50, S*0.40, S*0.84, S*0.80], start=260, end=110, fill=WHITE)
    d.rectangle([S*0.56, S*0.46, S*0.70, S*0.74], fill=WHITE)
    # notch the inner edges back to sapphire to suggest the two lungs
    d.polygon([(cx, S*0.46), (S*0.46, S*0.78), (cx, S*0.7)], fill=SAPPHIRE)
    d.polygon([(cx, S*0.46), (S*0.54, S*0.78), (cx, S*0.7)], fill=SAPPHIRE_DK)

    if pad_frac > 0:
        # Maskable: shrink the drawn art into a centered safe zone on sapphire.
        bg = Image.new("RGBA", (S, S), SAPPHIRE + (255,))
        inner = int(S * (1 - 2 * pad_frac))
        art = img.resize((inner, inner), Image.LANCZOS)
        off = (S - inner) // 2
        bg.paste(art, (off, off), art)
        img = bg

    return img.resize((size, size), Image.LANCZOS)


def main():
    # Standard icons
    for sz in (192, 512):
        draw_lungs(sz, 0.0).save(OUT / f"icon-{sz}.png")
    # Maskable (safe-zone padded) for Android adaptive icons
    draw_lungs(512, 0.12).save(OUT / "icon-maskable-512.png")
    # Apple touch icon (no transparency, 180px)
    apple = draw_lungs(180, 0.0).convert("RGB")
    apple.save(OUT / "apple-touch-icon.png")
    # Favicon
    draw_lungs(32, 0.0).save(OUT / "favicon-32.png")
    print("Wrote:", ", ".join(p.name for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
