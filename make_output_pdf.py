"""
Bundle the key result figures and summary into a single PDF for sharing.

Run from project root:
    python make_output_pdf.py
Output:
    results/project_output.pdf
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

RESULTS = Path("results")

# 1. Read the summary markdown as page 1
summary_path = RESULTS / "m7_summary.md"
if summary_path.exists():
    summary_text = summary_path.read_text(encoding="utf-8")
else:
    summary_text = "m7_summary.md not found."

# 2. List of image files to include (in display order)
image_files = [
    "m7_robustness_plot.png",
    "m7_confusion_matrices.png",
    "demo_predictions.png",
    "m3_circuit.png",
    "m4_qtl_curves.png",
    "m2_classical_curves.png",
    "m6_qtl_robust_curves.png",
]

# 3. Build page 1: rendered summary text
# Use PIL's default font; small but readable. For a nicer look, provide a TTF.
try:
    font = ImageFont.truetype("arial.ttf", 18)
except Exception:
    font = ImageFont.load_default()

# Wrap text to ~90 chars per line
def wrap(text, width=90):
    out = []
    for line in text.splitlines():
        while len(line) > width:
            out.append(line[:width])
            line = line[width:]
        out.append(line)
    return "\n".join(out)

wrapped = wrap(summary_text, width=85)
page1 = Image.new("RGB", (1200, 1600), "white")
draw = ImageDraw.Draw(page1)
draw.multiline_text((40, 40), wrapped, fill="black", font=font, spacing=4)

# 4. Convert each result image to RGB and collect
pages = [page1]
for fname in image_files:
    fp = RESULTS / fname
    if not fp.exists():
        print(f"[skip] missing {fname}")
        continue
    img = Image.open(fp).convert("RGB")
    # Resize large images down so PDF stays small
    max_w = 1600
    if img.width > max_w:
        ratio = max_w / img.width
        img = img.resize((int(img.width * ratio), int(img.height * ratio)))
    pages.append(img)
    print(f"[add]  {fname}  {img.size}")

# 5. Save the combined PDF
out_path = RESULTS / "project_output.pdf"
pages[0].save(out_path, save_all=True, append_images=pages[1:])
print(f"Saved: {out_path}  ({len(pages)} pages)")