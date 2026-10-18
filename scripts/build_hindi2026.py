"""Build the Hindi twins of the JEE Advanced 2026 questions (v3, Phase H).

The HF dataset (revision ab587a1) holds 2026 in English only. The official Hindi papers
(jeeadv.ac.in/documents/p{1,2}_hindi.pdf) are text PDFs with "Q.n" markers, so the Hindi
questions can be cropped automatically, like the dataset authors' tool did by hand
(PyMuPDF, zoom 2x, clip rectangle):

* question k = from its "Q.k" marker down to the next marker or section header ("खंड");
  a question that runs past the page bottom continues at the top of the next page
  (the parts are stacked vertically);
* page header (y < 50) and footer (y > 790) are excluded;
* numbering restarts at Q.1 for each subject (order: Mathematics, Physics, Chemistry);
  Q -> (type, qN) follows the English dataset rows of the same paper/subject;
* answers, types and requires_image are copied from the English twin.

Runs with any Python that has PyMuPDF (the project venv does not need it):
    python3 scripts/build_hindi2026.py <dir with p1_hindi.pdf p2_hindi.pdf> <artifacts_v3>
Writes <artifacts_v3>/data/images/2026_P*_Hindi_*.png, appends the rows to
<artifacts_v3>/data/questions.parquet (column crop_source = "v3_auto") and writes
<artifacts_v3>/data/hindi2026_report.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pymupdf
from PIL import Image

SUBJECT_ORDER = ["Mathematics", "Physics", "Chemistry"]
HEADER_Y, FOOTER_Y = 50.0, 790.0
X0, X1 = 50.0, 550.0
PAD = 4.0
ZOOM = 2.0


SECTION_RE = re.compile(r"^(खंड|SECTION)\s*\d")
STEM_RE = re.compile(r"^(प्रश्\s*न\s*संख्या|Question Stem for)")


def markers(doc) -> list[dict]:
    """Items in reading order: "Q.n" markers, section headers ("खंड n") and shared question
    stems ("... (Question Stem) ..."), as dicts(kind, n, page, y)."""
    out = []
    for pno, page in enumerate(doc):
        for w in page.get_text("words"):
            if re.fullmatch(r"Q\.\d+", w[4]):
                out.append({"kind": "q", "n": int(w[4][2:]), "page": pno, "y": w[1]})
        for b in page.get_text("blocks"):
            text = b[4].strip()
            if b[1] <= HEADER_Y:
                continue
            if SECTION_RE.match(text) or "END OF THE QUESTION PAPER" in text:
                out.append({"kind": "section", "n": None, "page": pno, "y": b[1]})
            elif STEM_RE.match(text):   # "प्रश् न संख्या(ओं) 15 एवं 16 के लिए प्रश् न स्तम्भ"
                out.append({"kind": "stem", "n": None, "page": pno, "y": b[1]})
    return sorted(out, key=lambda m: (m["page"], m["y"]))


def content_bottom(page, y0: float, y1: float) -> float:
    """Lowest drawn content (text, vector drawings, images) between y0 and y1."""
    bottoms = [b[3] for b in page.get_text("blocks") if b[1] >= y0 - 1 and b[3] <= y1 + 1
               and b[4].strip()]
    bottoms += [d["rect"].y1 for d in page.get_drawings()
                if d["rect"].y0 >= y0 - 1 and d["rect"].y1 <= y1 + 1]
    bottoms += [i["bbox"][3] for i in page.get_image_info()
                if i["bbox"][1] >= y0 - 1 and i["bbox"][3] <= y1 + 1]
    return min(y1, max(bottoms) + PAD) if bottoms else y0


def crop(page, y0: float, y1: float) -> Image.Image | None:
    y1 = content_bottom(page, y0, y1)
    if y1 - y0 < 8:
        return None
    pix = page.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM),
                          clip=pymupdf.Rect(X0, y0, X1, y1))
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def trim_bottom(img: Image.Image, margin: int = 8) -> Image.Image:
    """Drop trailing white rows (blank space before a page break)."""
    from PIL import ImageOps

    box = ImageOps.invert(img.convert("L")).point(lambda v: 255 if v > 20 else 0).getbbox()
    return img if box is None else img.crop((0, 0, img.width, min(img.height, box[3] + margin)))


def stack(parts: list[Image.Image]) -> Image.Image:
    w = max(p.width for p in parts)
    out = Image.new("RGB", (w, sum(p.height for p in parts)), "white")
    y = 0
    for p in parts:
        out.paste(p, (0, y))
        y += p.height
    return out


def region(doc, ms: list[dict], i: int) -> Image.Image | None:
    """Crop from item i down to the next item, continuing over page breaks."""
    m = ms[i]
    nxt = ms[i + 1] if i + 1 < len(ms) else None
    parts, page, y0 = [], m["page"], m["y"] - PAD
    while True:
        end = nxt["y"] - PAD if nxt and nxt["page"] == page else FOOTER_Y
        img = crop(doc[page], y0, end)
        if img is not None:
            parts.append(img)
        if nxt is None or nxt["page"] == page or page + 1 >= len(doc):
            break
        page, y0 = page + 1, HEADER_Y          # continuation on the next page
    return stack(parts) if parts else None


def question_images(doc) -> list[tuple[int, int, Image.Image, int]]:
    """[(subject index, Q number, image, page)] in paper order. A shared stem is stacked
    above both questions that follow it (as in the dataset's English crops)."""
    ms = markers(doc)
    out, subj, stem, stem_left = [], -1, None, 0
    for i, m in enumerate(ms):
        if m["kind"] == "stem":
            stem, stem_left = trim_bottom(region(doc, ms, i)), 2
            continue
        if m["kind"] != "q":
            continue
        if m["n"] == 1:
            subj += 1
        img = region(doc, ms, i)
        if stem is not None and stem_left > 0:
            img = stack([stem, img])
            stem_left -= 1
        out.append((subj, m["n"], trim_bottom(img), m["page"] + 1))
    return out


def main(pdf_dir: str, artifacts: str) -> None:
    art = Path(artifacts)
    qs = pd.read_parquet(art / "data" / "questions.parquet")
    en = qs[(qs["year"] == 2026) & (qs["language"] == "English")]
    if (qs["year"].eq(2026) & qs["language"].eq("Hindi")).any():
        raise SystemExit("Hindi 2026 rows already present")
    rows, report = [], {"papers": {}}
    for paper in ("P1", "P2"):
        doc = pymupdf.open(Path(pdf_dir) / f"p{paper[1]}_hindi.pdf")
        imgs = question_images(doc)
        ep = en[en["paper"] == paper]
        report["papers"][paper] = {"markers": len(imgs), "english_rows": len(ep)}
        assert len(imgs) == len(ep), (paper, len(imgs), len(ep))
        for s_idx, subject in enumerate(SUBJECT_ORDER):
            sub_en = ep[ep["subject"] == subject].copy()
            # English rows in paper order: type blocks in the order they appear, then qnum.
            sub_en["page_no"] = sub_en["question_id"].str.extract(r"page(\d+)$")[0].astype(int)
            type_order = sub_en.groupby("question_type")["page_no"].min().sort_values().index
            sub_en["t_rank"] = sub_en["question_type"].map({t: i for i, t in
                                                           enumerate(type_order)})
            sub_en = sub_en.sort_values(["t_rank", "qnum"]).reset_index(drop=True)
            sub_hi = [x for x in imgs if x[0] == s_idx]
            assert [x[1] for x in sub_hi] == list(range(1, len(sub_en) + 1)), (paper, subject)
            for (_, n, img, page), (_, e) in zip(sub_hi, sub_en.iterrows()):
                t1, t2 = e["question_id"].split("_")[4], e["question_id"].split("_")[6]
                qid = f"2026_{paper}_Hindi_{subject}_{t1}_q{e['qnum']}_{t2}_page{page}"
                img_rel = f"data/images/{qid}.png"
                img.save(art / img_rel)
                r = e.to_dict()
                r.update({"question_id": qid, "uid": qid, "language": "Hindi",
                          "image_path": img_rel})
                r.pop("page_no", None)
                r.pop("t_rank", None)
                rows.append(r)
    new = pd.DataFrame(rows)
    assert (new["twin_key"].isin(en["twin_key"])).all() and new["twin_key"].is_unique
    qs["crop_source"] = qs.get("crop_source", "dataset")
    new["crop_source"] = "v3_auto"
    out = pd.concat([qs, new[qs.columns]], ignore_index=True)
    out.to_parquet(art / "data" / "questions.parquet", index=False)
    report["rows_added"] = len(new)
    report["sizes"] = {r["uid"]: list(Image.open(art / r["image_path"]).size) for r in rows}
    (art / "data" / "hindi2026_report.json").write_text(json.dumps(report, indent=1))
    print(f"added {len(new)} Hindi 2026 rows")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
