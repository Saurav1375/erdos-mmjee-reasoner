"""Final-submission deck in the Phase-2 house style (black / white / grey, Arial), editorial layout.

The Phase-2 deck is only read (for its theme and slide size); the output is a new file. Every
number comes from the result files or the reports; charts are report/final_paper/figures/deck.

usage: <python with python-pptx> scripts/final_deck.py [OUT.pptx]
"""
import sys
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT
SRC = ROOT / "report" / "phase2" / "Erdos_Phase2_Presentation.pptx"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "report" / "final_presentation" / "Erdos_Final_Submission_Presentation.pptx"
FIG = PROJ / "report" / "final_paper" / "figures" / "deck"
IMG = PROJ / "artifacts_v3" / "data" / "images"
Q_EN = IMG / "2026_P1_English_Physics_MCQ-Single_q1_MCQ-Single_page12.png"
Q_HI = IMG / "2026_P1_Hindi_Physics_MCQ-Single_q1_MCQ-Single_page13.png"
Q_NUM = IMG / "2026_P1_English_Physics_Numerical_q2_Numerical_page19.png"

SANS, SERIF, MONO = "Arial", "Georgia", "Consolas"
BLACK = RGBColor(0x11, 0x11, 0x11)
INK = RGBColor(0x26, 0x26, 0x26)
GREY = RGBColor(0x59, 0x59, 0x59)
MID = RGBColor(0x8C, 0x8C, 0x8C)
LIGHT = RGBColor(0xB4, 0xB4, 0xB4)
HAIR = RGBColor(0xD9, 0xD9, 0xD9)
TINT = RGBColor(0xF4, 0xF4, 0xF4)
TINT2 = RGBColor(0xE9, 0xE9, 0xE9)
CHAR = RGBColor(0x2B, 0x2B, 0x2B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


# ---------------------------------------------------------------- text helpers
def style(run, size, bold=False, color=INK, italic=False, font=SANS, spacing=None):
    f = run.font
    f.name, f.size, f.bold, f.italic = font, Pt(size), bold, italic
    f.color.rgb = color
    if spacing is not None:
        run._r.get_or_add_rPr().set("spc", str(spacing))


def R(t, size=11, bold=False, color=INK, italic=False, font=SANS, spacing=None):
    return (t, size, bold, color, italic, font, spacing)


def P(*runs, align=PP_ALIGN.LEFT, after=0, before=0, bullet=False, line=None, indent=0.16):
    return {"runs": runs, "align": align, "after": after, "before": before, "bullet": bullet,
            "line": line, "indent": indent}


def T(t, size=11, bold=False, color=INK, **kw):
    """Single-run paragraph."""
    return P(R(t, size, bold, color), **kw)


def Bl(t, size=10, color=INK, after=4, lead=None):
    runs = [R(lead, size, True, BLACK)] if lead else []
    runs.append(R(t, size, False, color))
    return P(*runs, bullet=True, after=after)


def _bullet(para, indent):
    pPr = para._p.get_or_add_pPr()
    pPr.set("marL", str(Emu(Inches(indent))))
    pPr.set("indent", str(-Emu(Inches(indent))))
    bc = etree.SubElement(pPr, qn("a:buClr"))
    etree.SubElement(bc, qn("a:srgbClr"), val="8C8C8C")
    bf = etree.SubElement(pPr, qn("a:buFont"))
    bf.set("typeface", SANS)
    ch = etree.SubElement(pPr, qn("a:buChar"))
    ch.set("char", "–")


def text(S, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, margin=0.0):
    tb = S.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(margin))
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = p["align"]
        para.space_after = Pt(p["after"])
        para.space_before = Pt(p["before"])
        if p["line"]:
            para.line_spacing = p["line"]
        if p["bullet"]:
            _bullet(para, p["indent"])
        for t, size, bold, color, italic, font, spacing in p["runs"]:
            for k, piece in enumerate(t.split("\n")):
                if k:
                    para.add_line_break()
                run = para.add_run()
                run.text = piece
                style(run, size, bold, color, italic, font, spacing)
    return tb


# ---------------------------------------------------------------- shape helpers
def _nostyle(sh):
    st = sh._element.find(qn("p:style"))
    if st is not None:
        sh._element.remove(st)


def rect(S, x, y, w, h, fill=None, line=None, lw=0.75, shape=MSO_SHAPE.RECTANGLE, dash=False,
         radius=None):
    s = S.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    _nostyle(s)
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
        if dash:
            s.line.dash_style = 4
    if radius is not None:
        s.adjustments[0] = radius
    return s


def line(S, x1, y1, x2, y2, color=HAIR, width=0.75, head=False, dash=False):
    c = S.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    _nostyle(c)
    c.line.color.rgb = color
    c.line.width = Pt(width)
    if dash:
        c.line.dash_style = 4
    if head:
        ln = c.line._get_or_add_ln()
        tail = etree.SubElement(ln, qn("a:tailEnd"))
        tail.set("type", "triangle")
        tail.set("w", "med")
        tail.set("len", "med")
    return c


def arrow(S, x1, y1, x2, y2, color=GREY, width=1.0, dash=False):
    return line(S, x1, y1, x2, y2, color, width, head=True, dash=dash)


def path(S, pts, color=GREY, width=1.0, dash=False):
    """Orthogonal arrow through points; the arrowhead is on the last segment."""
    for i in range(len(pts) - 1):
        (x1, y1), (x2, y2) = pts[i], pts[i + 1]
        line(S, x1, y1, x2, y2, color, width, head=(i == len(pts) - 2), dash=dash)


def node(S, x, y, w, h, title, detail="", fill=WHITE, border=GREY, bw=0.75, ts=9, ds=7.5,
         dash=False, tcolor=BLACK, dcolor=GREY, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    rect(S, x, y, w, h, fill, border, bw, shape=shape, dash=dash,
         radius=0.08 if shape == MSO_SHAPE.ROUNDED_RECTANGLE else None)
    paras = [T(title, ts, True, tcolor, align=PP_ALIGN.CENTER, after=1)]
    if detail:
        paras.append(T(detail, ds, False, dcolor, align=PP_ALIGN.CENTER))
    text(S, x + 0.05, y + 0.02, w - 0.1, h - 0.04, paras, anchor=MSO_ANCHOR.MIDDLE)


def dot(S, x, y, n, d=0.22, fill=CHAR, color=WHITE):
    rect(S, x, y, d, d, fill, None, shape=MSO_SHAPE.OVAL)
    text(S, x, y, d, d, [T(str(n), 8, True, color, align=PP_ALIGN.CENTER)], anchor=MSO_ANCHOR.MIDDLE)


def kicker(S, x, y, w, label, color=MID, size=7.5, align=PP_ALIGN.LEFT):
    text(S, x, y, w, 0.22, [P(R(label.upper(), size, True, color, spacing=120), align=align)])


def rule(S, x, y, w, heavy=False):
    line(S, x, y, x + w, y, BLACK if heavy else HAIR, 1.5 if heavy else 0.6)


def bignum(S, x, y, w, num, caption, size=34, cap=8.5, color=BLACK, align=PP_ALIGN.LEFT):
    text(S, x, y, w, size / 72 * 1.25, [P(R(num, size, False, color, font=SERIF), align=align)])
    text(S, x, y + size / 72 * 1.2, w, 0.6, [T(caption, cap, False, GREY, align=align)])


def quote(S, x, y, w, t, size=12.5, h=0.6):
    rect(S, x, y + 0.02, 0.045, h - 0.04, BLACK)
    text(S, x + 0.18, y, w - 0.18, h, [P(R(t, size, False, BLACK, italic=True, font=SERIF))],
         anchor=MSO_ANCHOR.MIDDLE)


def code(S, x, y, w, h, title, lines, size=7.5):
    rect(S, x, y, w, h, TINT)
    rect(S, x, y, 0.04, h, CHAR)
    kicker(S, x + 0.15, y + 0.07, w - 0.3, title, GREY, 6.5)
    text(S, x + 0.15, y + 0.3, w - 0.25, h - 0.35,
         [P(R(l, size, False, INK, font=MONO), line=1.0) for l in lines])


def spec(S, x, y, w, rows, label_w=0.95, size=8.5, gap=0.06):
    """Label / value rows separated by hairlines (a 'spec sheet')."""
    cy = y
    for lab, val, h in rows:
        text(S, x, cy + 0.03, label_w, h, [P(R(lab.upper(), 6.5, True, MID, spacing=80))])
        text(S, x + label_w, cy + 0.02, w - label_w, h, [T(val, size)])
        cy += h + gap
        line(S, x, cy - gap / 2, x + w, cy - gap / 2, HAIR, 0.5)
    return cy


def picture(S, path_, x, y, w=None, h=None, border=False, rot=0):
    iw, ih = Image.open(path_).size
    if w is not None and h is not None:
        if iw / ih > w / h:
            h2 = w * ih / iw
            y += (h - h2) / 2
            h = h2
        else:
            w2 = h * iw / ih
            x += (w - w2) / 2
            w = w2
    elif w is not None:
        h = w * ih / iw
    else:
        w = h * iw / ih
    pic = S.add_picture(str(path_), Inches(x), Inches(y), Inches(w), Inches(h))
    if border:
        pic.line.color.rgb = LIGHT
        pic.line.width = Pt(0.75)
    if rot:
        pic.rotation = rot
    return pic


def fig(S, name, x, y, w):
    return picture(S, FIG / f"{name}.png", x, y, w)


def _cell_border(cell, side, width=None, color="000000"):
    tcPr = cell._tc.get_or_add_tcPr()
    tag = {"L": "a:lnL", "R": "a:lnR", "T": "a:lnT", "B": "a:lnB"}[side]
    for el in tcPr.findall(qn(tag)):
        tcPr.remove(el)
    ln = etree.SubElement(tcPr, qn(tag))
    if width is None:
        ln.set("w", "0")
        etree.SubElement(ln, qn("a:noFill"))
    else:
        ln.set("w", str(Emu(Pt(width))))
        sf = etree.SubElement(ln, qn("a:solidFill"))
        etree.SubElement(sf, qn("a:srgbClr"), val=color)


def table(S, x, y, col_w, header, rows, font=9, row_h=0.3, hi=(), align=None, head_h=None):
    """Book-style table: heavy top and bottom rules, hairline rows, no vertical lines."""
    n_r, n_c = len(rows) + 1, len(header)
    gt = S.add_table(n_r, n_c, Inches(x), Inches(y), Inches(sum(col_w)), Inches(row_h * n_r))
    tbl = gt.table
    tblPr = tbl._tbl.tblPr
    for attr in ("firstRow", "bandRow"):
        tblPr.set(attr, "0")
    st = tblPr.find(qn("a:tableStyleId"))
    if st is not None:
        st.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
    for j, w in enumerate(col_w):
        tbl.columns[j].width = Inches(w)
    align = align or (["l"] + ["c"] * (n_c - 1))
    for i in range(n_r):
        tbl.rows[i].height = Inches(head_h if (i == 0 and head_h) else row_h)
        for j in range(n_c):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            strong = (i - 1) in hi
            c.fill.solid()
            c.fill.fore_color.rgb = TINT if strong else WHITE
            val = header[j] if i == 0 else rows[i - 1][j]
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = ""
            p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align[j]]
            for k, piece in enumerate(str(val).split("\n")):
                if k:
                    p.add_line_break()
                r = p.add_run()
                r.text = piece
                if i == 0:
                    style(r, font - 1.5, True, GREY, spacing=60)
                else:
                    style(r, font, strong, BLACK if strong else INK)
            for side in ("L", "R"):
                _cell_border(c, side)
            _cell_border(c, "T", 1.5 if i == 0 else None, "111111")
            if i == 0:
                _cell_border(c, "B", 0.75, "111111")
            elif i == n_r - 1:
                _cell_border(c, "B", 1.5, "111111")
            else:
                _cell_border(c, "B", 0.5, "DDDDDD")
    return gt


# ---------------------------------------------------------------- slide frame
prs = Presentation(str(SRC))
sldIdLst = prs.slides._sldIdLst
for sldId in list(sldIdLst):
    prs.part.drop_rel(sldId.rId)
    sldIdLst.remove(sldId)
BLANK = prs.slide_layouts[5]
SLIDES = []


def blank():
    s = prs.slides.add_slide(BLANK)
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    return s


def slide(section, title, label=None):
    s = blank()
    SLIDES.append((s, section))
    S = s.shapes
    text(S, 0.4, 0.42, 9.2, 0.58, [T(title, 22, False, BLACK)], anchor=MSO_ANCHOR.BOTTOM)
    line(S, 0.4, 1.06, 9.6, 1.06, HAIR, 0.6)
    rect(S, 0.4, 1.045, 0.75, 0.035, BLACK)
    return S


def finish():
    n = len(SLIDES)
    for i, (s, section) in enumerate(SLIDES, 1):
        if section is None:
            continue
        S = s.shapes
        text(S, 8.4, 5.24, 1.2, 0.28, [P(R(f"{i:02d}", 10, False, BLACK, font=SERIF),
                                          R(f"  /  {n}", 7, False, MID), align=PP_ALIGN.RIGHT)])


# ================================================================ 1. title
s = blank()
SLIDES.append((s, None))
S = s.shapes
rect(S, 6.55, 0, 3.45, 5.625, CHAR)
picture(S, Q_EN, 6.85, 1.05, w=2.85, border=False, rot=-2.5)
text(S, 6.85, 3.35, 2.85, 0.5, [T("A question from our clean test set: JEE Advanced 2026, "
                                  "Paper 1. The model sees only this image.", 7.5, False, LIGHT)])
kicker(S, 6.85, 4.95, 2.85, "IIT Bhilai  ·  Machine Learning", LIGHT, 6.5)
kicker(S, 0.55, 0.55, 5.5, "Final submission  ·  Team Erdős", MID, 8)
text(S, 0.55, 0.9, 5.7, 1.9, [T("Improving Scientific Reasoning of Vision-Language Models on "
                                "JEE Advanced", 28, False, BLACK, line=0.95)])
rect(S, 0.55, 2.78, 0.75, 0.04, BLACK)
text(S, 0.55, 2.92, 5.6, 0.8, [P(R("A code sandbox, agentic correction, a learned answer selector "
                                   "and retrieval around a frozen 8B model on one GPU, tested on "
                                   "an exam the model has never seen.", 11.5, False, GREY,
                                   italic=True, font=SERIF), line=1.1)])
for i, (name, roll, role) in enumerate([
        ("Arpit Kumar", "12340350", "Retrieval module · code sandbox module"),
        ("Saurav Gupta", "12341940", "Agentic correction module · learned verifier")]):
    x = 0.55 + i * 2.85
    rule(S, x, 4.05, 2.6)
    text(S, x, 4.12, 2.6, 0.3, [T(name, 12, True, BLACK)])
    text(S, x, 4.4, 2.6, 0.25, [T(f"Roll No. {roll}", 8.5, False, GREY)])
    text(S, x, 4.62, 2.6, 0.4, [T(role, 8, False, MID)])
text(S, 0.55, 5.18, 5.8, 0.25, [T("Base paper: Mukherjee & Ghosh, mmJEE-Eval, Findings of "
                                  "IJCNLP-AACL 2025", 7, False, MID)])

# ================================================================ 2. problem
S = slide("Problem", "Open models still fail most JEE Advanced problems")
text(S, 0.4, 1.22, 5.0, 0.7, [T("mmJEE-Eval tests vision-language models on 1,460 JEE Advanced "
                                "questions (2019–2025) in English and Hindi. Each question is an "
                                "image: text, formulas, options and often a diagram.", 10.5,
                                False, INK, line=1.1)])
stats = [("37–45%", "of the 2025 questions solved by open models; the best closed models solve "
          "77–84%"),
         ("44.7%", "of the benchmark is numerical questions, where small calculation slips cost "
          "the whole answer"),
         ("1.1–5.2%", "of their own errors fixed by models, although they notice an error in "
          "21–73% of cases")]
for i, (n, cap) in enumerate(stats):
    x = 0.4 + i * 1.72
    if i:
        line(S, x - 0.12, 2.15, x - 0.12, 3.55, HAIR, 0.6)
    text(S, x, 2.05, 1.6, 0.55, [P(R(n, 24, False, BLACK, font=SERIF))])
    text(S, x, 2.65, 1.55, 1.0, [T(cap, 8.5, False, GREY, line=1.05)])
kicker(S, 0.4, 3.85, 5.0, "Our goal")
quote(S, 0.4, 4.1, 5.0, "Help a free, open model solve more of these questions without "
      "retraining it, and test this fairly on exams it has never seen.", 12, 0.85)
picture(S, Q_NUM, 5.75, 1.25, w=3.85, border=True)
text(S, 5.75, 3.45, 3.85, 0.5, [T("JEE Advanced 2026, Paper 1, Q.10 (numerical, with a diagram): "
                                  "one of the 204 questions in our clean test set.", 7.5, False,
                                  MID)])
text(S, 5.75, 4.95, 3.85, 0.25, [T("Statistics on this slide are from the base paper.", 7,
                                   False, MID)])

# ================================================================ 3. why
S = slide("Problem", "Why this problem is worth solving")
reasons = [
    ("A measured gap, not yet fixed", "The base paper only diagnoses. Two of its gaps are "
     "concrete and measurable: detecting vs. fixing errors, and weak numerical accuracy."),
    ("Fits real hardware", "Test-time methods need engineering and careful evaluation, not "
     "large-scale training. A frozen 4-bit model runs on one 12 GB lab GPU."),
    ("Free labels for learning", "Gold answers let us mark every sampled solution right or "
     "wrong automatically: thousands of training examples at no annotation cost."),
    ("A fair test is possible", "Exams are dated, so we can test on a paper written after the "
     "model was released, with the paper's own scoring code."),
]
for i, (t, d) in enumerate(reasons):
    x = 0.4 + (i % 2) * 4.7
    y = 1.35 + (i // 2) * 1.75
    text(S, x, y - 0.05, 0.75, 0.7, [P(R(f"{i + 1:02d}", 30, False, LIGHT, font=SERIF))])
    text(S, x + 0.8, y, 3.6, 0.3, [T(t, 13, True, BLACK)])
    text(S, x + 0.8, y + 0.36, 3.6, 1.0, [T(d, 10, False, GREY, line=1.1)])
    rule(S, x + 0.8, y + 1.42, 3.6)

# ================================================================ 4. journey
S = slide("Problem", "Five steps, each one prompted by the last")
steps = [("Phase 1", "Proposal", "Four modules planned\naround a frozen model", "SOP"),
         ("Phase 2", "Build and test", "Reproduce the paper, pilot,\n7 fixes, full run on 2025",
          "65.3%"),
         ("Step 3", "Clean test", "New 2026 exam, rules fixed\nin advance, stronger system",
          "58.8%"),
         ("Step 4", "Better selection", "Six ideas, each with a\npass rule set first",
          "0 of 6"),
         ("Step 5", "Diagnose", "Where does the gain\nreally come from?", "pool")]
LY = 2.45
line(S, 0.85, LY, 9.15, LY, BLACK, 1.25)
for i, (ph, t, d, res) in enumerate(steps):
    cx = 0.85 + i * 2.075
    rect(S, cx - 0.11, LY - 0.11, 0.22, 0.22, BLACK if i else WHITE, BLACK, 1.25,
         shape=MSO_SHAPE.OVAL)
    kicker(S, cx - 0.9, 1.35, 1.8, ph, MID, 7, PP_ALIGN.CENTER)
    text(S, cx - 0.95, 1.6, 1.9, 0.35, [T(t, 13, True, BLACK, align=PP_ALIGN.CENTER)])
    text(S, cx - 0.95, 2.7, 1.9, 0.6, [T(d, 8.5, False, GREY, align=PP_ALIGN.CENTER)])
    text(S, cx - 0.95, 3.3, 1.9, 0.5, [P(R(res, 20, False, BLACK, font=SERIF),
                                         align=PP_ALIGN.CENTER)])
caps = ["", "on 2025", "on unseen 2026", "passed", "not selection"]
for i, c in enumerate(caps):
    cx = 0.85 + i * 2.075
    text(S, cx - 0.95, 3.72, 1.9, 0.25, [T(c, 8, False, MID, align=PP_ALIGN.CENTER)])
quote(S, 0.4, 4.25, 9.2, "Phase 2 worked, but 2025 had shaped the design, so we tested on 2026. "
      "There the selector stopped helping, so we tried better selectors. None helped, so we "
      "measured where the gain comes from.", 11, 0.75)

# ================================================================ 5. data & rules
S = slide("Method", "A fair test needs strict rules")
kicker(S, 0.4, 1.22, 6, "How we split the exams")
years = [(2019, 216), (2020, 212), (2021, 228), (2022, 210), (2023, 200), (2024, 204),
         (2025, 190), (2026, 204)]
bw, gap = 1.1, 0.0429
for i, (yr, n) in enumerate(years):
    x = 0.4 + i * (bw + gap)
    fill, tc = (TINT2, BLACK) if yr < 2024 else (LIGHT, BLACK) if yr == 2024 else \
        (GREY, WHITE) if yr == 2025 else (BLACK, WHITE)
    rect(S, x, 1.5, bw, 0.62, fill)
    text(S, x, 1.52, bw, 0.6, [P(R(str(yr), 14, False, tc, font=SERIF), align=PP_ALIGN.CENTER),
                               T(f"{n} questions", 7, False, tc, align=PP_ALIGN.CENTER)],
         anchor=MSO_ANCHOR.MIDDLE)
for x0, x1, lab in [(0.4, 0.4 + 6 * (bw + gap) - gap, "Training and development: 1,270 "
                     "questions (2024 = dev year)"),
                    (0.4 + 6 * (bw + gap), 0.4 + 7 * (bw + gap) - gap, "Phase-2 test"),
                    (0.4 + 7 * (bw + gap), 9.6, "Clean final test")]:
    line(S, x0, 2.22, x1, 2.22, BLACK, 0.75)
    line(S, x0, 2.18, x0, 2.26, BLACK, 0.75)
    line(S, x1, 2.18, x1, 2.26, BLACK, 0.75)
    text(S, x0, 2.27, x1 - x0, 0.25, [T(lab, 7.5, False, GREY, align=PP_ALIGN.CENTER)])
rules = [("Same scorer", "The paper's own answer-extraction and scoring code, copied unchanged."),
         ("Same compute", "Our baseline is a majority vote over plain answers with as many "
          "generated tokens as our system."),
         ("Honest intervals", "English and Hindi twins are near-duplicates, so intervals treat "
          "each pair as one unit (cluster bootstrap)."),
         ("No leakage", "No test question or near-copy in any training data or retrieval base."),
         ("Choices on old years", "Every design choice uses 2019–2024 only; the final system is "
          "frozen before the 2026 run."),
         ("Rules before runs", "Later ideas and their pass rules are committed to GitHub before "
          "they are run.")]
for i, (t, d) in enumerate(rules):
    x = 0.4 + (i % 3) * 3.1
    y = 2.75 + (i // 3) * 1.18
    text(S, x, y, 0.4, 0.4, [P(R(f"{i + 1}", 18, False, LIGHT, font=SERIF))])
    text(S, x + 0.35, y + 0.03, 2.6, 0.25, [T(t, 10, True, BLACK)])
    text(S, x + 0.35, y + 0.3, 2.6, 0.8, [T(d, 8.5, False, GREY, line=1.05)])

# ================================================================ 6. architecture
S = slide("Method", "One frozen model, four modules, one trained part")
kicker(S, 0.4, 1.2, 5, "Answering one test question")
Y1, H1 = 1.5, 0.82
cols = [(0.4, 1.3, "Question image", "English or Hindi"),
        (1.95, 1.55, "Read the question", "topic tag + English\ntranscript"),
        (3.75, 1.6, "4  Retrieval", "top-2 similar solved\nproblems, if close"),
        (5.6, 1.65, "1  Code sandbox", "Python for every\ncalculation; 8 answers")]
for i, (x, w, t, d) in enumerate(cols):
    hot = i >= 2
    node(S, x, Y1, w, H1, t, d, fill=WHITE, border=BLACK if hot else LIGHT,
         bw=1.0 if hot else 0.75)
    if i:
        px = cols[i - 1][0] + cols[i - 1][1]
        arrow(S, px, Y1 + H1 / 2, x, Y1 + H1 / 2, GREY)
rect(S, 7.5, Y1 - 0.12, 2.1, 1.06, None, GREY, 0.75, dash=True)
text(S, 7.5, Y1 - 0.1, 2.1, 0.2, [T("2  Agentic correction, ≤ 3 rounds", 7, True, GREY,
                                    align=PP_ALIGN.CENTER)])
node(S, 7.58, Y1 + 0.15, 0.92, 0.72, "Critic", "names the first\nwrong step", ts=8.5, ds=6.5,
     border=BLACK, bw=1.0)
node(S, 8.6, Y1 + 0.15, 0.92, 0.72, "Corrector", "redoes from\nthat step", ts=8.5, ds=6.5,
     border=BLACK, bw=1.0)
arrow(S, 7.25, Y1 + H1 / 2, 7.5, Y1 + H1 / 2, GREY)
Y2 = 2.75
node(S, 7.5, Y2, 2.1, 0.75, "16 candidate answers", "8 original + 8 corrected", border=LIGHT)
arrow(S, 8.55, Y1 + 0.94, 8.55, Y2, GREY)
node(S, 4.45, Y2, 2.7, 0.75, "3  Learned verifier", "scores every answer; the best score wins",
     fill=TINT, border=BLACK, bw=1.25)
arrow(S, 7.5, Y2 + 0.375, 7.15, Y2 + 0.375, GREY)
node(S, 1.95, Y2, 2.15, 0.75, "Final answer", "marked by the paper's scorer", border=LIGHT)
arrow(S, 4.45, Y2 + 0.375, 4.1, Y2 + 0.375, GREY)
rect(S, 0.4, 3.82, 9.2, 1.25, TINT)
kicker(S, 0.55, 3.88, 6, "Prepared in advance from 2019–2024 only", GREY, 6.5)
for x, w, t, d in [(0.55, 2.85, "Labelled answers", "1,270 past questions × 8 answers, marked "
                    "right or wrong by the scorer"),
                   (3.55, 2.75, "Knowledge base", "verified correct solutions (+ JEEBench "
                    "2016–2018), test years removed"),
                   (6.45, 3.0, "Verifier training", "logistic regression vs. XGBoost, chosen "
                    "by leave-one-year-out")]:
    text(S, x, 4.12, w, 0.25, [T(t, 9.5, True, BLACK)])
    text(S, x, 4.37, w, 0.6, [T(d, 8, False, GREY)])
text(S, 0.4, 3.55, 4.0, 0.25, [T("Numbers 1–4 mark the four modules (next four slides).", 7.5,
                                 False, MID)])

# ================================================================ 7. module 1
S = slide("Method", "Module 1  ·  Code makes the arithmetic exact")
NY, NH, NW = 1.55, 0.72, 1.25
xs = [0.4, 1.85, 3.3, 4.75]
labels = [("Solver", "writes Step k and a\nPython code block"),
          ("Safety check", "parse the code; allowed\nmodules only"),
          ("Isolated run", "python -I · 10 s · 1 GB\n≤ 2,000 output chars"),
          ("Output back", "added as an output block;\nthe solver continues")]
for i, (x, (t, d)) in enumerate(zip(xs, labels)):
    node(S, x, NY, NW, NH, t, d, border=BLACK if i == 2 else GREY, bw=1.0 if i == 2 else 0.75,
         fill=TINT if i == 2 else WHITE)
    if i:
        arrow(S, xs[i - 1] + NW, NY + NH / 2, x, NY + NH / 2)
path(S, [(xs[3] + NW / 2, NY), (xs[3] + NW / 2, 1.3), (xs[0] + NW / 2, 1.3), (xs[0] + NW / 2, NY)])
text(S, 1.3, 1.17, 3.5, 0.18, [T("loop: at most 3 code blocks and 8 model calls per answer", 7,
                                 False, GREY, align=PP_ALIGN.CENTER)])
node(S, 0.4, 2.65, NW, 0.55, "Final answer", "one \\boxed{} answer", ts=8.5, ds=7)
arrow(S, xs[0] + NW / 2, NY + NH, xs[0] + NW / 2, 2.65)
node(S, 1.85, 2.65, NW, 0.55, "CoT fallback", "if no boxed answer", ts=8.5, ds=7, dash=True)
arrow(S, xs[0] + NW, 2.92, 1.85, 2.92, GREY, dash=True)
text(S, 3.3, 2.62, 2.7, 0.65, [T("Earlier blocks of the same answer are replayed first, so "
                                 "variables carry over (this fixed 47 NameErrors in the pilot).",
                                 8, False, GREY)])
code(S, 0.4, 3.45, 5.6, 1.62, "What the model writes (format example from our prompt)", [
    "Step 1: <which law / concept applies and the equation to use>",
    "```python",
    "import sympy as sp",
    "x = sp.symbols('x')",
    "print(sp.solve(2*x - 6, x))",
    "```",
    "```output",
    "[3]",
    "```",
    "Step 2: <interpret the printed result, check the options>   \\boxed{<answer>}"], 7)
kicker(S, 6.3, 1.2, 3.3, "Technical details")
yy = spec(S, 6.3, 1.45, 3.3, [
    ("Model", "Qwen3-VL-8B, 4-bit AWQ, frozen; 8 answers per question, temperature 0.7", 0.4),
    ("Allowed", "math, sympy, numpy, scipy, fractions, statistics, itertools …; no files, no "
     "network, no parsers", 0.52),
    ("Limits", "10 s, 1,024 MB and 2,000 output characters per block", 0.27),
    ("Budget", "4,096 tokens; the final system continues a cut-off answer once (+4,096)", 0.4),
])
kicker(S, 6.3, yy + 0.08, 3.3, "Result (2025, Phase 2)")
bignum(S, 6.3, yy + 0.3, 1.6, "40→50%", "numerical questions\nsolved", 20, 7.5)
bignum(S, 8.0, yy + 0.3, 1.6, "+5.0", "points over one answer\n(p = 0.012)", 20, 7.5)

# ================================================================ 8. module 2
S = slide("Method", "Module 2  ·  A second model points at the first wrong step")
NY = 1.6
node(S, 0.4, NY, 1.3, 0.8, "Solver answer", "Steps 1 … n\n(8 per question)")
node(S, 2.1, NY, 1.5, 0.8, "Critic", "InternVL3.5-8B:\na different model", fill=TINT,
     border=BLACK, bw=1.25)
node(S, 4.55, NY, 1.45, 0.8, "Corrector", "keeps Steps 1 … k−1,\nredoes from Step k",
     border=BLACK, bw=1.0)
arrow(S, 1.7, NY + 0.4, 2.1, NY + 0.4)
arrow(S, 3.6, NY + 0.4, 4.55, NY + 0.4)
text(S, 3.62, NY + 0.1, 0.92, 0.3, [T("ERROR in\nStep k", 7, True, BLACK, align=PP_ALIGN.CENTER)])
path(S, [(5.275, NY), (5.275, 1.32), (2.85, 1.32), (2.85, NY)])
text(S, 3.0, 1.17, 2.2, 0.18, [T("revised answer, checked again (≤ 3 rounds)", 7, False, GREY,
                                 align=PP_ALIGN.CENTER)])
node(S, 2.1, 2.8, 1.5, 0.55, "Stop", "CORRECT, or the answer\nis unchanged twice", ts=8.5, ds=6.5)
arrow(S, 2.85, NY + 0.8, 2.85, 2.8)
node(S, 4.55, 2.8, 1.45, 0.55, "Answer pool", "8 original + 8 corrected", ts=8.5, ds=6.5)
arrow(S, 3.6, 3.075, 4.55, 3.075)
text(S, 0.4, 2.62, 1.6, 0.85, [T("A cut-off critic reply gets a short cue to force a verdict, "
                                 "so every check ends with a decision.", 7.5, False, GREY)])
code(S, 0.4, 3.6, 5.6, 1.47, "How every critic reply must end (from our prompt)", [
    "VERDICT: CORRECT or ERROR",
    "FIRST_ERROR_STEP: <step number, or NONE>",
    "EXPLANATION: <one or two sentences: what is wrong and how to fix it>",
    "",
    "If the reply is cut off, we append:",
    "\"I must stop checking now and give my decision.  VERDICT:\""], 7)
kicker(S, 6.3, 1.2, 3.3, "Technical details")
yy = spec(S, 6.3, 1.45, 3.3, [
    ("Critic", "InternVL3.5-8B, 4-bit, greedy; short check, up to 1,536 tokens", 0.4),
    ("Why two", "a different model family reduces shared blind spots", 0.27),
    ("Corrector", "the solver itself, with the same code rules and the critic's explanation", 0.4),
    ("Gate", "final system: keep a fix only if the verifier score rises (δ = −0.08)", 0.4),
])
kicker(S, 6.3, yy + 0.08, 3.3, "Result (2025, Phase 2)")
bignum(S, 6.3, yy + 0.3, 1.6, "14–19%", "of detected errors fixed\n(paper: 1.1–5.2%)", 20, 7.5)
bignum(S, 8.0, yy + 0.3, 1.6, "87 / 39", "wrong→right vs.\nright→wrong flips", 20, 7.5)

# ================================================================ 9. module 3
S = slide("Method", "Module 3  ·  A small classifier picks the final answer")
kicker(S, 0.4, 1.2, 5, "Training, on 2019–2024 only", GREY, 6.5)
tr = [("1,270 past\nquestions", ""), ("8 answers each", "+ corrected ones"),
      ("Auto-label", "paper's scorer\nvs. gold answer"), ("Features", "simple signals\nper answer"),
      ("XGBoost ranker", "chosen by leave-\none-year-out")]
xs = [0.4 + i * 1.15 for i in range(5)]
for i, (x, (t, d)) in enumerate(zip(xs, tr)):
    node(S, x, 1.45, 1.0, 0.75, t, d, ts=8.5, ds=6.5, fill=TINT if i == 4 else WHITE,
         border=BLACK if i == 4 else GREY, bw=1.25 if i == 4 else 0.75)
    if i:
        arrow(S, xs[i - 1] + 1.0, 1.825, x, 1.825)
kicker(S, 0.4, 2.42, 5, "At test time, for every question", GREY, 6.5)
te = [(2.7, "16 candidates", "8 original +\n8 corrected"), (3.85, "Score each", "probability\nof being right"),
      (5.0, "Pick the best", "highest score\nwins")]
for i, (x, t, d) in enumerate(te):
    node(S, x, 2.65, 1.0, 0.7, t, d, ts=8.5, ds=6.5)
    if i:
        arrow(S, te[i - 1][0] + 1.0, 3.0, x, 3.0)
arrow(S, xs[4] + 0.5, 2.2, 4.35, 2.65)
code(S, 0.4, 3.6, 5.6, 1.47, "Features (one row per candidate answer)", [
    "length · number of steps · code ran OK · cut off at the limit",
    "hedging words · valid answer format",
    "agreement with the other answers (whole pool and each half)",
    "is the majority answer · critic flagged it · rounds · answer changed",
    "Phase 2 also: a sentence embedding of the answer (bge-small, 32 dims)"], 7.5)
kicker(S, 6.3, 1.2, 3.3, "Technical details")
yy = spec(S, 6.3, 1.45, 3.3, [
    ("Data", "10,160 labelled answers in Phase 2 (45.5% right); 12,096 incl. corrected later",
     0.4),
    ("Models", "logistic regression vs. XGBoost; final: pairwise ranker, 400 trees, depth 4",
     0.4),
    ("Split", "by exam year, so answers to one question never sit on both sides", 0.4),
    ("Pick", "highest score among the 16 candidates", 0.27),
])
kicker(S, 6.3, yy + 0.08, 3.3, "Result")
bignum(S, 6.3, yy + 0.3, 1.6, "+11.8", "points: biggest single\nstep (2025, Phase 2)", 20, 7.5)
bignum(S, 8.0, yy + 0.3, 1.6, "62.0%", "pick accuracy vs. 53.7%\nfor majority vote (LOYO)", 20,
       7.5)

# ================================================================ 10. module 4
S = slide("Method", "Module 4  ·  Solved past problems as worked examples")
r1 = [("Question\nimage", ""), ("Concept tagger", "one line: subject,\ntopic, method"),
      ("Embed", "bge-small-en-v1.5"), ("Search the KB", "FAISS, same\nsubject only"),
      ("Similar enough?", "cosine ≥ τ")]
xs = [0.4, 1.52, 2.64, 3.76, 4.88]
ws = [0.95, 0.95, 0.95, 0.95, 1.12]
for i, ((t, d), x, w) in enumerate(zip(r1, xs, ws)):
    node(S, x, 1.4, w, 0.75, t, d, ts=8.5, ds=6.5, fill=TINT if i == 4 else WHITE,
         border=BLACK if i in (3, 4) else GREY, bw=1.0 if i in (3, 4) else 0.75)
    if i:
        arrow(S, xs[i - 1] + ws[i - 1], 1.775, x, 1.775)
node(S, 4.88, 2.45, 1.12, 0.62, "Yes: add top 2", "worked solutions\nas method examples",
     ts=8, ds=6.5, border=BLACK, bw=1.0)
arrow(S, 5.44, 2.15, 5.44, 2.45)
text(S, 4.88, 3.1, 1.12, 0.4, [T("No: prompt unchanged, so weak matches add nothing", 6.5, False,
                                 GREY, align=PP_ALIGN.CENTER)])
node(S, 0.4, 2.45, 1.85, 0.62, "Verified solutions", "2019–2024 answers marked right\n"
     "(+ JEEBench 2016–2018 later)", ts=8, ds=6.5)
node(S, 2.45, 2.45, 1.6, 0.62, "Remove test overlap", "cosine ≥ 0.90 to any\n2025 / 2026 question",
     ts=8, ds=6.5)
arrow(S, 2.25, 2.76, 2.45, 2.76)
path(S, [(4.05, 2.76), (4.235, 2.76), (4.235, 2.15)])
code(S, 0.4, 3.6, 5.6, 1.47, "Tagger output and exemplar header (from our prompts)", [
    "SUBJECT: <Physics/Chemistry/Mathematics> | TOPIC: <2-6 words>",
    "| METHOD: <key concept or technique, at most 15 words>",
    "",
    "\"Below are worked solutions of earlier JEE Advanced problems that",
    " use a similar method. Use them only as guidance on the method.",
    " They are DIFFERENT problems: do not copy their numbers or answers.\""], 7)
kicker(S, 6.3, 1.2, 3.3, "Technical details")
yy = spec(S, 6.3, 1.45, 3.3, [
    ("Query", "the model's own tag, greedy, ≤ 160 tokens (no OCR needed)", 0.4),
    ("KB", "Phase 2: 454 verified entries; later + JEEBench and tag + text matching", 0.4),
    ("Threshold", "τ tuned on old years for 70% proxy precision (Phase 2: 0.885)", 0.4),
    ("Coverage", "final recipe 11.7% vs. 8.4% of questions on old years", 0.27),
])
kicker(S, 6.3, yy + 0.08, 3.3, "Result (2025, Phase 2)")
bignum(S, 6.3, yy + 0.3, 1.6, "30 / 190", "questions found a close\nexample", 20, 7.5)
bignum(S, 8.0, yy + 0.3, 1.6, "30→38%", "accuracy on those\nquestions (code solver)", 20, 7.5)

# ================================================================ 11. setup check
S = slide("Phase 2", "First, prove that our measuring tool is right")
bignum(S, 0.4, 1.35, 2.2, "13.8%", "the paper's Qwen2.5-VL-7B, run\nthrough our code (10 runs)", 40)
bignum(S, 2.75, 1.35, 2.2, "12.4%", "the same model, as reported\nin the paper", 40, color=MID)
text(S, 0.4, 2.65, 4.6, 0.6, [T("The paper's value lies inside our 95% interval (10.7–17.3), so "
                                "prompts, image handling and scoring match the original.", 9.5,
                                False, INK)])
line(S, 5.3, 1.35, 5.3, 4.9, HAIR, 0.6)
kicker(S, 5.6, 1.3, 4, "Our solver before any module")
facts = [("43.4%", "Qwen3-VL-8B, one answer (16 runs)"),
         ("53.7%", "majority vote over 16 answers: the strong rival"),
         ("−10 / −18", "points for Hindi / for questions that need the diagram"),
         ("40%", "of answers cut off at 4,096 tokens")]
for i, (n, d) in enumerate(facts):
    y = 1.6 + i * 0.75
    text(S, 5.6, y, 1.4, 0.5, [P(R(n, 18, False, BLACK, font=SERIF))])
    text(S, 7.05, y + 0.08, 2.55, 0.55, [T(d, 9, False, GREY)])
quote(S, 0.4, 3.55, 4.6, "Our tool matches the paper, and the baseline to beat is already "
      "strong.", 11.5, 0.75)

# ================================================================ 12. pilot
S = slide("Phase 2", "A 40-question pilot: every module adds accuracy")
table(S, 0.4, 1.3, [2.85, 0.85, 1.1], ["Pilot setup (40 test questions)", "Accuracy", "95% CI"], [
    ["Single answer (CoT)", "29.5", "17.8–41.6"], ["Majority vote, same compute", "35.0", "20.0–52.5"],
    ["+ Code sandbox", "41.2", "28.1–54.4"], ["+ Correction, same-model critic", "46.6", "32.8–60.3"],
    ["+ Correction, cross-model critic", "43.1", "29.7–56.2"], ["+ Learned verifier", "50.0", "32.5–67.5"],
    ["Full framework", "57.5", "40.0–72.5"]], font=9.5, row_h=0.36, hi=(6,))
line(S, 5.3, 1.35, 5.3, 4.9, HAIR, 0.6)
kicker(S, 5.6, 1.3, 4, "Why a pilot first")
text(S, 5.6, 1.6, 4.0, 2.2, [
    Bl("The full run costs about 54 GPU-hours, so mistakes are expensive.", 10, after=6),
    Bl("Full framework vs. majority vote: +22.5 points (interval +7.5 to +37.5). With 40 "
       "questions this is only a first signal.", 10, after=6),
    Bl("Its real value: reading 1,520 solution traces and 640 correction chains showed us what "
       "was broken.", 10)])
quote(S, 5.6, 4.15, 4.0, "Seven problems found and fixed before the 54-hour full run.", 11.5, 0.6)

# ================================================================ 13. fixes
S = slide("Phase 2", "Seven problems found in the pilot, fixed before the full run")
table(S, 0.4, 1.25, [1.7, 2.35, 2.35], ["Problem", "Evidence", "Fix and effect"], [
    ["Code state lost", "47 NameErrors: each block ran fresh", "replay earlier blocks → 0"],
    ["Library blocked", "86 blocks needed SciPy", "allow SciPy, no file I/O → 0"],
    ["No critic verdict", "76 replies cut off while re-solving", "short check + forced verdict:\n1 of 501 left; detection 35→56%"],
    ["Many fallbacks", "480 of 539 were long reasoning\nloops", "forcing an answer tested:\n21.9% vs. 30.0% → rejected"],
    ["Baseline too small", "system used ≈ 18 answers' tokens,\nonly 16 drawn", "draw 20 answers for a fair vote"],
    ["Selector = vote", "dev pick equal to majority vote", "reported honestly; 8× more data"],
    ["Batch aborted", "one 12,303-token prompt", "retry request by request"],
], font=8, row_h=0.47, align=["l", "l", "l"])
fig(S, "pilot_fixes", 6.6, 1.3, 3.0)
text(S, 6.6, 3.0, 3.0, 1.0, [T("Same 40 questions, before vs. after the fixes.", 8, False, GREY,
                               after=6),
                             T("A three-person code review also found 11 bugs before any GPU "
                               "run, e.g. a verifier crash, a train/test feature mismatch and "
                               "intervals that were too narrow.", 8, False, GREY)])

# ================================================================ 14. phase-2 results
S = slide("Phase 2", "Phase 2 on the 2025 exam: 65.3%, +12.1 over majority vote")
fig(S, "phase2_ladder", 0.35, 1.2, 5.4)
line(S, 6.0, 1.3, 6.0, 4.95, HAIR, 0.6)
bignum(S, 6.25, 1.25, 3.3, "65.3%", "full system, up from 43.2% for one answer", 36)
text(S, 6.25, 2.35, 3.35, 2.6, [
    Bl("+12.1 points over majority vote at equal compute (interval +4.2 to +20.5, p = 0.001).",
       9.5, after=4),
    Bl("Numerical questions: 40.2% → 50.4% with the code sandbox.", 9.5, after=4),
    Bl("Learned verifier: the biggest single step, +11.8.", 9.5, after=4),
    Bl("Retrieval: +2.1, not significant.", 9.5, after=4),
    Bl("No memorisation sign: 43.9% on 2019–24 vs. 42.6% on 2025.", 9.5)])
text(S, 0.4, 4.45, 5.4, 0.5, [T("190 questions, English + Hindi. Bars show 95% intervals; each "
                                "“+” row adds one module.", 7.5, False, MID)])

# ================================================================ 15. what phase 2 left open
S = slide("Phase 2", "What Phase 2 left open")
fig(S, "correction", 0.35, 1.2, 5.35)
text(S, 0.4, 3.35, 5.3, 0.4, [T("Left: detected errors that end correct after up to 3 rounds "
                                "(grey band: the paper's single-pass rate). Right: answers "
                                "flipped by correction.", 7.5, False, MID)])
text(S, 0.4, 3.85, 5.3, 1.2, [
    Bl("Our loop fixes 14–19% of the errors it detects, well above the paper's 1.1–5.2%: the "
       "critic points at one step and the solver can run code.", 9.5, after=4),
    Bl("But the critic also flags 19–26% of correct answers, so some right answers turn wrong.",
       9.5)])
line(S, 6.0, 1.3, 6.0, 4.95, HAIR, 0.6)
opens = [("Picking", "The verifier picked 65.3%, about the same as a plain vote over the same "
          "16 answers (65.8%). The gain came from the varied pool."),
         ("Retrieval", "Only 30 of 190 questions found a close example."),
         ("Budget", "36% of solutions fell back to plain reasoning, mostly at the token limit."),
         ("Fairness", "2025 had been used for the pilot, and the model may have seen it.")]
for i, (t, d) in enumerate(opens):
    y = 1.3 + i * 0.92
    text(S, 6.25, y, 0.5, 0.45, [P(R(f"{i + 1}", 20, False, LIGHT, font=SERIF))])
    text(S, 6.7, y + 0.02, 2.9, 0.25, [T(t, 10, True, BLACK)])
    text(S, 6.7, y + 0.27, 2.9, 0.65, [T(d, 8.5, False, GREY)])

# ================================================================ 16. clean test year
S = slide("Clean test", "A clean test: the 2026 exam, in two languages")
picture(S, Q_EN, 0.4, 1.25, w=4.5, border=True)
picture(S, Q_HI, 5.1, 1.25, w=4.5, border=True)
kicker(S, 0.4, 4.1, 4.5, "English · from the dataset", GREY, 6.5)
kicker(S, 5.1, 4.25, 4.5, "Hindi · cropped by us from the official paper", GREY, 6.5)
text(S, 0.4, 4.42, 9.2, 0.65, [
    Bl("Held after Qwen3-VL was released, so the model cannot have seen it. The dataset had "
       "only the 102 English questions.", 8.5, after=2),
    Bl("We built the 102 Hindi twins: matched by paper, subject and number, answers from the "
       "English key, automatic checks on every pair, 14 crops checked by hand.", 8.5)])

# ================================================================ 17. dev a/b
S = slide("Clean test", "Every improvement was chosen on old exams only")
fig(S, "dev_ab", 0.35, 1.2, 5.5)
text(S, 0.4, 4.0, 5.4, 0.5, [T("2024 dev year, 204 questions. Rule fixed in advance: keep a "
                               "change only if it gains and the lower end of its 95% interval "
                               "is above zero (black = passed).", 7.5, False, MID)])
line(S, 6.05, 1.3, 6.05, 4.95, HAIR, 0.6)
kicker(S, 6.3, 1.25, 3.3, "What went into the final system")
items = [("Transcript", "the model writes the question out in English first (Hindi gain "
          "+8.7)"),
         ("Continuation", "a cut-off answer continues once for 4,096 more tokens, in every row"),
         ("Hybrid retrieval", "tag + problem text, + JEEBench: coverage 8.4 → 11.7%"),
         ("Ranking verifier", "XGBoost ranker on pools with corrected answers: 62.0% vs. 53.7% "
          "for a vote"),
         ("Correction gate", "keep a fix only if the verifier score rises")]
for i, (t, d) in enumerate(items):
    y = 1.55 + i * 0.68
    text(S, 6.3, y, 3.3, 0.25, [T(t, 9.5, True, BLACK)])
    text(S, 6.3, y + 0.24, 3.3, 0.45, [T(d, 8, False, GREY)])
    if i < len(items) - 1:
        line(S, 6.3, y + 0.64, 9.6, y + 0.64, HAIR, 0.5)

# ================================================================ 18. main result
S = slide("Clean test", "58.8% on an exam the model has never seen")
fig(S, "main_2026", 0.3, 1.2, 6.0)
text(S, 0.4, 3.95, 5.9, 0.4, [T("Bars with 95% intervals. Majority vote gets as many generated "
                                "tokens as the full system (18 answers).", 7.5, False, MID)])
line(S, 6.5, 1.3, 6.5, 4.95, HAIR, 0.6)
bignum(S, 6.75, 1.2, 2.9, "58.8%", "final system; one answer: 44.3%", 40)
bignum(S, 6.75, 2.3, 2.9, "+6.4", "points over majority vote at equal compute\n(+1.5 to +10.8, "
       "p = 0.011)", 26)
text(S, 6.75, 3.35, 2.85, 1.6, [
    Bl("The Phase-2 system, re-run unchanged, also beats its vote by +6.4 (p = 0.015).", 8.5,
       after=3),
    Bl("Stronger solver: +6.8 over the Phase-2 solver (p < 0.001).", 8.5, after=3),
    Bl("Final vs. Phase-2 system: +3.4, not significant on 204 questions.", 8.5)])
quote(S, 0.4, 4.4, 5.9, "Our main claim holds on unseen data.", 12, 0.55)

# ================================================================ 19. compute
S = slide("Clean test", "The gain is not just more compute")
fig(S, "compute_2026", 0.35, 1.2, 5.0)
line(S, 5.65, 1.3, 5.65, 4.95, HAIR, 0.6)
kicker(S, 5.9, 1.25, 3.7, "How to read it")
text(S, 5.9, 1.55, 3.7, 1.9, [
    Bl("Dashed lines: majority vote over n plain answers as n grows. This is what extra "
       "compute buys without our modules.", 9.5, after=5),
    Bl("Dots: our rows (small) and full systems (large). Every dot lies above the dashed line "
       "at its cost.", 9.5)])
bignum(S, 5.9, 3.3, 1.7, "58.8%", "final system,\n80k tokens / question", 24)
bignum(S, 7.75, 3.3, 1.85, "52.5%", "majority vote with\nthe same tokens", 24, color=MID)

# ================================================================ 20. breakdown
S = slide("Clean test", "Gains everywhere, but diagrams stay hard")
fig(S, "breakdown_2026", 0.3, 1.2, 6.2)
line(S, 6.75, 1.3, 6.75, 4.95, HAIR, 0.6)
bignum(S, 7.0, 1.2, 2.6, "30% vs. 72%", "diagram vs. text-only questions,\nfinal system, 2026", 22)
text(S, 7.0, 2.25, 2.6, 2.6, [
    Bl("Physics, which has the most diagrams: 35%; Maths: 81%.", 9, after=4),
    Bl("Hindi 54% vs. English 64%; part of this may come from our own crops.", 9, after=4),
    Bl("Reasoning and code help; they do not fix reading the figure.", 9)])

# ================================================================ 21. correction on clean data
S = slide("Clean test", "Correction still works on unseen data")
table(S, 0.4, 1.3, [3.05, 1.35], ["Final system, 2026 (1,632 chains)", "Value"], [
    ["Wrong answers flagged by the critic (round 1)", "52.9%"],
    ["Flagged errors fixed in round 1", "13.2%"],
    ["Correct answers wrongly flagged", "19.9%"],
    ["Accuracy per chain, before → after", "49.8 → 52.0%"]], font=9, row_h=0.36)
text(S, 0.4, 3.2, 4.4, 0.3, [T("Base paper, single pass: 1.1–5.2% of detected errors fixed.", 8,
                               False, MID)])
table(S, 5.2, 1.3, [2.0, 1.2, 1.2], ["Correction gate, 2026", "No gate", "Gate"], [
    ["Right → wrong flips", "33", "9"], ["Wrong → right flips", "68", "54"],
    ["Accuracy per chain", "52.0%", "52.6%"]], font=9, row_h=0.36, hi=(0,))
text(S, 5.2, 2.85, 4.4, 0.5, [T("The gate keeps a correction only if the verifier score rises.",
                                8, False, MID)])
quote(S, 0.4, 3.75, 9.2, "Pointing at the first wrong step keeps working; the gate removes most "
      "harmful flips (33 → 9) but also blocks good fixes, so accuracy moves only +0.6.", 11.5,
      0.85)

# ================================================================ 22. honesty
S = slide("Clean test", "No memorisation, and the method needs a capable solver")
fig(S, "memorisation_probe", 0.35, 1.2, 4.6)
text(S, 0.4, 3.62, 4.6, 0.6, [T("Answer-only probe (no reasoning). If old papers were memorised, "
                                "old years would score higher than 2026. They do not.", 8, False,
                                MID)])
line(S, 5.25, 1.3, 5.25, 4.95, HAIR, 0.6)
kicker(S, 5.5, 1.25, 4.1, "With the paper's weaker model (2025)")
table(S, 5.5, 1.5, [2.6, 1.5], ["Qwen2.5-VL-7B", "Accuracy"], [
    ["Single answer", "14.0%"], ["Majority vote, same compute", "16.3%"],
    ["Full framework", "15.3%"]], font=9, row_h=0.33)
text(S, 5.5, 2.95, 4.1, 2.0, [
    Bl("No gain: when only ~14% of answers are right, agreement, critique and verification "
       "carry little signal.", 9, after=4),
    Bl("2025, used during design, shows bigger gains than 2026 (+13.7 vs. +6.4 over a vote), "
       "consistent with some optimism on a seen year.", 9)])

# ================================================================ 23. errors
S = slide("Diagnosis", "Where the remaining errors are")
fig(S, "errors_2026", 0.3, 1.2, 5.9)
line(S, 6.45, 1.3, 6.45, 4.95, HAIR, 0.6)
bignum(S, 6.7, 1.2, 1.4, "42", "right answer in the\npool, not picked", 30)
bignum(S, 8.2, 1.2, 1.4, "42", "no right answer\nin the pool", 30, color=MID)
text(S, 6.7, 2.45, 2.9, 2.5, [
    Bl("120 of 204 questions solved.", 9, after=4),
    Bl("In the missed pools only 17% of answers are right: the right answer is a small "
       "minority.", 9, after=4),
    Bl("Diagram questions: 35% missed by the selector, 35% never solved.", 9, after=4),
    Bl("Longer answers are not the main cause: they are longer in both years, yet the "
       "verifier helps only on 2025.", 9)])

# ================================================================ 24. selectors + retrieval
S = slide("Diagnosis", "Six fixes for selection and retrieval, none passed its rule")
fig(S, "selectors", 0.3, 1.2, 5.4)
text(S, 0.4, 4.25, 5.3, 0.7, [T("Pick accuracy, leave-one-year-out over 2019–2024 (no test "
                                "data). Rule set in advance: beat the current ranker on every "
                                "pool type. Ranker noise ≈ 1 point.", 7.5, False, MID)])
line(S, 5.95, 1.3, 5.95, 4.95, HAIR, 0.6)
kicker(S, 6.2, 1.25, 3.4, "Selectors")
text(S, 6.2, 1.5, 3.4, 1.45, [
    Bl("Fine-tuned neural verifier (ModernBERT): learns a signal (AUC 0.79), ranks worse.", 8.5,
       after=3),
    Bl("3,184 new training answers matching the test pools (≈ 30 GPU-hours): 61.2%, within "
       "noise.", 8.5)])
kicker(S, 6.2, 3.0, 3.4, "Retrieval")
text(S, 6.2, 3.25, 3.4, 1.75, [
    Bl("Method cards (method, plan, key equations, common mistake): +3.6 where retrieval fires "
       "(−0.8 to +8.5), so not adopted.", 8.5, after=3),
    Bl("207 open-licensed OlympiadBench solutions: coverage 15.0 → 13.5%, 2026 accuracy "
       "unchanged.", 8.5)])

# ================================================================ 25. decomposition
S = slide("Diagnosis", "The gain comes from the pool, not the picker")
fig(S, "decomposition", 0.35, 1.2, 4.9)
table(S, 5.5, 1.3, [2.05, 1.0, 1.05], ["", "2025", "2026"], [
    ["Pool gain\n(vote on our pool − plain vote)", "+7.4", "+6.9"],
    ["Selection gain\n(verifier pick − vote)", "+6.3", "−0.5"],
    ["Verifier AUC, all answers", "0.87", "0.90"],
    ["AUC inside mixed pools", "0.80", "0.75"]], font=9, row_h=0.42, hi=(1,))
text(S, 5.5, 3.5, 4.1, 0.6, [T("The overall AUC rises on 2026 because it mostly separates easy "
                               "questions from hard ones. Inside pools with both right and wrong "
                               "answers, the only place picking matters, it falls.", 8.5, False,
                               GREY)])
quote(S, 5.5, 4.3, 4.1, "A high AUC is not proof of good selection.", 12, 0.6)

# ================================================================ 26. last two ideas
S = slide("Diagnosis", "Two last ideas, two clear answers")
kicker(S, 0.4, 1.25, 3.4, "A · Answer-level verifier")
bignum(S, 0.4, 1.5, 3.3, "57.9%", "vs. 62.0% for the current ranker\n(worst year 50.0 vs. 52.5)",
       28)
text(S, 0.4, 2.55, 3.3, 2.4, [
    Bl("One row per distinct answer, not per solution; pick the most likely answer.", 9, after=3),
    Bl("5,335 answer rows from 15,280 solutions; model chosen by its worst year.", 9, after=3),
    Bl("Beats voting (53.7%) but loses the detail of single solutions.", 9)])
line(S, 3.95, 1.3, 3.95, 4.95, HAIR, 0.6)
kicker(S, 4.2, 1.25, 5.4, "B · Diagram-first transcript (2024 dev)")
fig(S, "diagram_first", 4.2, 1.5, 5.4)
text(S, 4.2, 3.55, 5.4, 1.45, [
    Bl("A second, itemised reading of the figure for half of each pool, same seeds.", 9, after=3),
    Bl("Majority vote −2.0 (−5.4 to +1.0); per answer −3.4, worst on diagram questions.", 9,
       after=3),
    Bl("Longer prompt, more cut-offs (37.3 → 42.6%): the solver misreads the figure whichever "
       "description it gets.", 9)])

# ================================================================ 27. findings
S = slide("Diagnosis", "Everything we learned, in one table")
table(S, 0.4, 1.22, [2.0, 2.75, 4.45], ["Idea", "Result", "Why (from our analysis)"], [
    ["Code sandbox", "numerical 40 → 50% (2025)", "exact arithmetic instead of mental maths"],
    ["Agentic correction", "fixes 14–19% of detected errors", "critic names one step; but flags 19–26% of right answers"],
    ["Learned verifier", "+11.8 on 2025; ±0 vs. vote on 2026", "ranking inside mixed pools got worse on the new year"],
    ["Retrieval", "+0.2 to +2.1, not significant", "fires for only 15–16% of questions"],
    ["Transcript", "+4.9 on dev (Hindi +8.7)", "writing the question out helps reading, most of all in Hindi"],
    ["Continue cut-off answers", "+2.3 (code), +6.4 (plain) on dev", "many answers were cut off at 4,096 tokens"],
    ["Correction gate", "bad flips 33 → 9, accuracy +0.6", "blocks good fixes as well as bad ones"],
    ["New selectors (4)", "all below 62.0 (LOYO)", "right answer is a small minority in missed pools"],
    ["Method cards / open KB", "+3.6 (not sure) / +0.0", "too few questions where retrieval fires"],
    ["Diagram-first reading", "−2.0 on dev", "longer prompt, more cut-offs; solver misreads figures"],
    ["Weaker solver (7B)", "no gain (15.3 vs. 16.3)", "too few right answers for voting or checking"],
], font=8, row_h=0.315, align=["l", "l", "l"])

# ================================================================ 28. engineering problems
S = slide("Engineering", "Running 8B models on one shared 12 GB GPU")
table(S, 0.4, 1.22, [2.3, 3.4, 3.5], ["Problem", "What happened", "What we did"], [
    ["Model did not fit", "only 0.36 GB left for the attention cache", "smaller images, 8-bit cache, batch limit: 1.9× faster"],
    ["Runs killed silently", "out-of-memory daemon killed a session (10.5 h lost)", "detached supervisor, 3 retries per stage, quiet watcher"],
    ["Three lab shutdowns", "PC powered off; driver missing after reboot", "driver installed, auto power-on; resumed with identical results"],
    ["GPU crash at start", "out of memory, then a CUDA error in a forked worker", "embedding model on CPU; workers started with spawn"],
    ["Second PC unusable", "no GPU driver; campus proxy blocked downloads", "packages copied over LAN; pre-compiled kernel installed"],
    ["One prompt, whole batch lost", "a 12,303-token prompt crashed hours of work", "catch the error, retry request by request"],
    ["Sandbox could eat RAM", "a printing loop used 8 GB in a test", "output written to size-capped files"],
    ["No Hindi 2026 questions", "the dataset had English only", "cropped 102 Hindi questions from official papers"],
], font=8.5, row_h=0.43, align=["l", "l", "l"])

# ================================================================ 29. optimisations
S = slide("Engineering", "Optimisations that made it fit, and kept it honest")
opts = [("4-bit models", "AWQ weights let two 8B models share one 12 GB GPU, one at a time."),
        ("Fast batched inference", "vLLM batching, 8-bit attention cache, capped image size: "
         "1.9× throughput."),
        ("Cache every call", "each answer stored with a fixed seed: crashes and re-runs cost "
         "nothing; samples reused across experiments."),
        ("Cached embeddings", "knowledge-base rebuild: 38 min → 1.8 min."),
        ("Matched seeds", "with and without retrieval share seeds, so only questions where it "
         "fires can change."),
        ("Extend only what is cut off", "instead of doubling every answer's budget (≈ 55 "
         "GPU-hours).")]
for i, (t, d) in enumerate(opts):
    x = 0.4 + (i % 3) * 3.1
    y = 1.3 + (i // 3) * 1.22
    rule(S, x, y, 2.9, heavy=True)
    text(S, x, y + 0.1, 2.9, 0.28, [T(t, 10.5, True, BLACK)])
    text(S, x, y + 0.4, 2.9, 0.75, [T(d, 8.5, False, GREY)])
kicker(S, 0.4, 3.85, 9, "Testing and reproducibility")
qa = [("115", "automated tests"), ("11", "bugs found in review\nbefore the first GPU run"),
      ("3", "freeze / pre-registration\nfiles, committed first"), ("240+", "hours of logged\nGPU runs")]
for i, (n, d) in enumerate(qa):
    x = 0.4 + i * 2.33
    text(S, x, 4.1, 0.95, 0.5, [P(R(n, 22, False, BLACK, font=SERIF))])
    text(S, x + 0.95, 4.16, 1.35, 0.5, [T(d, 8, False, GREY)])

# ================================================================ 30. impact
S = slide("Impact", "What this project achieved")
imp = [("+14.6", "points on an unseen exam", "44.3% → 58.8% on JEE Advanced 2026 with a frozen "
        "8B model, no retraining."),
       ("+6.4", "points at equal compute", "beats majority voting with the same tokens "
        "(p = 0.011), for both of our systems."),
       ("14–19%", "of detected errors fixed", "vs. 1.1–5.2% reported by the base paper for "
        "single-pass correction."),
       ("102", "Hindi 2026 questions built", "a clean bilingual test year that did not exist in "
        "the dataset."),
       ("8", "ideas tested and reported", "pre-registered negative results show what does not "
        "help, and why."),
       ("13.8%", "vs. 12.4% reproduced", "the paper's own number, matched before we built on "
        "it; open repository with 115 tests.")]
for i, (big, cap, d) in enumerate(imp):
    x = 0.4 + (i % 3) * 3.1
    y = 1.25 + (i // 3) * 1.9
    text(S, x, y, 2.9, 0.65, [P(R(big, 32, False, BLACK, font=SERIF))])
    kicker(S, x, y + 0.7, 2.9, cap, GREY, 7)
    line(S, x, y + 0.98, x + 2.6, y + 0.98, HAIR, 0.6)
    text(S, x, y + 1.05, 2.7, 0.75, [T(d, 8.5, False, INK)])

# ================================================================ 31. conclusion
S = slide("Impact", "Conclusion")
text(S, 0.4, 1.3, 5.6, 2.8, [
    P(R("On an exam the model cannot have seen, our framework lifts a frozen 8B model from 44.3% "
        "to 58.8% and beats majority voting at the same compute by 6.4 points.", 13, False, BLACK,
        font=SERIF), after=10, line=1.1),
    P(R("The gain comes from a richer pool of candidate answers, and it is stable across years. "
        "Picking the best answer is the fragile part, and diagrams remain the main open "
        "problem.", 11, False, GREY, font=SERIF), line=1.1)])
line(S, 6.3, 1.3, 6.3, 4.1, HAIR, 0.6)
kicker(S, 6.55, 1.25, 3, "Limitations")
text(S, 6.55, 1.5, 3.05, 1.35, [
    Bl("one solver, one clean year of 204 questions (intervals ≈ ±8 points)", 9, after=3),
    Bl("Hindi 2026 crops are our own; results after the freeze are post-hoc", 9)])
kicker(S, 6.55, 2.7, 3, "Next steps")
text(S, 6.55, 2.95, 3.05, 1.2, [
    Bl("better perception for diagrams and Hindi, to put the right answer into more pools", 9,
       after=3),
    Bl("larger dev sets to settle method cards", 9)])
quote(S, 0.4, 4.3, 9.2, "Every claim was tested against a fair baseline, with rules fixed in "
      "advance, and every negative result is reported.", 12, 0.7)

# ================================================================ 32. thank you
s = blank()
SLIDES.append((s, None))
S = s.shapes
rect(S, 6.55, 0, 3.45, 5.625, CHAR)
picture(S, Q_HI, 6.85, 1.05, w=2.85, border=False, rot=2.5)
text(S, 6.85, 3.4, 2.85, 0.5, [T("The Hindi twin of the title slide's question, cropped by us "
                                 "from the official 2026 paper.", 7.5, False, LIGHT)])
kicker(S, 6.85, 4.95, 2.85, "IIT Bhilai  ·  Machine Learning", LIGHT, 6.5)
kicker(S, 0.55, 0.55, 5.5, "Final submission  ·  Team Erdős", MID, 8)
text(S, 0.55, 1.2, 5.7, 1.2, [P(R("Thank you", 54, False, BLACK, font=SERIF))])
rect(S, 0.55, 2.45, 0.75, 0.04, BLACK)
text(S, 0.55, 2.6, 5.6, 0.5, [P(R("Questions and discussion are welcome.", 13, False, GREY,
                                  italic=True, font=SERIF))])
for i, (name, roll) in enumerate([("Arpit Kumar", "12340350"), ("Saurav Gupta", "12341940")]):
    x = 0.55 + i * 2.85
    rule(S, x, 3.6, 2.6)
    text(S, x, 3.67, 2.6, 0.3, [T(name, 12, True, BLACK)])
    text(S, x, 3.95, 2.6, 0.25, [T(f"Roll No. {roll}", 8.5, False, GREY)])
text(S, 0.55, 4.75, 5.8, 0.25, [T("Code, results and pre-registrations: "
                                  "github.com/Saurav1375/erdos-mmjee-reasoner", 8, False, MID)])

finish()
prs.save(str(OUT))
print(f"wrote {OUT} ({len(SLIDES)} slides)")
