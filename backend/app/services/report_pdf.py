"""The take-away report as a PDF (A4), built with ReportLab.

Same content as the HTML report, plus the student's answers to the resume questions.
render_pdf() returns bytes; the caller decides where (if anywhere) to save them.
"""

from datetime import date
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from xml.sax.saxutils import escape

from app.services.analysis import Analysis
from app.services.learning import LEVEL_NAMES
from app.services.report import NOTE, STATUS, _todo, blocker_lines

INK, MUTED, LINE, SOFT = colors.HexColor("#0F1115"), colors.HexColor("#5C6370"), colors.HexColor("#E3E6EA"), colors.HexColor("#F5F6F8")
KIND_LABEL = {"skill": "Skill", "task": "Experience", "soft": "Way of working"}
FIT_COLOUR = {"Strong fit": "#1E7A46", "Possible fit": "#2B4ACB", "Stretch": "#9A5B00", "Not yet": "#B42318"}
STATUS_COLOUR = {"below": "#9A5B00", "missing": "#B42318", "meets": "#1E7A46"}
FONT_FILES = [  # (regular, bold): the first pair found is used, so every character prints
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
]


def _fonts() -> tuple[str, str, bool]:
    """Use a full Unicode font if the computer has one; otherwise the built-in Helvetica."""
    from pathlib import Path

    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for regular, bold in FONT_FILES:
        if Path(regular).exists() and Path(bold).exists():
            pdfmetrics.registerFont(TTFont("SB", regular))
            pdfmetrics.registerFont(TTFont("SB-Bold", bold))
            addMapping("SB", 0, 0, "SB")
            addMapping("SB", 1, 0, "SB-Bold")      # so <b> inside text uses the bold file
            return "SB", "SB-Bold", True
    return "Helvetica", "Helvetica-Bold", False


FONT, BOLD, UNICODE = _fonts()

S = {
    "eyebrow": ParagraphStyle("eyebrow", fontName=BOLD, fontSize=8, textColor=MUTED, leading=10),
    "h1": ParagraphStyle("h1", fontName=BOLD, fontSize=18, leading=22, textColor=INK),
    "h2": ParagraphStyle("h2", fontName=BOLD, fontSize=11.5, leading=14, textColor=INK, spaceBefore=12, spaceAfter=5),
    "body": ParagraphStyle("body", fontName=FONT, fontSize=9.5, leading=13.5, textColor=INK),
    "small": ParagraphStyle("small", fontName=FONT, fontSize=8.5, leading=11.5, textColor=MUTED),
    "cell": ParagraphStyle("cell", fontName=FONT, fontSize=8.5, leading=11, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName=BOLD, fontSize=8.5, leading=11, textColor=INK),
    "head": ParagraphStyle("head", fontName=BOLD, fontSize=8, leading=10, textColor=MUTED),
    "right": ParagraphStyle("right", fontName=FONT, fontSize=8.5, leading=11, textColor=MUTED, alignment=TA_RIGHT),
}

# Characters we always swap (arrows render inconsistently). Without a Unicode font, anything
# outside Western European characters becomes "?" instead of an invisible box.
SWAPS = {"→": "->", "—": "-", "–": "-", "“": '"', "”": '"', "‘": "'", "’": "'", "…": "...", "•": "·"}


def t(text) -> str:
    """Safe text for a Paragraph: swap unsupported characters, then escape <, >, &."""
    text = "".join(SWAPS.get(ch, ch) for ch in str(text))
    if not UNICODE:
        text = text.encode("latin-1", "replace").decode("latin-1")
    return escape(text)


def P(text, style="body") -> Paragraph:
    return Paragraph(text, S[style])


def _table(rows, widths, header=True) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE),
             ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style.append(("LINEBELOW", (0, 0), (-1, 0), 0.8, INK))
    table.setStyle(TableStyle(style))
    return table


def _facts(a: Analysis, width: float) -> Table:
    must = sum(1 for m in a.gaps if m.must_have)
    cells = [[P(label, "small") for label in ("Requirements", "Already meet", "Gaps", "Must-have gaps")],
             [P(f"<font size=15><b>{n}</b></font>") for n in (len(a.matches), len(a.matches) - len(a.gaps), len(a.gaps), must)]]
    table = Table(cells, colWidths=[width / 4] * 4)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SOFT),
                               ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
                               ("LEFTPADDING", (0, 0), (-1, -1), 8),
                               ("LINEAFTER", (0, 0), (-2, -1), 0.5, colors.white)]))
    return table


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "SkillBridge · keeps no copy of your resume or this report")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def render_pdf(a: Analysis, on: date | None = None) -> bytes:
    on = on or date.today()
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm,
                            title=f"SkillBridge report · {a.title}", author="SkillBridge")
    width = doc.width - 12          # the page frame has 6pt padding on each side
    level = f" · {a.seniority.title()}" if a.seniority else ""
    story = []

    head = Table([[[P("SKILLBRIDGE REPORT", "eyebrow"), P(t(a.title + level), "h1"),
                    P("Your resume compared with this job description", "small")],
                   P(on.strftime("%d %B %Y"), "right")]], colWidths=[width - 40 * mm, 40 * mm])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 1, INK),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story += [head, Spacer(1, 10), _facts(a, width)]

    if a.fit:
        colour = FIT_COLOUR[a.fit.verdict]
        box = Table([[[P(f'<font color="{colour}"><b>{t(a.fit.verdict)}</b></font>', "h2"),
                       P(t(a.fit.advice)), Spacer(1, 3),
                       P(t(" ".join(a.fit.reasons)), "small")]]], colWidths=[width])
        box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.HexColor(colour)),
                                 ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9)]))
        story += [Spacer(1, 10), box]

    story += [P("Summary", "h2"), P(t(a.summary))]

    if a.focus:
        story += [P("Start here", "h2"),
                  P("<br/>".join(f"<b>{i}. {t(f)}</b>" for i, f in enumerate(a.focus, 1))),
                  Spacer(1, 3), P(t(a.focus_reason), "small")]

    if a.blockers:
        story.append(P("Likely auto-filters: check these first", "h2"))
        story.append(P("Employers' systems often filter on these before anyone reads your skills. A hint, not a verdict.", "small"))
        for label, st, q, note in blocker_lines(a):
            colour = "#B42318" if st == "Check" else "#1E7A46"
            story.append(P(f'<font color="{colour}"><b>{st}</b></font> · <b>{t(label)}</b>: "{t(q)}" {t(note)}'))
    if a.soft["shown"] or a.soft["interview"]:
        story.append(P("Ways of working the job mentions (not scored)", "h2"))
        if a.soft["shown"]:
            story.append(P("Shown on your resume: " + t(", ".join(m.skill for m in a.soft["shown"]))))
        if a.soft["interview"]:
            story.append(P("Prepare one real example for the interview: " + t(", ".join(m.skill for m in a.soft["interview"]))
                           + ". You don't need a soft-skills list on your resume."))
    story.append(P("Learn: what's missing", "h2"))
    if a.gaps:
        rows = [[P(h, "head") for h in ("Skill", "Status", "The job asks", "Your resume", "What to do")]]
        for m in a.gaps:
            status = (f'<font color="{STATUS_COLOUR[m.status]}"><b>{STATUS[m.status]}</b></font>'
                      f"<br/>{LEVEL_NAMES[m.current]} -&gt; {LEVEL_NAMES[m.needed]}"
                      + ("<br/>Must-have" if m.must_have else "<br/>A plus"))
            rows.append([P(t(m.skill) + (f'<br/><font size=7 color="#5C6370">{KIND_LABEL[m.kind]}</font>'
                                            if m.kind != "skill" else ""), "cellb"), P(status, "cell"), P(f'"{t(m.job_quote)}"', "cell"),
                         P(f'"{t(m.resume_quote)}"' if m.resume_quote else "Not shown", "cell"),
                         P(t(_todo(m, a)), "cell")])
        story.append(_table(rows, [26 * mm, 26 * mm, 40 * mm, 40 * mm, width - 132 * mm]))
    else:
        story.append(P("Your resume shows every skill this job asks for."))
    if a.show_better:
        story += [P("Show better: you have it, your resume hides it", "h2"),
                  P("Same facts, presented for this job. Nothing new is added.", "small"), Spacer(1, 3)]
        for b in a.show_better:
            line = f'<br/><font color="#5C6370">Your line: "{t(b.resume_line)}"</font>' if b.resume_line else ""
            story += [KeepTogether([P(f"<b>{t(b.skill)}</b>. {t(b.text)}{line}"), Spacer(1, 5)])]
    if a.strengths:
        story.append(P("Strengths", "h2"))
        rows = [[P(t(m.skill), "cellb"), P(KIND_LABEL.get(m.kind, ""), "cell"),
                 P(f'"{t(m.resume_quote)}"' if m.resume_quote else "", "cell")] for m in a.strengths]
        story.append(_table(rows, [40 * mm, 22 * mm, width - 62 * mm], header=False))

    story.append(P("Learning plan", "h2"))
    if a.learning:
        for i, s in enumerate(a.learning, 1):
            topics = f"<br/><font color='#5C6370'>Key topics: {t(', '.join(s.topics))}</font>" if s.topics else ""
            if s.suggested:
                topics += "<br/><font color='#5C6370' size=8>Suggested by AI (no curated guide for this yet)</font>"
            story.append(KeepTogether([P(f"<b>{i}. {t(s.text)}</b> <font color='#5C6370'>· {t(s.reason)}</font>{topics}"),
                                       Spacer(1, 4)]))
    else:
        story.append(P("Nothing to learn for this job."))

    if a.projects:
        story.append(P("Portfolio projects", "h2"))
        for p in a.projects:
            story.append(KeepTogether([P(f"<b>{t(p.title)}</b>"), P(t(p.build)),
                                       P(f"Shows {t(', '.join(p.skills))} · The job says \"{t(p.job_quote)}\"", "small"),
                                       Spacer(1, 6)]))

    story.append(P("Resume checks", "h2"))
    rows = [[P(f'<font color="{"#1E7A46" if c["ok"] else "#9A5B00"}"><b>{"Pass" if c["ok"] else "Review"}</b></font>', "cell"),
             P(t(c["area"]), "cellb"), P(t(c["text"]), "cell")] for c in a.checks]
    if rows:
        story.append(_table(rows, [18 * mm, 36 * mm, width - 54 * mm], header=False))

    if a.changes:
        story += [P("What changed after your answers", "h2"),
                  P("These levels changed because of what you told us. Your plan above already "
                    "reflects them.", "small"), Spacer(1, 4)]
        rows = [[P(h, "head") for h in ("Skill", "Level", "You said", "On your resume")]]
        rows += [[P(t(c.skill), "cellb"), P(f"{LEVEL_NAMES[c.before]} -&gt; <b>{LEVEL_NAMES[c.after]}</b>", "cell"),
                  P(f'"{t(c.said)}"', "cell"), P(t(c.where), "cell")] for c in a.changes]
        story.append(_table(rows, [28 * mm, 30 * mm, 55 * mm, width - 113 * mm]))

    asked = a.asked()
    if asked:
        story += [P("Your questions and answers", "h2"),
                  P("Only your answers can add new facts to your resume. Reword each line using only "
                    "what you wrote here.", "small"), Spacer(1, 4)]
        for q in asked:
            reply = (f"<b>Your answer:</b> {t(q['answer'])}" if q["answer"] else
                     "<font color='#5C6370'><b>Not answered.</b> Leave this as it is until you can answer it.</font>")
            parts = [P(t(q["topic"]), "small"), P(t(q["issue"]), "small"), Spacer(1, 2),
                     P(f"<b>{t(q['question'])}</b>"), Spacer(1, 3), P(reply)]
            if q["takeaway"]:
                parts += [Spacer(1, 4), P(f"<b>What this means:</b> {t(q['takeaway'])}")]
            if q["effects"]:
                parts.append(P(f"<b>Changed your plan:</b> {t('; '.join(q['effects']))}"))
            if q["rewrite"]:
                parts += [Spacer(1, 3), P("<b>Suggested line</b> (only your facts, check it before using):"),
                          P(f'<font color="#1E7A46">"{t(q["rewrite"])}"</font>')]
            box = Table([[parts]], colWidths=[width])
            box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, LINE), ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                                     ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                                     ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
            story += [KeepTogether(box), Spacer(1, 6)]

    note = NOTE
    if a.review:
        note += (" Checked by an automatic reviewer." if a.review["passed"]
                 else " An automatic reviewer flagged parts of this advice; read it critically.")
    story += [Spacer(1, 12), P(t(note), "small")]

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
