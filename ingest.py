"""
Ingestion pipeline shared by index.py (main.py index) and server.py (add_document).

This first part parses an EUR-Lex HTML file into structural *units*: one
recital, one article paragraph (or one definition point), or one annex point.
Every unit keeps its source and location. Chunking designs, embedding and
storage are built on top of these units.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Tag

# Known acts: official number -> (short name, CELEX). Unknown acts fall back to the title.
KNOWN_ACTS = {
    "2016/679": ("GDPR", "32016R0679"),
    "2024/1689": ("AI Act", "32024R1689"),
    "2023/2854": ("Data Act", "32023R2854"),
    "2022/2555": ("NIS2", "32022L2555"),
}

# Units without paragraph structure longer than this are split into their points.
SPLIT_POINTS_ABOVE_WORDS = 300

PARA_ID = re.compile(r"^\d{3}\.\d{3}$")
ARTICLE_ID = re.compile(r"^art_(\d+[a-z]?)$")
ANNEX_ID = re.compile(r"^anx_([IVXLC]+)$")
RECITAL_ID = re.compile(r"^rct_(\d+)$")
LEADING_LABEL = re.compile(r"^\(?([0-9]+[a-z]?|[a-z]{1,4})[\).]\s*")


@dataclass
class Unit:
    doc: str            # short name, e.g. "GDPR"
    celex: str
    kind: str           # recital | article | annex
    ref: str            # article number, recital number or annex roman numeral
    title: str          # article or annex title ("" for recitals)
    chapter: str        # e.g. "CHAPTER III Rights of the data subject" ("" if none)
    para: str           # paragraph number or point label ("" if the whole unit)
    text: str
    order: int = 0      # position in the document, for windowing and sorting
    words: int = field(init=False)

    def __post_init__(self) -> None:
        self.words = len(self.text.split())

    @property
    def location(self) -> str:
        """Human-readable citation, e.g. 'Article 6(1)', 'Article 4, point (11)', 'Recital 32'."""
        if self.kind == "recital":
            return f"Recital {self.ref}"
        base = f"Article {self.ref}" if self.kind == "article" else f"Annex {self.ref}"
        if not self.para:
            return base
        if self.para.isdigit() and self.kind == "article":
            return f"{base}({self.para})"
        return f"{base}, point {self.para}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["location"] = self.location
        return d


# ---------------------------------------------------------------- cleaning

def clean(text: str) -> str:
    """Normalise Unicode (non-breaking and em spaces from EUR-Lex) and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:)])", r"\1", text)   # 'Union .' -> 'Union.' (left by inline tags)
    return re.sub(r"\(\s+", "(", text).strip()


def element_lines(node: Tag) -> list[str]:
    """Text of a block as lines: one per paragraph, one per list row ('(a) text')."""
    lines = []
    for child in node.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "table":
            for row in child.find("tbody", recursive=False).find_all("tr", recursive=False) if child.find("tbody", recursive=False) else child.find_all("tr", recursive=False):
                cells = [clean(td.get_text(" ")) for td in row.find_all("td", recursive=False)]
                line = " ".join(c for c in cells if c)
                if line:
                    lines.append(line)
        elif child.name == "p":
            line = clean(child.get_text(" "))
            if line:
                lines.append(line)
        elif child.name == "div":
            lines.extend(element_lines(child))
    return lines


def strip_noise(soup: BeautifulSoup) -> None:
    """Remove footnotes and footnote markers, which are not part of the legal text."""
    for note in soup.select("p.oj-note, hr.oj-note"):
        note.decompose()
    for marker in soup.find_all("a", id=re.compile(r"^ntr")):
        marker.decompose()
    for marker in soup.select("span.oj-note-tag"):
        marker.decompose()


# ---------------------------------------------------------------- document identity

def identify(soup: BeautifulSoup, path: Path) -> tuple[str, str, str]:
    """Return (short name, CELEX, full title) of the act."""
    title_div = soup.find(id="tit_1")
    title = clean(title_div.get_text(" ")) if title_div else path.stem
    match = re.search(r"\(EU\)\s*(?:No\s*)?(\d{4}/\d+)", title)
    if match and match.group(1) in KNOWN_ACTS:
        short, celex = KNOWN_ACTS[match.group(1)]
        return short, celex, title
    paren = [p for p in re.findall(r"\(([^()]+)\)", title) if "EEA" not in p and not p.startswith("EU")]
    short = paren[-1] if paren else (match.group(1) if match else path.stem)
    return short, (match.group(1) if match else ""), title


# ---------------------------------------------------------------- structure

def chapter_of(node: Tag) -> str:
    """Title of the chapter that contains this node, e.g. 'CHAPTER III Rights of the data subject'."""
    for parent in node.parents:
        pid = parent.get("id", "") if isinstance(parent, Tag) else ""
        if re.fullmatch(r"cpt_[IVXLC]+", pid):
            heading = parent.find(id=f"{pid}.tit_1")
            label = parent.find("p", class_="oj-ti-section-1")
            parts = [clean(label.get_text(" ")) if label else "", clean(heading.get_text(" ")) if heading else ""]
            return " ".join(p for p in parts if p)
    return ""


def split_points(lines: list[str]) -> list[tuple[str, str]]:
    """Split the lines of an unstructured block into (point label, text) pairs.

    A line that starts with a label like '(11)' or '1.' opens a new point; lines
    without a label belong to the previous point. Text before the first label
    becomes an unlabelled intro point.
    """
    points: list[tuple[str, list[str]]] = []
    for line in lines:
        m = LEADING_LABEL.match(line)
        top_level = m and (m.group(1).isdigit() or not points)
        if top_level:
            points.append((f"({m.group(1)})" if m.group(1).isdigit() else m.group(1), [line]))
        elif points:
            points[-1][1].append(line)
        else:
            points.append(("", [line]))
    return [(label, " ".join(text)) for label, text in points]


def parse_recitals(soup: BeautifulSoup, doc: str, celex: str) -> list[Unit]:
    units = []
    for div in soup.find_all(id=RECITAL_ID):
        num = RECITAL_ID.match(div["id"]).group(1)
        text = " ".join(element_lines(div))
        text = re.sub(rf"^\({num}\)\s*", "", text)
        units.append(Unit(doc, celex, "recital", num, "", "", "", text))
    return units


def parse_articles(soup: BeautifulSoup, doc: str, celex: str) -> list[Unit]:
    units = []
    for div in soup.find_all(id=ARTICLE_ID):
        num = ARTICLE_ID.match(div["id"]).group(1)
        sub = div.find("p", class_="oj-sti-art")
        title = clean(sub.get_text(" ")) if sub else ""
        chapter = chapter_of(div)
        paras = [p for p in div.find_all("div", id=PARA_ID, recursive=False)]
        if paras:
            for p in paras:
                pnum = str(int(p["id"].split(".")[1]))
                units.append(Unit(doc, celex, "article", num, title, chapter, pnum, " ".join(element_lines(p))))
            continue
        # No numbered paragraphs: the body is everything except the two headings.
        body = [c for c in div.children if isinstance(c, Tag) and "oj-ti-art" not in (c.get("class") or []) and c.get("id") != f"{div['id']}.tit_1"]
        lines = []
        for c in body:
            lines.extend(element_lines(c) if c.name == "div" else element_lines(_wrap(c)))
        text = " ".join(lines)
        if len(text.split()) > SPLIT_POINTS_ABOVE_WORDS:
            for label, ptext in split_points(lines):
                units.append(Unit(doc, celex, "article", num, title, chapter, label, ptext))
        else:
            units.append(Unit(doc, celex, "article", num, title, chapter, "", text))
    return units


def parse_annexes(soup: BeautifulSoup, doc: str, celex: str) -> list[Unit]:
    units = []
    for div in soup.find_all(id=ANNEX_ID):
        roman = ANNEX_ID.match(div["id"]).group(1)
        heads = div.find_all("p", class_="oj-doc-ti", limit=2)
        title = clean(heads[1].get_text(" ")) if len(heads) > 1 else ""
        lines = []
        for c in div.children:
            if isinstance(c, Tag) and "oj-doc-ti" not in (c.get("class") or []):
                lines.extend(element_lines(c) if c.name == "div" else element_lines(_wrap(c)))
        text = " ".join(lines)
        if len(text.split()) > SPLIT_POINTS_ABOVE_WORDS:
            for label, ptext in split_points(lines):
                units.append(Unit(doc, celex, "annex", roman, title, "", label, ptext))
        else:
            units.append(Unit(doc, celex, "annex", roman, title, "", "", text))
    return units


def _wrap(tag: Tag) -> Tag:
    """Wrap a single element in a div so element_lines can walk it."""
    holder = BeautifulSoup("<div></div>", "html.parser").div
    holder.append(tag.__copy__())
    return holder


def parse_file(path: str | Path) -> tuple[dict, list[Unit]]:
    """Parse one EUR-Lex HTML file into (document metadata, ordered units)."""
    path = Path(path)
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    strip_noise(soup)
    doc, celex, title = identify(soup, path)
    units = parse_recitals(soup, doc, celex) + parse_articles(soup, doc, celex) + parse_annexes(soup, doc, celex)
    for i, u in enumerate(units):
        u.order = i
    meta = {"doc": doc, "celex": celex, "title": title, "file": path.name}
    return meta, units
