"""
HTML -> readable article text, with no third-party dependency.

News pages are mostly navigation, scripts, cookie banners and footers. This
keeps the text of the main content: the <article> element if the page has
one, otherwise <main>, otherwise <body>, minus non-content elements.
"""
from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from typing import List, Optional

# Elements whose text is never article content.
_SKIP = {"script", "style", "noscript", "template", "svg", "nav", "header", "footer",
         "aside", "form", "button", "iframe", "figure", "figcaption"}
# Elements that end a paragraph/line of text.
_BLOCK = {"p", "div", "section", "article", "main", "li", "ul", "ol", "br", "tr",
          "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "table"}
_CONTAINERS = ("article", "main", "body")
_VOID = {"br", "img", "hr", "meta", "link", "input", "source", "wbr", "area", "base", "col", "embed"}


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.depth = {c: 0 for c in _CONTAINERS}
        self.parts = {c: [] for c in _CONTAINERS}

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in _VOID:
            if tag == "br":
                self._emit("\n")
            return
        if tag in _SKIP:
            self.skip_depth += 1
        if tag in self.depth:
            self.depth[tag] += 1
        if tag in _BLOCK:
            self._emit("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return
        if tag in _BLOCK:
            self._emit("\n")
        if tag in self.depth and self.depth[tag] > 0:
            self.depth[tag] -= 1
        if tag in _SKIP and self.skip_depth > 0:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0:
            self._emit(data)

    def _emit(self, text: str) -> None:
        for container in _CONTAINERS:
            if self.depth[container] > 0:
                self.parts[container].append(text)


def _tidy(raw: str) -> str:
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in unescape(raw).splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def html_to_text(html: str, *, min_chars: int = 200) -> str:
    """
    Return the readable text of the page's main content. Prefers <article>,
    then <main>, then <body>; a container is only used if it holds at least
    `min_chars` of text (some pages wrap a teaser in <article>).
    """
    parser = _Extractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed HTML: use what was parsed
        pass
    best: Optional[str] = None
    for container in _CONTAINERS:
        text = _tidy("".join(parser.parts[container]))
        if len(text) >= min_chars:
            return text
        if text and best is None:
            best = text
    return best or ""


def strip_tags(fragment: str) -> str:
    """Plain text of a small HTML fragment (e.g. an RSS summary)."""
    parts: List[str] = []

    class _P(HTMLParser):
        def handle_data(self, data: str) -> None:
            parts.append(data)

    p = _P(convert_charrefs=True)
    p.feed(fragment)
    p.close()
    return _tidy(" ".join(parts))
