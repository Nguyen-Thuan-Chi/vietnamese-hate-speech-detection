# src/services/highlighter.py
import html
import re
from typing import Dict, List, Union

from src.config.toxic_keywords import TOXIC_KEYWORDS


# =============================================================================
# KEYWORD-BASED HIGHLIGHTER (PRIMARY – sentence-level model)
# =============================================================================

class KeywordHighlighter:
    HIGHLIGHT_STYLE = "color:orange;font-weight:bold;text-decoration:underline"

    def __init__(
        self,
        keywords: List[str] | None = None,
        style: str | None = None
    ) -> None:
        self._keywords = keywords if keywords is not None else TOXIC_KEYWORDS
        self._style = style if style is not None else self.HIGHLIGHT_STYLE
        self._pattern = self._compile_pattern()

    def _compile_pattern(self) -> re.Pattern:
        if not self._keywords:
            return re.compile(r"(?!)")

        keywords = sorted(self._keywords, key=len, reverse=True)
        escaped = [re.escape(k) for k in keywords]

        pattern = r"(?:^|(?<=\s))(" + "|".join(escaped) + r")(?:$|(?=\s))"
        return re.compile(pattern, re.IGNORECASE)

    def highlight(self, text: str) -> str:
        if not text:
            return ""

        matches = list(self._pattern.finditer(text))
        if not matches:
            return html.escape(text)

        return self._render(text, matches)

    def _render(self, text: str, matches: List[re.Match]) -> str:
        out: List[str] = []
        pos = 0

        for m in matches:
            if pos < m.start():
                out.append(html.escape(text[pos:m.start()]))

            out.append(
                f'<span style="{self._style}">{html.escape(m.group(0))}</span>'
            )
            pos = m.end()

        if pos < len(text):
            out.append(html.escape(text[pos:]))

        return "".join(out)


# =============================================================================
# SPAN-BASED HIGHLIGHTER (RESERVED FOR FUTURE TOKEN-LEVEL MODELS)
# =============================================================================

class ToxicHighlighter:
    SPAN_STYLE = "color:red;font-weight:bold"

    def __init__(self, style: str | None = None) -> None:
        self._style = style if style is not None else self.SPAN_STYLE

    def highlight(
        self,
        text: str,
        spans: List[Dict[str, Union[int, str]]]
    ) -> str:
        if not text or not spans:
            return html.escape(text)

        spans = self._merge(self._normalize(spans))
        return self._render(text, spans)

    def _normalize(
        self,
        spans: List[Dict[str, Union[int, str]]]
    ) -> List[Dict[str, int]]:
        return sorted(
            [{"start": int(s["start"]), "end": int(s["end"])} for s in spans],
            key=lambda s: s["start"]
        )

    def _merge(self, spans: List[Dict[str, int]]) -> List[Dict[str, int]]:
        merged: List[Dict[str, int]] = []
        cur = spans[0]

        for s in spans[1:]:
            if s["start"] <= cur["end"]:
                cur["end"] = max(cur["end"], s["end"])
            else:
                merged.append(cur)
                cur = s

        merged.append(cur)
        return merged

    def _render(
        self,
        text: str,
        spans: List[Dict[str, int]]
    ) -> str:
        out: List[str] = []
        pos = 0
        n = len(text)

        for s in spans:
            start = max(0, s["start"])
            end = min(n, s["end"])

            if pos < start:
                out.append(html.escape(text[pos:start]))

            out.append(
                f'<span style="{self._style}">{html.escape(text[start:end])}</span>'
            )
            pos = end

        if pos < n:
            out.append(html.escape(text[pos:]))

        return "".join(out)
