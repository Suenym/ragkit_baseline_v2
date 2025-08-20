import re
import unicodedata
from typing import Iterable, List, Tuple


class RelaxedAnswerMatch:
    """Utility to compile relaxed answer patterns and match text."""

    GAP = r"[ \t\r\n\.,;:!\?-–—\"'()]{0,3}"
    _NBSP = "\u00A0"
    _THIN = "\u2009"

    def _normalize(self, text: str) -> str:
        if text is None:
            return ""
        text = str(text)
        text = (
            text.replace("«", '"')
            .replace("»", '"')
            .replace("“", '"')
            .replace("”", '"')
        )
        text = text.replace(self._NBSP, " ").replace(self._THIN, " ")
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _is_number(self, value) -> bool:
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            s = value.replace(self._NBSP, "").replace(" ", "").replace(",", ".")
            try:
                float(s)
                return True
            except ValueError:
                return False
        return False

    def _digits_with_gap(self, digits: str) -> str:
        gap = r"[\s\u00A0\u2009]*"
        return gap.join(list(digits))

    def _compile_number_patterns(self, answer) -> List[re.Pattern]:
        s = str(answer)
        s = s.replace(self._NBSP, "").replace(" ", "").replace(",", ".")
        num = float(s)
        if num.is_integer():
            num_str = str(int(num))
        else:
            num_str = ("%f" % num).rstrip("0").rstrip(".")
        patterns: List[re.Pattern] = []
        if "." in num_str:
            int_part, frac_part = num_str.split(".")
            exact = rf"(?<!\d)(?:{int_part}[.,]{frac_part})(?!\d)"
            gap = r"[\s\u00A0\u2009]*"
            spaced = (
                rf"(?<!\d){self._digits_with_gap(int_part)}{gap}[.,]{gap}{self._digits_with_gap(frac_part)}(?!\d)"
            )
        else:
            exact = rf"(?<!\d){num_str}(?!\d)"
            spaced = rf"(?<!\d){self._digits_with_gap(num_str)}(?!\d)"
        patterns.append(re.compile(exact))
        patterns.append(re.compile(spaced))
        return patterns

    def _mojibake_variants(self, text: str) -> List[str]:
        base = self._normalize(text)
        variants = [base]
        encodings = ["latin1", "cp1252", "cp1251"]
        for enc in encodings:
            try:
                v = base.encode("utf-8").decode(enc, errors="ignore")
                v = self._normalize(v)
                if v and v not in variants:
                    variants.append(v)
            except Exception:
                pass
            try:
                v = base.encode(enc, errors="ignore").decode("utf-8", errors="ignore")
                v = self._normalize(v)
                if v and v not in variants:
                    variants.append(v)
            except Exception:
                pass
        return variants[:3]

    def _compile_text_patterns(self, answer: str) -> List[re.Pattern]:
        patterns: List[re.Pattern] = []
        for variant in self._mojibake_variants(answer):
            tokens = [re.escape(t) for t in variant.split()]
            if not tokens:
                continue
            pat = self.GAP.join(tokens)
            patterns.append(re.compile(pat, flags=re.IGNORECASE))
        return patterns

    def compile_patterns(self, answer) -> List[re.Pattern]:
        if self._is_number(answer):
            return self._compile_number_patterns(answer)
        return self._compile_text_patterns(str(answer))

    def any_match(self, text: str, patterns: Iterable[re.Pattern]) -> Tuple[bool, str]:
        norm_text = self._normalize(text)
        for pat in patterns:
            if pat.search(norm_text):
                return True, pat.pattern
        return False, ""


relaxed_answer_match = RelaxedAnswerMatch()
