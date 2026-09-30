"""Language detection for FR / AR / EN questions (pure functions, no dependency)."""

import re

from reglens.models import Language

_ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN = re.compile(r"[A-Za-zÀ-ÿ]")
_FR_HINTS = {
    "le",
    "la",
    "les",
    "des",
    "du",
    "de",
    "un",
    "une",
    "est",
    "et",
    "quel",
    "quelle",
    "quels",
    "quelles",
    "pour",
    "dans",
    "sur",
    "avec",
    "mon",
    "ma",
    "mes",
    "je",
    "vous",
    "tu",
    "qui",
    "que",
    "quoi",
    "comment",
    "combien",
    "pourquoi",
    "banque",
    "délai",
    "taux",
    "au",
    "aux",
    "ce",
    "cette",
    "il",
    "elle",
    "sont",
    "doit",
    "peut",
    "bonjour",
    "merci",
    "salut",
}
_EN_HINTS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "what",
    "which",
    "how",
    "why",
    "who",
    "when",
    "does",
    "do",
    "can",
    "of",
    "for",
    "in",
    "on",
    "with",
    "my",
    "i",
    "you",
    "your",
    "bank",
    "must",
    "should",
    "rate",
    "deadline",
    "hello",
    "hi",
    "thanks",
    "and",
    "to",
    "it",
}


def detect_language(text: str, default: Language = "fr") -> Language:
    """Arabic by script; French vs English by function words and accents."""
    arabic, latin = len(_ARABIC.findall(text)), len(_LATIN.findall(text))
    if arabic and arabic >= latin:
        return "ar"
    words = re.findall(r"[a-zà-ÿ']+", text.lower())
    fr = sum(w in _FR_HINTS for w in words) + 2 * len(re.findall(r"[éèêàùçôîâ]", text.lower()))
    en = sum(w in _EN_HINTS for w in words)
    if fr == en == 0:
        return default
    return "fr" if fr >= en else "en"
