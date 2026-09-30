"""Input guardrails, applied before retrieval or any LLM call.

1. Sanitisation: control characters stripped, whitespace collapsed, length capped.
2. Meta questions: anything about how the assistant is built (model, provider, prompt,
   architecture, creators) or attempts to override its instructions gets a fixed answer that
   discloses nothing. Deterministic multilingual patterns.
3. Greetings: a short introduction instead of a refusal.
4. Scope: a question must be closer to the regulatory domain than to everyday topics. The
   question embedding is compared with in-scope and out-of-scope reference phrases.

Fixed answers are returned in the user's language (FR / EN / AR).
"""

import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import structlog

from reglens.models import Language
from reglens.retrieval.embeddings import Embedder
from reglens.text import detect_language

log = structlog.get_logger(__name__)

Verdict = Literal["ok", "empty", "too_long", "meta", "greeting", "off_topic"]

MAX_QUESTION_CHARS = 1000

# --- 1. Sanitisation ---------------------------------------------------------------------


def sanitize(text: str) -> str:
    """Remove control/format characters (incl. zero-width and bidi overrides), trim spaces."""
    text = unicodedata.normalize("NFC", text).replace("’", "'").replace("‘", "'")
    text = "".join(c for c in text if unicodedata.category(c)[0] != "C" or c in "\n\t")
    return re.sub(r"\s+", " ", text).strip()


# --- 2. Meta questions and instruction overrides -------------------------------------------

_META_PATTERNS = [
    # Patterns need an AI/system context: "quel modèle de convention" or "which model
    # prospectus" are regulatory questions and must pass.
    # French
    r"\bquel(?:le)?s? (?:llm|ia\b|intelligence artificielle|mod[eè]le (?:d[' ]?ia\b|d[' ]?intelligence|de langage|utilises|utilisez|es-tu|êtes-vous|te fait|vous fait))",
    r"\b(?:tu utilises|vous utilisez|utilises-tu|utilisez-vous) (?:quel|quelle|chatgpt|gpt|openai|une ia)",
    r"\b(?:qui|quelle (?:entreprise|soci[eé]t[eé])) (?:t'a|vous a) (?:cr[eé]{2}|con[çc]u|d[eé]velopp[eé]|programm[eé]|entra[iî]n[eé])",
    r"\bcomment (?:as-tu|avez-vous|es-tu|êtes-vous|tu as|vous avez) (?:[eé]t[eé] )?(?:con[çc]u|cr[eé]{2}|construit|d[eé]velopp[eé]|entra[iî]n[eé])",
    r"\b(?:tu es|es-tu|êtes-vous|vous êtes) (?:chatgpt|gpt|claude|gemini|mistral|llama|une ia|un robot)",
    r"\b(?:ton|votre|tes|vos) (?:prompt|instructions? syst[eè]me|consignes internes|code source|architecture technique)",
    r"\b(?:ignore|oublie|oubliez|ignorez) (?:tes|vos|toutes tes|toutes vos|les) (?:instructions|consignes|r[eè]gles)",
    r"\bmode d[eé]veloppeur\b",
    # English
    r"\b(?:what|which) (?:llm|ai model|language model|ai are you|model (?:are you|do you use|powers you|is behind you)|provider do you)",
    r"\bwhat (?:embedding model|vector (?:db|database)|tech(?:nology)? stack) (?:do you|are you)",
    r"\bwho (?:made|built|created|developed|trained|designed) you\b",
    r"\bhow (?:were|are) you (?:made|built|created|developed|trained|designed)",
    r"\bare you (?:chatgpt|gpt|claude|gemini|mistral|llama|an? ai\b|a bot|openai)",
    r"\byour (?:system prompt|prompt|instructions|source code|architecture|training data)",
    r"\bthe system prompt\b",
    r"\bignore (?:all |your |the )?(?:previous |prior |above )?(?:instructions|rules|prompts?)",
    r"\b(?:developer|dan|jailbreak) mode\b",
    r"\bpretend (?:to be|you are)\b",
    # Arabic
    r"(?:نموذج|النموذج) (?:اللغوي|لغوي|الذكاء الاصطناعي|ذكاء اصطناعي)",
    r"(?:تستعمل|تستخدم)(?:ه)? (?:أي|اي) (?:نموذج|تقنية|ذكاء)",
    r"(?:ما هو|ما هي|أي) (?:الذكاء الاصطناعي|التقنية التي)",
    r"من (?:صنعك|طورك|برمجك|أنشأك|صممك|درّبك|دربك)",
    r"كيف (?:تم|تمّ) (?:صنعك|تطويرك|برمجتك|إنشاؤك|تصميمك|تدريبك)",
    r"هل (?:أنت|انت) (?:شات ?جي ?بي ?تي|chatgpt|gpt|كلود|claude|ذكاء اصطناعي|روبوت)",
    r"(?:تعليماتك|التعليمات السابقة|موجهك|برومبت)",
    r"(?:تجاهل|انس) (?:التعليمات|تعليماتك|القواعد)",
    # Vendor names in a question are about the system, never about regulation
    r"\b(?:chatgpt|openai|anthropic|claude|gemini|llama|gpt-?\d)\b",
]
_META = re.compile("|".join(f"(?:{p})" for p in _META_PATTERNS), re.IGNORECASE)

_GREETING = re.compile(
    r"^(?:bonjour|bonsoir|salut|coucou|hello|hi|hey|good (?:morning|afternoon|evening)|"
    r"merci|thanks?(?: you)?|السلام عليكم|سلام|مرحبا|مرحباً|صباح الخير|مساء الخير|شكرا|شكراً)"
    r"[\s!.,؟?]*(?:(?:à vous|à toi|everyone|all))?[\s!.,؟?]*$",
    re.IGNORECASE,
)

# --- 3. Scope -------------------------------------------------------------------------------

IN_SCOPE = [
    "réglementation bancaire et prudentielle au Maroc, circulaires de Bank Al-Maghrib",
    "ratio de solvabilité, fonds propres, ratio de liquidité, ratio de levier des banques",
    "gestion des risques, contrôle interne, gouvernance des établissements de crédit",
    "lutte contre le blanchiment de capitaux, obligation de vigilance, connaissance du client",
    "droits des clients des banques : comptes, frais, réclamations, médiation, crédit",
    "marché des capitaux, AMMC, bourse, émetteurs, prospectus, conseillers en investissement",
    "assurance, ACAPS, garantie des dépôts, fonds de garantie, conglomérats financiers",
    "banking regulation in Morocco, prudential rules, capital and liquidity requirements",
    "bank customers' rights, accounts, fees, complaints, loans, deposit guarantee",
    "capital markets regulation, securities, stock exchange, listed companies, investors",
    "anti-money laundering, customer due diligence, beneficial owners, compliance",
    "insurance supervision, financial conglomerates, crowdfunding, fintech regulation",
    "التنظيم البنكي في المغرب، دوريات بنك المغرب، الأموال الذاتية ونسبة السيولة",
    "حقوق عملاء البنوك، الحسابات، العمولات، الشكايات، القروض",
    "سوق الرساميل، الهيئة المغربية لسوق الرساميل، البورصة، الاستثمار",
    "مكافحة غسل الأموال، واجب اليقظة، التأمين، ضمان الودائع",
]
OUT_OF_SCOPE = [
    "recette de cuisine, restaurant, plat, gâteau",
    "football, match, sport, équipe, joueur",
    "météo, voyage, vacances, hôtel, tourisme",
    "programmation informatique, code Python, bug, développement web",
    "santé, médecin, maladie, symptômes, régime",
    "films, séries, musique, jeux vidéo, célébrités",
    "histoire, géographie, mathématiques, devoirs scolaires",
    "politique, élections, actualité internationale",
    "cooking recipe, food, restaurant",
    "sports, football match, team, player",
    "weather, travel, holidays, hotels",
    "programming, Python code, software bug, web development",
    "health, doctor, illness, symptoms",
    "movies, music, video games, celebrities",
    "homework, mathematics, history, general knowledge",
    "وصفة طبخ، مطعم، أكل",
    "كرة القدم، مباراة، رياضة",
    "الطقس، السفر، العطلة، الفنادق",
    "البرمجة، الحاسوب، الألعاب",
    "الصحة، الطبيب، المرض",
]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class ScopeClassifier:
    """In scope when the question is closer to the domain phrases than to everyday topics.

    ``margin`` is how much closer to an out-of-scope phrase the question must be before it is
    refused; a positive margin favours answering (a false refusal costs more than a retrieval
    that later abstains).
    """

    def __init__(self, embedder: Embedder, margin: float = 0.0) -> None:
        self._embedder = embedder
        self._margin = margin
        self._inside = embedder.embed_documents(IN_SCOPE)
        self._outside = embedder.embed_documents(OUT_OF_SCOPE)

    def scores(self, question: str) -> tuple[float, float]:
        vector = self._embedder.embed_query(question)
        inside = max(_cosine(vector, v) for v in self._inside)
        outside = max(_cosine(vector, v) for v in self._outside)
        return inside, outside

    def in_scope(self, question: str) -> bool:
        inside, outside = self.scores(question)
        return outside <= inside + self._margin


# --- 4. Decision and fixed answers ----------------------------------------------------------


@dataclass(frozen=True)
class GuardrailResult:
    verdict: Verdict
    language: Language
    question: str  # sanitised

    @property
    def allowed(self) -> bool:
        return self.verdict == "ok"


def check(question: str, scope: ScopeClassifier | None = None) -> GuardrailResult:
    clean = sanitize(question)
    language = detect_language(clean)
    if not clean:
        verdict: Verdict = "empty"
    elif len(clean) > MAX_QUESTION_CHARS:
        verdict = "too_long"
    elif _META.search(clean):
        verdict = "meta"
    elif _GREETING.match(clean):
        verdict = "greeting"
    elif scope is not None and not scope.in_scope(clean):
        verdict = "off_topic"
    else:
        verdict = "ok"
    if verdict != "ok":
        log.info("guardrail_triggered", verdict=verdict, language=language, chars=len(clean))
    return GuardrailResult(verdict, language, clean)


FIXED_ANSWERS: dict[Verdict, dict[Language, str]] = {
    "meta": {
        "fr": "Je suis RegLens, un assistant spécialisé dans la réglementation financière "
        "marocaine. Je ne communique pas d'informations sur ma conception ni sur les "
        "technologies utilisées. Je peux en revanche répondre à vos questions sur les textes "
        "de Bank Al-Maghrib et de l'AMMC.",
        "en": "I'm RegLens, an assistant specialised in Moroccan financial regulation. I don't "
        "share details about how I'm built or the technology behind me. I can answer your "
        "questions about Bank Al-Maghrib and AMMC texts.",
        "ar": "أنا RegLens، مساعد متخصص في التنظيم المالي المغربي. لا أقدم معلومات حول طريقة "
        "تصميمي أو التقنيات المستعملة. يمكنني في المقابل الإجابة عن أسئلتكم حول نصوص بنك "
        "المغرب والهيئة المغربية لسوق الرساميل.",
    },
    "off_topic": {
        "fr": "Je réponds uniquement aux questions sur la réglementation financière marocaine "
        "(Bank Al-Maghrib, AMMC). Posez-moi par exemple une question sur un ratio "
        "prudentiel, un délai de réclamation ou une obligation de vigilance.",
        "en": "I only answer questions about Moroccan financial regulation (Bank Al-Maghrib, "
        "AMMC). For example, ask me about a prudential ratio, a complaint deadline or a due "
        "diligence obligation.",
        "ar": "أجيب فقط عن الأسئلة المتعلقة بالتنظيم المالي المغربي (بنك المغرب، الهيئة المغربية "
        "لسوق الرساميل). يمكنكم مثلاً السؤال عن نسبة احترازية أو أجل معالجة شكاية أو واجب "
        "اليقظة.",
    },
    "greeting": {
        "fr": "Bonjour ! Je suis RegLens, votre assistant pour la réglementation financière "
        "marocaine. Posez-moi une question, par exemple : « Quel est le ratio de liquidité "
        "minimum des banques ? »",
        "en": "Hello! I'm RegLens, your assistant for Moroccan financial regulation. Ask me a "
        'question, for example: "What is the minimum liquidity ratio for banks?"',
        "ar": "مرحباً! أنا RegLens، مساعدكم في التنظيم المالي المغربي. اطرحوا سؤالكم، مثلاً: "
        "« ما هي النسبة الدنيا للسيولة المفروضة على البنوك؟ »",
    },
    "too_long": {
        "fr": f"Votre question dépasse {MAX_QUESTION_CHARS} caractères. Merci de la raccourcir.",
        "en": f"Your question is longer than {MAX_QUESTION_CHARS} characters. Please shorten it.",
        "ar": f"يتجاوز سؤالكم {MAX_QUESTION_CHARS} حرف. يرجى اختصاره.",
    },
    "empty": {
        "fr": "Merci de poser une question.",
        "en": "Please ask a question.",
        "ar": "يرجى طرح سؤال.",
    },
}


def fixed_answer(result: GuardrailResult) -> str:
    return FIXED_ANSWERS[result.verdict][result.language]
