"""RegLens Maroc: a small chat interface over the answering pipeline.

Run from the project folder:  .venv/bin/streamlit run ui/app.py
"""

import html

import streamlit as st

from reglens.rag import Answer, RagPipeline
from reglens.service import EmptyIndexError, build_pipeline
from reglens.text import detect_language

st.set_page_config(page_title="RegLens Maroc", page_icon="⚖️", layout="centered")

st.markdown(
    """
<style>
.badge { display:inline-block; font-size:0.78rem; font-weight:600; padding:2px 8px;
         border-radius:4px; margin-bottom:6px; letter-spacing:0.02em; }
.badge-ok    { background:#e3f3ea; color:#1e6b43; }
.badge-none  { background:#fbf0d9; color:#7a5200; }
.badge-scope { background:#e7eaf7; color:#2f3f8f; }
.source-ref  { font-weight:600; }
.source-meta { color:#6b7280; font-size:0.85rem; }
.rtl { direction:rtl; text-align:right; }
.quote { border-left:3px solid #2f3f8f; padding:4px 10px; margin:6px 0 2px;
         font-family: Georgia, serif; font-size:0.92rem; color:#374151; }
</style>
""",
    unsafe_allow_html=True,
)

EXAMPLES = [
    "Quel est le ratio de liquidité minimum des banques ?",
    "Ma banque veut fermer mon compte : quel préavis doit-elle respecter ?",
    "What is the capital conservation buffer?",
    "ما هو الأجل الذي يجب أن يرد فيه البنك على شكاية العميل؟",
    "Quel est le taux directeur de Bank Al-Maghrib ?",
]


@st.cache_resource(show_spinner="Chargement de RegLens…")
def pipeline() -> RagPipeline:
    return build_pipeline(warm=True)


def badge(answer: Answer) -> str:
    if answer.guardrail != "ok":
        return '<span class="badge badge-scope">Hors périmètre</span>'
    if answer.abstention != "none":
        return '<span class="badge badge-none">Non trouvé dans les textes</span>'
    n = len(answer.citations)
    return f'<span class="badge badge-ok">Sourcé · {n} citation{"s" if n > 1 else ""}</span>'


def render(answer: Answer, show_all: bool) -> None:
    rtl = detect_language(answer.question) == "ar"
    st.markdown(badge(answer), unsafe_allow_html=True)
    text = answer.text or "_Aucun modèle de langage configuré : voici les passages trouvés._"
    if rtl:
        st.markdown(f'<div class="rtl">{html.escape(text)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(text)

    if answer.guardrail != "ok" or answer.abstention != "none":
        return
    cited = {c.number for c in answer.citations}
    sources = [
        (n, s)
        for n, s in enumerate(answer.sources, start=1)
        if show_all or n in cited or answer.text is None
    ]
    if sources:
        label = "Sources citées" if not show_all else "Passages retrouvés"
        with st.expander(f"{label} ({len(sources)})", expanded=False):
            for n, scored in sources:
                c = scored.chunk
                pages = f"p. {c.page_start}" + (
                    f"–{c.page_end}" if c.page_end != c.page_start else ""
                )
                where = " · ".join(x for x in (c.section, pages) if x)
                st.markdown(
                    f'<div><span class="source-ref">[{n}] {html.escape(c.issuer)} '
                    f"{html.escape(c.reference or '')}</span> "
                    f'<span class="source-meta">· {html.escape(where)}</span></div>'
                    f'<div class="source-meta">{html.escape(c.title)}</div>'
                    f'<div class="quote">{html.escape(c.text[:420].strip())}…</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"[Ouvrir le texte officiel]({c.url})")
    seconds = (answer.retrieval_ms + (answer.generation_ms or 0.0)) / 1000
    st.caption(f"{seconds:.1f} s")


with st.sidebar:
    st.markdown("### RegLens Maroc")
    st.caption(
        "Réponses sourcées sur la réglementation de Bank Al-Maghrib et de l'AMMC "
        "(47 textes). Français, English, العربية."
    )
    show_all = st.toggle("Afficher tous les passages retrouvés", value=False)
    st.markdown("**Exemples**")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending = example
    if st.button("Effacer la conversation", type="secondary"):
        st.session_state.history = []
    st.caption(
        "Les réponses citent les textes officiels. Vérifiez toujours le texte source avant "
        "toute décision."
    )

st.title("RegLens Maroc")
st.caption("Assistant sur la réglementation financière marocaine")

if "history" not in st.session_state:
    st.session_state.history = []

try:
    rag = pipeline()
except EmptyIndexError as error:
    st.error(f"{error}")
    st.stop()

for question, answer in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        render(answer, show_all)

question = st.chat_input("Posez votre question (FR, EN, AR)…") or st.session_state.pop(
    "pending", None
)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.caption("Recherche dans les textes…")
        streamed = ""
        answer: Answer | None = None
        for piece in rag.stream(question):  # words appear as the model writes them
            if isinstance(piece, Answer):
                answer = piece
            else:
                streamed += piece
                placeholder.markdown(streamed + " ▌")
        placeholder.empty()
        if answer is not None:
            render(answer, show_all)
    if answer is not None:
        st.session_state.history.append((question, answer))
