/* RegLens Maroc web page: streams answers from POST /v1/query/stream (same origin). */
"use strict";

const STRINGS = {
  fr: {
    skip: "Aller à la question", newChat: "Nouvelle conversation", language: "Langue de l'interface",
    eyebrow: "Réglementation financière marocaine",
    title: "Posez une question sur les textes officiels.",
    lede: "Chaque réponse cite la circulaire, l'article et la page du texte de Bank Al-Maghrib ou de l'AMMC. Quand les textes ne répondent pas, RegLens le dit.",
    examples: "Exemples de questions", placeholder: "Posez votre question en français, en anglais ou en arabe…",
    send: "Envoyer", stop: "Arrêter", close: "Fermer",
    hint: "Entrée pour envoyer · Maj+Entrée pour une nouvelle ligne",
    disclaimer: "RegLens aide à lire les textes officiels ; il ne remplace pas un avis juridique.",
    documents: (n) => `<strong>${n}</strong> textes officiels`,
    years: (a, b) => `Publiés de <strong>${a}</strong> à <strong>${b}</strong>`,
    langs: "Français · English · العربية",
    searching: "Recherche dans les textes…", writing: "Rédaction de la réponse…",
    answered: (n) => `Réponse sourcée · ${n} source${n > 1 ? "s" : ""}`,
    passages: "Passages retrouvés", notFound: "Non trouvé dans les textes", outOfScope: "Hors périmètre",
    failed: "Réponse indisponible", stopped: "Réponse interrompue",
    understood: "Question comprise :", sources: "Sources citées",
    others: (n) => `Autres passages retrouvés (${n})`, hideOthers: "Masquer les autres passages",
    source: (n) => `Source ${n}`, openPdf: (p) => `Ouvrir le texte officiel, p. ${p}`,
    bamNote: (p) => `Le site de Bank Al-Maghrib télécharge le PDF : allez ensuite à la page ${p}.`,
    pages: (a, b) => (a === b ? `p. ${a}` : `p. ${a}–${b}`),
    copy: "Copier", copied: "Copié", helpful: "Réponse utile", notHelpful: "Réponse à améliorer",
    thanks: "Merci", retry: "Réessayer", noLlm: "Aucun modèle de langage configuré : voici les passages retrouvés.",
    errUnavailable: "Le service de réponse est momentanément indisponible. Réessayez dans quelques minutes.",
    errRate: (s) => `Trop de questions d'affilée. Réessayez dans ${s} s.`,
    errNetwork: "Connexion perdue. Vérifiez votre réseau puis réessayez.",
    errAccess: "L'accès à ce service est restreint.", errTooLong: "Question trop longue (1 000 caractères au maximum).",
    seconds: (s) => `${s.toFixed(1).replace(".", ",")} s`,
  },
  en: {
    skip: "Skip to the question", newChat: "New conversation", language: "Interface language",
    eyebrow: "Moroccan financial regulation",
    title: "Ask a question about the official texts.",
    lede: "Every answer cites the circular, article and page of the Bank Al-Maghrib or AMMC text. When the texts do not answer, RegLens says so.",
    examples: "Example questions", placeholder: "Ask your question in English, French or Arabic…",
    send: "Send", stop: "Stop", close: "Close",
    hint: "Enter to send · Shift+Enter for a new line",
    disclaimer: "RegLens helps you read the official texts; it is not legal advice.",
    documents: (n) => `<strong>${n}</strong> official texts`,
    years: (a, b) => `Published <strong>${a}</strong>–<strong>${b}</strong>`,
    langs: "English · Français · العربية",
    searching: "Searching the texts…", writing: "Writing the answer…",
    answered: (n) => `Sourced answer · ${n} source${n > 1 ? "s" : ""}`,
    passages: "Passages found", notFound: "Not found in the texts", outOfScope: "Out of scope",
    failed: "Answer unavailable", stopped: "Answer stopped",
    understood: "Understood as:", sources: "Cited sources",
    others: (n) => `Other passages found (${n})`, hideOthers: "Hide the other passages",
    source: (n) => `Source ${n}`, openPdf: (p) => `Open the official text, p. ${p}`,
    bamNote: (p) => `Bank Al-Maghrib's site downloads the PDF: then go to page ${p}.`,
    pages: (a, b) => (a === b ? `p. ${a}` : `pp. ${a}–${b}`),
    copy: "Copy", copied: "Copied", helpful: "Helpful answer", notHelpful: "Answer needs work",
    thanks: "Thanks", retry: "Try again", noLlm: "No language model configured: here are the passages found.",
    errUnavailable: "The answering service is temporarily unavailable. Try again in a few minutes.",
    errRate: (s) => `Too many questions in a row. Try again in ${s} s.`,
    errNetwork: "Connection lost. Check your network and try again.",
    errAccess: "Access to this service is restricted.", errTooLong: "Question too long (1,000 characters at most).",
    seconds: (s) => `${s.toFixed(1)} s`,
  },
  ar: {
    skip: "الانتقال إلى السؤال", newChat: "محادثة جديدة", language: "لغة الواجهة",
    eyebrow: "التنظيم المالي المغربي",
    title: "اطرح سؤالك حول النصوص الرسمية.",
    lede: "كل جواب يحيل إلى الدورية والمادة والصفحة من نص بنك المغرب أو الهيئة المغربية لسوق الرساميل. وعندما لا تتضمن النصوص الجواب، يصرّح RegLens بذلك.",
    examples: "أمثلة على الأسئلة", placeholder: "اطرح سؤالك بالعربية أو الفرنسية أو الإنجليزية…",
    send: "إرسال", stop: "إيقاف", close: "إغلاق",
    hint: "Enter للإرسال · Shift+Enter لسطر جديد",
    disclaimer: "يساعد RegLens على قراءة النصوص الرسمية، ولا يغني عن الاستشارة القانونية.",
    documents: (n) => `<strong>${n}</strong> نصاً رسمياً`,
    years: (a, b) => `صادرة بين <strong>${a}</strong> و<strong>${b}</strong>`,
    langs: "العربية · Français · English",
    searching: "جارٍ البحث في النصوص…", writing: "جارٍ تحرير الجواب…",
    answered: (n) => `جواب موثّق · ${n} ${n > 2 && n < 11 ? "مصادر" : "مصدر"}`,
    passages: "مقاطع مسترجعة", notFound: "غير موجود في النصوص", outOfScope: "خارج النطاق",
    failed: "الجواب غير متاح", stopped: "توقف الجواب",
    understood: "السؤال كما فُهم:", sources: "المصادر المستشهد بها",
    others: (n) => `مقاطع أخرى مسترجعة (${n})`, hideOthers: "إخفاء المقاطع الأخرى",
    source: (n) => `المصدر ${n}`, openPdf: (p) => `فتح النص الرسمي، ص. ${p}`,
    bamNote: (p) => `موقع بنك المغرب يحمّل الملف: انتقل بعد ذلك إلى الصفحة ${p}.`,
    pages: (a, b) => (a === b ? `ص. ${a}` : `ص. ${a}–${b}`),
    copy: "نسخ", copied: "تم النسخ", helpful: "جواب مفيد", notHelpful: "جواب يحتاج إلى تحسين",
    thanks: "شكراً", retry: "إعادة المحاولة", noLlm: "لا يوجد نموذج لغوي مُعدّ: إليك المقاطع المسترجعة.",
    errUnavailable: "خدمة الإجابة غير متاحة مؤقتاً. أعد المحاولة بعد دقائق.",
    errRate: (s) => `عدد كبير من الأسئلة المتتالية. أعد المحاولة بعد ${s} ثانية.`,
    errNetwork: "انقطع الاتصال. تحقق من الشبكة ثم أعد المحاولة.",
    errAccess: "الولوج إلى هذه الخدمة مقيّد.", errTooLong: "السؤال طويل جداً (1000 حرف كحد أقصى).",
    seconds: (s) => `${s.toFixed(1)} ث`,
  },
};

const EXAMPLES = {
  fr: [
    ["Prudentiel", "Quel est le ratio de liquidité minimum des banques ?"],
    ["Clientèle", "Ma banque veut fermer mon compte : quel préavis doit-elle respecter ?"],
    ["Marché des capitaux", "Quel est le délai de règlement-livraison en bourse ?"],
    ["Prudentiel", "Quel est le niveau minimum du ratio de levier imposé aux banques ?"],
  ],
  en: [
    ["Prudential", "What is the capital conservation buffer?"],
    ["Customers", "How long does a bank have to answer a customer complaint?"],
    ["Capital markets", "How long does the AMMC have to handle a complaint?"],
    ["Governance", "How often must the audit committee of a credit institution meet?"],
  ],
  ar: [
    ["العملاء", "ما هو الأجل الذي يجب أن يرد فيه البنك على شكاية العميل؟"],
    ["الحكامة", "من يجب أن يرأس لجنة التدقيق في مؤسسة الائتمان؟"],
    ["الحكامة", "بأي جهاز يجب أن ترتبط وظيفة الامتثال داخل مؤسسة الائتمان؟"],
    ["العملاء", "بعد أي مدة من عدم تشغيل الحساب المدين يجب على البنك إغلاقه؟"],
  ],
};
const ISSUERS = {
  fr: { BAM: "Bank Al-Maghrib", AMMC: "AMMC", ACAPS: "ACAPS" },
  en: { BAM: "Bank Al-Maghrib", AMMC: "AMMC", ACAPS: "ACAPS" },
  ar: { BAM: "بنك المغرب", AMMC: "الهيئة المغربية لسوق الرساميل", ACAPS: "هيئة مراقبة التأمينات" },
};
const MAX_CHARS = 1000;
const HISTORY_TURNS = 3;
const STORE_KEY = "reglens.chat.v1";
const LANG_KEY = "reglens.lang";
const ARABIC = /[؀-ۿ]/;

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};
const escapeHtml = (s) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const icon = (path) =>
  `<svg viewBox="0 0 24 24" aria-hidden="true">${path}</svg>`;
const ICONS = {
  copy: icon('<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h8"/>'),
  check: icon('<path d="M5 12.5l4.5 4.5L19 7.5"/>'),
  up: icon('<path d="M7 11v9H4v-9zM7 11l4-7a2 2 0 0 1 3 1.7V9h5a2 2 0 0 1 2 2.3l-1.2 7A2 2 0 0 1 17.8 20H7"/>'),
  down: icon('<path d="M17 13V4h3v9zM17 13l-4 7a2 2 0 0 1-3-1.7V15H5a2 2 0 0 1-2-2.3l1.2-7A2 2 0 0 1 6.2 4H17"/>'),
  open: icon('<path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>'),
  retry: icon('<path d="M4 12a8 8 0 1 0 2.3-5.7M4 4v4h4"/>'),
};

const state = {
  lang: "fr",
  turns: [], // {id, question, status, text, answer, error, ms, rating, showOthers}
  busy: false,
  controller: null,
  active: null, // {turn, n} shown in the source panel
};

/* ---------- storage (per-viewer convenience; the page works without it) ---------- */
function load(key, fallback) {
  try {
    const raw = key === STORE_KEY ? sessionStorage.getItem(key) : localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}
function save() {
  try {
    const done = state.turns.filter((t) => t.status !== "streaming");
    sessionStorage.setItem(STORE_KEY, JSON.stringify(done.slice(-20)));
  } catch { /* storage unavailable: the conversation lives in memory only */ }
}

/* ---------- language ---------- */
const t = (key, ...args) => {
  const value = STRINGS[state.lang][key];
  return typeof value === "function" ? value(...args) : value;
};
function setLang(lang) {
  state.lang = STRINGS[lang] ? lang : "fr";
  try { localStorage.setItem(LANG_KEY, JSON.stringify(state.lang)); } catch { /* ignore */ }
  document.documentElement.lang = state.lang;
  document.documentElement.dir = state.lang === "ar" ? "rtl" : "ltr";
  document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
    node.placeholder = t(node.dataset.i18nPlaceholder);
  });
  document.querySelectorAll("[data-i18n-label]").forEach((node) => {
    node.setAttribute("aria-label", t(node.dataset.i18nLabel));
    node.title = t(node.dataset.i18nLabel);
  });
  document.querySelectorAll("[data-lang]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.lang === state.lang));
  });
  renderExamples();
  renderFacts();
  renderThread();
  if (state.active) openSource(state.active.turn, state.active.n, false);
  updateComposer();
}

/* ---------- welcome ---------- */
let corpus = null;
function renderFacts() {
  const list = $("facts");
  list.innerHTML = "";
  const items = [];
  if (corpus) {
    items.push(t("documents", corpus.documents));
    items.push(corpus.issuers.map((i) => escapeHtml(ISSUERS[state.lang][i] || i)).join(" · "));
    if (corpus.first_year && corpus.last_year) items.push(t("years", corpus.first_year, corpus.last_year));
  }
  items.push(escapeHtml(t("langs")));
  for (const html of items) {
    const li = el("li");
    li.innerHTML = html; // built from numbers and escaped names only
    list.append(li);
  }
}
function renderExamples() {
  const box = $("examples");
  box.innerHTML = "";
  for (const [tag, question] of EXAMPLES[state.lang]) {
    const button = el("button", "example");
    button.type = "button";
    button.append(el("span", "example-tag", tag), el("span", "example-text", question));
    button.addEventListener("click", () => ask(question));
    box.append(button);
  }
}

/* ---------- rendering answers ---------- */
function inline(text, turn, sourceCount) {
  let html = escapeHtml(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  return html.replace(/\[(\d{1,2})\]/g, (match, digits) => {
    const n = Number(digits);
    if (sourceCount !== null && (n < 1 || n > sourceCount)) return "";
    return `<button type="button" class="cite" data-turn="${turn.id}" data-n="${n}" aria-label="${escapeHtml(t("source", n))}">${n}</button>`;
  });
}
function renderMarkdown(text, turn, sourceCount) {
  const out = [];
  let list = null;
  let para = [];
  const flush = () => {
    if (para.length) out.push(`<p dir="auto">${para.join("<br>")}</p>`);
    para = [];
  };
  const closeList = () => {
    if (list) out.push(`</${list}>`);
    list = null;
  };
  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      flush();
      const kind = bullet ? "ul" : "ol";
      if (list !== kind) { closeList(); out.push(`<${kind} dir="auto">`); list = kind; }
      out.push(`<li>${inline((bullet || numbered)[1], turn, sourceCount)}</li>`);
    } else if (!line.trim()) {
      flush(); closeList();
    } else {
      closeList();
      para.push(inline(line, turn, sourceCount));
    }
  }
  flush(); closeList();
  return out.join("");
}

function statusOf(turn) {
  if (turn.status === "streaming") return ["wait", turn.text ? t("writing") : t("searching")];
  if (turn.status === "error") return ["error", t("failed")];
  if (turn.status === "stopped") return ["none", t("stopped")];
  const a = turn.answer;
  if (a.status === "out_of_scope") return ["scope", t("outOfScope")];
  if (a.status === "not_found") return ["none", t("notFound")];
  if (a.status === "sources_only") return ["ok", t("passages")];
  return ["ok", t("answered", a.citations.length)];
}

/* "Article 8 · p. 2": each part isolated so French section names and Arabic page labels
   keep their own reading order. */
function sourceLocation(source, cls) {
  const box = el("span", cls);
  const parts = [source.section, t("pages", source.page_start, source.page_end)].filter(Boolean);
  parts.forEach((part, index) => {
    if (index) box.append(" · ");
    box.append(el("bdi", "", part));
  });
  return box;
}

function sourceRow(turn, number, source) {
  const row = el("button", "src");
  row.type = "button";
  row.dataset.turn = turn.id;
  row.dataset.n = number;
  const main = el("span", "src-main");
  const ref = el("span", "src-ref");
  ref.append(el("span", `issuer ${source.issuer}`, source.issuer), el("code", "", source.reference || source.title));
  main.append(ref, el("span", "src-title", source.title));
  row.append(el("span", "src-num", String(number)), main, sourceLocation(source, "src-where"));
  return row;
}

function renderTurn(turn) {
  const li = el("li");
  li.className = "turn";
  li.dataset.turn = turn.id;
  const q = el("div", "turn-q", turn.question);
  q.dir = "auto";
  const card = el("article", "turn-a");
  li.append(q, card);

  const [kind, label] = statusOf(turn);
  const meta = el("div", "meta");
  meta.append(el("span", `status ${kind}`, label));
  card.append(meta);

  const a = turn.answer;
  if (a && a.standalone_question) {
    const understood = el("p", "understood");
    understood.append(`${t("understood")} `, el("q", "", a.standalone_question));
    card.append(understood);
  }

  const body = el("div", "answer");
  body.dir = "auto";
  if (turn.status === "streaming") {
    body.classList.add("streaming");
    body.innerHTML = renderMarkdown(turn.text, turn, null);
  } else if (a) {
    body.innerHTML = renderMarkdown(a.answer ?? t("noLlm"), turn, a.sources.length);
  } else if (turn.text) {
    body.innerHTML = renderMarkdown(turn.text, turn, null);
  }
  if (turn.status === "error") {
    body.append(el("p", "", turn.error));
  }
  card.append(body);

  if (turn.status === "error") {
    const retry = el("button", "btn-ghost retry");
    retry.type = "button";
    retry.innerHTML = `${ICONS.retry}<span>${escapeHtml(t("retry"))}</span>`;
    retry.addEventListener("click", () => {
      state.turns = state.turns.filter((x) => x !== turn);
      renderThread();
      ask(turn.question);
    });
    card.append(retry);
  }

  if (a && (a.status === "answered" || a.status === "sources_only")) {
    const numbered = a.sources.map((s) => [s.number, s]);
    const shown = numbered.filter(([, s]) => s.cited || a.status === "sources_only");
    const others = numbered.filter(([, s]) => !(s.cited || a.status === "sources_only"));
    if (shown.length || others.length) {
      const box = el("div", "sources");
      if (shown.length) {
        box.append(el("h3", "sources-title", t("sources")));
        for (const [n, s] of shown) box.append(sourceRow(turn, n, s));
      }
      if (others.length) {
        const toggle = el("button", "more", turn.showOthers ? t("hideOthers") : t("others", others.length));
        toggle.type = "button";
        toggle.setAttribute("aria-expanded", String(Boolean(turn.showOthers)));
        toggle.addEventListener("click", () => { turn.showOthers = !turn.showOthers; replaceTurn(turn); });
        box.append(toggle);
        if (turn.showOthers) {
          const list = el("div", "more-list");
          for (const [n, s] of others) list.append(sourceRow(turn, n, s));
          box.append(list);
        }
      }
      card.append(box);
    }
  }

  if (a && turn.status === "done") card.append(actions(turn));
  return li;
}

function actions(turn) {
  const bar = el("div", "actions");
  const copy = el("button", "icon-btn");
  copy.type = "button";
  copy.innerHTML = `${ICONS.copy}<span>${escapeHtml(t("copy"))}</span>`;
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(plainText(turn));
      copy.innerHTML = `${ICONS.check}<span>${escapeHtml(t("copied"))}</span>`;
      setTimeout(() => { copy.innerHTML = `${ICONS.copy}<span>${escapeHtml(t("copy"))}</span>`; }, 1800);
    } catch { /* clipboard refused: nothing to do */ }
  });
  bar.append(copy);
  if (turn.answer.status === "answered") {
    for (const [rating, key] of [["up", "helpful"], ["down", "notHelpful"]]) {
      const button = el("button", "icon-btn");
      button.type = "button";
      button.innerHTML = ICONS[rating];
      button.setAttribute("aria-label", t(key));
      button.title = t(key);
      button.setAttribute("aria-pressed", String(turn.rating === rating));
      button.addEventListener("click", () => rate(turn, rating));
      bar.append(button);
    }
    if (turn.rating) bar.append(el("span", "time", t("thanks")));
  }
  if (turn.ms && !turn.rating) bar.append(el("span", "time", t("seconds", turn.ms / 1000)));
  return bar;
}

function plainText(turn) {
  const a = turn.answer;
  const lines = [a.answer || ""];
  const cited = a.sources.filter((s) => s.cited);
  if (cited.length) {
    lines.push("");
    for (const s of cited) {
      const where = [s.section, t("pages", s.page_start, s.page_end)].filter(Boolean).join(", ");
      lines.push(`[${s.number}] ${s.issuer} ${s.reference || s.title}, ${where} : ${s.link}`);
    }
  }
  return lines.join("\n");
}

function renderThread() {
  const thread = $("thread");
  thread.innerHTML = "";
  for (const turn of state.turns) thread.append(renderTurn(turn));
  const empty = state.turns.length === 0;
  $("welcome").hidden = !empty;
  $("new-chat").hidden = empty;
  markActive();
}
function replaceTurn(turn) {
  const old = document.querySelector(`.turn[data-turn="${turn.id}"]`);
  if (!old) return renderThread();
  old.replaceWith(renderTurn(turn));
  markActive();
}

/* ---------- source panel ---------- */
function findSource(turnId, n) {
  const turn = state.turns.find((x) => x.id === turnId);
  const source = turn?.answer?.sources.find((s) => s.number === n);
  return turn && source ? [turn, source] : null;
}
function openSource(turnId, n, focus = true) {
  const found = findSource(turnId, n);
  if (!found) return;
  const [, s] = found;
  state.active = { turn: turnId, n };
  $("panel-title").textContent = t("source", n);
  const body = $("panel-body");
  body.innerHTML = "";
  const ref = el("div", "panel-ref");
  ref.append(el("span", `issuer ${s.issuer}`, ISSUERS[state.lang][s.issuer] || s.issuer));
  if (s.reference) ref.append(el("code", "", s.reference));
  const excerpt = el("blockquote", "excerpt", s.excerpt);
  excerpt.dir = ARABIC.test(s.excerpt) ? "rtl" : "ltr";
  const link = el("a", "open-pdf");
  link.href = s.link;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.innerHTML = `${ICONS.open}<span>${escapeHtml(t("openPdf", s.page_start))}</span>`;
  const where = el("p", "panel-where");
  where.append(sourceLocation(s, ""));
  body.append(ref, el("p", "panel-title", s.title), where, excerpt, link);
  if (s.issuer === "BAM") body.append(el("p", "panel-note", t("bamNote", s.page_start)));
  $("panel").hidden = false;
  document.querySelector(".layout").classList.add("with-panel");
  ensureScrim();
  markActive();
  if (focus) $("panel-close").focus({ preventScroll: true });
}
function closeSource() {
  const active = state.active;
  state.active = null;
  $("panel").hidden = true;
  document.querySelector(".layout").classList.remove("with-panel");
  document.querySelector(".scrim")?.remove();
  markActive();
  if (active) document.querySelector(`.src[data-turn="${active.turn}"][data-n="${active.n}"]`)?.focus();
}
function ensureScrim() {
  if (!window.matchMedia("(max-width: 64rem)").matches || document.querySelector(".scrim")) return;
  const scrim = el("div", "scrim");
  scrim.addEventListener("click", closeSource);
  document.body.append(scrim);
}
function markActive() {
  document.querySelectorAll(".cite.active, .src.active").forEach((node) => node.classList.remove("active"));
  if (!state.active) return;
  const { turn, n } = state.active;
  document.querySelectorAll(`[data-turn="${turn}"][data-n="${n}"]`).forEach((node) => node.classList.add("active"));
}

/* ---------- asking ---------- */
function history() {
  return state.turns
    .filter((x) => x.status === "done" && ["answered", "not_found"].includes(x.answer.status))
    .slice(-HISTORY_TURNS)
    .map((x) => ({
      question: (x.answer.standalone_question || x.question).slice(0, 2000),
      answer: (x.answer.answer || "").slice(0, 4000),
    }));
}

function nearBottom() {
  return window.innerHeight + window.scrollY >= document.body.scrollHeight - 240;
}
function scrollToEnd() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

async function ask(text) {
  const question = text.trim();
  if (!question || state.busy) return;
  if (question.length > MAX_CHARS) { updateComposer(); return; }
  const turn = { id: `t${Date.now().toString(36)}`, question, status: "streaming", text: "", answer: null };
  const previous = history();
  state.turns.push(turn);
  state.busy = true;
  state.controller = new AbortController();
  $("composer-input").value = "";
  updateComposer();
  renderThread();
  scrollToEnd();

  const started = performance.now();
  let frame = 0;
  const paint = () => {
    frame = 0;
    const follow = nearBottom();
    replaceTurn(turn);
    if (follow) window.scrollTo({ top: document.body.scrollHeight });
  };
  try {
    const response = await fetch("/v1/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: previous }),
      signal: state.controller.signal,
    });
    if (!response.ok) throw await httpError(response);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        const event = parseEvent(block);
        if (event.type === "token") {
          turn.text += JSON.parse(event.data);
          if (!frame) frame = requestAnimationFrame(paint);
        } else if (event.type === "answer") {
          turn.answer = JSON.parse(event.data);
          turn.status = "done";
          turn.ms = performance.now() - started;
        } else if (event.type === "error") {
          throw Object.assign(new Error("stream"), { message: t("errUnavailable") });
        }
      }
    }
    if (turn.status === "streaming") throw new Error(t("errNetwork"));
  } catch (error) {
    if (error.name === "AbortError") {
      turn.status = "stopped";
    } else {
      turn.status = "error";
      turn.error = error.message && error.message !== "Failed to fetch" ? error.message : t("errNetwork");
    }
  } finally {
    if (frame) cancelAnimationFrame(frame);
    state.busy = false;
    state.controller = null;
    replaceTurn(turn);
    updateComposer();
    save();
    if (nearBottom()) scrollToEnd();
  }
}

function parseEvent(block) {
  let type = "message";
  const data = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) type = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  return { type, data: data.join("\n") };
}

async function httpError(response) {
  if (response.status === 429) {
    const wait = Number(response.headers.get("Retry-After")) || 30;
    return new Error(t("errRate", wait));
  }
  if (response.status === 401 || response.status === 403) return new Error(t("errAccess"));
  if (response.status === 422) return new Error(t("errTooLong"));
  return new Error(t("errUnavailable"));
}

async function rate(turn, rating) {
  turn.rating = turn.rating === rating ? null : rating;
  replaceTurn(turn);
  save();
  if (!turn.rating) return;
  try {
    await fetch("/v1/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request_id: turn.answer.request_id, rating }),
    });
  } catch { /* a lost rating is not worth bothering the reader */ }
}

/* ---------- composer ---------- */
function updateComposer() {
  const input = $("composer-input");
  const send = $("send");
  const length = input.value.trim().length;
  const count = $("count");
  count.textContent = input.value.length > MAX_CHARS * 0.8 ? `${input.value.length} / ${MAX_CHARS}` : "";
  count.classList.toggle("over", input.value.length > MAX_CHARS);
  send.classList.toggle("busy", state.busy);
  send.disabled = !state.busy && (length === 0 || input.value.length > MAX_CHARS);
  const label = state.busy ? t("stop") : t("send");
  send.setAttribute("aria-label", label);
  send.title = label;
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}

function newChat() {
  state.controller?.abort();
  state.turns = [];
  closeSource();
  save();
  renderThread();
  window.scrollTo({ top: 0 });
  $("composer-input").focus();
}

/* ---------- start ---------- */
function start() {
  const saved = load(LANG_KEY, null);
  const browser = (navigator.language || "fr").slice(0, 2);
  state.turns = load(STORE_KEY, []).filter((x) => x && x.id && x.question);
  setLang(saved || (STRINGS[browser] ? browser : "fr"));

  document.querySelectorAll("[data-lang]").forEach((button) => {
    button.addEventListener("click", () => setLang(button.dataset.lang));
  });
  $("new-chat").addEventListener("click", newChat);
  $("brand").addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
  $("panel-close").addEventListener("click", closeSource);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.active) closeSource();
  });
  $("thread").addEventListener("click", (event) => {
    const target = event.target.closest(".cite, .src");
    if (target) openSource(target.dataset.turn, Number(target.dataset.n));
  });

  const input = $("composer-input");
  input.addEventListener("input", updateComposer);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  });
  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    if (state.busy) state.controller?.abort();
    else ask(input.value);
  });

  fetch("/v1/corpus")
    .then((response) => (response.ok ? response.json() : null))
    .then((data) => { corpus = data; renderFacts(); })
    .catch(() => { /* facts are optional */ });

  if (state.turns.length) window.scrollTo({ top: document.body.scrollHeight });
  input.focus({ preventScroll: true });
}

document.addEventListener("DOMContentLoaded", start);
