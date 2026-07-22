"""
Explainable Hallucination Detection Framework
-----------------------------------------------
Internship project — 1st year

WORKFLOW (as designed):
Upload CSV -> Extract Text -> Create Embeddings -> Store in FAISS
-> User asks Question -> Retrieve Context -> LLM Generates Answer
-> Verification Module (Uploaded Dataset + Wikipedia)
-> Semantic Comparison -> Confidence Score -> Explanation + Trusted Answer

This version is written for a TruthfulQA-style CSV with columns:
Type, Category, Question, Best Answer, Best Incorrect Answer,
Correct Answers, Incorrect Answers, Source

It will also work with any CSV that has at least a "Question" style
column and an "Answer" style column — see find_column() below.
"""

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import requests

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------------------------------------------
# STEP 0: Load Models (cached so they only load once)
#
# NOTE: we load the tokenizer + model directly instead of using
# transformers.pipeline("text2text-generation", ...). Some newer
# transformers versions don't register that task name in their
# pipeline registry unless extra optional packages are installed,
# which throws: KeyError: "Unknown task text2text-generation".
# Loading the model/tokenizer directly avoids that entirely.
# ------------------------------------------------------------

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_generator():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model


def generate_answer(prompt, max_length=100):
    """Runs the local free flan-t5 model on a prompt and returns the decoded text."""
    tokenizer, model = generator
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def call_openai(prompt, api_key, model="gpt-4o-mini"):
    """
    Optional second LLM opinion using the OpenAI API.
    Only runs if the user enters their own API key in the sidebar — this
    is a paid API, so it's never called unless a key is provided.
    Returns the answer text, or raises an Exception with a readable message.
    """
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0,
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(f"OpenAI API error {response.status_code}: {response.text[:200]}")
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


embedder = load_embedder()
generator = load_generator()

# ------------------------------------------------------------
# Helper: find the right columns automatically
# ------------------------------------------------------------

def find_column(df, candidates):
    """Return the first column in df that matches one of the candidate names
    (case-insensitive, ignoring leading/trailing spaces). Returns None if
    nothing matches."""
    lower_map = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.strip().lower() in lower_map:
            return lower_map[cand.strip().lower()]
    return None


# ------------------------------------------------------------
# Wikipedia verification function
# ------------------------------------------------------------

def get_wikipedia_answer(question, sentences=2):
    """
    Looks up a Wikipedia summary using Wikipedia's own REST API directly
    (instead of the old 'wikipedia' PyPI package, which is unmaintained
    and frequently fails silently on cloud hosts due to parsing/SSL issues).
    `sentences` controls how much text is returned — use more when this
    text will be fed to the LLM as grounding context, fewer when it's just
    for display.
    """
    headers = {"User-Agent": "HallucinationDetectionApp/1.0 (student project)"}

    try:
        # STEP 1: search for the best matching page title
        search_resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": question,
                "format": "json",
                "srlimit": 1,
            },
            headers=headers,
            timeout=10,
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()
        hits = search_data.get("query", {}).get("search", [])
        if not hits:
            return "No Wikipedia information found (no search results)."
        title = hits[0]["title"]

        # STEP 2: fetch a short summary for that page title
        summary_resp = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
            headers=headers,
            timeout=10,
        )
        if summary_resp.status_code == 404:
            return f"No Wikipedia information found (page '{title}' not found)."
        summary_resp.raise_for_status()
        summary_data = summary_resp.json()
        extract = summary_data.get("extract", "").strip()
        if not extract:
            return "No Wikipedia information found (empty summary)."
        result_sentences = extract.split(". ")
        return ". ".join(result_sentences[:sentences]).strip().rstrip(".") + "."

    except requests.exceptions.RequestException as e:
        # Network-level failure (timeout, DNS, blocked outbound request, etc.)
        return f"No Wikipedia information found (network error: {type(e).__name__}: {e})."
    except Exception as e:
        return f"No Wikipedia information found (error: {type(e).__name__}: {e})."


# ------------------------------------------------------------
# Build the FAISS knowledge base from the uploaded CSV
#
# NOTE: this is intentionally NOT wrapped in @st.cache_resource.
# Passing a big serialized JSON string into a cached function is
# fragile and can throw confusing errors. Instead we cache the
# result manually in st.session_state further down, keyed by the
# file name + chosen columns, so it still only rebuilds when needed.
# ------------------------------------------------------------

def build_knowledge_base(df, question_col, answer_col):
    # Each "knowledge entry" pairs the question with its trusted answer.
    # This gives FAISS much more meaningful chunks to retrieve than
    # dumping every raw cell in the sheet.
    knowledge = []
    for _, row in df.iterrows():
        q = str(row[question_col])
        a = str(row[answer_col])
        knowledge.append(f"Q: {q}\nA: {a}")

    embeddings = embedder.encode(knowledge, convert_to_numpy=True, show_progress_bar=False)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return knowledge, index


# ------------------------------------------------------------
# Core pipeline — runs once per question, returns a result dict.
# Pulled into its own function so the chat loop below can call it
# for every new message without repeating code.
# ------------------------------------------------------------

def run_pipeline(question, df, knowledge, index, question_col, answer_col, top_k, openai_api_key=None):

    # STEP 1: Retrieve Context (Semantic Search over the CSV)
    q_embed = embedder.encode([question], convert_to_numpy=True)
    distances, idx = index.search(q_embed, top_k)
    retrieved = [knowledge[i] for i in idx[0]]
    trusted_context = "\n".join(retrieved)

    best_match_row = df.iloc[idx[0][0]]
    trusted_answer = str(best_match_row[answer_col])

    # --- Relevance check ---
    # FAISS always returns the closest row, even if nothing in the dataset
    # is actually related to the question. Without this check, an unrelated
    # row (e.g. a "religion" row for a "capital city" question) would get
    # treated as ground truth. We compare the question's embedding directly
    # against the matched row's own embedding (reconstructed from the FAISS
    # index) and only trust the dataset if they're genuinely close.
    top_match_embedding = index.reconstruct(int(idx[0][0])).reshape(1, -1)
    dataset_relevance = cosine_similarity(q_embed, top_match_embedding)[0][0]
    RELEVANCE_THRESHOLD = 0.45
    dataset_is_relevant = dataset_relevance >= RELEVANCE_THRESHOLD

    # STEP 2: Verification Module — fetch Wikipedia BEFORE generating, so that
    # if the dataset isn't relevant we can ground the LLM's answer on Wikipedia
    # instead of on an unrelated dataset row (which was producing vague,
    # ungrounded answers like "The President of India" instead of a real name).
    wiki_answer_short = get_wikipedia_answer(question, sentences=2)  # for display/scoring
    wiki_is_available = not wiki_answer_short.startswith("No Wikipedia information found")
    wiki_answer = wiki_answer_short

    if dataset_is_relevant:
        generation_context = trusted_context
    elif wiki_is_available:
        # Pull a longer excerpt to actually ground the answer, not just the
        # short 2-sentence version used for display/scoring.
        generation_context = get_wikipedia_answer(question, sentences=5)
    else:
        generation_context = trusted_context  # last resort, nothing better available

    # STEP 3: LLM(s) Generate an Answer using whichever context is actually relevant
    prompt = f"""Answer the question using ONLY the context below.

Context:
{generation_context}

Question:
{question}

Answer:"""

    # Model 1 (always runs): the free local flan-t5 model
    flant5_answer = generate_answer(prompt, max_length=100)

    # Model 2 (optional): OpenAI, only if the user supplied an API key
    gpt_answer = None
    gpt_error = None
    if openai_api_key:
        try:
            gpt_answer = call_openai(prompt, openai_api_key)
        except Exception as e:
            gpt_error = str(e)

    # STEP 4: Semantic Comparison — score every available AI answer against
    # whichever trusted sources are actually usable for this question.
    trusted_embedding = embedder.encode([trusted_answer])
    wiki_embedding = embedder.encode([wiki_answer])

    candidates = []  # each item: {"source": name, "answer": text, "confidence": int}

    for source_name, answer_text in [("flan-t5 (free, local)", flant5_answer), ("OpenAI GPT", gpt_answer)]:
        if not answer_text:
            continue
        ans_embedding = embedder.encode([answer_text])
        dataset_sim = cosine_similarity(ans_embedding, trusted_embedding)[0][0]
        wiki_sim = cosine_similarity(ans_embedding, wiki_embedding)[0][0] if wiki_is_available else None

        # Only average in the sources that are actually usable for this question
        usable_sims = []
        if dataset_is_relevant:
            usable_sims.append(dataset_sim)
        if wiki_is_available:
            usable_sims.append(wiki_sim)

        conf = int((sum(usable_sims) / len(usable_sims)) * 100) if usable_sims else 0
        conf = max(0, min(conf, 100))
        candidates.append({
            "source": source_name,
            "answer": answer_text,
            "confidence": conf,
            "dataset_similarity": dataset_sim,
            "wiki_similarity": wiki_sim,
        })

    # Pick whichever AI answer scored highest
    best_candidate = max(candidates, key=lambda c: c["confidence"]) if candidates else None

    # STEP 5: Decide the Final Answer.
    # If the best AI answer is well supported by the *usable* trusted sources, use it.
    # Otherwise, self-correct — but only fall back to a source that's actually
    # relevant/available. Never fall back to an irrelevant dataset row.
    CONFIDENCE_THRESHOLD = 60

    if best_candidate and best_candidate["confidence"] >= CONFIDENCE_THRESHOLD:
        final_answer = best_candidate["answer"]
        final_source = best_candidate["source"]
        confidence = best_candidate["confidence"]
    elif dataset_is_relevant:
        final_answer = trusted_answer
        final_source = "Trusted Dataset (auto-corrected)"
        confidence = best_candidate["confidence"] if best_candidate else 0
    elif wiki_is_available:
        final_answer = wiki_answer
        final_source = "Wikipedia (auto-corrected — question not covered by dataset)"
        confidence = best_candidate["confidence"] if best_candidate else 0
    else:
        final_answer = best_candidate["answer"] if best_candidate else "Unable to generate or verify an answer."
        final_source = "AI Answer (unverified — no trusted source available)"
        confidence = best_candidate["confidence"] if best_candidate else 0

    # STEP 6: Explanation + Hallucination Level
    if not dataset_is_relevant:
        explanation = "This question isn't covered by the uploaded dataset, so the dataset was excluded from scoring; verification relied on Wikipedia only."
    elif confidence >= 85:
        explanation = "The AI response matches the uploaded dataset and Wikipedia."
    elif confidence >= CONFIDENCE_THRESHOLD:
        explanation = "The AI response is partially supported by trusted knowledge."
    else:
        explanation = "The AI response(s) contradicted trusted knowledge, so the Final Answer was corrected."

    if confidence >= 85:
        level = "LOW"
    elif confidence >= CONFIDENCE_THRESHOLD:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return {
        "final_answer": final_answer,
        "final_source": final_source,
        "candidates": candidates,
        "gpt_error": gpt_error,
        "trusted_answer": trusted_answer,
        "trusted_context": trusted_context,
        "wiki_answer": wiki_answer,
        "dataset_is_relevant": dataset_is_relevant,
        "dataset_relevance": dataset_relevance,
        "confidence": confidence,
        "explanation": explanation,
        "level": level,
    }


def format_reply(result):
    """Turn the pipeline result dict into one Markdown chat bubble.
    Everything relevant — including the Wikipedia cross-check — is shown
    directly here (not hidden in an expander) so it's never missed."""
    lines = [f"**✅ Final (Correct) Answer:** {result['final_answer']}",
             f"*Source: {result['final_source']} — Confidence: {result['confidence']}%*",
             "",
             "---",
             "**Cross-checked against:**", ""]

    for c in result["candidates"]:
        lines.append(f"- **{c['source']} said:** {c['answer']}  _(confidence: {c['confidence']}%)_")

    if result["dataset_is_relevant"]:
        lines.append(f"- **Trusted Dataset says:** {result['trusted_answer']}")
    else:
        lines.append(
            f"- **Trusted Dataset:** _not relevant to this question_ "
            f"(closest row was only {result['dataset_relevance']*100:.0f}% similar — excluded from scoring)"
        )
    lines.append(f"- **Wikipedia says:** {result['wiki_answer']}")

    if result.get("gpt_error"):
        lines.append(f"\n_Note: OpenAI call failed ({result['gpt_error']}), so only the free model was used._")

    lines.append("")
    lines.append(f"**Explanation:** {result['explanation']}")
    lines.append(f"**Hallucination Level:** {result['level']}")

    return "\n".join(lines)


# ------------------------------------------------------------
# Streamlit UI — Chatbox layout
# ------------------------------------------------------------

st.set_page_config(page_title="Explainable Hallucination Detection", layout="wide")
st.title("🔍 Explainable Hallucination Detection — Chatbot")
st.caption("RAG + Semantic Verification against a trusted dataset and Wikipedia")

# ---- Sidebar: upload + settings (kept out of the chat flow) ----
with st.sidebar:
    st.header("Setup")
    uploaded = st.file_uploader("Upload CSV (e.g. TruthfulQA.csv)", type=["csv"])

    st.markdown("---")
    st.caption("Optional: add your own OpenAI API key to get a second LLM opinion (paid, never required).")
    openai_api_key = st.text_input("OpenAI API Key (optional)", type="password")

    st.markdown("---")
    st.caption("Debug: test the Wikipedia connection on its own, without uploading a CSV.")
    if st.button("Test Wikipedia connection"):
        with st.spinner("Calling Wikipedia..."):
            test_result = get_wikipedia_answer("Albert Einstein")
        st.write(test_result)

    question_col = None
    answer_col = None
    top_k = 3

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(df.head(3), height=120)

        question_col = find_column(df, ["Question"])
        answer_col = find_column(df, ["Best Answer", "Answer", "Correct Answers"])

        question_col = st.selectbox(
            "Question column", df.columns,
            index=list(df.columns).index(question_col) if question_col else 0,
        )
        answer_col = st.selectbox(
            "Trusted answer column", df.columns,
            index=list(df.columns).index(answer_col) if answer_col else 0,
        )
        top_k = st.slider("Context entries to retrieve", 1, 5, 3)

        # Only rebuild the FAISS index if the file or chosen columns changed —
        # otherwise reuse what's already in session_state.
        kb_key = (uploaded.name, question_col, answer_col)
        if st.session_state.get("kb_key") != kb_key:
            with st.spinner("Building embeddings and FAISS index..."):
                knowledge, index = build_knowledge_base(df, question_col, answer_col)
            st.session_state["kb_key"] = kb_key
            st.session_state["knowledge"] = knowledge
            st.session_state["index"] = index
        else:
            knowledge = st.session_state["knowledge"]
            index = st.session_state["index"]

        st.success(f"Knowledge base ready — {len(knowledge)} entries indexed.")

    if st.button("Clear chat"):
        st.session_state.messages = []

# ---- Session state: chat history persists across reruns ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Replay previous messages so the conversation looks continuous ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Chat input (disabled until a CSV is uploaded) ----
if not uploaded:
    st.info("Upload a CSV file in the sidebar to start chatting (try your TruthfulQA.csv).")
else:
    question = st.chat_input("Ask a question about your dataset...")

    if question:
        # Show the user's message immediately
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Run the pipeline and show the assistant's reply
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context, generating answer(s), verifying..."):
                result = run_pipeline(
                    question, df, knowledge, index, question_col, answer_col, top_k,
                    openai_api_key=openai_api_key,
                )
                reply = format_reply(result)

            st.markdown(reply)

            with st.expander("Show raw retrieved dataset context (debug detail)"):
                st.text(result["trusted_context"])

        st.session_state.messages.append({"role": "assistant", "content": reply})