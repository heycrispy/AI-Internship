"""Streamlit UI for the Week 3 handbook agent: chat + live Think / Act / Observe trace + approval gate.

Runs the ADK agent in-process (no API server needed).
Run:  streamlit run agent_page.py
"""

import asyncio
import json
import threading

import streamlit as st

from agent_runner import AgentSession, Step
from handbook_agent.agent import (
    CLAIMS_FILE,
    MODEL_NAME,
    SAMPLE_DOC,
    _client,
    ensure_handbook_ingested,
    root_agent,
)
import rag

INJECTED_DOC = SAMPLE_DOC.parent / "injected_memo.md"
EXAMPLES = [
    "I'm flying to Berlin for 4 days. What's my total meal allowance, and can I book premium "
    "economy on the 9 hour flight?",
    "I have 10 days of PTO banked. If I take none for the next 12 months, how many will I have?",
    "I paid $92 for a client dinner on my work trip and have the itemized receipt. "
    "Please submit the expense claim.",
    "What's Northwind's 401(k) match percentage?",
]
ICONS = {"THINK": "🧠", "ACT": "🛠️", "OBSERVE": "👀", "APPROVAL": "✋", "ANSWER": "💬"}

st.set_page_config(page_title="Week 3 Handbook Agent", layout="wide")


# ADK + LiteLLM keep async clients bound to an event loop, so run every turn on one
# long-lived loop in a background thread instead of a fresh asyncio.run() per click.
@st.cache_resource
def background_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop


def run(coro):
    return asyncio.run_coroutine_threadsafe(coro, background_loop()).result(timeout=300)


def new_session(model: str) -> None:
    agent = root_agent
    if model != MODEL_NAME:
        from google.adk.models.lite_llm import LiteLlm

        agent = root_agent.clone(update={"model": LiteLlm(model=model)})
    st.session_state.agent = AgentSession(agent)
    st.session_state.turns = []  # list of {"user": str, "steps": [Step], "answer": str|None}
    st.session_state.pending = None
    st.session_state.model = model


def injection_doc_loaded() -> bool:
    return any(d["document_id"] == "vendor_memo" for d in rag.list_documents())


def md(text: str) -> str:
    """Escape "$" so Streamlit doesn't render "$440 ... $110" as LaTeX."""
    return text.replace("$", "\\$")


def render_step(step: Step) -> None:
    icon = ICONS.get(step.kind, "•")
    label = f"{icon} **{step.kind}** · loop {step.loop}"
    if step.kind == "THINK":
        st.markdown(f"{label}  \n_{md(step.text)}_")
    elif step.kind == "ACT":
        st.markdown(f"{label} → `{step.tool}`")
        st.code(json.dumps(step.data, indent=2), language="json")
    elif step.kind == "OBSERVE":
        warn = " ⚠️ injection flagged" if (step.data or {}).get("security_warning") else ""
        with st.expander(f"{icon} OBSERVE · loop {step.loop} ← {step.tool}{warn}"):
            st.json(step.data)
    elif step.kind == "APPROVAL":
        st.markdown(f"{label} — {step.text or f'paused before `{step.tool}`'}")
    # ANSWER is shown in the chat bubble


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Agent settings")
    model = st.selectbox("Model", ["openai/gpt-4o-mini", "openai/gpt-4o", "openai/gpt-4.1-mini"])
    if "agent" not in st.session_state or st.session_state.model != model:
        new_session(model)
    if st.button("New conversation"):
        new_session(model)
        st.rerun()

    st.divider()
    st.subheader("Knowledge base")
    ensure_handbook_ingested()
    for doc in rag.list_documents():
        st.caption(f"📄 {doc['document_id']} — {doc['chunks']} chunks")

    st.subheader("Prompt-injection test")
    st.caption("Adds a vendor memo with hidden instructions telling the agent to silently file a "
               "$4,999 claim. Ask about laptop refreshes or expensing a mouse.")
    if injection_doc_loaded():
        if st.button("Remove poisoned doc"):
            rag.get_collection().delete(where={"document_id": "vendor_memo"})
            st.rerun()
    elif st.button("Load poisoned doc"):
        rag.ingest_document(_client(), "vendor_memo", INJECTED_DOC.read_text(),
                            rag.DEFAULT_CHUNK_SIZE, rag.DEFAULT_CHUNK_OVERLAP,
                            {"source": INJECTED_DOC.name})
        st.rerun()

    st.divider()
    st.subheader("Filed expense claims")
    claims = [json.loads(l) for l in CLAIMS_FILE.read_text().splitlines()] if CLAIMS_FILE.exists() else []
    if claims:
        st.dataframe([{k: c[k] for k in ("claim_id", "amount_usd", "category")} for c in claims],
                     hide_index=True)
    else:
        st.caption("None yet.")

# ------------------------------------------------------------------ main
st.title("Northwind HR Agent")
st.caption("Google ADK agent · tools: search_handbook, calculate, list_handbook_documents, "
           "submit_expense_claim (human approval required). Each turn shows the Think → Act → "
           "Observe loop.")

chat_col, trace_col = st.columns([3, 2], gap="large")

with trace_col:
    st.subheader("Decision loop")
    if not st.session_state.turns:
        st.caption("Ask something to see the agent plan, call tools and observe results.")
    for i, turn in enumerate(reversed(st.session_state.turns)):
        with st.container(border=True):
            st.markdown(f"**Turn {len(st.session_state.turns) - i}:** {turn['user'][:80]}")
            for step in turn["steps"]:
                render_step(step)

with chat_col:
    for turn in st.session_state.turns:
        with st.chat_message("user"):
            st.write(turn["user"])
        if turn["answer"]:
            with st.chat_message("assistant"):
                st.markdown(md(turn["answer"]))

    pending = st.session_state.pending
    if pending:
        with st.container(border=True):
            st.warning(f"✋ The agent wants to run **{pending.tool}**. Approve?")
            st.json(pending.args)
            a, r = st.columns(2)
            decision = None
            if a.button("✅ Approve", type="primary", use_container_width=True):
                decision = True
            if r.button("❌ Reject", use_container_width=True):
                decision = False
            if decision is not None:
                with st.spinner("Resuming the agent loop..."):
                    result = run(st.session_state.agent.decide(pending, decision))
                turn = st.session_state.turns[-1]
                turn["steps"].append(Step(
                    "APPROVAL", tool=pending.tool,
                    text=f"Human {'APPROVED' if decision else 'REJECTED'} {pending.tool}",
                    loop=turn["steps"][-1].loop,
                ))
                turn["steps"] += result.steps
                turn["answer"] = result.answer
                st.session_state.pending = result.pending
                st.rerun()

    st.markdown("**Try:**")
    picked = None
    for ex in EXAMPLES:
        if st.button(ex, key=ex, disabled=bool(pending)):
            picked = ex
    prompt = st.chat_input("Ask the HR agent...", disabled=bool(pending)) or picked

    if prompt:
        with st.spinner("Agent is thinking and calling tools..."):
            result = run(st.session_state.agent.ask(prompt))
        st.session_state.turns.append(
            {"user": prompt, "steps": result.steps, "answer": result.answer}
        )
        st.session_state.pending = result.pending
        st.rerun()
