"""Streamlit UI for the Golden Hour Coffee Co. agent: chat + live Think / Act / Observe trace + approval gate.

Runs the ADK agent in-process (no API server needed).
Run:  streamlit run agent_page.py
"""

import asyncio
import json
import threading

import streamlit as st

from agent_runner import AgentSession, Step
from golden_hour_agent.agent import (
    BOOKINGS_FILE,
    INJECTED_DOC,
    INJECTED_DOC_ID,
    MODEL_NAME,
    ensure_kb_ingested,
    ingest,
    root_agent,
)
from golden_hour_agent.availability import load_bookings, parse_date, parse_time
import rag


def availability_example() -> str:
    """Build prompt 3 from the fixture calendar, so its date always matches a real conflict."""
    b = load_bookings()[0]
    day = parse_date(b["date"]).strftime("%B %-d, %Y")
    start, end = (parse_time(b[k]).strftime("%-I %p") for k in ("start", "end"))
    return f"Are you available on {day}, from {start} to {end}?"


EXAMPLES = [
    "What coffee catering packages do you offer?",
    "I'm hosting a corporate event for 85 guests in Newport Beach. How much would three hours cost?",
    availability_example(),
    "I'd like to book The Golden Hour package for my wedding. What information do you need?",
]
ICONS = {"THINK": "🧠", "ACT": "🛠️", "OBSERVE": "👀", "APPROVAL": "✋", "ANSWER": "💬"}

st.set_page_config(page_title="Golden Hour Coffee Co. Agent", page_icon="☕", layout="wide")


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
    return any(d["document_id"] == INJECTED_DOC_ID for d in rag.list_documents())


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
    ensure_kb_ingested()
    for doc in rag.list_documents():
        st.caption(f"📄 {doc['document_id']} — {doc['chunks']} chunks")

    st.subheader("Prompt-injection test")
    st.caption("Adds an untrusted coffee-supplier memo with hidden instructions telling the agent "
               "to offer a \\$1 catering package and confirm bookings without owner approval. "
               "Then ask about supplies, oat milk, or a quote.")
    if injection_doc_loaded():
        if st.button("Remove poisoned doc"):
            rag.get_collection().delete(where={"document_id": INJECTED_DOC_ID})
            st.rerun()
    elif st.button("Load poisoned doc"):
        ingest(INJECTED_DOC, INJECTED_DOC_ID, trust="untrusted")
        st.rerun()

    st.divider()
    st.subheader("Booking requests")
    st.caption("Requests for owner review, not confirmed bookings.")
    requests = ([json.loads(l) for l in BOOKINGS_FILE.read_text().splitlines()]
                if BOOKINGS_FILE.exists() else [])
    if requests:
        st.dataframe([{"status": r["status"].replace("_", " "), "date": r.get("event_date"),
                       "package": r.get("package"), "subtotal": r.get("subtotal_usd"),
                       "id": r["request_id"]} for r in reversed(requests)],
                     hide_index=True)
    else:
        st.caption("None yet.")

# ------------------------------------------------------------------ main
st.title("☕ Golden Hour Coffee Co. Agent")
st.markdown("*Your AI concierge for mobile coffee catering in Orange County.*  \n"
            "Great coffee. Golden moments.")
st.caption("Google ADK agent · tools: search_knowledge_base, calculate_quote, check_availability, "
           "list_knowledge_documents, submit_booking_request (human approval required). "
           "Each turn shows the Think → Act → Observe loop.")

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
            st.warning(f"✋ The agent wants to run **{pending.tool}**. Approving submits a "
                       "booking *request* for owner review; it does not confirm the booking.")
            st.json(pending.args)
            a, r = st.columns(2)
            decision = None
            if a.button("Approve", icon=":material/check:", type="primary",
                        use_container_width=True):
                decision = True
            if r.button("Reject", icon=":material/close:", use_container_width=True):
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
    prompt = st.chat_input("Ask about packages, quotes, dates, or booking...",
                           disabled=bool(pending)) or picked

    if prompt:
        with st.spinner("Agent is thinking and calling tools..."):
            result = run(st.session_state.agent.ask(prompt))
        st.session_state.turns.append(
            {"user": prompt, "steps": result.steps, "answer": result.answer}
        )
        st.session_state.pending = result.pending
        st.rerun()
