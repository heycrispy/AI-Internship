"""Run the Golden Hour agent one turn at a time and turn ADK events into a Think / Act / Observe trace.

Used by both the CLI proof script (run_agent_demo.py) and the Streamlit UI (agent_page.py).

Mapping from ADK events to the decision loop:
  THINK    model text sent in the same turn as tool calls (the "Thought:" line)
  ACT      a function_call part            -> the tool the model chose + its arguments
  OBSERVE  a function_response part        -> what the tool returned
  APPROVAL an adk_request_confirmation call -> the loop is paused for a human decision
  ANSWER   final model text with no tool calls
Every step is also written to a log file (proof/agent_trace.log by default).
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from golden_hour_agent.agent import record_human_rejection, root_agent

APP_NAME = "golden_hour_agent"
USER_ID = "employee"
CONFIRM_CALL = "adk_request_confirmation"
LOG_FILE = Path(__file__).resolve().parent / "proof" / "agent_trace.log"

log = logging.getLogger("agent_trace")


def configure_logging(path: Path = LOG_FILE) -> None:
    if log.handlers:
        return
    path.parent.mkdir(exist_ok=True)
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)


@dataclass
class Step:
    kind: str  # THINK | ACT | OBSERVE | APPROVAL | ANSWER
    text: str = ""
    tool: str | None = None
    data: dict | None = None
    loop: int = 0  # which model call (iteration of the loop) produced this step

    def one_line(self) -> str:
        if self.kind == "ACT":
            return f"ACT      {self.tool}({json.dumps(self.data)})"
        if self.kind == "OBSERVE":
            return f"OBSERVE  {self.tool} -> {summarize(self.tool, self.data)}"
        if self.kind == "APPROVAL":
            return f"APPROVAL paused: human must approve {self.tool}({json.dumps(self.data)})"
        return f"{self.kind:<8} {self.text.strip()}"


@dataclass
class PendingApproval:
    confirmation_id: str
    tool: str
    args: dict


@dataclass
class TurnResult:
    steps: list[Step] = field(default_factory=list)
    answer: str | None = None
    pending: PendingApproval | None = None


def summarize(tool: str | None, data: dict | None) -> str:
    """Short, readable version of a tool result for logs."""
    data = data or {}
    if tool == "calculate_quote" and "subtotal_usd" in data:
        return (f"{data['package']}: subtotal ${data['subtotal_usd']}, deposit "
                f"${data['estimated_deposit_usd']}, balance ${data['remaining_balance_usd']}, "
                f"unresolved: {len(data['unresolved_charges'])}")
    if tool == "calculate_quote" and "recommended_quote" in data:
        r = data["recommended_quote"]
        return f"recommended {r['package']}: subtotal ${r['subtotal_usd']}, deposit ${r['estimated_deposit_usd']}"
    if tool == "check_availability" and "status" in data:
        conflicts = "; ".join(f"{c['event_type']} {c['date']} {c['start']}-{c['end']}"
                              for c in data.get("conflicts", []))
        return f"status={data['status']} {conflicts or ' '.join(data.get('reasons', []))}"
    if tool == "search_knowledge_base" and "chunks" in data:
        hits = ", ".join(f"{c['chunk_id']} ({c['score']:.2f})" for c in data["chunks"][:3])
        warn = "  !! SECURITY WARNING: injection-like text flagged" if "security_warning" in data else ""
        return f"relevant={data.get('relevant')} best={data.get('best_score')} top: {hits}{warn}"
    text = json.dumps(data)
    return text if len(text) <= 220 else text[:220] + "..."


def retrieved_sources(steps: list[Step]) -> str:
    """Deterministic citations: the relevant, non-quarantined chunks that search_knowledge_base
    actually returned this turn. The model can't fabricate these."""
    ids: list[str] = []
    for s in steps:
        if s.kind == "OBSERVE" and s.tool == "search_knowledge_base" and s.data:
            quarantined = str(s.data.get("security_warning", ""))
            for c in s.data.get("chunks", []):
                if (c["score"] >= s.data.get("threshold", 0) and c["chunk_id"] not in quarantined
                        and c["chunk_id"] not in ids):
                    ids.append(c["chunk_id"])
    return f"\n\nRetrieved sources: {', '.join(ids)}" if ids else ""


class AgentSession:
    """One conversation with the agent (keeps ADK session state across turns)."""

    def __init__(self, agent=root_agent):
        self.service = InMemorySessionService()
        self.runner = Runner(agent=agent, app_name=APP_NAME, session_service=self.service)
        self.session_id: str | None = None
        self.loop = 0
        configure_logging()

    async def _ensure_session(self) -> str:
        if self.session_id is None:
            session = await self.service.create_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=str(uuid.uuid4())
            )
            self.session_id = session.id
        return self.session_id

    async def ask(self, text: str) -> TurnResult:
        self.loop = 0  # loop numbers restart each user turn (a resumed approval continues it)
        log.info("USER     %s", text)
        message = types.Content(role="user", parts=[types.Part(text=text)])
        return await self._run(message)

    async def decide(self, pending: PendingApproval, approved: bool) -> TurnResult:
        """Resume a paused loop with the human's approve/reject decision."""
        log.info("HUMAN    %s %s", "APPROVED" if approved else "REJECTED", pending.tool)
        if not approved:
            # ADK never runs a rejected tool, so record the outcome here for the history.
            record_human_rejection(pending.tool, pending.args)
        message = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=pending.confirmation_id,
                        name=CONFIRM_CALL,
                        response={"confirmed": approved},
                    )
                )
            ],
        )
        return await self._run(message)

    async def _run(self, message: types.Content) -> TurnResult:
        session_id = await self._ensure_session()
        result = TurnResult()

        async for event in self.runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=message
        ):
            parts = (event.content.parts if event.content else None) or []
            texts = [p.text for p in parts if p.text and not p.function_call]
            calls = [p.function_call for p in parts if p.function_call]
            responses = [p.function_response for p in parts if p.function_response]

            # One model call = one iteration of the loop. ADK's confirmation request is
            # generated by the framework, not the model, so it doesn't count.
            if any(c.name != CONFIRM_CALL for c in calls) or (texts and not calls):
                self.loop += 1

            for call in calls:
                if call.name == CONFIRM_CALL:
                    original = (call.args or {}).get("originalFunctionCall", {})
                    pending_args = dict(original.get("args", {}))
                    pending_args.pop("reason", None)
                    result.pending = PendingApproval(
                        confirmation_id=call.id,
                        tool=original.get("name", "?"),
                        args=pending_args,
                    )
                    self._add(result, Step("APPROVAL", tool=result.pending.tool,
                                           data=result.pending.args, loop=self.loop))
                    continue
                args = dict(call.args or {})
                # Think = any free text the model sent with the call + the required `reason` arg.
                thought = " ".join(texts + [args.pop("reason", "")]).strip()
                texts = []
                if thought:
                    self._add(result, Step("THINK", text=thought, loop=self.loop))
                self._add(result, Step("ACT", tool=call.name, data=args, loop=self.loop))

            for resp in responses:
                payload = dict(resp.response or {})
                if resp.name == CONFIRM_CALL:
                    continue
                if "requires confirmation" in str(payload.get("error", "")):
                    continue  # ADK's placeholder while waiting for the human; not a real result
                self._add(result, Step("OBSERVE", tool=resp.name, data=payload, loop=self.loop))

            if texts and not calls and event.is_final_response():
                answer = " ".join(texts).strip() + retrieved_sources(result.steps)
                result.answer = answer
                self._add(result, Step("ANSWER", text=answer, loop=self.loop))

        if result.pending:
            log.info("--- loop paused, waiting for human approval ---")
        return result

    @staticmethod
    def _add(result: TurnResult, step: Step) -> None:
        result.steps.append(step)
        log.info("[loop %d] %s", step.loop, step.one_line())
