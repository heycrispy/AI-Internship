# AI Engineering Bootcamp v2

Hands-on course materials for building production-style LLM APIs with **FastAPI**, **OpenAI**, **Pydantic**, and **Streamlit**.

## Weeks

| Week | Topic | Location |
|------|-------|----------|
| 1 | `/ask` endpoint — typed I/O, structured output, guardrails, model selection, cost | [`week-1/`](week-1/) |
| 1 v2 | Simplified class-ready `/ask` demo with one final API and optional stage references | [`week-1v2/`](week-1v2/) |
| 2 | RAG and vector databases | [`week-2/`](week-2/) |
| 3 | Capstone as an agent (Google ADK): tools, Think/Act/Observe trace, human approval, injection test | [`week-1v2/`](week-1v2/README.md#week-3-the-assistant-as-an-agent-google-adk) |

## Tech stack

- **FastAPI** — HTTP API with automatic OpenAPI docs
- **OpenAI Python SDK** — chat completions and structured output (`response_format`)
- **Pydantic** — request/response schemas and validation guardrails
- **python-dotenv** — load `OPENAI_API_KEY` from `.env`
- **Streamlit** — interactive demo runner (`demo_page.py`)
- **httpx** — HTTP client for tests and the Streamlit UI

## Quick start

```bash
cd week-1v2
cp .env.example .env          # add your OPENAI_API_KEY
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See [week-1v2/README.md](week-1v2/README.md) for the simplified class demo, or
[week-1/README.md](week-1/README.md) for the original five-stage version.
