"""Run the stock ADK Demo 1 (ai-engineering-bootcamp/adk-multi-agent-systems/demo1_routing.py).

The sample hardcodes MODEL = "gemini-2.5-flash". If GOOGLE_API_KEY is set we run it as-is;
otherwise we swap only that constant for an OpenAI model through ADK's LiteLlm wrapper.
Nothing else in the demo changes.

Run: python run_adk_demo1.py | tee proof/adk_demo1_output.txt
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

THIS_DIR = Path(__file__).resolve().parent
load_dotenv(THIS_DIR / ".env")

DEMO1 = THIS_DIR.parents[1] / "ai-engineering-bootcamp/adk-multi-agent-systems/demo1_routing.py"
source = DEMO1.read_text()

if os.getenv("GOOGLE_API_KEY"):
    print("Model: gemini-2.5-flash (stock)")
else:
    print("Model: openai/gpt-4o-mini via LiteLlm (no GOOGLE_API_KEY set)")
    source = source.replace(
        'MODEL = "gemini-2.5-flash"',
        'from google.adk.models.lite_llm import LiteLlm\nMODEL = LiteLlm(model="openai/gpt-4o-mini")',
    )

namespace = {"__name__": "demo1_routing"}
exec(compile(source, str(DEMO1), "exec"), namespace)
asyncio.run(namespace["main"]())
