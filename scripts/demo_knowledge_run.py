"""G8.1a end-to-end proof: drive ONE legal question through the REAL agent loop with the
live `search_knowledge` tool against the ingested Constitution (kb_in).

Shows the WHEN (the model chooses to call search_knowledge) and HOW (it quotes + cites
the retrieved Article; the output gate binds it). API-burning — one standard-mode run on
the env model (gpt-5.4). Run only on Jeel's go.

The scope deliberately has NO vault (no collection/retrieval_manager) — only the KB
manager — so the agent must reach for the law, not the user's docs.

Usage:  USE_KNOWLEDGE=true python -u scripts/demo_knowledge_run.py "your legal question"
"""

import os
import sys
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("USE_KNOWLEDGE", "true")  # the flag that offers the tool

from copy import copy

from src.components.config import Config
from src.components.retrieval import RetrievalManager
from src.components.agent_core.registry import RunScope, REGISTRY
from src.components.agent_core.budgets import budget_for
from src.components.agent_core.model import build_model
from src.components.agent_core.prompt import system_prompt
from src.components.agent_core.loop import run_agent


def main() -> int:
    question = sys.argv[1] if len(sys.argv) > 1 else \
        "What does the Constitution of India guarantee about the right to life and personal liberty?"
    mode = "standard"
    config = Config()

    # KB-scoped retrieval manager (kb_in), NO vault — force the law path.
    kb_cfg = copy(config)
    kb_cfg.PINECONE_NAMESPACE = getattr(config, "KNOWLEDGE_NAMESPACE", "kb_in")
    kb = RetrievalManager(kb_cfg)

    scope = RunScope(kb_retrieval_manager=kb, question=question, config=config)
    budget = budget_for(mode, config)
    sys_prompt = system_prompt("v1")
    model = build_model(mode, budget, config, system=sys_prompt)

    print(f"Q: {question}\n" + "=" * 72)
    answer = []
    for ev in run_agent(question, model=model, scope=scope, budget=budget,
                        system_prompt=sys_prompt, registry=REGISTRY):
        t = ev.get("type")
        if t == "tool_call":
            print(f"  → CALL {ev.get('name')}  {str(ev.get('args_summary'))[:110]}")
        elif t == "tool_result":
            print(f"     ok={ev.get('ok')}  {str(ev.get('summary'))[:110]}")
        elif t == "gate":
            print(f"  [GATE {ev.get('name')}] pass={ev.get('pass')}  {str(ev.get('detail'))[:90]}")
        elif t == "token":
            answer.append(ev.get("text", ""))
        elif t == "meta":
            print("=" * 72)
            print(f"  steps={ev.get('steps')} tokens={ev.get('tokens')} abstained={ev.get('abstained')}")
    print("\nANSWER:\n" + "".join(answer).strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
