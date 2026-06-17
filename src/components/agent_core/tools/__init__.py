"""Agent-core tool adapters (AGENT_CORE_PLAN §3.3, task A1).

Four thin adapters over existing proven code, each emitting the uniform `_envelope`
shape and NEVER raising:

  search_vault   → RetrievalManager.retrieve / retrieve_table_chunks
  read_document  → table_intent.load_grids_for_docs (+ page text)
  table_lookup   → perception.grounding.ground_metric
  compute        → analyst.compute (the deterministic kernel)

Each module also exposes a `SCHEMA` dict (name + JSON input_schema) for the registry
(A2). No registry/loop here — A1 is the adapters only.
"""

from .compute import SCHEMA as COMPUTE_SCHEMA, compute
from .knowledge import SCHEMA as KNOWLEDGE_SCHEMA, search_knowledge
from .metrics import SCHEMA as METRICS_SCHEMA, list_metrics
from .read import SCHEMA as READ_SCHEMA, read_document
from .search import SCHEMA as SEARCH_SCHEMA, search_vault
from .survey import SCHEMA as SURVEY_SCHEMA, survey_collection
from .table import SCHEMA as TABLE_SCHEMA, table_lookup

SCHEMAS = {
    "search_vault": SEARCH_SCHEMA,
    "read_document": READ_SCHEMA,
    "list_metrics": METRICS_SCHEMA,
    "table_lookup": TABLE_SCHEMA,
    "compute": COMPUTE_SCHEMA,
    # G5: the broad whole-vault pass — deep mode only (see registry._MODE_TOOLS).
    "survey_collection": SURVEY_SCHEMA,
    # G8: the agent's hand into the shared legal Knowledge Base. Offered only when the
    # run threads a kb_retrieval_manager (behind USE_KNOWLEDGE); off ⇒ never wired in.
    "search_knowledge": KNOWLEDGE_SCHEMA,
}

__all__ = [
    "search_vault", "read_document", "list_metrics", "table_lookup", "compute",
    "survey_collection", "search_knowledge",
    "SEARCH_SCHEMA", "READ_SCHEMA", "METRICS_SCHEMA", "TABLE_SCHEMA", "COMPUTE_SCHEMA",
    "SURVEY_SCHEMA", "KNOWLEDGE_SCHEMA",
    "SCHEMAS",
]
