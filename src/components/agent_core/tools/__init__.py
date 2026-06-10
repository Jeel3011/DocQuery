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
from .read import SCHEMA as READ_SCHEMA, read_document
from .search import SCHEMA as SEARCH_SCHEMA, search_vault
from .table import SCHEMA as TABLE_SCHEMA, table_lookup

SCHEMAS = {
    "search_vault": SEARCH_SCHEMA,
    "read_document": READ_SCHEMA,
    "table_lookup": TABLE_SCHEMA,
    "compute": COMPUTE_SCHEMA,
}

__all__ = [
    "search_vault", "read_document", "table_lookup", "compute",
    "SEARCH_SCHEMA", "READ_SCHEMA", "TABLE_SCHEMA", "COMPUTE_SCHEMA", "SCHEMAS",
]
