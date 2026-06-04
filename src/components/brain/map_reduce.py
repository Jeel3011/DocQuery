"""
Stage-2 Brain: T1 map-reduce synthesis — Phase 4 / §4 of the Brain plan.

Architecture:
  MAP    — for each routed doc, retrieve its best chunks and run an extraction
           prompt → list of Claim objects with verbatim evidence spans.
           Runs in parallel (bounded concurrency). Per-doc try/except — a failed
           doc is recorded, not fatal (quorum logic, §6.3).
  REDUCE — feed per-doc extracts into a single synthesis call using a large-context
           model. Because each extract is small (one claim per sentence), dozens of
           docs fit in one context. Produces a set of synthesized Claims.
  VERIFY — claim-by-claim entailment check via an independent verifier (§4a.3).
           Below-threshold claims are dropped; if too many are dropped the answer
           is flagged low-confidence or the Brain abstains (§4a.4).

The Brain hands the caller a BrainResult with:
  - a prose answer rendered from verified claims
  - the structured claims (with citations) for the Trust UI
  - coverage ledger (docs_routed / docs_read / docs_relevant / docs_failed)
  - confidence score

Non-regression guarantee:
  The Brain is ONLY called for collection queries above ROUTING_MAX_FANOUT or
  with explicit synthesis intent.  Single-doc and simple queries continue to use
  the T0 fast path (generate() in generation.py) — byte-for-byte unchanged.
"""

from __future__ import annotations

import json
import time
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, Optional, Callable

from langchain_core.documents import Document

from src.components.config import Config
from src.components.brain.claims import (
    Claim, EvidenceSpan, PerDocExtract, BrainResult,
)
from src.components.brain.verifier import verify_claims, ABSTAIN_THRESHOLD
from src.logger import get_logger

logger = get_logger(__name__)


# ── Default concurrency / quorum ──────────────────────────────────────────────
MAP_CONCURRENCY = 8      # parallel MAP workers (OpenAI rate-limit safe)
MAP_QUORUM = 0.90        # proceed to REDUCE if ≥ 90% of docs MAP'd successfully

# Confidence below this → Brain abstains rather than guessing
BRAIN_ABSTAIN_THRESHOLD = 0.45


# ── Prompt builders ───────────────────────────────────────────────────────────
# IMPORTANT: these are plain string builders, NOT LangChain ChatPromptTemplates.
# The prompts contain literal JSON examples (e.g. {"claim": ...}) and ChatPromptTemplate
# would mis-parse those braces as template variables and crash every call. We build
# the final strings here (interpolating only real variables) and pass them straight
# to llm.invoke([SystemMessage, HumanMessage]).

def _map_messages(question: str, doc_id: str, filename: str, context: str) -> tuple[str, str]:
    system = (
        "You are a precise research assistant extracting evidence from a document.\n"
        "Your job: find ALL facts relevant to the question and express each as a separate claim.\n\n"
        "Rules:\n"
        "- Each claim must be one sentence.\n"
        "- Every claim MUST include a verbatim quote (exact words) from the document proving it.\n"
        "- If the document contains nothing relevant, respond with exactly: NOTHING_RELEVANT\n"
        "- Do NOT add information from outside the document.\n"
        "- Output ONLY a JSON array (no prose, no markdown fences) in this exact shape:\n"
        '  [{"claim": "...", "verbatim_span": "...", "confidence": 0.0-1.0}]\n\n'
        f"Question: {question}"
    )
    user = (
        f"Document [{doc_id} | {filename}]:\n{context}\n\n"
        "Extract relevant claims (or respond NOTHING_RELEVANT):"
    )
    return system, user


def _reduce_messages(question: str, n_docs: int, extracts: str) -> tuple[str, str]:
    system = (
        "You are a senior analyst synthesizing research findings into a comprehensive answer.\n\n"
        "You will receive extracted claims grouped by source document. Your task:\n"
        "1. Synthesize them into a coherent answer.\n"
        "2. After each claim, cite the supporting document using EXACTLY this format: "
        "[Source: <filename>] — copy the filename verbatim from the source header above the claims.\n"
        "3. Highlight contradictions or conflicts between documents explicitly.\n"
        "4. If evidence is thin or uncertain on a point, say so — do NOT guess.\n"
        "5. Do NOT add information not present in the claims below.\n"
        "6. When the claims contain the numbers needed for a calculation the question asks for, "
        "perform the arithmetic and show it.\n"
        "7. When the answer compares items across two or more dimensions (e.g. metrics across "
        "documents, clauses across agreements, values vs thresholds), present that comparison as a "
        "GitHub-flavored Markdown table: a header row, a `|---|` separator row, then one row per "
        "item. The app renders these as interactive sortable tables, so prefer a table over bullets "
        "whenever the data is naturally tabular. Keep cells short; put the [Source: <filename>] "
        "citations in the prose around the table, not inside cells.\n\n"
        "Output format:\n"
        "- A direct, well-structured answer with [Source: <filename>] inline citations, using a "
        "Markdown table for any multi-dimensional comparison.\n"
        '- End with a "## Confidence" section: overall confidence 0.0-1.0 and a one-line reason.\n\n'
        f"Question: {question}"
    )
    user = (
        f"Extracted claims from {n_docs} document(s):\n\n{extracts}\n\n"
        "Synthesize a comprehensive answer:"
    )
    return system, user


def _clean_snippet(text: str, limit: int = 300) -> str:
    """Make a verbatim span presentable: collapse whitespace/newlines (PDF table
    extraction leaves '\\n\\n' artifacts like 'Total revenue\\n\\n211,915'), strip,
    and cap length on a word boundary so the citation reads cleanly to the user.
    """
    if not text:
        return ""
    cleaned = " ".join(text.split())  # collapses all runs of whitespace/newlines
    if len(cleaned) > limit:
        cleaned = cleaned[:limit].rsplit(" ", 1)[0] + "…"
    return cleaned


def _clean_page(page):
    """Display page numbers as integers: 35.0 -> 35, keep 'N/A'/strings as-is."""
    try:
        f = float(page)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return page


def _context_snippet(span: str, chunk_text: str, window: int = 140) -> str:
    """Render a readable citation: show the verbatim span inside its surrounding
    sentence/context from the source chunk, instead of a bare fragment like
    '$ (4,051)'. Falls back to the cleaned span if context isn't available.
    """
    span_clean = " ".join((span or "").split())
    text_clean = " ".join((chunk_text or "").split())
    if not text_clean or not span_clean:
        return _clean_snippet(span)
    probe = span_clean[:40].lower()
    idx = text_clean.lower().find(probe)
    if idx == -1:
        return _clean_snippet(span)
    start = max(0, idx - window)
    end = min(len(text_clean), idx + len(span_clean) + window)
    snippet = text_clean[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(text_clean):
        snippet = snippet + "…"
    return snippet


def _locate_evidence(span: str, chunks: list, fallback_doc_id: str) -> tuple[str, str]:
    """Find which chunk a verbatim span came from.

    Returns (chunk_id, doc_id).  Falls back to the first chunk if the span can't
    be matched (e.g. the model lightly paraphrased it).
    """
    if span and chunks:
        probe = span.strip()[:60].lower()
        if probe:
            for c in chunks:
                content = (c.page_content or "").lower()
                if probe in content:
                    return (
                        c.metadata.get("chunk_id", "") or "",
                        c.metadata.get("doc_id", fallback_doc_id),
                    )
    first = chunks[0] if chunks else None
    if first is not None:
        return (
            first.metadata.get("chunk_id", "") or "",
            first.metadata.get("doc_id", fallback_doc_id),
        )
    return "", fallback_doc_id


class Brain:
    """Stage-2 map-reduce synthesis Brain.

    Usage:
        brain = Brain(config)
        result = brain.run(query, routed_docs, retrieval_mgr)
        # or streaming:
        for event in brain.run_stream(query, routed_docs, retrieval_mgr):
            yield event  # SSE events: map_progress, reduce, verify, done
    """

    def __init__(self, config: Config):
        self.config = config
        self._map_llm = None    # cheap model for MAP (gpt-4o-mini)
        self._reduce_llm = None  # larger model for REDUCE (gpt-4o or claude)
        self._verify_llm = None  # independent model for VERIFY

    def _get_map_llm(self):
        if self._map_llm is None:
            from langchain_openai import ChatOpenAI
            self._map_llm = ChatOpenAI(
                model=self.config.LLM_MODEL_NAME,  # gpt-4o-mini by default
                temperature=0.0,
                api_key=self.config.OPENAI_API_KEY,
                request_timeout=45,
            )
        return self._map_llm

    def _get_reduce_llm(self):
        if self._reduce_llm is None:
            from langchain_openai import ChatOpenAI
            # REDUCE uses a stronger model; fall back to the same model if not configured
            reduce_model = getattr(self.config, "REDUCE_LLM_MODEL", None) or "gpt-4o"
            self._reduce_llm = ChatOpenAI(
                model=reduce_model,
                temperature=0.1,
                api_key=self.config.OPENAI_API_KEY,
                request_timeout=90,
            )
        return self._reduce_llm

    def _get_verify_llm(self):
        if self._verify_llm is None:
            from langchain_openai import ChatOpenAI
            # Verifier uses a different model from REDUCE to de-correlate errors (§4a.3)
            verify_model = getattr(self.config, "VERIFY_LLM_MODEL", None) or "gpt-4o-mini"
            self._verify_llm = ChatOpenAI(
                model=verify_model,
                temperature=0.0,
                api_key=self.config.OPENAI_API_KEY,
                request_timeout=30,
            )
        return self._verify_llm

    # ── MAP step ──────────────────────────────────────────────────────────────

    def _map_single_doc(
        self,
        query: str,
        doc_id: str,
        filename: str,
        chunks: list[Document],
    ) -> PerDocExtract:
        """Extract claims from a single document's chunks."""
        if not chunks:
            return PerDocExtract(doc_id=doc_id, filename=filename, nothing_relevant=True)

        context = "\n---\n".join(
            f"[chunk {c.metadata.get('chunk_id', i)}]\n{c.page_content}"
            for i, c in enumerate(chunks)
        )

        try:
            from langchain_core.messages import SystemMessage, HumanMessage

            system, user = _map_messages(query, doc_id, filename, context)
            raw = self._get_map_llm().invoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            ).content

            raw = (raw or "").strip()
            if raw.upper() == "NOTHING_RELEVANT":
                return PerDocExtract(doc_id=doc_id, filename=filename, nothing_relevant=True)

            # Parse JSON claims
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            items = json.loads(raw)

            claims = []
            for item in items:
                claim_text = item.get("claim", "").strip()
                span = item.get("verbatim_span", "").strip()
                conf = float(item.get("confidence", 0.8))
                if not claim_text:
                    continue
                # Attribute the evidence to the chunk the span actually came from,
                # not blindly to chunk[0] — so the Trust UI can highlight the span
                # in the right source chunk.
                ev_chunk_id, ev_doc_id = _locate_evidence(span, chunks, doc_id)
                claims.append(Claim(
                    text=claim_text,
                    evidence=[EvidenceSpan(
                        doc_id=ev_doc_id,
                        chunk_id=ev_chunk_id,
                        verbatim_span=span,
                    )] if span else [],
                    confidence=conf,
                    derivation="extracted",
                ))

            logger.info(
                "[Brain MAP] %s: %d claims extracted from %d chunks",
                filename, len(claims), len(chunks),
            )
            return PerDocExtract(
                doc_id=doc_id,
                filename=filename,
                claims=claims,
                nothing_relevant=len(claims) == 0,
            )

        except Exception as exc:
            logger.warning("[Brain MAP] Failed for %s: %s", filename, exc)
            return PerDocExtract(doc_id=doc_id, filename=filename, error=str(exc))

    def _map_all_docs(
        self,
        query: str,
        doc_chunks: dict[str, tuple[str, list[Document]]],
        on_progress: Optional[Callable[[str, PerDocExtract], None]] = None,
    ) -> list[PerDocExtract]:
        """Run MAP over all docs in parallel (bounded by MAP_CONCURRENCY).

        doc_chunks: {doc_id: (filename, [chunks])}
        on_progress: callback(doc_id, extract) called as each doc completes (for SSE)
        """
        results: list[PerDocExtract] = []
        concurrency = min(MAP_CONCURRENCY, len(doc_chunks))

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    self._map_single_doc,
                    query, doc_id, filename, chunks,
                ): doc_id
                for doc_id, (filename, chunks) in doc_chunks.items()
            }
            for future in as_completed(futures):
                doc_id = futures[future]
                try:
                    extract = future.result()
                except Exception as exc:
                    doc_id_val = doc_id
                    filename = doc_chunks[doc_id_val][0]
                    extract = PerDocExtract(doc_id=doc_id_val, filename=filename, error=str(exc))
                results.append(extract)
                if on_progress:
                    on_progress(doc_id, extract)

        return results

    # ── REDUCE step ───────────────────────────────────────────────────────────

    def _reduce(
        self,
        query: str,
        extracts: list[PerDocExtract],
    ) -> tuple[str, float, list[Claim]]:
        """Synthesize per-doc extracts into one answer.

        Returns: (prose_answer, confidence, synthesized_claims)
        """
        relevant = [e for e in extracts if not e.nothing_relevant and not e.error]
        if not relevant:
            return (
                "I couldn't find relevant information across the documents for this question.",
                0.0,
                [],
            )

        # Build the extracts block
        extracts_text_parts = []
        for ext in relevant:
            claims_text = "\n".join(
                f"  - {c.text}" + (f' [evidence: "{c.evidence[0].verbatim_span[:120]}"]' if c.evidence else "")
                for c in ext.claims
            )
            extracts_text_parts.append(
                f"[Source: {ext.filename}]\n{claims_text or '  (no specific claims)'}"
            )
        extracts_text = "\n\n".join(extracts_text_parts)

        try:
            from langchain_core.messages import SystemMessage, HumanMessage

            system, user = _reduce_messages(query, len(relevant), extracts_text)
            raw_answer = self._get_reduce_llm().invoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            ).content
            raw_answer = raw_answer or ""

            # Parse confidence from answer
            confidence = 0.7  # default
            answer = raw_answer
            if "## Confidence" in raw_answer:
                parts = raw_answer.rsplit("## Confidence", 1)
                answer = parts[0].strip()
                conf_text = parts[1].strip()
                import re
                m = re.search(r"(\d+\.?\d*)", conf_text)
                if m:
                    val = float(m.group(1))
                    confidence = val if val <= 1.0 else val / 100.0

            # Build synthesized claims from all per-doc claims (for citation UI)
            all_claims = []
            for ext in relevant:
                all_claims.extend(ext.claims)

            return answer, confidence, all_claims

        except Exception as exc:
            logger.error("[Brain REDUCE] Failed: %s", exc)
            # Degraded: return per-doc summaries as fallback answer
            fallback = "\n\n".join(
                f"**{ext.filename}:** " + " ".join(c.text for c in ext.claims[:3])
                for ext in relevant[:5]
            )
            return fallback, 0.4, []

    def _group_claims_by_doc(
        self,
        claims: list[Claim],
        doc_chunks: dict[str, tuple[str, list[Document]]],
    ) -> list[PerDocExtract]:
        """Regroup a flat list of claims back into per-doc extracts (by evidence doc_id).

        Used to feed REDUCE only the VERIFIED claims, grouped by source document.
        """
        by_doc: dict[str, list[Claim]] = {}
        for c in claims:
            did = c.evidence[0].doc_id if c.evidence else "unknown"
            by_doc.setdefault(did, []).append(c)
        out: list[PerDocExtract] = []
        for did, cl in by_doc.items():
            filename = doc_chunks.get(did, (did, None))[0]
            out.append(PerDocExtract(doc_id=did, filename=filename, claims=cl))
        return out

    def _build_sources(
        self,
        verified_claims: list[Claim],
        doc_chunks: dict[str, tuple[str, list[Document]]],
    ) -> list[dict]:
        """Build the sources list from verified claims only.

        Matches the exact shape emitted by generate_stream() — source_id, filename,
        page, chunk_type, chunk_id, content — so the existing frontend citation UI
        renders Brain sources with no special-casing.
        """
        sources: list[dict] = []
        seen: set = set()
        sid = 1
        for claim in verified_claims:
            for ev in claim.evidence:
                key = (ev.doc_id, ev.chunk_id)
                if key in seen:
                    continue
                seen.add(key)
                filename, chunks = doc_chunks.get(ev.doc_id, (ev.doc_id, []))
                page, chunk_type, chunk_text = "N/A", "text", ""
                for c in (chunks or []):
                    if str(c.metadata.get("chunk_id", "")) == str(ev.chunk_id):
                        page = c.metadata.get("page_number", "N/A")
                        chunk_type = c.metadata.get("chunk_type", "text")
                        chunk_text = c.page_content or ""
                        break
                sources.append({
                    "source_id": sid,
                    "filename": filename,
                    "page": _clean_page(page),
                    "chunk_type": chunk_type,
                    "chunk_id": ev.chunk_id,
                    "content": _context_snippet(ev.verbatim_span, chunk_text),
                })
                sid += 1
        return sources

    # ── Full Brain run ────────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        doc_chunks: dict[str, tuple[str, list[Document]]],
        user_id: Optional[str] = None,
        collection_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> BrainResult:
        """Run the full MAP → VERIFY → REDUCE pipeline.

        Order matters (§4a.2): claims are verified BEFORE synthesis so the prose
        is rendered from verified claims only — never from claims an entailment
        check would have dropped.

        Args:
            query:      User query.
            doc_chunks: {doc_id: (filename, [Document chunks])} from the retrieval layer.

        Returns:
            BrainResult with answer, claims, confidence, coverage ledger.
        """
        docs_routed = len(doc_chunks)
        t0 = time.perf_counter()

        # ── MAP ──────────────────────────────────────────────────────────────
        logger.info("[Brain] MAP start: %d docs", docs_routed)
        extracts = self._map_all_docs(query, doc_chunks)

        docs_read = len([e for e in extracts if e.error is None])
        docs_relevant = len([e for e in extracts if not e.nothing_relevant and not e.error])
        docs_failed = len([e for e in extracts if e.error is not None])

        # Quorum check (§6.3)
        if docs_read < docs_routed * MAP_QUORUM:
            logger.warning(
                "[Brain] MAP quorum not met: %d/%d succeeded (need %.0f%%)",
                docs_read, docs_routed, MAP_QUORUM * 100,
            )

        # ── VERIFY (before REDUCE — §4a.2) ─────────────────────────────────────
        all_claims = [c for e in extracts if e.error is None for c in e.claims]
        logger.info("[Brain] VERIFY start: %d claims", len(all_claims))
        verified_claims, dropped_claims = verify_claims(all_claims, self._get_verify_llm())
        if dropped_claims:
            logger.info(
                "[Brain] VERIFY: %d/%d claims dropped (below threshold)",
                len(dropped_claims), len(all_claims),
            )

        # ── REDUCE (synthesise prose from verified claims only) ────────────────
        if not verified_claims:
            answer = (
                "I couldn't verify any claims against the source documents for this "
                "question, so I can't give a grounded answer."
            )
            confidence = 0.0
        else:
            logger.info("[Brain] REDUCE start: %d verified claims", len(verified_claims))
            verified_extracts = self._group_claims_by_doc(verified_claims, doc_chunks)
            answer, confidence, _ = self._reduce(query, verified_extracts)

        # Abstain if confidence is too low (§4a.4)
        abstained = False
        abstain_reason = None
        if confidence < BRAIN_ABSTAIN_THRESHOLD:
            abstained = True
            abstain_reason = (
                f"Brain confidence {confidence:.2f} below abstention threshold "
                f"{BRAIN_ABSTAIN_THRESHOLD}.  Insufficient evidence to give a reliable answer."
            )
            answer = (
                "I can only partially answer this question based on the available documents. "
                f"Here is what I found, but please treat it as preliminary:\n\n{answer}"
            )

        # Build sources list (same shape as generate() for API compatibility)
        sources = self._build_sources(verified_claims, doc_chunks)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "[Brain] Done: routed=%d read=%d relevant=%d failed=%d "
            "claims=%d confidence=%.2f elapsed=%dms",
            docs_routed, docs_read, docs_relevant, docs_failed,
            len(verified_claims), confidence, elapsed_ms,
        )

        result = BrainResult(
            answer=answer,
            claims=verified_claims,
            confidence=confidence,
            abstained=abstained,
            abstain_reason=abstain_reason,
            sources=sources,
            docs_routed=docs_routed,
            docs_read=docs_read,
            docs_relevant=docs_relevant,
            docs_failed=docs_failed,
            per_doc_extracts=extracts,
        )

        self._record_ledger(
            result, query, user_id, collection_id, conversation_id, elapsed_ms,
        )
        return result

    @staticmethod
    def _record_ledger(result, query, user_id, collection_id, conversation_id, wall_ms):
        """Best-effort coverage-ledger write (§3.2).  Never raises."""
        try:
            from src.components.brain.ledger import record_run
            record_run(
                user_id=user_id,
                collection_id=collection_id,
                conversation_id=conversation_id,
                query_text=query,
                run_type="map_reduce",
                docs_routed=result.docs_routed,
                docs_read=result.docs_read,
                docs_relevant=result.docs_relevant,
                docs_failed=result.docs_failed,
                confidence=result.confidence,
                wall_ms=wall_ms,
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("[Brain] ledger write failed (non-fatal): %s", exc)

    # ── Streaming run (SSE step events) ──────────────────────────────────────

    def run_stream(
        self,
        query: str,
        doc_chunks: dict[str, tuple[str, list[Document]]],
        user_id: Optional[str] = None,
        collection_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Iterator[str]:
        """Generator yielding SSE events as the Brain works.

        Pipeline order is MAP → VERIFY → REDUCE (§4a.2: prose is synthesised from
        verified claims only).  MAP progress is streamed live, per document, as each
        finishes.

        Events emitted (JSON, same wire format as the existing chat stream):
          {"type": "brain_start",    "docs_routed": N}
          {"type": "brain_map",      "filename": ..., "claims": N, "relevant": bool, "progress": "k/N"}
          {"type": "brain_verify",   "claims_total": N, "claims_verified": N}
          {"type": "brain_reduce",   "docs_relevant": N}
          {"type": "sources",        "sources": [...]}   ← same as fast path
          {"type": "token",          "content": "..."}   ← streamed answer tokens
          {"type": "brain_meta",     "confidence": ..., "abstained": ..., "coverage": {...}}
          [DONE]
        """
        import json as _json

        t0 = time.perf_counter()
        docs_routed = len(doc_chunks)
        yield f"data: {_json.dumps({'type': 'brain_start', 'docs_routed': docs_routed})}\n\n"

        # ── MAP with live per-doc progress (driven directly so we can yield) ────
        extracts: list[PerDocExtract] = []
        concurrency = max(1, min(MAP_CONCURRENCY, len(doc_chunks)))
        done = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(self._map_single_doc, query, doc_id, filename, chunks): doc_id
                for doc_id, (filename, chunks) in doc_chunks.items()
            }
            for future in as_completed(futures):
                doc_id = futures[future]
                try:
                    ext = future.result()
                except Exception as exc:
                    ext = PerDocExtract(
                        doc_id=doc_id, filename=doc_chunks[doc_id][0], error=str(exc)
                    )
                extracts.append(ext)
                done += 1
                yield f"data: {_json.dumps({'type': 'brain_map', 'filename': ext.filename, 'claims': len(ext.claims), 'relevant': not ext.nothing_relevant and not ext.error, 'progress': f'{done}/{docs_routed}'})}\n\n"

        docs_read = len([e for e in extracts if e.error is None])
        docs_relevant = len([e for e in extracts if not e.nothing_relevant and not e.error])
        docs_failed = len([e for e in extracts if e.error is not None])

        # ── VERIFY (before REDUCE) ─────────────────────────────────────────────
        all_claims = [c for e in extracts if e.error is None for c in e.claims]
        verified_claims, _dropped = verify_claims(all_claims, self._get_verify_llm())
        yield f"data: {_json.dumps({'type': 'brain_verify', 'claims_total': len(all_claims), 'claims_verified': len(verified_claims)})}\n\n"

        # ── REDUCE from verified claims only ───────────────────────────────────
        if not verified_claims:
            answer, confidence = (
                "I couldn't verify any claims against the source documents for this "
                "question, so I can't give a grounded answer.",
                0.0,
            )
        else:
            answer, confidence, _ = self._reduce(
                query, self._group_claims_by_doc(verified_claims, doc_chunks)
            )
        yield f"data: {_json.dumps({'type': 'brain_reduce', 'docs_relevant': docs_relevant})}\n\n"

        abstained = confidence < BRAIN_ABSTAIN_THRESHOLD
        if abstained:
            answer = (
                "I can only partially answer this question based on the available documents. "
                f"Here is what I found, but please treat it as preliminary:\n\n{answer}"
            )

        sources = self._build_sources(verified_claims, doc_chunks)
        yield f"data: {_json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # Stream the answer. Markdown tables only parse correctly when each table
        # row arrives as a whole line, so we stream line-by-line: words within a
        # line stream for the live "typing" feel, but every newline is preserved
        # and flushed intact — keeping the `| col | col |` rows un-shattered so the
        # frontend renders them as proper sortable tables (not broken fragments).
        for line in answer.splitlines(keepends=True):
            if line.strip().startswith("|"):
                # Table row — emit the whole line atomically so GFM can parse it.
                yield f"data: {_json.dumps({'type': 'token', 'content': line})}\n\n"
            else:
                # Prose — stream word-by-word, preserving the trailing newline.
                stripped = line.rstrip("\n")
                newline = line[len(stripped):]
                for word in stripped.split(" "):
                    yield f"data: {_json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
                if newline:
                    yield f"data: {_json.dumps({'type': 'token', 'content': newline})}\n\n"

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        yield f"data: {_json.dumps({'type': 'brain_meta', 'confidence': confidence, 'abstained': abstained, 'coverage': {'docs_routed': docs_routed, 'docs_read': docs_read, 'docs_relevant': docs_relevant, 'docs_failed': docs_failed}})}\n\n"

        result = BrainResult(
            answer=answer, claims=verified_claims, confidence=confidence,
            abstained=abstained, sources=sources, docs_routed=docs_routed,
            docs_read=docs_read, docs_relevant=docs_relevant, docs_failed=docs_failed,
        )
        self._record_ledger(
            result, query, user_id, collection_id, conversation_id, elapsed_ms,
        )
        yield "data: [DONE]\n\n"
