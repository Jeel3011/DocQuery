"""F-A standing coverage gate — every mutating / cross-user route declares a cap (offline, $0).

The committed gate for plans/tool_hard.md §F-A. It does NOT test the authz DECISION (that is
`eval/test_authz.py`) — it tests that the decision is actually WIRED onto every route that needs
it. The failure class it guards against is the one the F-A forensics found: a NEW route that
mutates or reads cross-user data ships WITHOUT a `require_cap(verb)` dependency, opening a silent
hole the offline authz gate can't see (it only tests routes it's told about). Here, a new un-capped
mutating route is a RED test, not a silent gap.

HOW (deliberately static, no app import): importing the FastAPI app pulls in langchain /
transformers (heavy, slow, $). This gate instead parses each `src/api/routes/*.py` file's AST and,
for every `@router.<method>`-decorated handler, asserts it EITHER declares a `require_cap(...)`
dependency in its signature OR is on the EXPLICIT_ALLOWLIST below (with a written reason). Caps are
always declared as `… = Depends(require_cap("verb"))` params (verified across all 14 route files —
no router-level `dependencies=[…]` style is used), so a signature scan is faithful to what fires on
the request path.

  - MUTATING method = POST / PUT / PATCH / DELETE → must be capped or allow-listed.
  - GET that reads CROSS-USER / CROSS-FIRM data → must be capped or allow-listed. (Plain
    per-user GETs — list your own conversations/documents — are isolated by `.eq(user_id)` + RLS
    and need only authentication; they are NOT in scope here.)

The ALLOWLIST is the *justified* set: auth (signup/login can't require a firm cap — you have no
firm yet), health, and the PERSONAL per-user conversation CRUD (user_id-scoped, never firm/vault
data). Each entry carries its reason so removing a cap is a deliberate, reviewed act.

Run:  python -u eval/test_route_authz_coverage.py

> LIVE gate this does NOT replace (offline-green ≠ done — tool_hard.md governing rule): this proves
> the dependency is DECLARED, not that it FIRES. Jeel must still, on the real stack, hit each
> newly-capped endpoint as a lesser-role user (e.g. a `client`/`guest` for the `ask`-gated query
> endpoints; an `associate` lacking `ingest` for the connector imports — or whatever role lacks the
> verb) and confirm a real 403. The newly-capped endpoints to verify live:
>   chat.py:        POST /query, /query/stream, /query/agent, /query/agent/stream,
>                   /query/brain/stream, /conversations/{id}/messages   → cap "ask"
>   connectors.py:  POST /connectors/google-drive/import, /connectors/email/import → cap "ingest"
>   document_comparison.py: POST /documents/compare → cap "ask"
> A real 403 for a verb-lacking role on each = F-A live-closed. (See tool_hard.md §F-A + §6.1.)

OFFLINE: pure AST parse, no Supabase, no app, no LLM, no network. Touches neither extraction nor
the kernel — run only the security/authz suite for this slice ([[run-only-relevant-gates]]).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROUTES_DIR = Path(__file__).resolve().parent.parent / "src" / "api" / "routes"

_MUTATING = {"post", "put", "patch", "delete"}

# ─────────────────────────────────────────
# EXPLICIT ALLOW-LIST — routes that mutate / read but legitimately carry NO firm capability.
# Keyed by (filename, function_name) → the WRITTEN reason. Anything not here that mutates / reads
# cross-user data MUST declare require_cap(...) or this gate goes red.
# ─────────────────────────────────────────
EXPLICIT_ALLOWLIST: dict[tuple[str, str], str] = {
    # ── auth.py — pre-firm / self-service. signup/login/refresh happen BEFORE the caller has a
    # firm/role, so a firm cap is impossible and incorrect (the F-A forensics' justified carve-out).
    # accept_invite assigns the INVITED role server-side (db.accept_invite — the joiner can never
    # self-escalate), so it is self-scoped, not a privilege grant the caller controls.
    ("auth.py", "signup"): "pre-firm: no membership exists yet to cap against",
    ("auth.py", "login"): "pre-firm: authenticates to MINT the session; nothing to cap",
    ("auth.py", "logout"): "ends the caller's OWN session; no cross-user effect",
    ("auth.py", "accept_invite"): "joins at the SERVER-assigned invited role (no self-escalation)",
    ("auth.py", "update_preferences"): "edits the caller's OWN preferences (user_id-scoped)",
    ("auth.py", "my_firm"): "reads the caller's OWN firm membership (user_id-scoped)",
    # ── admin.py — a roster READ any member may do (mutations promote/demote/remove ARE cap-gated,
    # verified above). Server-filtered to the caller's own firm (T2).
    ("admin.py", "list_firm_members"): "roster read any member may do; mutations are cap-gated",
    # ── audit.py — the caller's OWN audit log (user_id-scoped read).
    ("audit.py", "get_audit_log"): "the caller's own audit log (user_id-scoped)",
    # ── collections.py — scan_conflicts is a metadata-only pre-create ethical-wall screen (no
    # vault content, no persisted mutation of a shared resource).
    ("collections.py", "scan_conflicts"): "metadata-only conflict pre-screen; no vault mutation",
    # ── dpdp.py — DPDP data-principal SELF-SERVICE (§12 erase / §13 grievance). These are the
    # caller exercising their OWN statutory rights; gating them behind a firm verb would be WRONG
    # (a principal must always be able to act on their own data). Firm resolved server-side (T3).
    ("dpdp.py", "erase_my_data"): "DPDP §12 self-service on the caller's OWN data (must not be cap-gated)",
    ("dpdp.py", "file_grievance"): "DPDP §13 self-service; caller files their own grievance",
    # ── notifications.py — the caller's OWN rows (server-scoped to user_id; a guessed id touches
    # nothing — T2/T8).
    ("notifications.py", "mark_read"): "caller's own notification rows (user_id-scoped)",
    ("notifications.py", "set_preferences"): "caller's own preference row (user_id-scoped)",
    # ── chat.py — conversations are PERSONAL, per-user threads (user_id-scoped, never firm/vault-
    # stamped — see [[f2f-rls-backstop-live-schema]]). They carry no cross-user data; the route's
    # .eq(user_id) + RLS is the isolation. (The message-SEND route IS capped — it runs the agent.)
    ("chat.py", "create_conversation"): "personal per-user thread (user_id-scoped, no firm data)",
    ("chat.py", "delete_conversation"): "personal per-user thread (user_id-scoped, no firm data)",
    ("chat.py", "rename_conversation"): "personal per-user thread (user_id-scoped, no firm data)",
}

# KNOWN DEFERRALS — mutations on a FIRM-SHARED resource that are currently isolated only by the
# data layer's `.eq(user_id)`, OUTSIDE F-A's 3-file scope (chat/connectors/document_comparison).
# They are NOT "safe-and-self-scoped"; they are open items a later F2 slice should cap (e.g.
# documents.py::update_document's OWN docstring already says "F2 will partner-gate WHO may set
# [privileged]"). Listed SEPARATELY from the allow-list so the gate stays green while the debt is
# tracked honestly and visibly — not buried under a false "no cap needed" reason.
KNOWN_DEFERRALS: dict[tuple[str, str], str] = {
    ("collections.py", "update_collection"):
        "renames/retypes a matter; sibling of delete (capped) — cap in a later F2 slice",
    ("documents.py", "update_document"):
        "sets the privileged flag; its docstring defers 'WHO may set this' to F2 — cap then",
}

# GET handlers whose NAME signals they read across users/firms (so they need a cap even though GET).
# Kept conservative: most GETs are per-user reads (RLS-isolated) and need only auth. We flag the
# ones that read the FIRM's shared surface. (None are un-capped today — this list is the tripwire
# for a future cross-firm GET that forgets its cap.)
_CROSS_USER_GET_HINTS = ("firm", "members", "billing", "audit_log", "all_")


def _decorator_methods(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return the HTTP methods this handler is decorated with (e.g. ['post'])."""
    methods = []
    for dec in func.decorator_list:
        # @router.post("/x")  → dec is a Call whose func is an Attribute .post on a Name `router`
        call = dec if isinstance(dec, ast.Call) else None
        target = call.func if call else dec
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            if target.value.id == "router" and target.attr.lower() in (
                _MUTATING | {"get"}
            ):
                methods.append(target.attr.lower())
    return methods


def _declares_require_cap(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if any parameter default is `Depends(require_cap(...))`."""
    defaults = list(func.args.defaults) + list(func.args.kw_defaults)
    for d in defaults:
        if d is None:
            continue
        # Depends(require_cap("verb"))  → Call(Depends)[ Call(require_cap)[...] ]
        for node in ast.walk(d):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "require_cap"
            ):
                return True
    return False


def _audit_file(path: Path) -> list[tuple[str, str, str]]:
    """Return a list of (kind, func_name, detail) findings for one route file.
    kind ∈ {'UNCAPPED_MUTATING', 'UNCAPPED_CROSS_GET'}."""
    tree = ast.parse(path.read_text())
    findings: list[tuple[str, str, str]] = []
    fname = path.name
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        methods = _decorator_methods(node)
        if not methods:
            continue
        if (fname, node.name) in EXPLICIT_ALLOWLIST:
            continue
        if (fname, node.name) in KNOWN_DEFERRALS:
            continue  # tracked debt — reported as a WARN below, never a silent pass
        if _declares_require_cap(node):
            continue
        mutating = [m for m in methods if m in _MUTATING]
        if mutating:
            findings.append(
                ("UNCAPPED_MUTATING", node.name, f"{','.join(mutating).upper()} (no require_cap)")
            )
            continue
        # GET-only: flag only if the name hints at a cross-user/firm read.
        if any(h in node.name.lower() for h in _CROSS_USER_GET_HINTS):
            findings.append(
                ("UNCAPPED_CROSS_GET", node.name, "cross-user/firm GET (no require_cap)")
            )
    return findings


def _all_route_handlers() -> set[tuple[str, str]]:
    """(filename, func_name) for every @router.<method>-decorated handler across the routes dir."""
    handlers: set[tuple[str, str]] = set()
    for path in ROUTES_DIR.glob("*.py"):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _decorator_methods(node):
                handlers.add((path.name, node.name))
    return handlers


def _assert_no_stale_entries() -> None:
    """Every allow-list / deferral key must name a REAL handler. A stale entry (renamed/deleted
    route) would silently excuse a future route that happens to reuse the name — so it is a red
    test, keeping the lists honest."""
    real = _all_route_handlers()
    stale_allow = [k for k in EXPLICIT_ALLOWLIST if k not in real]
    stale_defer = [k for k in KNOWN_DEFERRALS if k not in real]
    assert not stale_allow, f"stale EXPLICIT_ALLOWLIST entries (no such handler): {stale_allow}"
    assert not stale_defer, f"stale KNOWN_DEFERRALS entries (no such handler): {stale_defer}"


def main() -> int:
    assert ROUTES_DIR.is_dir(), f"routes dir not found: {ROUTES_DIR}"
    route_files = sorted(p for p in ROUTES_DIR.glob("*.py") if p.name != "__init__.py")
    assert route_files, "no route files discovered"
    _assert_no_stale_entries()

    all_findings: dict[str, list[tuple[str, str, str]]] = {}
    total = 0
    for path in route_files:
        findings = _audit_file(path)
        if findings:
            all_findings[path.name] = findings
            total += len(findings)

    # Report
    print("=" * 72)
    print("F-A route-authz coverage gate")
    print(f"  scanned {len(route_files)} route files; "
          f"{len(EXPLICIT_ALLOWLIST)} explicit allow-list entries; "
          f"{len(KNOWN_DEFERRALS)} tracked deferrals")
    print("=" * 72)

    if KNOWN_DEFERRALS:
        print("WARN — tracked deferrals (mutations isolated only by .eq(user_id), to cap in a "
              "later F2 slice; NOT a clean pass):")
        for (fn, func), reason in KNOWN_DEFERRALS.items():
            print(f"  [DEFERRED] {fn}::{func} — {reason}")
        print("-" * 72)

    if not all_findings:
        print("PASS — every mutating / cross-user route declares require_cap(...) or is "
              "explicitly allow-listed with a reason.")
        # Positive assertion: confirm the F-A target routes are actually capped now (a guard
        # against the allow-list silently swallowing a route it shouldn't).
        _assert_target_capped()
        print("PASS — the F-A target endpoints (chat query path, connectors, compare) are capped.")
        return 0

    print("FAIL — uncapped routes found (add require_cap(verb) or justify onto EXPLICIT_ALLOWLIST):")
    for fname, findings in all_findings.items():
        for kind, func, detail in findings:
            print(f"  [{kind}] {fname}::{func} — {detail}")
    print(f"\n{total} uncapped route(s).")
    return 1


def _assert_target_capped() -> None:
    """Belt-and-suspenders: the exact endpoints F-A closed must parse as capped. If a refactor
    drops one of these caps, this fails even if it also (wrongly) got allow-listed."""
    targets = {
        "chat.py": {"query", "query_stream", "query_agent", "agent_query_stream",
                    "brain_query_stream", "send_message"},
        "connectors.py": {"import_google_drive_folder", "import_email_attachments"},
        "document_comparison.py": {"compare_documents"},
    }
    for fname, funcs in targets.items():
        tree = ast.parse((ROUTES_DIR / fname).read_text())
        capped = {
            n.name
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and _declares_require_cap(n)
        }
        missing = funcs - capped
        assert not missing, f"F-A regression: {fname} endpoints lost their cap: {sorted(missing)}"


if __name__ == "__main__":
    sys.exit(main())
