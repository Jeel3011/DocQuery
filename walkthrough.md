# FastAPI Backend Server — Walkthrough

## What Was Built

A complete FastAPI REST API wrapping DocQuery's existing RAG pipeline — **8 new files, 16 endpoints, zero modifications to existing components**.

### New Files

| File | Purpose |
|---|---|
| [server.py](file:///Users/jeelthummar/Desktop/DocQuery/src/api/server.py) | FastAPI app, CORS, lifespan, route includes |
| [dependencies.py](file:///Users/jeelthummar/Desktop/DocQuery/src/api/dependencies.py) | DI layer: config singleton, Bearer auth, user-scoped RAG components |
| [schemas.py](file:///Users/jeelthummar/Desktop/DocQuery/src/api/schemas.py) | 16 Pydantic models for request/response validation |
| [routes/health.py](file:///Users/jeelthummar/Desktop/DocQuery/src/api/routes/health.py) | `GET /health` |
| [routes/auth.py](file:///Users/jeelthummar/Desktop/DocQuery/src/api/routes/auth.py) | signup, login, logout, me |
| [routes/documents.py](file:///Users/jeelthummar/Desktop/DocQuery/src/api/routes/documents.py) | upload (full pipeline), list, delete |
| [routes/chat.py](file:///Users/jeelthummar/Desktop/DocQuery/src/api/routes/chat.py) | query, query/stream (SSE), conversations CRUD, messages |

### Endpoints Overview

```
Health:     GET  /api/v1/health
Auth:       POST /api/v1/auth/signup
            POST /api/v1/auth/login
            POST /api/v1/auth/logout          🔒
            GET  /api/v1/auth/me               🔒
Documents:  POST /api/v1/documents/upload      🔒
            GET  /api/v1/documents             🔒
            DELETE /api/v1/documents/{filename} 🔒
Chat:       POST /api/v1/query                 🔒
            POST /api/v1/query/stream          🔒 (SSE)
            POST /api/v1/conversations         🔒
            GET  /api/v1/conversations         🔒
            DELETE /api/v1/conversations/{id}  🔒
            GET  /api/v1/conversations/{id}/messages    🔒
            POST /api/v1/conversations/{id}/messages    🔒
Root:       GET  /
```

🔒 = requires Bearer token auth

---

## Verification Results

### ✅ Server Import
```
✅ Import OK - FastAPI app loaded
   Registered routes (20)
```

### ✅ Health Endpoint
```bash
$ curl http://localhost:8000/api/v1/health
{"status":"ok","version":"0.1.0"}
```

### ✅ Swagger UI

![Swagger UI showing all endpoint groups](swagger_ui_top_1774615996561.png)

### ✅ Server Logs
All requests returned HTTP 200:
```
INFO: "GET /api/v1/health HTTP/1.1" 200 OK
INFO: "GET / HTTP/1.1" 200 OK
INFO: "GET /docs HTTP/1.1" 200 OK
INFO: "GET /openapi.json HTTP/1.1" 200 OK
```

---

## How to Run

```bash
cd /Users/jeelthummar/Desktop/DocQuery
source venv/bin/activate
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

> [!NOTE]
> The remaining verification steps (auth flow, document upload + query, streaming) require valid Supabase credentials and can be tested via the Swagger UI at `/docs`.

> [!WARNING]
> The venv has a pre-existing `fsspec`/`importlib_metadata` corruption. Heavy RAG imports work at request-time but may fail if triggered at module-load time. A clean `pip install -r requirements.txt` in a fresh venv will resolve this.
