"""
DocQuery — FastAPI Middleware

CorrelationIDMiddleware: injects/forwards X-Correlation-ID on every request.
This ID is echoed back in the response header and should be passed to Celery
tasks so every log line across FastAPI → Celery shares the same trace ID.

SecurityHeadersMiddleware: adds standard browser security headers to all responses.
"""

import uuid
from starlette.middleware.base import BaseHTTPMiddleware


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Reads X-Correlation-ID from the incoming request.
    If absent, generates a new UUID4.
    Stores on request.state.correlation_id and echoes back in the response.
    """

    async def dispatch(self, request, call_next):
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds standard browser security headers to every response.
    HSTS is intentionally omitted here — Railway / CDN handles it at the proxy level.
    """

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response
