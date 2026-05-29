"""
DocQuery — Chat Export (Phase 3)

Export conversations as Markdown or PDF.
"""

import io
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_current_user
from src.api.routes.audit import log_audit
from src.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _format_conversation_md(title: str, messages: list) -> str:
    """Format a conversation as Markdown text."""
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"*Exported on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*")
    lines.append("")
    lines.append("---")
    lines.append("")

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        created = msg.get("created_at", "")
        sources = msg.get("sources", [])

        if role == "user":
            lines.append(f"## 🧑 You")
        else:
            lines.append(f"## 🤖 DocQuery")

        if created:
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                lines.append(f"*{dt.strftime('%I:%M %p')}*")
            except Exception:
                pass

        lines.append("")
        lines.append(content)
        lines.append("")

        # Add sources for assistant messages
        if sources and role == "assistant":
            lines.append("**Sources:**")
            for s in sources:
                filename = s.get("filename", "Unknown")
                page = s.get("page", "")
                page_str = f" (p. {page})" if page else ""
                lines.append(f"- {filename}{page_str}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _format_conversation_pdf(title: str, messages: list) -> bytes:
    """Format a conversation as a PDF document using fpdf2."""
    try:
        from fpdf import FPDF
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PDF export requires fpdf2. Install with: pip install fpdf2"
        )

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, title, ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 8, f"Exported on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        sources = msg.get("sources", [])

        # Role header
        pdf.set_font("Helvetica", "B", 12)
        label = "You" if role == "user" else "DocQuery"
        pdf.set_fill_color(240, 240, 240) if role == "user" else pdf.set_fill_color(245, 245, 255)
        pdf.cell(0, 8, label, ln=True, fill=True)

        # Content
        pdf.set_font("Helvetica", "", 10)
        pdf.ln(2)

        # Handle multi-line content safely
        for line in content.split("\n"):
            # fpdf2 multi_cell handles wrapping
            pdf.multi_cell(0, 5, line.encode("latin-1", "replace").decode("latin-1"))

        # Sources
        if sources and role == "assistant":
            pdf.ln(2)
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(100, 100, 100)
            for s in sources:
                filename = s.get("filename", "Unknown")
                page = s.get("page", "")
                page_str = f" (p. {page})" if page else ""
                pdf.cell(0, 5, f"Source: {filename}{page_str}", ln=True)
            pdf.set_text_color(0, 0, 0)

        pdf.ln(4)

        # Separator line
        pdf.set_draw_color(200, 200, 200)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(4)

    return pdf.output()


@router.get("/conversations/{conversation_id}/export")
async def export_conversation(
    conversation_id: str,
    format: Literal["md", "pdf"] = "md",
    sb=Depends(get_current_user),
):
    """Export a conversation as Markdown or PDF.

    Query params:
        format: 'md' or 'pdf' (default: 'md')
    """
    # Fetch conversation metadata
    convs = sb.get_conversations()
    conv = next((c for c in convs if c["id"] == conversation_id), None)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    title = conv.get("title", "Untitled Conversation")

    # Fetch messages
    messages = sb.get_messages(conversation_id)
    if not messages:
        raise HTTPException(status_code=404, detail="No messages in this conversation")

    if format == "md":
        md_content = _format_conversation_md(title, messages)
        safe_title = "".join(c for c in title if c.isalnum() or c in " -_").strip()[:50]
        filename = f"{safe_title or 'conversation'}.md"
        log_audit(sb, "export.conversation", "conversation", conversation_id, {"format": "md", "title": title})
        return StreamingResponse(
            io.BytesIO(md_content.encode("utf-8")),
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    elif format == "pdf":
        pdf_bytes = _format_conversation_pdf(title, messages)
        safe_title = "".join(c for c in title if c.isalnum() or c in " -_").strip()[:50]
        filename = f"{safe_title or 'conversation'}.pdf"
        log_audit(sb, "export.conversation", "conversation", conversation_id, {"format": "pdf", "title": title})
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
