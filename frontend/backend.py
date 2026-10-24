"""
backend.py

Thin backend bridge between the Streamlit UI and the RAG backend.
Contains NO AI logic.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from retrival.retrieval import RetrievalApp

ROOT = Path(_ROOT)
INGEST_SCRIPT = ROOT / "ingester" / "ingestion.py"


class Backend:

    def __init__(self):
        try:
            self.rag = RetrievalApp()
        except Exception:
            self.rag = None

    def ingest_documents(self) -> Dict[str, Any]:
        cmd = [sys.executable, str(INGEST_SCRIPT)]
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def ask(self, question: str) -> Dict[str, Any]:
        return self.rag.answer_question(question)

    def chunk_count(self) -> int:
        if self.rag is None:
            return 0
        try:
            return self.rag.chunk_count()
        except Exception:
            return 0

    def document_count(self) -> int:
        if self.rag is None:
            return 0
        try:
            return self.rag.document_count()
        except Exception:
            return 0

    def health(self) -> Dict[str, bool]:
        status = {"chroma": False, "ollama": False}
        if self.rag is None:
            return status
        try:
            hs = self.rag.health_status()
            status["chroma"] = hs["chroma"].get("ok", False)
            status["ollama"] = hs["ollama"].get("ok", False)
        except Exception:
            pass
        return status
