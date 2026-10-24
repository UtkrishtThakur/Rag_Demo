"""Command line interface for the local RAG retrieval system.

Usage:
    python retrieval.py
    python retrieval.py --help
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_BASE = str(Path(__file__).resolve().parent)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
for p in (_BASE, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
# Also add SDK src to path as fallback
_SDK_SRC = os.path.join(_PROJECT_ROOT, "rag-inspector-sdk", "src")
if _SDK_SRC not in sys.path:
    sys.path.insert(0, _SDK_SRC)

from embedder import QueryEmbedder
from llm import OllamaClient
from vectordb import ChromaVectorDB

try:
    from rag_inspector import RAGInspector
except Exception:
    RAGInspector = None


BASE_DIR = Path(_BASE)
DEFAULT_PROMPT_FILE = Path(os.getenv("PROMPT_FILE", str(BASE_DIR / "prompt.txt")))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
DEFAULT_LOG_LEVEL = os.getenv("RETRIEVAL_LOG_LEVEL", "INFO").upper()


@dataclass(frozen=True)
class RetrievedChunkView:
    id: str
    text: str
    distance: Optional[float]
    metadata: Dict[str, Any]


class RetrievalApp:
    def __init__(self, top_k: int = DEFAULT_TOP_K, prompt_file: Path = DEFAULT_PROMPT_FILE) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.top_k = top_k
        self.prompt_file = prompt_file

        self.embedder = QueryEmbedder()
        self.db = ChromaVectorDB()
        self.llm = OllamaClient()
        self.prompt_template = self._load_prompt_template(self.prompt_file)
        try:
            self.inspector = RAGInspector() if RAGInspector is not None else None
        except Exception as exc:
            self.logger.warning("RAGInspector initialization failed: %s", exc)
            self.inspector = None

    def run(self) -> int:
        self._print_banner()
        self._print_health()

        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                return 0

            if not question:
                continue

            if question.lower() in {"exit", "quit", "q"}:
                print("Exiting.")
                return 0

            try:
                self.answer_question(question)
            except Exception as exc:
                self.logger.exception("Question handling failed: %s", exc)
                print(f"[ERROR] {exc}")

    def answer_question(self, question: str, query_id: Optional[str] = None) -> Dict[str, Any]:
        self.logger.info("Question received: %s", question)
        run = None
        inspector_enabled = False

        if self.inspector is not None:
            try:
                run_metadata = {
                    "retrieval_backend": "chroma",
                    "generation_backend": "ollama",
                    "model": getattr(self.llm, "model", None) or os.getenv("OLLAMA_MODEL", "qwen3:8b"),
                    "top_k": self.top_k,
                    "embed_model": getattr(self.embedder, "model_name", None),
                }
                run = self.inspector.start_run(question=question, query_id=query_id, metadata=run_metadata)
                inspector_enabled = True
            except Exception as exc:
                self.logger.warning("Failed to start RAG inspector run: %s", exc)
                run = None
                inspector_enabled = False

        views: List[RetrievedChunkView] = []
        try:
            print("\n[1/4] Embedding query...")
            query_embedding = self.embedder.embed_query(question)
            self.logger.debug("Query embedding dimension=%d", len(query_embedding))

            print("[2/4] Retrieving chunks from Chroma...")
            t_retrieval_start = time.perf_counter()
            retrieved = self.db.search(query_embedding=query_embedding, top_k=self.top_k)
            retrieval_latency_ms = (time.perf_counter() - t_retrieval_start) * 1000.0
            views = self._normalize_results(retrieved)
            self._print_retrieval_debug(views)

            if run is not None and inspector_enabled:
                try:
                    retrieval_docs = []
                    for idx, v in enumerate(views):
                        doc: Dict[str, Any] = {
                            "id": v.id if v.id else None,
                            "text": v.text,
                            "metadata": dict(v.metadata) if v.metadata else {},
                            "rank": idx + 1,
                        }
                        if v.distance is not None:
                            doc["score"] = float(v.distance)
                        retrieval_docs.append(doc)
                    run.record_retrieval(retrieval_docs, latency_ms=retrieval_latency_ms)
                except Exception as exc:
                    self.logger.warning("Failed to record retrieval in RAG inspector: %s", exc)

            context = self._build_context(views)
            if run is not None and inspector_enabled:
                try:
                    run.record_context(context)
                except Exception:
                    pass
            final_prompt = self._build_prompt(context=context, question=question)

            print("[3/4] Generating answer with Ollama...")
            t_gen_start = time.perf_counter()
            answer = self.llm.generate(final_prompt)
            generation_latency_ms = (time.perf_counter() - t_gen_start) * 1000.0

            if run is not None and inspector_enabled:
                try:
                    run.record_generation(
                        answer,
                        context=context,
                        latency_ms=generation_latency_ms,
                    )
                except Exception as exc:
                    self.logger.warning("Failed to record generation in RAG inspector: %s", exc)

        except Exception as exc:
            if run is not None and inspector_enabled:
                try:
                    failed_record = run.record_failure(exc)
                    exc.sdk_record = failed_record
                    self.logger.info("Recorded pipeline failure in RAG inspector run: %s", failed_record.run_id)
                except Exception:
                    pass
            raise

        print("[4/4] Done.\n")
        self._print_answer(answer, views)

        seen = set()
        sources = []
        for chunk in views:
            meta = chunk.metadata
            file_name = meta.get("file_name") or meta.get("filename") or "Unknown document"
            page_number = meta.get("page_number")
            key = (file_name, page_number)
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "file_name": file_name,
                "page_number": page_number if page_number not in (None, -1) else None,
            })

        chunks_data = [
            {"id": c.id, "text": c.text, "distance": c.distance, "metadata": dict(c.metadata)}
            for c in views
        ]

        result: Dict[str, Any] = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks": chunks_data,
            "prompt": final_prompt,
            "context": context,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": generation_latency_ms,
        }

        if run is not None and inspector_enabled:
            try:
                execution = run.finish()
                result["rag_inspector"] = execution.to_dict()
                result["sdk_record"] = execution
            except Exception as exc:
                self.logger.warning("Failed to finish/serialize RAG inspector run: %s", exc)

        return result

    def health_status(self) -> Dict[str, Any]:
        return {
            "chroma": self.db.health_check(),
            "ollama": self.llm.health_check(),
        }

    def chunk_count(self) -> int:
        return self.db.count()

    def document_count(self) -> int:
        return self.db.document_count()

    def _load_prompt_template(self, prompt_file: Path) -> str:
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        text = prompt_file.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Prompt file is empty: {prompt_file}")
        self.logger.debug("Loaded prompt template from %s (%d chars)", prompt_file, len(text))
        return text

    def _build_prompt(self, context: str, question: str) -> str:
        template = self.prompt_template
        if "{context}" not in template or "{question}" not in template:
            raise ValueError(
                "prompt.txt must contain both {context} and {question} placeholders"
            )
        return template.replace("{context}", context).replace("{question}", question)

    def _normalize_results(self, retrieved: List[Any]) -> List[RetrievedChunkView]:
        views: List[RetrievedChunkView] = []
        for item in retrieved:
            if isinstance(item, dict):
                chunk_id = str(item.get("id") or "")
                text = str(item.get("text", ""))
                distance = item.get("distance")
                metadata = dict(item.get("metadata") or {})
            else:
                chunk_id = str(getattr(item, "id", "") or "")
                text = str(getattr(item, "text", ""))
                distance = getattr(item, "distance", None)
                metadata = dict(getattr(item, "metadata", {}) or {})
            views.append(RetrievedChunkView(id=chunk_id, text=text, distance=distance, metadata=metadata))
        return views

    def _build_context(self, chunks: List[RetrievedChunkView]) -> str:
        if not chunks:
            return "No relevant context found."

        blocks: List[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata
            file_name = meta.get("file_name") or meta.get("filename") or "Unknown document"
            page_number = meta.get("page_number")
            chunk_index = meta.get("chunk_index")
            distance = chunk.distance

            header_parts = [f"Source: {file_name}"]
            if page_number is not None and page_number != -1:
                header_parts.append(f"Page: {page_number}")
            if chunk_index is not None:
                header_parts.append(f"Chunk: {chunk_index}")
            if distance is not None:
                header_parts.append(f"Distance: {distance:.4f}")

            header = " | ".join(header_parts)
            body = chunk.text.strip()
            blocks.append(f"{header}\n\n{body}")

        return "\n\n" + ("\n\n" + ("-" * 80) + "\n\n").join(blocks)

    def _print_banner(self) -> None:
        total = None
        try:
            total = self.db.count()
        except Exception:
            total = "unknown"

        print("=" * 60)
        print("Local RAG Retrieval CLI")
        print(f"Indexed chunks: {total}")
        print(f"Top K: {self.top_k}")
        print(f"Prompt file: {self.prompt_file}")
        print("Type 'exit' to quit")
        print("=" * 60)

    def _print_health(self) -> None:
        print("\nChecking services...")
        db_health = self.db.health_check()
        llm_health = self.llm.health_check()
        print(f"Chroma: {'OK' if db_health.get('ok') else 'FAIL'}")
        print(f"Ollama: {'OK' if llm_health.get('ok') else 'FAIL'}")

    def _print_retrieval_debug(self, chunks: List[RetrievedChunkView]) -> None:
        print(f"\nRetrieved {len(chunks)} chunks:")
        for idx, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata
            file_name = meta.get("file_name") or meta.get("filename") or "Unknown document"
            page_number = meta.get("page_number")
            page_str = f", page {page_number}" if page_number is not None and page_number != -1 else ""
            dist_str = f"distance={chunk.distance:.4f}" if chunk.distance is not None else "distance=None"
            id_str = f"id={chunk.id}" if chunk.id else "id=None"
            print(f"  [{idx}] {id_str} {file_name}{page_str} ({dist_str})")

    def _print_answer(self, answer: str, chunks: List[RetrievedChunkView]) -> None:
        print("=" * 60)
        print("ANSWER:")
        print(answer)
        print("=" * 60)
        print("SOURCES:")
        seen = set()
        count = 0
        for chunk in chunks:
            meta = chunk.metadata
            file_name = meta.get("file_name") or meta.get("filename") or "Unknown document"
            page_number = meta.get("page_number")
            key = (file_name, page_number)
            if key in seen:
                continue
            seen.add(key)
            page_str = f", page {page_number}" if page_number is not None and page_number != -1 else ""
            print(f"  - {file_name}{page_str}")
            count += 1
        if count == 0:
            print("  (no sources)")
        print("=" * 60)


def configure_logging(level_name: str) -> None:
    numeric_level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local RAG retrieval application.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help=f"Path to prompt template file (default: {DEFAULT_PROMPT_FILE})",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help=f"Log level (default: {DEFAULT_LOG_LEVEL})",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Optional single question to ask in non-interactive mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    app = RetrievalApp(top_k=args.top_k, prompt_file=args.prompt_file)

    if args.question:
        app.answer_question(args.question)
        return 0

    return app.run()


if __name__ == "__main__":
    sys.exit(main())
