"""CLI retrieval loop for the local RAG demo.

This file orchestrates:
- Query embedding via embedder.py
- Similarity search via vectordb.py
- Prompt assembly via prompt.txt
- Answer generation via llm.py

It is intentionally CLI-only and interactive.

Run:
    python retrieval.py

Optional env vars:
    TOP_K=5
    RETRIEVAL_LOG_LEVEL=INFO
    PROMPT_FILE=./prompt.txt
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from embedder import QueryEmbedder
from llm import OllamaClient
from vectordb import ChromaVectorDB


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILE = Path(os.getenv("PROMPT_FILE", str(BASE_DIR / "prompt.txt")))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
DEFAULT_LOG_LEVEL = os.getenv("RETRIEVAL_LOG_LEVEL", "INFO").upper()


@dataclass(frozen=True)
class RetrievedChunkView:
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

    def answer_question(self, question: str) -> str:
        self.logger.info("Question received: %s", question)
        print("\n[1/4] Embedding query...")
        query_embedding = self.embedder.embed_query(question)
        self.logger.debug("Query embedding dimension=%d", len(query_embedding))

        print("[2/4] Retrieving chunks from Chroma...")
        retrieved = self.db.search(query_embedding=query_embedding, top_k=self.top_k)
        views = self._normalize_results(retrieved)
        self._print_retrieval_debug(views)

        context = self._build_context(views)
        final_prompt = self._build_prompt(context=context, question=question)

        print("[3/4] Generating answer with Ollama...")
        answer = self.llm.generate(final_prompt)

        print("[4/4] Done.\n")
        self._print_answer(answer, views)
        return answer

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
                text = str(item.get("text", ""))
                distance = item.get("distance")
                metadata = dict(item.get("metadata") or {})
            else:
                text = str(getattr(item, "text", ""))
                distance = getattr(item, "distance", None)
                metadata = dict(getattr(item, "metadata", {}) or {})
            views.append(RetrievedChunkView(text=text, distance=distance, metadata=metadata))
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
        if not db_health.get("ok"):
            print(f"  Chroma error: {db_health.get('error')}")
        if not llm_health.get("ok"):
            print(f"  Ollama error: {llm_health.get('error')}")

    def _print_retrieval_debug(self, chunks: List[RetrievedChunkView]) -> None:
        print("[2/4] Top matches:")
        if not chunks:
            print("  No chunks returned.")
            return

        for idx, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata
            file_name = meta.get("file_name") or meta.get("filename") or "Unknown document"
            page_number = meta.get("page_number")
            distance = chunk.distance
            dist_text = f"{distance:.4f}" if distance is not None else "n/a"
            page_text = f", page {page_number}" if page_number not in (None, -1) else ""
            print(f"  {idx}. {file_name}{page_text} | distance={dist_text}")

    def _print_answer(self, answer: str, chunks: List[RetrievedChunkView]) -> None:
        print("\n" + "=" * 80)
        print("Answer")
        print("=" * 80)
        print(answer.strip())

        print("\n" + "=" * 80)
        print("Sources")
        print("=" * 80)
        seen = set()
        for chunk in chunks:
            meta = chunk.metadata
            file_name = meta.get("file_name") or meta.get("filename") or "Unknown document"
            page_number = meta.get("page_number")
            key = (file_name, page_number)
            if key in seen:
                continue
            seen.add(key)
            if page_number in (None, -1):
                print(f"- {file_name}")
            else:
                print(f"- {file_name}, page {page_number}")
        print("=" * 80)


def build_arg_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Local CLI retrieval for the RAG demo")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT_FILE,
        help="Path to prompt.txt template",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        app = RetrievalApp(top_k=args.top_k, prompt_file=args.prompt_file)
        return app.run()
    except KeyboardInterrupt:
        print("\nExiting.")
        return 0
    except Exception as exc:
        logging.exception("Fatal error: %s", exc)
        print(f"[FATAL] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
