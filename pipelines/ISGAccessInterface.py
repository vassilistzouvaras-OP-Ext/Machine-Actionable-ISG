"""
title: ISG Access Interface
author: Ottobot
version: 2.0.0
description: |
    Retrieval-grounded access interface for the Interinstitutional Style
    Guide (ISG) corpus. Loads MinerU-style chunk JSONL files, builds FAISS +
    BM25 retrieval, reranks with a CrossEncoder, checks context sufficiency
    with OpenAI, and answers with inline citations that are backed by the
    verbatim retrieved chunks actually used in the answer.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

from openai import OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModel(str, Enum):
    MINILM = "sentence-transformers/all-MiniLM-L6-v2"
    OPENAI_SMALL = "text-embedding-3-small"


EMBEDDING_MODEL_CONFIGS = {
    EmbeddingModel.MINILM: {
        "name": "all-MiniLM-L6-v2",
        "dimension": 384,
        "batch_size": 64,
        "faiss_batch_size": 1000,
        "min_gpu_memory_gb": 0.5,
        "provider": "huggingface",
    },
    EmbeddingModel.OPENAI_SMALL: {
        "name": "text-embedding-3-small",
        "dimension": 1536,
        "batch_size": 512,
        "faiss_batch_size": 512,
        "min_gpu_memory_gb": 0.0,
        "provider": "openai",
    },
}


class OpenAIEmbeddingWrapper:
    """LangChain-compatible wrapper around the OpenAI embeddings API."""

    def __init__(self, client: OpenAI, model: str, batch_size: int = 512):
        self._client = client
        self._model = model
        self._batch_size = batch_size

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        texts = [text if text.strip() else " " for text in texts]
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for start in range(0, len(texts), self._batch_size):
            embeddings.extend(self._embed_batch(texts[start : start + self._batch_size]))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)



ANSWER_SYSTEM_PROMPT = """You are the ISG Access Interface, a strict retrieval-grounded assistant for the Publications Office of the European Union / Interinstitutional Style Guide (ISG) corpus.

Use only the retrieved context. Do not use outside knowledge, memory, common sense, or legal/style assumptions to fill gaps. Treat the Interinstitutional Style Guide as the only authority unless the user explicitly asks for an answer outside the Style Guide.

Do not infer dates, founding years, legal effects, official status, or procedural steps from publication references, OJ references, examples, table placement, abbreviations, or section headings unless the context explicitly states that conclusion. Do not describe logos, emblems, symbols, or quotation marks if the retrieved text only contains placeholders or corrupted extraction.

Write in the user's language unless the user asks for another language.

Structure the visible answer using clear section headings (translated into the user's language), but only include a section when it actually adds something - never pad the answer with empty or repetitive sections just to fill a template. Format every section heading in markdown bold (wrap it in double asterisks, e.g. **Direct answer**), on its own line, so it always stands out visually:

1. **Direct answer** - always present, always first. Put the bolded section heading on its own line, then answer the user's actual question immediately and plainly underneath it, but only when the retrieved context directly supports it. If the context is missing, partial, ambiguous, visual-only, or only supports an adjacent fact, say plainly that the retrieved ISG context does not state the requested information, then mention only the nearest explicitly supported ISG fact, clearly labelled as such. For yes/no questions, the first word of the answer text itself (not the heading) must be "Yes" only if the supported rule is affirmative, or "No" only if the supported rule is negative.

2. **Detailed explanation** - include this section only if the context actually supports more than the direct answer already covered (how the rule works, why, related context). Skip it entirely for simple questions that the direct answer fully covers.

3. **Exceptions and what to watch out for** - include this section only if the retrieved context actually states specific exceptions, special cases, or pitfalls. Skip it if there are none.

4. **Summary** - include a short 1-2 sentence summary only if you used section 2 and/or 3 above (i.e. the answer became long or multi-part); skip it for short answers that are just the direct answer.

Citations: tag every factual claim with the source number it came from, inline, in square brackets right next to the claim, e.g. "Quotation marks must be doubled in this context [S2]." Use only source numbers that appear in the retrieved context below; never invent one, and never tag a claim with a source that does not support it. Do not write a separate "Sources" heading, a source list, or any quoted passages yourself; the exact quoted passages for every source number you used are attached automatically after your answer.

Avoid filler, generic caveats, unsupported tips, or examples that are not grounded in the retrieved context. Be as complete as the question genuinely calls for - no more, no less."""


CONTEXT_CHECK_SYSTEM_PROMPT = """You judge whether retrieved passages directly support an answer to a user's question about the Publications Office / Interinstitutional Style Guide corpus.

Mark the context sufficient only when it contains the rule, fact, definition, table row, exception, example, or visual/textual evidence needed to answer the requested question directly. Adjacent facts are not sufficient. For example, a reference act or OJ publication date is not sufficient to answer when an organisation was founded unless the context explicitly says founded/established/created. A table row naming an agency is not sufficient to describe its logo if the logo itself is not available in text. Corrupted table extraction is not sufficient for exact symbols.

Mark the context insufficient when the answer would require outside knowledge, inference, visual interpretation not present in the text, or a policy/legal/style conclusion not explicitly stated in the retrieved passages.

If more retrieval is needed, produce one targeted search query for the missing information. The new_query must be a retrieval query, not a request for the user to clarify. Avoid broad restatements of the original question.

Return only JSON with keys: sufficient, reason, new_query."""



class Pipeline:
    """ISG Access Interface: retrieval-grounded OpenWebUI pipeline for the ISG corpus."""

    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(default=os.getenv("OPENAI_API_KEY", ""))
        OPENAI_BASE_URL: str = Field(default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        OPENAI_MODEL: str = Field(default=os.getenv("OPENAI_MODEL", "gpt-5.5"))

        INGESTED_DATA_DIR: str = Field(default=os.getenv("INGESTED_DATA_DIR", "/app/pipelines/ingested_data_epo"))
        PDF_DIRS: str = Field(default=os.getenv("PDF_DIRS", os.getenv("SOURCE_DIRS", "/app/sources/source_1")))
        INGEST_OUTPUT_DIR: str = Field(default=os.getenv("INGEST_OUTPUT_DIR", "/app/pipelines/ingested_data_epo"))
        INGEST_SCRIPT_PATH: str = Field(default=os.getenv("INGEST_SCRIPT_PATH", ""))
        AUTO_INGEST_PDFS: bool = Field(default=True)
        INGEST_MODE: str = Field(default="all", description="one|all")
        ONE_FILE_PAGE_RANGE: str = Field(default="", description="Optional start:end zero-based page range for one-file ingest")
        PREBUILD_CHUNKS_ON_STARTUP: bool = Field(default=True)
        PDF_LANG: str = Field(default="en")
        INGEST_BACKEND: str = Field(default="auto", description="auto|mineru|simple")
        REINGEST_WITH_MINERU: bool = Field(default=True)
        AUTO_INSTALL_MINERU: bool = Field(default=True)
        MINERU_BINARY: str = Field(default=os.getenv("MINERU_BINARY", "mineru"))
        MINERU_PIP_PACKAGE: str = Field(default=os.getenv("MINERU_PIP_PACKAGE", "mineru[all]"))
        CHUNK_SIZE: int = Field(default=1800)
        CHUNK_OVERLAP: int = Field(default=250)
        VECTOR_STORE_DIR: str = Field(default="/app/pipelines/vector_stores")
        EMBEDDING_MODEL: str = Field(default=EmbeddingModel.OPENAI_SMALL.value)
        RERANKER_PROVIDER: str = Field(
            default="openai",
            description=(
                "local|openai. 'local' downloads and runs RERANKER_MODEL (a multi-GB "
                "CrossEncoder) on CPU/GPU - good quality but slow without a GPU and needs "
                "disk space. 'openai' asks an OpenAI chat model to rank the candidates "
                "instead - no local model/disk cost, much faster without a GPU."
            ),
        )
        RERANKER_MODEL: str = Field(default="BAAI/bge-reranker-v2-m3")
        RERANKER_OPENAI_MODEL: str = Field(
            default="gpt-4o-mini", description="Cheap/fast OpenAI model used only when RERANKER_PROVIDER=openai."
        )
        REBUILD_VECTOR_STORE: bool = Field(default=False)

        TOP_K: int = Field(default=15)
        CANDIDATES_K: int = Field(default=40)
        FUSION_ALPHA: float = Field(default=0.55)
        MAX_ITERATIONS: int = Field(default=3)
        CONTEXT_CHAR_LIMIT: int = Field(default=60000)
        MAX_QUERY_LENGTH: int = Field(default=int(os.getenv("MAX_QUERY_LENGTH", "12000")))
        TEMPERATURE: float = Field(default=0.2)
        SEED: int = Field(
            default=42,
            description=(
                "Seed passed to every OpenAI chat completion call (answer generation and the "
                "context-sufficiency check) so that, combined with the deterministic retrieval "
                "pipeline, repeated runs of the same question return the same answer and take the "
                "same retrieval path. Set to 0 to disable."
            ),
        )
        STREAM: bool = Field(default=True)
        STREAM_DELAY_SECONDS: float = Field(default=0.03)
        DEBUG: bool = Field(default=False)

    def __init__(self):
        self.name = "ISG Access Interface"
        self.valves = self.Valves()
        self.documents: List[Document] = []
        self.chunk_paths: List[Path] = []
        self.chunk_fingerprint = ""
        self.index_name = ""
        self.vector_store_path: Optional[Path] = None
        self.index_meta_path: Optional[Path] = None
        self.embeddings: Optional[Any] = None
        self.vector_store: Optional[FAISS] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.cross_encoder: Optional[CrossEncoder] = None
        self.initialized = False
        self._openai_client: Optional[OpenAI] = None
        self._mineru_install_attempted = False
        self._torch_device: Optional[str] = None

    def _get_openai_client(self) -> OpenAI:
        if not self.valves.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        if (
            self._openai_client is None
            or getattr(self._openai_client, "_api_key", None) != self.valves.OPENAI_API_KEY
            or getattr(self._openai_client, "_base_url_value", None) != self.valves.OPENAI_BASE_URL
        ):
            self._openai_client = OpenAI(
                api_key=self.valves.OPENAI_API_KEY,
                base_url=self.valves.OPENAI_BASE_URL or "https://api.openai.com/v1",
            )
            self._openai_client._api_key = self.valves.OPENAI_API_KEY
            self._openai_client._base_url_value = self.valves.OPENAI_BASE_URL
        return self._openai_client

    def _openai_chat_temperature_kwargs(self, temperature: Optional[float]) -> Dict[str, Any]:
        if temperature is None:
            return {}
        model_name = (self.valves.OPENAI_MODEL or "").strip().lower()
        if model_name.startswith("gpt-5"):
            return {}
        return {"temperature": float(temperature)}

    def _openai_chat_seed_kwargs(self) -> Dict[str, Any]:
        if not self.valves.SEED:
            return {}
        return {"seed": int(self.valves.SEED)}

    def _split_paths(self, value: str) -> List[Path]:
        paths: List[Path] = []
        for raw in (value or "").replace(";", ":").replace(",", ":").split(":"):
            raw = raw.strip()
            if raw:
                paths.append(Path(raw))
        return paths

    def _has_chunk_files(self, ingested_dir: Path) -> bool:
        chunks_dir = ingested_dir / "chunks"
        return (chunks_dir / "all_chunks.jsonl").exists() or bool(list(chunks_dir.glob("*.jsonl")))

    def _existing_ingested_dirs(self) -> List[Path]:
        discovered: List[Path] = []
        seen = set()
        for root in self._split_paths(self.valves.INGESTED_DATA_DIR):
            candidates = [root]
            if root.exists():
                candidates.extend(sorted(root.glob("*")))
            for candidate in candidates:
                if not candidate.is_dir() or not self._has_chunk_files(candidate):
                    continue
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                discovered.append(candidate)
        return discovered

    def _detect_chunk_backend(self, ingested_dir: Path) -> str:
        try:
            chunk_paths = self._discover_chunk_paths(ingested_dir)
        except FileNotFoundError:
            return "missing"

        for chunk_path in chunk_paths:
            with chunk_path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    metadata = payload.get("metadata") or {}
                    parser = str(metadata.get("parser") or "").strip().lower()
                    ingest_backend = str(metadata.get("ingest_backend") or "").strip().lower()
                    if parser == "mineru":
                        return "mineru"
                    if "pypdf" in ingest_backend or "simple" in ingest_backend:
                        return "simple"
        return "unknown"

    def _mineru_is_available(self) -> bool:
        return shutil.which(self.valves.MINERU_BINARY) is not None

    def _ensure_mineru_available(self) -> bool:
        if self._mineru_is_available():
            return True
        if not self.valves.AUTO_INSTALL_MINERU or self._mineru_install_attempted:
            return False

        self._mineru_install_attempted = True
        logger.info("MinerU CLI was not found. Installing %s with pip.", self.valves.MINERU_PIP_PACKAGE)
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    self.valves.MINERU_PIP_PACKAGE,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.warning("MinerU installation failed: %s", exc.stderr.strip() or exc.stdout.strip() or exc)
            return False
        except Exception as exc:
            logger.warning("MinerU installation failed: %s", exc)
            return False

        return self._mineru_is_available()

    def _mineru_failure_marker(self, output_dir: Path) -> Path:
        return output_dir / ".mineru_unavailable"

    def _should_regenerate_with_mineru(self, ingested_dir: Path) -> bool:
        backend = self.valves.INGEST_BACKEND.strip().lower()
        if backend not in {"auto", "mineru"} or not self.valves.REINGEST_WITH_MINERU:
            return False
        if not self._ensure_mineru_available():
            return False
        if self._mineru_failure_marker(ingested_dir).exists():
            # MinerU already failed once for this corpus (e.g. it needs a network
            # service that isn't reachable here). Don't retry - and don't re-embed -
            # on every single startup; delete the marker file to force a retry.
            return False

        existing_backend = self._detect_chunk_backend(ingested_dir)
        if existing_backend == "mineru":
            return False

        logger.info(
            "Existing chunks under %s use backend '%s'; regenerating them with MinerU.",
            ingested_dir,
            existing_backend,
        )
        return True

    def _ingest_script_candidates(self) -> List[Path]:
        if self.valves.INGEST_SCRIPT_PATH:
            return [Path(self.valves.INGEST_SCRIPT_PATH).expanduser()]

        pipeline_stem = Path(__file__).stem
        candidates = [
            Path(__file__).with_name("ingest.py"),
            Path(__file__).with_name(pipeline_stem) / "ingest.py",
        ]
        imported_root = Path("/app/imported-pipelines")
        if imported_root.exists():
            candidates.extend(sorted(imported_root.glob("pipeline_*/ingest.py")))
        candidates.extend(
            [
                Path("/app/pipelines/ingest.py"),
                Path.cwd() / "ingest.py",
            ]
        )

        unique: List[Path] = []
        seen = set()
        for candidate in candidates:
            key = candidate.as_posix()
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def _load_ingest_class(self):
        ingest_path = next((path for path in self._ingest_script_candidates() if path.exists()), None)
        if ingest_path is None:
            searched = ", ".join(str(path) for path in self._ingest_script_candidates())
            raise RuntimeError(f"ingest.py was not found. Searched: {searched}")

        logger.info("Loading ingest module from %s", ingest_path)
        spec = importlib.util.spec_from_file_location("ottobot_epo_ingest", ingest_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load ingest module from {ingest_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        ingest_class = getattr(module, "INGEST", None)
        if ingest_class is None:
            raise RuntimeError(f"INGEST class was not found in {ingest_path}")
        return ingest_class

    def _resolve_torch_device(self) -> str:
        if self._torch_device:
            return self._torch_device

        requested_accelerator = os.getenv("PIPELINES_ACCELERATOR", "").strip().lower()
        if requested_accelerator == "cpu":
            self._torch_device = "cpu"
            return self._torch_device

        try:
            import torch
        except Exception as exc:
            logger.warning("PyTorch import failed while checking CUDA; using CPU. Error: %s", exc)
            self._torch_device = "cpu"
            return self._torch_device

        if not torch.cuda.is_available():
            self._torch_device = "cpu"
            return self._torch_device

        try:
            probe = torch.tensor([1.0], device="cuda")
            _ = (probe + 1).item()
            self._torch_device = "cuda"
            return self._torch_device
        except Exception as exc:
            logger.warning("CUDA is present but unusable for this Torch build; falling back to CPU. Error: %s", exc)
            self._torch_device = "cpu"
            return self._torch_device

    def _discover_pdf_paths(self) -> List[Path]:
        pdfs: List[Path] = []
        for directory in self._split_paths(self.valves.PDF_DIRS):
            if directory.is_file() and directory.suffix.lower() == ".pdf":
                pdfs.append(directory)
                continue
            if directory.is_dir():
                pdfs.extend(path for path in directory.glob("**/*.pdf") if path.is_file())
        return sorted({path.resolve() for path in pdfs})

    def _parse_page_range(self) -> Optional[Tuple[int, int]]:
        value = (self.valves.ONE_FILE_PAGE_RANGE or "").strip()
        if not value:
            return None
        for separator in (":", "-", ","):
            if separator in value:
                start, end = value.split(separator, 1)
                return int(start.strip()), int(end.strip())
        raise ValueError("ONE_FILE_PAGE_RANGE must be formatted as start:end")

    def _clean_pdf_text(self, text: str) -> str:
        return " ".join((text or "").replace("\x00", " ").split())

    def _chunk_text(self, text: str) -> List[str]:
        chunk_size = max(300, int(self.valves.CHUNK_SIZE))
        overlap = max(0, min(int(self.valves.CHUNK_OVERLAP), chunk_size - 1))
        step = chunk_size - overlap
        chunks: List[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            if start + chunk_size >= len(text):
                break
        return chunks

    def _write_jsonl(self, path: Path, chunks: Sequence[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    def _simple_ingest_pdf(
        self,
        pdf_path: Path,
        output_dir: Path,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        reader = PdfReader(str(pdf_path))
        start_page = page_range[0] if page_range else 0
        end_page = page_range[1] if page_range else len(reader.pages)
        start_page = max(0, start_page)
        end_page = min(len(reader.pages), end_page)

        chunks: List[Dict[str, Any]] = []
        pdf_stem = pdf_path.stem
        chunks_path = output_dir / "chunks" / f"{pdf_stem}.jsonl"

        for page_index in range(start_page, end_page):
            page = reader.pages[page_index]
            text = self._clean_pdf_text(page.extract_text() or "")
            if not text:
                continue
            page_number = page_index + 1
            for chunk_index, chunk_text in enumerate(self._chunk_text(text)):
                chunks.append(
                    {
                        "id": f"{pdf_stem}-p{page_number}-c{chunk_index}",
                        "text": chunk_text,
                        "metadata": {
                            "source_pdf": pdf_path.name,
                            "source_pdf_path": str(pdf_path),
                            "page_start": page_number,
                            "page_end": page_number,
                            "chunk_type": "text",
                            "chunk_index": chunk_index,
                            "ingest_backend": "simple_pypdf",
                        },
                    }
                )

        self._write_jsonl(chunks_path, chunks)
        result = {
            "source_pdf": str(pdf_path),
            "chunks_path": str(chunks_path),
            "chunk_count": len(chunks),
            "table_count": 0,
            "image_count": 0,
        }
        return result, chunks

    def _run_simple_ingest(self, pdf_paths: Sequence[Path], output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "chunks").mkdir(parents=True, exist_ok=True)
        all_chunks_path = output_dir / "chunks" / "all_chunks.jsonl"
        manifest_path = output_dir / "manifest.json"

        mode = self.valves.INGEST_MODE.strip().lower()
        selected_pdfs = list(pdf_paths[:1] if mode == "one" else pdf_paths)
        page_range = self._parse_page_range() if mode == "one" else None

        all_chunks: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []
        for pdf_path in selected_pdfs:
            result, chunks = self._simple_ingest_pdf(pdf_path, output_dir, page_range=page_range)
            results.append(result)
            all_chunks.extend(chunks)

        self._write_jsonl(all_chunks_path, all_chunks)
        manifest_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Simple PDF ingest wrote {len(all_chunks)} chunks to {all_chunks_path}")
        return output_dir

    def _run_mineru_ingest(self, pdf_paths: Sequence[Path], output_dir: Path) -> Path:
        INGEST = self._load_ingest_class()

        if not self._ensure_mineru_available():
            raise RuntimeError(f"MinerU CLI was not found in PATH as '{self.valves.MINERU_BINARY}'.")

        mode = self.valves.INGEST_MODE.strip().lower()
        page_range = self._parse_page_range() if mode == "one" else None
        extra_mineru_args: List[str] = []
        if page_range is not None:
            start_page, end_page = page_range
            extra_mineru_args.extend(["-s", str(start_page), "-e", str(end_page)])

        mineru_env = {
            "MINERU_PROCESSING_WINDOW_SIZE": "8",
            "MINERU_PDF_RENDER_THREADS": "1",
            "MINERU_API_MAX_CONCURRENT_REQUESTS": "1",
            "MINERU_INTRA_OP_NUM_THREADS": "1",
            "MINERU_INTER_OP_NUM_THREADS": "1",
        }
        ingest = INGEST(
            pdf_dir=str(pdf_paths[0].parent),
            output_dir=str(output_dir),
            mineru_binary=self.valves.MINERU_BINARY,
            backend="pipeline",
            method="auto",
            lang=self.valves.PDF_LANG,
            table_row_threshold=25,
            table_char_threshold=4000,
            skip_existing_mineru=True,
            extra_mineru_args=extra_mineru_args,
            mineru_env=mineru_env,
        )

        selected_pdfs = list(pdf_paths[:1] if mode == "one" else pdf_paths)
        if mode not in {"one", "all"}:
            raise ValueError('INGEST_MODE must be "one" or "all"')

        ingest._prepare_output_dirs()
        results = []
        all_chunks: List[Dict[str, Any]] = []
        for pdf_path in selected_pdfs:
            result, chunks = ingest.ingest_pdf(pdf_path)
            results.append(result)
            all_chunks.extend(chunks)

        ingest._write_jsonl(ingest.all_chunks_path, all_chunks)
        ingest._write_manifest(results)
        logger.info("MinerU ingest wrote %s chunks to %s", len(all_chunks), ingest.all_chunks_path)
        return output_dir

    def _ensure_ingested_data_dir(self) -> Path:
        for path in self._existing_ingested_dirs():
            if not self._should_regenerate_with_mineru(path):
                return path

        if not self.valves.AUTO_INGEST_PDFS:
            searched = ", ".join(str(path) for path in self._split_paths(self.valves.INGESTED_DATA_DIR))
            raise FileNotFoundError(f"No chunk JSONL files found and AUTO_INGEST_PDFS is disabled. Searched: {searched}")

        pdf_paths = self._discover_pdf_paths()
        if not pdf_paths:
            searched = ", ".join(str(path) for path in self._split_paths(self.valves.PDF_DIRS))
            raise FileNotFoundError(f"No PDFs found for ingestion under: {searched}")

        output_dir = Path(self.valves.INGEST_OUTPUT_DIR)
        if self._has_chunk_files(output_dir) and not self._should_regenerate_with_mineru(output_dir):
            return output_dir

        backend = self.valves.INGEST_BACKEND.strip().lower()
        if backend in {"auto", "mineru"}:
            try:
                return self._run_mineru_ingest(pdf_paths, output_dir)
            except Exception as exc:
                if backend == "mineru":
                    raise
                logger.warning(f"MinerU ingest unavailable, falling back to simple PyPDF ingest: {exc}")
                output_dir.mkdir(parents=True, exist_ok=True)
                self._mineru_failure_marker(output_dir).write_text(str(exc), encoding="utf-8")

        return self._run_simple_ingest(pdf_paths, output_dir)

    def _resolve_ingested_data_dir(self) -> Path:
        return self._ensure_ingested_data_dir()

    def _discover_chunk_paths(self, ingested_dir: Path) -> List[Path]:
        chunks_dir = ingested_dir / "chunks"
        all_chunks = chunks_dir / "all_chunks.jsonl"
        if all_chunks.exists():
            return [all_chunks]
        paths = sorted(path for path in chunks_dir.glob("*.jsonl") if path.name != "all_chunks.jsonl")
        if not paths:
            raise FileNotFoundError(f"No chunk JSONL files found under {chunks_dir}")
        return paths

    def _fingerprint_files(self, paths: Sequence[Path]) -> str:
        digest = hashlib.sha1()
        for path in paths:
            digest.update(path.as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _load_documents_from_chunks(self, ingested_dir: Path) -> Tuple[List[Document], List[Path], str]:
        chunk_paths = self._discover_chunk_paths(ingested_dir)
        fingerprint = self._fingerprint_files(chunk_paths)
        documents: List[Document] = []

        for chunk_path in chunk_paths:
            with chunk_path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = (chunk.get("text") or chunk.get("content") or "").strip()
                    if not text:
                        continue
                    metadata = dict(chunk.get("metadata") or {})
                    metadata.update(
                        {
                            "chunk_id": chunk.get("id") or metadata.get("chunk_id"),
                            "chunk_file": str(chunk_path),
                            "index_name": ingested_dir.name,
                        }
                    )
                    documents.append(Document(page_content=text, metadata=metadata))

        if not documents:
            raise ValueError(f"Chunk files were found under {ingested_dir}, but no text chunks were loaded.")
        return documents, chunk_paths, fingerprint

    def _load_index_meta(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _get_embedding_model_config(self, model_name: str) -> Dict[str, Any]:
        for model_enum, config in EMBEDDING_MODEL_CONFIGS.items():
            if model_enum.value == model_name:
                return config
        if model_name.startswith("text-embedding-"):
            return {
                "name": model_name,
                "dimension": 1536,
                "batch_size": 512,
                "faiss_batch_size": 512,
                "min_gpu_memory_gb": 0.0,
                "provider": "openai",
            }
        return {
            "name": model_name,
            "batch_size": 32,
            "faiss_batch_size": 500,
            "min_gpu_memory_gb": 0.5,
            "provider": "huggingface",
        }

    def _embedding_model_cache_key(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.valves.EMBEDDING_MODEL.lower()).strip("_")

    def _initialize_embeddings(self) -> Any:
        model_config = self._get_embedding_model_config(self.valves.EMBEDDING_MODEL)
        provider = model_config.get("provider", "huggingface")

        if provider == "openai":
            logger.info(
                "Initializing OpenAI embeddings: %s (batch size %s)",
                self.valves.EMBEDDING_MODEL,
                model_config.get("batch_size", 512),
            )
            return OpenAIEmbeddingWrapper(
                client=self._get_openai_client(),
                model=self.valves.EMBEDDING_MODEL,
                batch_size=int(model_config.get("batch_size", 512)),
            )

        torch_device = self._resolve_torch_device()
        logger.info("Using torch device for embeddings: %s", torch_device)
        return HuggingFaceEmbeddings(
            model_name=self.valves.EMBEDDING_MODEL,
            model_kwargs={"device": torch_device},
        )

    def _initialize_reranker(self) -> CrossEncoder:
        device = self._resolve_torch_device()
        try:
            return CrossEncoder(
                self.valves.RERANKER_MODEL,
                model_kwargs={"torch_dtype": "auto"},
                trust_remote_code=True,
                device=device,
            )
        except TypeError:
            return CrossEncoder(
                self.valves.RERANKER_MODEL,
                automodel_args={"torch_dtype": "auto"},
                trust_remote_code=True,
                device=device,
            )

    def _initialize(self) -> None:
        if self.initialized:
            return

        ingested_dir = self._resolve_ingested_data_dir()
        self.index_name = ingested_dir.name
        vector_store_dir = Path(self.valves.VECTOR_STORE_DIR)
        vector_store_dir.mkdir(parents=True, exist_ok=True)
        embedding_cache_key = self._embedding_model_cache_key()
        self.vector_store_path = vector_store_dir / f"{self.index_name}_{embedding_cache_key}_faiss_reranker"
        self.index_meta_path = self.vector_store_path / "index_meta.json"

        self.documents, self.chunk_paths, self.chunk_fingerprint = self._load_documents_from_chunks(ingested_dir)
        current_meta = {
            "index_name": self.index_name,
            "ingested_data_dir": str(ingested_dir),
            "chunk_paths": [str(path) for path in self.chunk_paths],
            "chunk_count": len(self.documents),
            "chunk_fingerprint": self.chunk_fingerprint,
            "embedding_model": self.valves.EMBEDDING_MODEL,
        }

        logger.info(f"Corpus: {self.index_name}")
        logger.info(f"Loaded chunks: {len(self.documents)}")
        logger.info(f"Loading embeddings: {self.valves.EMBEDDING_MODEL}")
        if self.valves.RERANKER_PROVIDER.strip().lower() != "openai":
            logger.info("Using reranker torch device: %s", self._resolve_torch_device())
        self.embeddings = self._initialize_embeddings()

        existing_meta = self._load_index_meta(self.index_meta_path)
        index_is_current = (
            self.vector_store_path.exists()
            and existing_meta.get("chunk_fingerprint") == self.chunk_fingerprint
            and existing_meta.get("embedding_model") == self.valves.EMBEDDING_MODEL
        )

        if index_is_current and not self.valves.REBUILD_VECTOR_STORE:
            logger.info(f"Loading existing FAISS index: {self.vector_store_path}")
            self.vector_store = FAISS.load_local(
                str(self.vector_store_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            logger.info(f"Building FAISS index: {self.vector_store_path}")
            self.vector_store = FAISS.from_documents(documents=self.documents, embedding=self.embeddings)
            self.vector_store.save_local(str(self.vector_store_path))
            self.index_meta_path.write_text(json.dumps(current_meta, indent=2), encoding="utf-8")

        logger.info("Building BM25 retriever from the same chunks")
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = int(self.valves.CANDIDATES_K)

        if self.valves.RERANKER_PROVIDER.strip().lower() == "openai":
            logger.info("Reranking with OpenAI model: %s (no local reranker loaded)", self.valves.RERANKER_OPENAI_MODEL)
            self.cross_encoder = None
        else:
            logger.info(f"Loading reranker: {self.valves.RERANKER_MODEL}")
            self.cross_encoder = self._initialize_reranker()
        self.initialized = True

    def _doc_key(self, doc: Document) -> str:
        return str(doc.metadata.get("chunk_id") or hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest())

    def _rerank_with_openai(self, query: str, candidates: List[Document]) -> List[Document]:
        """Ask a cheap OpenAI chat model to rank candidates, instead of a local CrossEncoder."""
        snippets = [f"[{idx}] {doc.page_content.strip()[:400].replace(chr(10), ' ')}" for idx, doc in enumerate(candidates)]
        prompt = (
            f"Question:\n{query}\n\nPassages:\n"
            + "\n\n".join(snippets)
            + '\n\nRank the passage numbers from most to least relevant to the question.\nReturn JSON only: {"ranking": [most relevant index, ..., least relevant index]}'
        )
        order: List[int] = list(range(len(candidates)))
        try:
            response = self._get_openai_client().chat.completions.create(
                model=self.valves.RERANKER_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You rank retrieved passages by relevance to a question. Reply with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                **self._openai_chat_seed_kwargs(),
            )
            ranking = json.loads(response.choices[0].message.content).get("ranking", [])
            seen_order = [i for i in ranking if isinstance(i, int) and 0 <= i < len(candidates)]
            seen = set(seen_order)
            order = seen_order + [i for i in range(len(candidates)) if i not in seen]
        except Exception as exc:
            logger.warning(f"OpenAI rerank failed, keeping fusion order: {exc}")

        total = len(order)
        ranked: List[Document] = []
        for rank, idx in enumerate(order):
            doc = candidates[idx]
            doc.metadata["rerank_score"] = float(total - rank)
            ranked.append(doc)
        return ranked

    def _retrieve_and_rerank(self, query: str, top_k: int, candidates_k: int, alpha: float) -> List[Document]:
        use_openai_reranker = self.valves.RERANKER_PROVIDER.strip().lower() == "openai"
        if not self.vector_store or not self.bm25_retriever or (not use_openai_reranker and not self.cross_encoder):
            return []
        candidates_k = min(max(1, int(candidates_k)), len(self.documents))

        vector_docs = self.vector_store.similarity_search(query, k=candidates_k)
        self.bm25_retriever.k = candidates_k
        bm25_docs = self.bm25_retriever.invoke(query)

        fused: Dict[str, Dict[str, Any]] = {}

        def add_score(found_docs: Sequence[Document], weight: float) -> None:
            for rank, doc in enumerate(found_docs):
                key = self._doc_key(doc)
                if key not in fused:
                    fused[key] = {"doc": doc, "score": 0.0}
                fused[key]["score"] += weight * (1.0 / (rank + 60))

        add_score(vector_docs, alpha)
        add_score(bm25_docs, 1 - alpha)

        candidates = [item["doc"] for item in sorted(fused.values(), key=lambda item: item["score"], reverse=True)]
        candidates = candidates[:candidates_k]
        if not candidates:
            return []

        if use_openai_reranker:
            ranked = self._rerank_with_openai(query, candidates)
            return ranked[:top_k]

        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.cross_encoder.predict(pairs)
        for doc, score in zip(candidates, scores):
            doc.metadata["rerank_score"] = float(score)

        return sorted(candidates, key=lambda doc: doc.metadata.get("rerank_score", 0.0), reverse=True)[:top_k]

    def _source_label(self, doc: Document) -> str:
        """Short, human-readable label: 'filename.pdf, page N'. Used for both the LLM
        context header and the citation shown to the end user - keep it simple."""
        meta = doc.metadata
        source = meta.get("source_pdf") or meta.get("source_pdf_path") or meta.get("chunk_file") or "source"
        source_name = Path(str(source)).name
        page_start = meta.get("page_start")
        page_end = meta.get("page_end")

        if page_start and page_end and page_start != page_end:
            pages = f"pages {page_start}-{page_end}"
        elif page_start:
            pages = f"page {page_start}"
        else:
            pages = "page unknown"
        return f"{source_name}, {pages}"

    def _build_context(self, docs: Sequence[Document], max_chars: int) -> Tuple[str, List[Document]]:
        sections: List[str] = []
        included_docs: List[Document] = []
        used_chars = 0
        for index, doc in enumerate(docs, start=1):
            header = f"Source S{index}: {self._source_label(doc)}"
            text = doc.page_content.strip()
            remaining = max_chars - used_chars - len(header) - 8
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[:remaining].rstrip() + "..."
            section = f"[{header}]\n{text}"
            sections.append(section)
            included_docs.append(doc)
            used_chars += len(section)
        return "\n\n".join(sections), included_docs

    def _unique_docs(self, existing: Sequence[Document], new_docs: Sequence[Document]) -> List[Document]:
        output: List[Document] = []
        seen = set()
        for doc in list(existing) + list(new_docs):
            key = doc.metadata.get("chunk_id") or doc.page_content
            if key in seen:
                continue
            output.append(doc)
            seen.add(key)
        return output

    def _check_if_context_sufficient(self, query: str, context: str) -> Dict[str, Any]:
        check_prompt = f"""Question:
{query}

Retrieved context:
{context}

Decide whether the context is enough for a useful answer. If it is not enough, write one focused retrieval query that could fetch the missing rule, section, table, example, or exception.

Return JSON only:
{{
  "sufficient": true,
  "reason": "short explanation",
  "new_query": ""
}}"""
        try:
            response = self._get_openai_client().chat.completions.create(
                model=self.valves.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": CONTEXT_CHECK_SYSTEM_PROMPT},
                    {"role": "user", "content": check_prompt},
                ],
                response_format={"type": "json_object"},
                **self._openai_chat_temperature_kwargs(0),
                **self._openai_chat_seed_kwargs(),
            )
            return json.loads(response.choices[0].message.content)
        except Exception as exc:
            logger.warning(f"Context check failed, continuing with retrieved context: {exc}")
            return {"sufficient": True, "reason": "Context check failed; using retrieved context.", "new_query": ""}

    def _build_final_prompt(self, query: str, context: str) -> str:

        return f"""Use the retrieved context below to answer the user's question.

Retrieved context. Each block is labelled with its source number (S1, S2, ...):
{context}

User question:
{query}

Answering requirements:
- Use only the retrieved context above. Do not use outside knowledge or assumptions.
- Follow the structure from your instructions: direct answer always first, then a detailed explanation only if it adds something, then exceptions/what to watch out for only if the context states any, then a short summary only if you used the explanation and/or exceptions sections. Do not add a "Sources" section or heading - citations are inline only.
- Answer directly first only if the retrieved context directly supports the answer. If it does not, say plainly that the retrieved ISG context does not state it, then mention only the nearest explicitly supported adjacent fact, clearly labelled as such.
- Do not infer founding dates from OJ/reference-act dates, official rules from examples, or design details from logo/emblem table placeholders.
- Do not add practical guidance, points of caution, suggested wording, or external tips unless the user asks for them or the context explicitly contains them.
- For yes/no questions, make the first word match the supported rule: "Yes" for affirmative support, "No" for negative support, and "The retrieved ISG context does not state this" when unsupported.
- Tag every claim inline with its source number in square brackets, e.g. [S2], using only source numbers that appear in the context above. Do not write your own list of sources or quote the passages yourself.
- Before finalising, silently remove any sentence that cannot be tied to a specific retrieved passage.
- Match the length of the answer to what the question actually needs - do not force every optional section to appear.
"""

    def _generate_answer(self, prompt: str) -> Generator[str, None, None]:
        try:
            stream = self._get_openai_client().chat.completions.create(
                model=self.valves.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                **self._openai_chat_temperature_kwargs(self.valves.TEMPERATURE),
                **self._openai_chat_seed_kwargs(),
                stream=bool(self.valves.STREAM),
            )
            if self.valves.STREAM:
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        delay = max(0.0, float(self.valves.STREAM_DELAY_SECONDS))
                        if delay:
                            time.sleep(delay)
                        yield delta.content
            else:
                yield stream.choices[0].message.content or ""
        except Exception as exc:
            yield f"OpenAI request failed: {exc}"

    _CITATION_PATTERN = re.compile(r"\[S(\d+)\]")

    def _format_cited_sources(self, answer_text: str, docs: Sequence[Document]) -> str:
        """Render only the sources the model actually tagged ([S#]), with the verbatim chunk text."""
        used_numbers: List[int] = []
        seen = set()
        for match in self._CITATION_PATTERN.finditer(answer_text or ""):
            number = int(match.group(1))
            if number not in seen and 1 <= number <= len(docs):
                seen.add(number)
                used_numbers.append(number)

        if not used_numbers:
            return ""

        lines = ["\n\n---\n\n**Sources**\n"]
        for number in used_numbers:
            doc = docs[number - 1]
            label = self._source_label(doc)
            quote = doc.page_content.strip()
            lines.append(f"[S{number}] {label}\n\n> {quote}\n")
        return "\n".join(lines)

    _SOURCES_BLOCK_MARKER = "\n\n---\n\n**Sources**\n"
    _SOURCE_ENTRY_PATTERN = re.compile(
        r"\[S(\d+)\]\s+(?P<label>[^\n]+)\n\n>\s*(?P<quote>.*?)(?=\n\n\[S\d+\]|\Z)", re.DOTALL
    )

    def _split_cited_sources_block(self, content: str) -> Tuple[str, List[Tuple[int, str, str]]]:
        """Pull the verbatim Sources block our own pipe() appended back apart into (number, label, quote)."""
        index = content.find(self._SOURCES_BLOCK_MARKER)
        if index == -1:
            return content, []
        visible = content[:index]
        block = content[index + len(self._SOURCES_BLOCK_MARKER) :]
        entries = [
            (int(match.group(1)), match.group("label").strip(), match.group("quote").strip())
            for match in self._SOURCE_ENTRY_PATTERN.finditer(block)
        ]
        return visible, entries

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Turn the trailing verbatim Sources block into OpenWebUI's native, clickable citation chips."""
        try:
            messages = body.get("messages") or []
            if not messages or messages[-1].get("role") != "assistant":
                return body

            content = messages[-1].get("content") or ""
            visible_text, entries = self._split_cited_sources_block(content)
            if not entries:
                return body

            order: List[int] = []
            seen = set()
            for match in self._CITATION_PATTERN.finditer(visible_text):
                number = int(match.group(1))
                if number not in seen:
                    seen.add(number)
                    order.append(number)

            entries_by_number = {number: (label, quote) for number, label, quote in entries}

            # OpenWebUI's frontend visually merges citation chips that share the same
            # source name into one chip. If two different [S#] tags map to the same
            # label (e.g. two chunks from the same page) but we numbered them as two
            # separate chips, the chip <-> [N] marker indices drift out of sync and
            # later citations stop being clickable. So group by label ourselves first,
            # and give every [S#] sharing a label the SAME final number.
            group_index_by_label: Dict[str, int] = {}
            groups: List[Dict[str, Any]] = []
            number_to_group_index: Dict[int, int] = {}
            for number in order:
                entry = entries_by_number.get(number)
                if entry is None:
                    continue
                label, quote = entry
                if label not in group_index_by_label:
                    group_index_by_label[label] = len(groups)
                    groups.append({"source": {"name": label}, "document": [], "metadata": []})
                group_idx = group_index_by_label[label]
                groups[group_idx]["document"].append(quote)
                groups[group_idx]["metadata"].append({"name": label})
                number_to_group_index[number] = group_idx

            renumbered_text = visible_text
            for number, group_idx in number_to_group_index.items():
                renumbered_text = renumbered_text.replace(f"[S{number}]", f"[{group_idx + 1}]")

            # OpenWebUI's frontend merges back-to-back "[1][2]" markers (no gap between
            # them) into a single multi-source token that uses a hover/LinkPreview
            # widget instead of the plain, reliably-clickable single-citation button.
            # The model often writes citations back-to-back, so insert a thin space
            # between adjacent markers to force them to stay separate, independently
            # clickable citations.
            renumbered_text = re.sub(r"\](?=\[\d+\])", "] ", renumbered_text)

            messages[-1]["content"] = renumbered_text.rstrip()
            messages[-1]["sources"] = groups
        except Exception as exc:
            logger.warning(f"outlet citation rewrite failed: {exc}")
        return body

    def _prepare_rag_answer(self, query: str) -> Tuple[str, List[Document]]:
        all_docs: List[Document] = []
        current_query = query
        context = ""
        included_docs: List[Document] = []

        for _iteration in range(max(1, int(self.valves.MAX_ITERATIONS))):
            docs = self._retrieve_and_rerank(
                current_query,
                top_k=int(self.valves.TOP_K),
                candidates_k=int(self.valves.CANDIDATES_K),
                alpha=float(self.valves.FUSION_ALPHA),
            )
            all_docs = self._unique_docs(all_docs, docs)
            context, included_docs = self._build_context(all_docs, max_chars=int(self.valves.CONTEXT_CHAR_LIMIT))

            check_result = self._check_if_context_sufficient(query, context)
            if self.valves.DEBUG:
                logger.info(f"Context check: {check_result.get('reason', 'No reason returned')}")
            if check_result.get("sufficient", False):
                break

            new_query = (check_result.get("new_query") or "").strip()
            if not new_query or new_query.lower() == current_query.lower():
                break
            current_query = new_query

        return self._build_final_prompt(query, context), included_docs

    def _validate_query(self, query: str) -> Tuple[bool, str]:
        if not query or not query.strip():
            return False, "Query cannot be empty."
        if len(query) > int(self.valves.MAX_QUERY_LENGTH):
            return False, f"Query too long. Maximum length is {self.valves.MAX_QUERY_LENGTH} characters."
        return True, ""

    def _message_content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part)
        return ""

    def _last_user_message(self, messages: Optional[Sequence[dict]]) -> str:
        for message in reversed(messages or []):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "user":
                return self._message_content_to_text(message.get("content")).strip()
        return ""

    def _extract_query(
        self,
        user_message: Union[str, dict],
        messages: Optional[List[dict]] = None,
        body: Optional[dict] = None,
    ) -> str:
        body_messages = body.get("messages") if isinstance(body, dict) else None
        query = self._last_user_message(messages) or self._last_user_message(body_messages)
        if query:
            return query

        if isinstance(user_message, str):
            return user_message.strip()
        if isinstance(user_message, dict):
            return self._message_content_to_text(user_message.get("content")).strip()
        return ""

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        if not self.valves.PREBUILD_CHUNKS_ON_STARTUP:
            return
        try:
            ingested_dir = self._ensure_ingested_data_dir()
            logger.info("Startup ingest ready under %s", ingested_dir)
        except Exception as exc:
            logger.warning("Startup ingest preparation failed: %s", exc)

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    def pipe(
        self,
        user_message: Union[str, dict],
        model_id: str = None,
        messages: List[dict] = None,
        body: dict = None,
    ) -> Generator[str, None, None]:
        query = self._extract_query(user_message, messages=messages, body=body)
        is_valid, error_message = self._validate_query(query)
        if not is_valid:
            yield f"Invalid query: {error_message}"
            return

        if not self.initialized:
            yield "Initializing ISG Access Interface index. First run can take a while...\n"
            try:
                self._initialize()
            except Exception as exc:
                yield f"Failed to initialize ISG Access Interface index: {exc}"
                return

        try:
            prompt, context_docs = self._prepare_rag_answer(query.strip())
        except Exception as exc:
            yield f"Pipeline error: {exc}"
            return

        answer_parts: List[str] = []
        for chunk in self._generate_answer(prompt):
            answer_parts.append(chunk)
            yield chunk
        yield self._format_cited_sources("".join(answer_parts), context_docs)
