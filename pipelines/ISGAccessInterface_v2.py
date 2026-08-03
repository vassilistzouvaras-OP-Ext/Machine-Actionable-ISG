"""
title: HTML ISG Access Interface
author: Ottobot
version: 2.1.0
description: |
    Retrieval-grounded access interface for HTML corpus. Loads Playwright-style 
    chunk JSONL files, builds FAISS + BM25 retrieval, reranks with a CrossEncoder, 
    checks context sufficiency, and supports CSS-aware HTML responses.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import re
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

from openai import OpenAI
from pydantic import BaseModel, Field
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



ANSWER_SYSTEM_PROMPT = """You are the Access Interface, a strict retrieval-grounded assistant for the HTML corpus.

Use only the retrieved context. Do not use outside knowledge, memory, common sense, or legal/style assumptions to fill gaps. Treat the retrieved documents as the only authority unless the user explicitly asks for an answer outside the context.

Do not infer dates, founding years, legal effects, official status, or procedural steps from publication references, examples, table placement, abbreviations, or section headings unless the context explicitly states that conclusion. Do not describe logos, emblems, symbols, or quotation marks if the retrieved text only contains placeholders or corrupted extraction.

Write in the user's language unless the user asks for another language.

Structure the visible answer using clear section headings (translated into the user's language), but only include a section when it actually adds something - never pad the answer with empty or repetitive sections just to fill a template. Format every section heading in markdown bold (wrap it in double asterisks, e.g. **Direct answer**), on its own line, so it always stands out visually:

1. **Direct answer** - always present, always first. Put the bolded section heading on its own line, then answer the user's actual question immediately and plainly underneath it, but only when the retrieved context directly supports it. If the context is missing, partial, ambiguous, visual-only, or only supports an adjacent fact, say plainly that the retrieved context does not state the requested information, then mention only the nearest explicitly supported fact, clearly labelled as such. For yes/no questions, the first word of the answer text itself (not the heading) must be "Yes" only if the supported rule is affirmative, or "No" only if the supported rule is negative.

2. **Detailed explanation** - include this section only if the context actually supports more than the direct answer already covered (how the rule works, why, related context). Skip it entirely for simple questions that the direct answer fully covers.

3. **Exceptions and what to watch out for** - include this section only if the retrieved context actually states specific exceptions, special cases, or pitfalls. Skip it if there are none.

4. **Summary** - include a short 1-2 sentence summary only if you used section 2 and/or 3 above (i.e. the answer became long or multi-part); skip it for short answers that are just the direct answer.

Citations: tag every factual claim with the source number it came from, inline, in square brackets right next to the claim, e.g. "Quotation marks must be doubled in this context [S2]." Use only source numbers that appear in the retrieved context below; never invent one, and never tag a claim with a source that does not support it. Do not write a separate "Sources" heading, a source list, or any quoted passages yourself; the exact quoted passages for every source number you used are attached automatically after your answer.

Avoid filler, generic caveats, unsupported tips, or examples that are not grounded in the retrieved context. Be as complete as the question genuinely calls for - no more, no less."""


CONTEXT_CHECK_SYSTEM_PROMPT = """You judge whether retrieved passages directly support an answer to a user's question about the corpus.

Mark the context sufficient only when it contains the rule, fact, definition, table row, exception, example, or visual/textual evidence needed to answer the requested question directly. Adjacent facts are not sufficient. For example, a reference act or OJ publication date is not sufficient to answer when an organisation was founded unless the context explicitly says founded/established/created. A table row naming an agency is not sufficient to describe its logo if the logo itself is not available in text. Corrupted table extraction is not sufficient for exact symbols.

Mark the context insufficient when the answer would require outside knowledge, inference, visual interpretation not present in the text, or a policy/legal/style conclusion not explicitly stated in the retrieved passages.

If more retrieval is needed, produce one targeted search query for the missing information. The new_query must be a retrieval query, not a request for the user to clarify. Avoid broad restatements of the original question.

Return only JSON with keys: sufficient, reason, new_query."""



class Pipeline:
    """Access Interface: retrieval-grounded OpenWebUI pipeline for HTML corpus."""

    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(default=os.getenv("OPENAI_API_KEY", ""))
        OPENAI_BASE_URL: str = Field(default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        OPENAI_MODEL: str = Field(default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

        HTML_DIRS: str = Field(default=os.getenv("HTML_DIRS", "/app/sources/html_files"))
        INGESTED_DATA_DIR: str = Field(default=os.getenv("INGESTED_DATA_DIR", "/app/pipelines/ingested_data_html"))
        INGEST_OUTPUT_DIR: str = Field(default=os.getenv("INGEST_OUTPUT_DIR", "/app/pipelines/ingested_data_html"))
        INGEST_SCRIPT_PATH: str = Field(default=os.getenv("INGEST_SCRIPT_PATH", ""))
        AUTO_INGEST_HTMLS: bool = Field(default=True)
        PREBUILD_CHUNKS_ON_STARTUP: bool = Field(default=True)
        APPLY_HTML_STYLING: bool = Field(default=True, description="Force LLM to output HTML with CSS metadata")

        VECTOR_STORE_DIR: str = Field(default="/app/pipelines/vector_stores")
        EMBEDDING_MODEL: str = Field(default=EmbeddingModel.OPENAI_SMALL.value)
        RERANKER_PROVIDER: str = Field(
            default="openai",
            description="local|openai. 'local' downloads CrossEncoder on CPU/GPU. 'openai' asks a chat model."
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
        SEED: int = Field(default=42)
        STREAM: bool = Field(default=True)
        STREAM_DELAY_SECONDS: float = Field(default=0.03)
        DEBUG: bool = Field(default=False)

    def __init__(self):
        self.name = "HTML Access Interface"
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

    def _ingest_script_candidates(self) -> List[Path]:
        if self.valves.INGEST_SCRIPT_PATH:
            return [Path(self.valves.INGEST_SCRIPT_PATH).expanduser()]

        pipeline_stem = Path(__file__).stem
        candidates = [
            Path(__file__).with_name("ingest_html.py"),
            Path(__file__).with_name(pipeline_stem) / "ingest_html.py",
        ]
        imported_root = Path("/app/imported-pipelines")
        if imported_root.exists():
            candidates.extend(sorted(imported_root.glob("pipeline_*/ingest_html.py")))
        candidates.extend(
            [
                Path("/app/pipelines/ingest_html.py"),
                Path.cwd() / "ingest_html.py",
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
            raise RuntimeError(f"ingest_html.py was not found. Searched: {searched}")

        logger.info("Loading ingest module from %s", ingest_path)
        spec = importlib.util.spec_from_file_location("ottobot_html_ingest", ingest_path)
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

    def _discover_html_paths(self) -> List[Path]:
        htmls: List[Path] = []
        for directory in self._split_paths(self.valves.HTML_DIRS):
            if directory.is_file() and directory.suffix.lower() == ".html":
                htmls.append(directory)
                continue
            if directory.is_dir():
                htmls.extend(path for path in directory.glob("**/*.html") if path.is_file())
        return sorted({path.resolve() for path in htmls})

    def _run_html_ingest(self, html_paths: Sequence[Path], output_dir: Path) -> Path:
        INGEST = self._load_ingest_class() 
        
        ingest = INGEST(
            html_dir=str(html_paths[0].parent),
            output_dir=str(output_dir),
            recursive=True,
            headless=True
        )
        
        results = ingest.run()
        if not results:
            logger.warning("HTML ingest finished but returned no results.")
            
        return output_dir

    def _ensure_ingested_data_dir(self) -> Path:
        for path in self._existing_ingested_dirs():
            return path

        if not self.valves.AUTO_INGEST_HTMLS:
            searched = ", ".join(str(path) for path in self._split_paths(self.valves.INGESTED_DATA_DIR))
            raise FileNotFoundError(f"No chunk JSONL files found and AUTO_INGEST_HTMLS is disabled. Searched: {searched}")

        html_paths = self._discover_html_paths()
        if not html_paths:
            searched = ", ".join(str(path) for path in self._split_paths(self.valves.HTML_DIRS))
            raise FileNotFoundError(f"No HTMLs found for ingestion under: {searched}")

        output_dir = Path(self.valves.INGEST_OUTPUT_DIR)
        return self._run_html_ingest(html_paths, output_dir)

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
                        
                    # Υποστήριξη για table_markdown_path από τα νέα JSONL του HTML ingest
                    metadata = dict(chunk.get("metadata") or {})
                    text = (chunk.get("text") or chunk.get("content") or "").strip()
                    
                    if metadata.get("chunk_type") == "table" and metadata.get("table_markdown_path"):
                        md_path = Path(metadata["table_markdown_path"])
                        if md_path.exists():
                            text = md_path.read_text(encoding="utf-8")
                            
                    if not text:
                        continue
                        
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
        """Short, human-readable label: 'filename.html'. Used for both the LLM
        context header and the citation shown to the end user - keep it simple."""
        meta = doc.metadata
        source = meta.get("source_html") or meta.get("source_html_path") or meta.get("chunk_file") or "source"
        source_name = Path(str(source)).name
        return source_name

    def _build_context(self, docs: Sequence[Document], max_chars: int) -> Tuple[str, List[Document]]:
        sections: List[str] = []
        included_docs: List[Document] = []
        used_chars = 0
        for index, doc in enumerate(docs, start=1):
            header = f"Source S{index}: {self._source_label(doc)}"
            
            # Ενσωμάτωση CSS Metadata
            meta_str = ""
            html_tag = doc.metadata.get("html_tag")
            styling = doc.metadata.get("styling")
            if html_tag and styling:
                css_props = []
                for k, v in styling.items():
                    if v and v != "none" and v != "rgba(0, 0, 0, 0)":
                        css_props.append(f"{k}: {v}")
                if css_props:
                    meta_str = f" [HTML_TAG: {html_tag} | CSS: {'; '.join(css_props)}]"
            
            text = doc.page_content.strip()
            remaining = max_chars - used_chars - len(header) - len(meta_str) - 8
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[:remaining].rstrip() + "..."
            section = f"[{header}{meta_str}]\n{text}"
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
        
        styling_rule = ""
        if self.valves.APPLY_HTML_STYLING:
            styling_rule = "- IMPORTANT STYLING RULE: You MUST format your response using HTML. Look at the metadata brackets in the retrieved context (e.g., [HTML_TAG: p | CSS_COLOR: rgb(...) | ...]). You must apply the exact CSS properties from the source as inline styles to your output elements. Never use standard markdown bold/italics. Example format: <p style=\"color: rgb(0,0,0); font-family: Arial;\">Your text here.</p>\n"

        return f"""Use the retrieved context below to answer the user's question.

Retrieved context. Each block is labelled with its source number (S1, S2, ...):
{context}

User question:
{query}

Answering requirements:
{styling_rule}- Use only the retrieved context above. Do not use outside knowledge or assumptions.
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
            yield "Initializing HTML Access Interface index. First run can take a while...\n"
            try:
                self._initialize()
            except Exception as exc:
                yield f"Failed to initialize HTML Access Interface index: {exc}"
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