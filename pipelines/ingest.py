"""PDF ingest pipeline powered by MinerU.

This module only handles PDF ingestion:
- parse PDFs to Markdown/structured JSON with MinerU
- save Markdown, extracted images, and table CSV files
- create retrieval-ready chunks without building a vector store

MinerU itself is called through its CLI so the parsing engine can be upgraded
independently from this project.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
TABLE_MARKDOWN_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
LEGAL_HEADING_RE = re.compile(
    r"^\s*(ΜΕΡΟΣ|ΚΕΦΑΛΑΙΟ|ΑΡΘΡΟ|ΠΑΡΑΡΤΗΜΑ|ΕΝΟΤΗΤΑ|ΠΙΝΑΚΑΣ|"
    r"PART|CHAPTER|ARTICLE|SECTION|SUBSECTION|ANNEX|APPENDIX|TABLE|FIGURE)\b",
    re.IGNORECASE,
)
SKIP_BLOCK_TYPES = {
    "header",
    "footer",
    "page_header",
    "page_footer",
    "page_number",
    "page_aside_text",
    "aside_text",
}


@dataclass
class MinerUOutput:
    markdown_path: Optional[Path]
    content_list_path: Optional[Path]
    base_dir: Optional[Path]
    raw_output_dir: Path


@dataclass
class IngestedDocument:
    source_pdf: str
    markdown_path: Optional[str]
    content_list_path: Optional[str]
    chunks_path: str
    images_dir: str
    tables_dir: str
    chunk_count: int
    table_count: int
    image_count: int


@dataclass
class IngestConfig:
    pdf_dir: Path = field(default_factory=lambda: Path("pdf_files"))
    output_dir: Path = field(default_factory=lambda: Path("ingested_data"))
    mineru_output_dir: Optional[Path] = None
    mineru_binary: str = "mineru"
    backend: str = "pipeline"
    method: str = "auto"
    lang: str = "el"
    recursive: bool = True
    skip_existing_mineru: bool = True
    table_row_threshold: int = 25
    table_char_threshold: int = 4000
    formula: bool = True
    table: bool = True
    extra_mineru_args: Sequence[str] = field(default_factory=tuple)
    mineru_env: Dict[str, str] = field(default_factory=dict)


class _HTMLTableParser(HTMLParser):
    """Small HTML table parser that keeps rowspan/colspan mostly usable."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: List[List[str]] = []
        self._in_row = False
        self._in_cell = False
        self._cell_parts: List[str] = []
        self._row: List[str] = []
        self._col = 0
        self._cell_colspan = 1
        self._cell_rowspan = 1
        self._rowspans: Dict[int, Tuple[str, int]] = {}
        self._next_rowspans: Dict[int, Tuple[str, int]] = {}

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._in_row = True
            self._row = []
            self._col = 0
            self._next_rowspans = {}
            return
        if tag in {"td", "th"} and self._in_row:
            self._consume_rowspans_at_current_col()
            attr_map = {key.lower(): value for key, value in attrs}
            self._cell_colspan = self._safe_int(attr_map.get("colspan"), default=1)
            self._cell_rowspan = self._safe_int(attr_map.get("rowspan"), default=1)
            self._cell_parts = []
            self._in_cell = True
            return
        if tag == "br" and self._in_cell:
            self._cell_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._in_cell:
            text = _normalize_space(" ".join(self._cell_parts))
            for offset in range(self._cell_colspan):
                self._row.append(text)
                if self._cell_rowspan > 1:
                    self._next_rowspans[self._col + offset] = (text, self._cell_rowspan - 1)
            self._col += self._cell_colspan
            self._in_cell = False
            return
        if tag == "tr" and self._in_row:
            self._consume_trailing_rowspans()
            if any(cell.strip() for cell in self._row):
                self.rows.append(self._row)
            self._rowspans.update(self._next_rowspans)
            self._in_row = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data)

    def _consume_rowspans_at_current_col(self) -> None:
        while self._col in self._rowspans:
            self._consume_rowspan_cell()

    def _consume_trailing_rowspans(self) -> None:
        while self._rowspans and self._col <= max(self._rowspans):
            if self._col in self._rowspans:
                self._consume_rowspan_cell()
            else:
                self._row.append("")
                self._col += 1

    def _consume_rowspan_cell(self) -> None:
        text, remaining = self._rowspans.pop(self._col)
        self._row.append(text)
        if remaining > 1:
            self._next_rowspans[self._col] = (text, remaining - 1)
        self._col += 1

    @staticmethod
    def _safe_int(value: Optional[str], default: int = 1) -> int:
        try:
            parsed = int(value or default)
        except (TypeError, ValueError):
            return default
        return max(parsed, 1)


class INGEST:
    """Parse all PDFs in a folder with MinerU and create smart chunks.

    The produced chunk JSONL is intentionally vector-store agnostic. Each line is
    a dict with `id`, `text`, and `metadata`, so it can later be embedded by any
    retrieval pipeline.
    """

    def __init__(
        self,
        pdf_dir: str = "pdf_files",
        output_dir: str = "ingested_data",
        mineru_output_dir: Optional[str] = None,
        mineru_binary: str = "mineru",
        backend: str = "pipeline",
        method: str = "auto",
        lang: str = "el",
        recursive: bool = True,
        skip_existing_mineru: bool = True,
        table_row_threshold: int = 25,
        table_char_threshold: int = 4000,
        formula: bool = True,
        table: bool = True,
        extra_mineru_args: Optional[Sequence[str]] = None,
        mineru_env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.config = IngestConfig(
            pdf_dir=Path(pdf_dir),
            output_dir=Path(output_dir),
            mineru_output_dir=Path(mineru_output_dir) if mineru_output_dir else None,
            mineru_binary=mineru_binary,
            backend=backend,
            method=method,
            lang=lang,
            recursive=recursive,
            skip_existing_mineru=skip_existing_mineru,
            table_row_threshold=table_row_threshold,
            table_char_threshold=table_char_threshold,
            formula=formula,
            table=table,
            extra_mineru_args=tuple(extra_mineru_args or ()),
            mineru_env=dict(mineru_env or {}),
        )
        if self.config.mineru_output_dir is None:
            self.config.mineru_output_dir = self.config.output_dir / "mineru_raw"

        self.markdown_dir = self.config.output_dir / "markdown"
        self.chunks_dir = self.config.output_dir / "chunks"
        self.tables_dir = self.config.output_dir / "tables"
        self.images_dir = self.config.output_dir / "images"
        self.manifest_path = self.config.output_dir / "manifest.json"
        self.all_chunks_path = self.chunks_dir / "all_chunks.jsonl"

    def run(self) -> List[IngestedDocument]:
        """Ingest every PDF found under `pdf_dir`."""

        self._prepare_output_dirs()
        pdfs = self._discover_pdfs()
        if not pdfs:
            raise FileNotFoundError(f"No PDF files found under {self.config.pdf_dir}")

        results: List[IngestedDocument] = []
        all_chunks: List[Dict[str, Any]] = []

        for pdf_path in pdfs:
            result, chunks = self.ingest_pdf(pdf_path)
            results.append(result)
            all_chunks.extend(chunks)

        self._write_jsonl(self.all_chunks_path, all_chunks)
        self._write_manifest(results)
        return results

    def ingest_pdf(self, pdf_path: Path) -> Tuple[IngestedDocument, List[Dict[str, Any]]]:
        """Ingest one PDF and return its manifest row plus chunks."""

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        doc_key = _safe_filename(pdf_path.stem)
        mineru_output = self._ensure_mineru_output(pdf_path)
        blocks = self._load_content_blocks(mineru_output)
        image_map = self._copy_mineru_images(mineru_output, blocks, doc_key)
        markdown_path = self._copy_markdown(mineru_output.markdown_path, doc_key, image_map)

        chunks, table_count = self._build_chunks(pdf_path, doc_key, blocks, image_map)
        doc_chunks_path = self.chunks_dir / f"{doc_key}.jsonl"
        self._write_jsonl(doc_chunks_path, chunks)

        result = IngestedDocument(
            source_pdf=str(pdf_path),
            markdown_path=str(markdown_path) if markdown_path else None,
            content_list_path=str(mineru_output.content_list_path) if mineru_output.content_list_path else None,
            chunks_path=str(doc_chunks_path),
            images_dir=str(self.images_dir / doc_key),
            tables_dir=str(self.tables_dir / doc_key),
            chunk_count=len(chunks),
            table_count=table_count,
            image_count=len({Path(path).as_posix() for path in image_map.values()}),
        )
        return result, chunks

    def _prepare_output_dirs(self) -> None:
        for directory in (
            self.config.output_dir,
            self.config.mineru_output_dir,
            self.markdown_dir,
            self.chunks_dir,
            self.tables_dir,
            self.images_dir,
        ):
            if directory is not None:
                directory.mkdir(parents=True, exist_ok=True)

    def _discover_pdfs(self) -> List[Path]:
        pattern = "**/*.pdf" if self.config.recursive else "*.pdf"
        return sorted(path for path in self.config.pdf_dir.glob(pattern) if path.is_file())

    def _ensure_mineru_output(self, pdf_path: Path) -> MinerUOutput:
        if self.config.skip_existing_mineru:
            existing = self._locate_mineru_outputs(pdf_path, allow_single_fallback=False)
            if existing.markdown_path or existing.content_list_path:
                return existing

        if shutil.which(self.config.mineru_binary) is None:
            raise RuntimeError(
                "MinerU CLI was not found. Install MinerU in a Python 3.10-3.13 "
                'environment, then make sure the "mineru" command is on PATH.'
            )

        command = [
            self.config.mineru_binary,
            "-p",
            str(pdf_path),
            "-o",
            str(self.config.mineru_output_dir),
            "-m",
            self.config.method,
            "-b",
            self.config.backend,
            "-l",
            self.config.lang,
            "-f",
            _bool_cli(self.config.formula),
            "-t",
            _bool_cli(self.config.table),
        ]
        command.extend(self.config.extra_mineru_args)

        env = os.environ.copy()
        env.update(self.config.mineru_env)

        completed = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
        if completed.returncode != 0:
            raise RuntimeError(
                "MinerU failed for "
                f"{pdf_path}\n\nCommand:\n{' '.join(command)}\n\n"
                f"Return code: {completed.returncode}\n\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
            )

        output = self._locate_mineru_outputs(pdf_path, allow_single_fallback=True)
        if not output.markdown_path and not output.content_list_path:
            raise FileNotFoundError(
                f"MinerU finished but no Markdown/content_list output was found for {pdf_path}"
            )
        return output

    def _locate_mineru_outputs(self, pdf_path: Path, allow_single_fallback: bool = False) -> MinerUOutput:
        root = self.config.mineru_output_dir
        assert root is not None
        if not root.exists():
            return MinerUOutput(None, None, None, root)

        markdown = self._choose_best_candidate(
            root.rglob("*.md"),
            pdf_path,
            prefer_v2=False,
            allow_single_fallback=allow_single_fallback,
        )
        content_v2 = self._choose_best_candidate(
            root.rglob("*_content_list_v2.json"),
            pdf_path,
            prefer_v2=True,
            allow_single_fallback=allow_single_fallback,
        )
        content_v1 = self._choose_best_candidate(
            root.rglob("*_content_list.json"),
            pdf_path,
            prefer_v2=False,
            allow_single_fallback=allow_single_fallback,
        )
        content_list = content_v2 or content_v1
        base_dir = None
        if content_list:
            base_dir = content_list.parent
        elif markdown:
            base_dir = markdown.parent
        return MinerUOutput(markdown, content_list, base_dir, root)

    def _choose_best_candidate(
        self,
        candidates: Iterable[Path],
        pdf_path: Path,
        prefer_v2: bool,
        allow_single_fallback: bool,
    ) -> Optional[Path]:
        all_candidates = list(candidates)
        scored: List[Tuple[int, int, Path]] = []
        pdf_stem = pdf_path.stem
        safe_stem = _safe_filename(pdf_stem)
        for candidate in all_candidates:
            name = candidate.name
            path_text = candidate.as_posix()
            score = 0
            if pdf_stem and pdf_stem in name:
                score += 100
            if pdf_stem and pdf_stem in path_text:
                score += 80
            if safe_stem and safe_stem in path_text:
                score += 40
            if candidate.name == f"{pdf_stem}.md":
                score += 30
            if prefer_v2 and candidate.name.endswith("_content_list_v2.json"):
                score += 10
            try:
                size = candidate.stat().st_size
            except OSError:
                size = 0
            if score > 0:
                scored.append((score, size, candidate))

        if not scored:
            if not allow_single_fallback or len(all_candidates) != 1:
                return None
            return max(all_candidates, key=lambda path: path.stat().st_size if path.exists() else 0)

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2]

    def _load_content_blocks(self, output: MinerUOutput) -> List[Dict[str, Any]]:
        if output.content_list_path and output.content_list_path.exists():
            data = json.loads(output.content_list_path.read_text(encoding="utf-8"))
            if _looks_like_content_list_v2(data):
                return self._blocks_from_content_list_v2(data)
            if isinstance(data, list):
                return [block for block in data if isinstance(block, dict)]
        if output.markdown_path and output.markdown_path.exists():
            return self._blocks_from_markdown(output.markdown_path)
        return []

    def _blocks_from_content_list_v2(self, data: Any) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        if not isinstance(data, list):
            return blocks

        for page_idx, page_blocks in enumerate(data):
            if not isinstance(page_blocks, list):
                continue
            for item in page_blocks:
                if not isinstance(item, dict):
                    continue
                block_type = str(item.get("type") or "")
                content = item.get("content") if isinstance(item.get("content"), dict) else {}
                block: Dict[str, Any] = {
                    "type": self._legacy_type_from_v2(block_type),
                    "page_idx": page_idx,
                    "bbox": item.get("bbox"),
                }

                if block_type == "title":
                    block["text"] = _flatten_content(content.get("title_content"))
                    block["text_level"] = _safe_int(content.get("level"), default=1)
                elif block_type == "paragraph":
                    block["text"] = _flatten_content(content.get("paragraph_content"))
                elif block_type in {"list", "index"}:
                    items = content.get("list_items") or content.get("index_items")
                    block["list_items"] = _flatten_to_list(items)
                elif block_type in {"table", "chart"}:
                    block["table_body"] = self._first_content_value(
                        content,
                        keys=("html", "table_body", "table_content", "content", "chart_content"),
                    )
                    block["table_html"] = str(content.get("html") or "")
                    block["table_type"] = _flatten_content(content.get("table_type"))
                    block["img_path"] = self._first_content_value(
                        content,
                        keys=("img_path", "image_path", "table_img_path", "chart_img_path", "image_source"),
                    )
                    block["table_caption"] = _flatten_to_list(
                        _first_value(content, keys=("table_caption", "caption", "chart_caption"))
                    )
                    block["table_footnote"] = _flatten_to_list(
                        _first_value(content, keys=("table_footnote", "footnote", "chart_footnote"))
                    )
                elif block_type in {"image", "seal"}:
                    block["img_path"] = self._first_content_value(
                        content,
                        keys=("img_path", "image_path", "path", "image_source"),
                    )
                    block["image_caption"] = _flatten_to_list(
                        _first_value(content, keys=("image_caption", "caption"))
                    )
                    block["image_footnote"] = _flatten_to_list(
                        _first_value(content, keys=("image_footnote", "footnote"))
                    )
                elif block_type == "equation_interline":
                    block["type"] = "equation"
                    block["text"] = self._first_content_value(content, keys=("math_content", "content"))
                elif block_type in {"code", "algorithm"}:
                    block["type"] = "code"
                    block["code_body"] = self._first_content_value(
                        content,
                        keys=("code_content", "algorithm_content", "content"),
                    )
                else:
                    block["text"] = _flatten_content(content)

                blocks.append(block)
        return blocks

    @staticmethod
    def _legacy_type_from_v2(block_type: str) -> str:
        mapping = {
            "title": "text",
            "paragraph": "text",
            "equation_interline": "equation",
            "page_header": "header",
            "page_footer": "footer",
            "page_number": "page_number",
            "page_aside_text": "aside_text",
        }
        return mapping.get(block_type, block_type)

    @staticmethod
    def _first_content_value(content: Dict[str, Any], keys: Sequence[str]) -> str:
        for key in keys:
            if key in content and content[key] not in (None, "", []):
                return _flatten_content(content[key])
        return ""

    def _blocks_from_markdown(self, markdown_path: Path) -> List[Dict[str, Any]]:
        text = markdown_path.read_text(encoding="utf-8")
        blocks: List[Dict[str, Any]] = []
        paragraph: List[str] = []
        table_lines: List[str] = []

        def flush_paragraph() -> None:
            if paragraph:
                blocks.append({"type": "text", "text": "\n".join(paragraph).strip()})
                paragraph.clear()

        def flush_table() -> None:
            if table_lines:
                blocks.append({"type": "table", "table_body": "\n".join(table_lines).strip()})
                table_lines.clear()

        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if heading_match:
                flush_paragraph()
                flush_table()
                blocks.append(
                    {
                        "type": "text",
                        "text": heading_match.group(2),
                        "text_level": len(heading_match.group(1)),
                    }
                )
                continue

            if "|" in line and line.strip():
                flush_paragraph()
                table_lines.append(line)
                continue

            flush_table()
            if line.strip():
                paragraph.append(line)
            else:
                flush_paragraph()

        flush_paragraph()
        flush_table()
        return blocks

    def _copy_markdown(
        self,
        markdown_path: Optional[Path],
        doc_key: str,
        image_map: Dict[str, str],
    ) -> Optional[Path]:
        if not markdown_path or not markdown_path.exists():
            return None

        target = self.markdown_dir / f"{doc_key}.md"
        text = markdown_path.read_text(encoding="utf-8")
        text = self._rewrite_markdown_image_links(text, doc_key, image_map)
        target.write_text(text, encoding="utf-8")
        return target

    def _rewrite_markdown_image_links(
        self,
        markdown: str,
        doc_key: str,
        image_map: Dict[str, str],
    ) -> str:
        def replace(match: re.Match[str]) -> str:
            alt = match.group(1)
            target = match.group(2).strip()
            clean_target = target.split("#", 1)[0].split("?", 1)[0]
            mapped = image_map.get(target) or image_map.get(clean_target) or image_map.get(Path(clean_target).name)
            if mapped:
                rel = os.path.relpath(mapped, start=self.markdown_dir)
                return f"![{alt}]({rel})"
            if clean_target.startswith("images/"):
                rel = Path("..") / "images" / doc_key / clean_target[len("images/") :]
                return f"![{alt}]({rel.as_posix()})"
            return match.group(0)

        return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace, markdown)

    def _copy_mineru_images(
        self,
        output: MinerUOutput,
        blocks: Sequence[Dict[str, Any]],
        doc_key: str,
    ) -> Dict[str, str]:
        image_dir = self.images_dir / doc_key
        image_dir.mkdir(parents=True, exist_ok=True)
        image_map: Dict[str, str] = {}
        if not output.base_dir or not output.base_dir.exists():
            return image_map

        image_refs = self._image_refs_from_blocks(blocks)
        for ref in image_refs:
            source = self._resolve_asset(output.base_dir, ref)
            if source:
                target = self._copy_image_file(source, image_dir)
                image_map[ref] = str(target)
                image_map[Path(ref).name] = str(target)

        for source in output.base_dir.rglob("*"):
            if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES and "images" in source.parts:
                target = self._copy_image_file(source, image_dir)
                image_map[source.name] = str(target)
                image_map[f"images/{target.name}"] = str(target)

        return image_map

    @staticmethod
    def _image_refs_from_blocks(blocks: Sequence[Dict[str, Any]]) -> List[str]:
        refs: List[str] = []
        for block in blocks:
            for key in ("img_path", "image_path", "table_img_path", "chart_img_path"):
                value = block.get(key)
                if isinstance(value, str) and _looks_like_asset_file_ref(value):
                    refs.append(value.strip())
        return refs

    @staticmethod
    def _resolve_asset(base_dir: Path, ref: str) -> Optional[Path]:
        path = Path(ref)
        if path.is_absolute() and path.is_file():
            return path
        direct = base_dir / path
        if direct.is_file():
            return direct
        matches = [match for match in base_dir.rglob(path.name) if match.is_file()]
        return matches[0] if matches else None

    @staticmethod
    def _copy_image_file(source: Path, image_dir: Path) -> Path:
        target = image_dir / source.name
        counter = 2
        while target.exists() and source.exists() and target.read_bytes() != source.read_bytes():
            target = image_dir / f"{source.stem}_{counter}{source.suffix}"
            counter += 1
        if not target.exists():
            shutil.copy2(source, target)
        return target

    def _build_chunks(
        self,
        pdf_path: Path,
        doc_key: str,
        blocks: Sequence[Dict[str, Any]],
        image_map: Dict[str, str],
    ) -> Tuple[List[Dict[str, Any]], int]:
        chunks: List[Dict[str, Any]] = []
        chapter_stack: List[str] = []
        current: Optional[Dict[str, Any]] = None
        table_count = 0

        def new_chunk(text: str, chunk_type: str, metadata: Dict[str, Any]) -> None:
            clean = text.strip()
            if not clean:
                return
            chunk_id = f"{doc_key}:{len(chunks) + 1:05d}"
            chunk_metadata = {
                "source_pdf": pdf_path.name,
                "source_pdf_path": str(pdf_path),
                "parser": "mineru",
                "chunk_type": chunk_type,
                **metadata,
            }
            chunk_metadata.setdefault("is_table", chunk_type.startswith("table"))
            chunk_metadata.setdefault("embed", True)
            chunk_metadata["references"] = _build_references(chunk_metadata)
            chunks.append(
                {
                    "id": chunk_id,
                    "text": clean,
                    "content": clean,
                    "metadata": chunk_metadata,
                }
            )

        def flush_current() -> None:
            nonlocal current
            if not current:
                return
            lines = current.get("lines") or []
            if len(lines) > 1 or (lines and not str(lines[0]).startswith("#")):
                new_chunk(
                    "\n\n".join(lines),
                    "chapter_text",
                    {
                        "heading_path": current.get("heading_path", []),
                        "chapter_title": current.get("title"),
                        "page_start": current.get("page_start"),
                        "page_end": current.get("page_end"),
                    },
                )
            current = None

        def ensure_current(page_number: Optional[int]) -> Dict[str, Any]:
            nonlocal current
            if current is None:
                title = chapter_stack[-1] if chapter_stack else pdf_path.stem
                level = len(chapter_stack) if chapter_stack else 1
                current = {
                    "title": title,
                    "heading_path": chapter_stack[:] or [title],
                    "lines": [f"{'#' * min(level, 6)} {title}"],
                    "page_start": page_number,
                    "page_end": page_number,
                }
            return current

        for block in blocks:
            block_type = str(block.get("type") or "").strip()
            if block_type in SKIP_BLOCK_TYPES:
                continue

            page_number = _human_page_number(block.get("page_idx"))
            heading = self._heading_from_block(block)
            if heading:
                flush_current()
                level, heading_text = heading
                while len(chapter_stack) >= level:
                    chapter_stack.pop()
                chapter_stack.append(heading_text)
                current = {
                    "title": heading_text,
                    "heading_path": chapter_stack[:],
                    "lines": [f"{'#' * min(level, 6)} {heading_text}"],
                    "page_start": page_number,
                    "page_end": page_number,
                }
                continue

            if block_type in {"table", "chart"}:
                flush_current()
                table_count += 1
                context = {
                    "heading_path": chapter_stack[:],
                    "chapter_title": chapter_stack[-1] if chapter_stack else pdf_path.stem,
                    "page_start": page_number,
                    "page_end": page_number,
                }
                for table_chunk in self._table_chunks(pdf_path, doc_key, block, table_count, context, image_map):
                    new_chunk(table_chunk["text"], table_chunk["chunk_type"], table_chunk["metadata"])
                continue

            if block_type in {"image", "seal"}:
                flush_current()
                image_text, image_metadata = self._image_chunk(block, image_map, chapter_stack, page_number)
                if image_text:
                    new_chunk(image_text, "image", image_metadata)
                continue

            text = self._block_text(block)
            if text:
                active = ensure_current(page_number)
                active["lines"].append(text)
                if page_number is not None:
                    if active.get("page_start") is None:
                        active["page_start"] = page_number
                    active["page_end"] = page_number

        flush_current()
        return chunks, table_count

    def _heading_from_block(self, block: Dict[str, Any]) -> Optional[Tuple[int, str]]:
        text = _normalize_space(str(block.get("text") or ""))
        if not text:
            return None

        level = _safe_int(block.get("text_level"), default=0)
        if level > 0:
            return min(level, 6), text

        if self._looks_like_heading(text):
            return self._guess_heading_level(text), text

        return None

    @staticmethod
    def _looks_like_heading(text: str) -> bool:
        stripped = text.strip()
        if not (3 <= len(stripped) <= 180):
            return False
        if "|" in stripped:
            return False
        if LEGAL_HEADING_RE.match(stripped):
            return True

        letters = [char for char in stripped if char.isalpha()]
        if not letters:
            return False
        uppercase_ratio = sum(1 for char in letters if char.upper() == char) / len(letters)
        word_count = len(stripped.split())
        has_sentence_punctuation = stripped.endswith(".") and word_count > 4
        return uppercase_ratio >= 0.85 and word_count <= 16 and not has_sentence_punctuation

    @staticmethod
    def _guess_heading_level(text: str) -> int:
        upper = text.upper()
        if upper.startswith("ΜΕΡΟΣ"):
            return 1
        if upper.startswith("ΚΕΦΑΛΑΙΟ"):
            return 2
        if upper.startswith("ΑΡΘΡΟ"):
            return 3
        if upper.startswith("ΠΑΡΑΡΤΗΜΑ"):
            return 1
        if upper.startswith(("PART", "ANNEX", "APPENDIX")):
            return 1
        if upper.startswith("CHAPTER"):
            return 2
        if upper.startswith(("ARTICLE", "SECTION", "SUBSECTION")):
            return 3
        if upper.startswith(("TABLE", "FIGURE")):
            return 4
        return 2

    @staticmethod
    def _block_text(block: Dict[str, Any]) -> str:
        block_type = str(block.get("type") or "")
        if block_type == "list":
            items = _flatten_to_list(block.get("list_items"))
            return "\n".join(f"- {item}" for item in items if item)
        if block_type == "equation":
            return _normalize_space(str(block.get("text") or ""))
        if block_type == "code":
            return str(block.get("code_body") or "").strip()
        if block_type in {"page_footnote", "footnote"}:
            return _normalize_space(str(block.get("text") or _flatten_content(block)))
        return _normalize_space(str(block.get("text") or ""))

    def _table_chunks(
        self,
        pdf_path: Path,
        doc_key: str,
        block: Dict[str, Any],
        table_number: int,
        context: Dict[str, Any],
        image_map: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        body = self._table_body(block)
        caption = self._caption_text(block, "table")
        footnote = self._footnote_text(block, "table")
        rows = self._table_rows(body)
        table_text = _rows_to_markdown(rows) if rows else _strip_html(body)
        table_files = self._write_table_files(
            doc_key=doc_key,
            table_number=table_number,
            page_number=context.get("page_start"),
            body=body,
            rows=rows,
            caption=caption,
            footnote=footnote,
        )
        image_ref = str(block.get("img_path") or "").strip()
        image_path = image_map.get(image_ref) or image_map.get(Path(image_ref).name) if image_ref else None
        content_available = bool(table_text.strip() or table_files)
        is_large = bool(
            rows
            and (
                len(rows) >= self.config.table_row_threshold
                or len(body) >= self.config.table_char_threshold
            )
        )
        base_metadata = {
            **context,
            "table_number": table_number,
            "table_csv_path": table_files.get("csv"),
            "table_markdown_path": table_files.get("markdown"),
            "table_html_path": table_files.get("html"),
            "table_image_path": image_path,
            "table_row_count": len(rows) if rows else None,
            "table_col_count": max((len(row) for row in rows), default=None),
            "table_body_char_count": len(body),
            "table_text_char_count": len(table_text),
            "table_type": block.get("table_type"),
            "is_table": True,
            "content_available": content_available,
            "embed": content_available,
            "bbox": block.get("bbox"),
        }

        chunks: List[Dict[str, Any]] = []
        if not content_available:
            return []

        if is_large:
            headers, data_rows = _split_header(rows)
            for row_index, row in enumerate(data_rows, start=1):
                row_text = self._row_chunk_text(table_number, caption, headers, row, row_index, table_files.get("csv"))
                chunks.append(
                    {
                        "text": row_text,
                        "chunk_type": "table_row",
                        "metadata": {
                            **base_metadata,
                            "table_row_index": row_index,
                        },
                    }
                )
            return chunks

        lines = []
        if caption:
            lines.append(f"Caption: {caption}")
        if table_text:
            lines.append(table_text)
        if footnote:
            lines.append(f"Footnote: {footnote}")
        chunks.append(
            {
                "text": "\n\n".join(lines),
                "chunk_type": "table",
                "metadata": base_metadata,
            }
        )
        return chunks

    @staticmethod
    def _table_body(block: Dict[str, Any]) -> str:
        for key in ("table_body", "table_html", "html", "content", "text", "chart_content"):
            value = block.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _caption_text(block: Dict[str, Any], prefix: str) -> str:
        values = []
        for key in (f"{prefix}_caption", "caption", "chart_caption", "image_caption"):
            values.extend(_flatten_to_list(block.get(key)))
        return _normalize_space(" ".join(values))

    @staticmethod
    def _footnote_text(block: Dict[str, Any], prefix: str) -> str:
        values = []
        for key in (f"{prefix}_footnote", "footnote", "chart_footnote", "image_footnote"):
            values.extend(_flatten_to_list(block.get(key)))
        return _normalize_space(" ".join(values))

    @staticmethod
    def _table_rows(body: str) -> List[List[str]]:
        if not body.strip():
            return []
        if "<table" in body.lower() or "<tr" in body.lower():
            parser = _HTMLTableParser()
            parser.feed(body)
            return _normalize_rows(parser.rows)
        return _normalize_rows(_markdown_table_rows(body))

    def _write_table_files(
        self,
        doc_key: str,
        table_number: int,
        page_number: Optional[int],
        body: str,
        rows: Sequence[Sequence[str]],
        caption: str,
        footnote: str,
    ) -> Dict[str, str]:
        if not rows and not body.strip():
            return {}
        table_dir = self.tables_dir / doc_key
        table_dir.mkdir(parents=True, exist_ok=True)
        page_part = f"_page_{page_number:03d}" if page_number is not None else ""
        base = table_dir / f"table_{table_number:04d}{page_part}"
        files: Dict[str, str] = {}

        if rows:
            csv_path = base.with_suffix(".csv")
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(rows)
            files["csv"] = str(csv_path)

            markdown_path = base.with_suffix(".md")
            markdown_parts = []
            if caption:
                markdown_parts.append(f"Caption: {caption}")
            markdown_parts.append(_rows_to_markdown(rows))
            if footnote:
                markdown_parts.append(f"Footnote: {footnote}")
            markdown_path.write_text("\n\n".join(part for part in markdown_parts if part), encoding="utf-8")
            files["markdown"] = str(markdown_path)

        if body.strip() and ("<table" in body.lower() or "<tr" in body.lower()):
            html_path = base.with_suffix(".html")
            html_path.write_text(body.strip(), encoding="utf-8")
            files["html"] = str(html_path)

        return files

    @staticmethod
    def _row_chunk_text(
        table_number: int,
        caption: str,
        headers: Sequence[str],
        row: Sequence[str],
        row_index: int,
        csv_path: Optional[str],
    ) -> str:
        lines = []
        if caption:
            lines.append(f"Caption: {caption}")
        lines.append(_row_as_key_values(headers, row))
        return "\n".join(line for line in lines if line)

    @staticmethod
    def _image_chunk(
        block: Dict[str, Any],
        image_map: Dict[str, str],
        chapter_stack: Sequence[str],
        page_number: Optional[int],
    ) -> Tuple[str, Dict[str, Any]]:
        ref = str(block.get("img_path") or block.get("image_path") or "").strip()
        if not ref:
            return "", {}
        mapped = image_map.get(ref) or image_map.get(Path(ref).name)
        caption = INGEST._caption_text(block, "image")
        footnote = INGEST._footnote_text(block, "image")
        lines = []
        if caption:
            lines.append(f"Caption: {caption}")
        if footnote:
            lines.append(f"Footnote: {footnote}")
        if not lines:
            return "", {}
        metadata = {
            "heading_path": list(chapter_stack),
            "chapter_title": chapter_stack[-1] if chapter_stack else None,
            "page_start": page_number,
            "page_end": page_number,
            "image_path": mapped or ref,
            "bbox": block.get("bbox"),
        }
        return "\n".join(lines), metadata

    @staticmethod
    def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_manifest(self, results: Sequence[IngestedDocument]) -> None:
        payload = {
            "pdf_dir": str(self.config.pdf_dir),
            "output_dir": str(self.config.output_dir),
            "mineru_output_dir": str(self.config.mineru_output_dir),
            "mineru_env": self.config.mineru_env,
            "all_chunks_path": str(self.all_chunks_path),
            "documents": [result.__dict__ for result in results],
        }
        self.manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _looks_like_content_list_v2(data: Any) -> bool:
    return (
        isinstance(data, list)
        and bool(data)
        and isinstance(data[0], list)
        and (not data[0] or isinstance(data[0][0], dict))
    )


def _first_value(content: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in content and content[key] not in (None, "", []):
            return content[key]
    return None


def _looks_like_asset_file_ref(value: str) -> bool:
    stripped = value.strip()
    if not stripped or stripped.endswith("/"):
        return False
    return Path(stripped).suffix.lower() in IMAGE_SUFFIXES


def _build_references(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    references: List[Dict[str, Any]] = []

    source_ref: Dict[str, Any] = {
        "type": "pdf",
        "path": metadata.get("source_pdf_path"),
        "file": metadata.get("source_pdf"),
    }
    if metadata.get("page_start") is not None:
        source_ref["page_start"] = metadata.get("page_start")
    if metadata.get("page_end") is not None:
        source_ref["page_end"] = metadata.get("page_end")
    if metadata.get("bbox") is not None:
        source_ref["bbox"] = metadata.get("bbox")
    references.append(_drop_empty(source_ref))

    for key, ref_type in (
        ("table_csv_path", "table_csv"),
        ("table_markdown_path", "table_markdown"),
        ("table_html_path", "table_html"),
        ("table_image_path", "table_image"),
        ("image_path", "image"),
    ):
        value = metadata.get(key)
        if value:
            references.append({"type": ref_type, "path": value})

    if metadata.get("table_number") is not None:
        table_ref: Dict[str, Any] = {"type": "table", "table_number": metadata.get("table_number")}
        if metadata.get("table_row_index") is not None:
            table_ref["row_index"] = metadata.get("table_row_index")
        references.append(table_ref)

    return references


def _drop_empty(value: Dict[str, Any]) -> Dict[str, Any]:
    return {key: item for key, item in value.items() if item not in (None, "", [])}


def _safe_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    if cleaned:
        return cleaned[:160]
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"document_{digest}"


def _bool_cli(value: bool) -> str:
    return "true" if value else "false"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _human_page_number(page_idx: Any) -> Optional[int]:
    if page_idx is None:
        return None
    try:
        return int(page_idx) + 1
    except (TypeError, ValueError):
        return None


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def _flatten_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_flatten_content(item) for item in value if item is not None).strip()
    if isinstance(value, dict):
        preferred_keys = (
            "content",
            "html",
            "text",
            "path",
            "title_content",
            "paragraph_content",
            "table_body",
            "table_content",
            "math_content",
            "list_items",
            "code_content",
            "algorithm_content",
            "caption",
        )
        for key in preferred_keys:
            if key in value and value[key] not in (None, "", []):
                return _flatten_content(value[key])
        ignored = {"bbox", "page_idx", "level", "type", "img_path", "image_path"}
        return " ".join(_flatten_content(item) for key, item in value.items() if key not in ignored).strip()
    return str(value)


def _flatten_to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = []
        for item in value:
            flattened = _flatten_content(item)
            if flattened:
                items.append(_normalize_space(flattened))
        return items
    flattened = _flatten_content(value)
    return [_normalize_space(flattened)] if flattened else []


def _normalize_rows(rows: Sequence[Sequence[str]]) -> List[List[str]]:
    cleaned = [[_normalize_space(str(cell)) for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return []
    width = max(len(row) for row in cleaned)
    return [row + [""] * (width - len(row)) for row in cleaned]


def _markdown_table_rows(markdown: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped or "|" not in stripped:
            continue
        if TABLE_MARKDOWN_SEPARATOR_RE.match(stripped):
            continue
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        cells = [_normalize_space(cell) for cell in stripped.split("|")]
        rows.append(cells)
    return rows


def _split_header(rows: Sequence[Sequence[str]]) -> Tuple[List[str], List[List[str]]]:
    normalized = _normalize_rows(rows)
    if len(normalized) <= 1:
        return [], normalized
    header = normalized[0]
    data_rows = normalized[1:]
    return header, data_rows


def _rows_to_markdown(rows: Sequence[Sequence[str]]) -> str:
    normalized = _normalize_rows(rows)
    if not normalized:
        return ""
    header = [_escape_markdown_cell(cell) for cell in normalized[0]]
    separator = ["---"] * len(header)
    body = [[_escape_markdown_cell(cell) for cell in row] for row in normalized[1:]]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _row_as_key_values(headers: Sequence[str], row: Sequence[str]) -> str:
    if headers and len(headers) == len(row):
        pairs = []
        for header, value in zip(headers, row):
            label = header or "Column"
            pairs.append(f"{label}: {value}")
        return "\n".join(pairs)
    return "\n".join(f"Column {index + 1}: {value}" for index, value in enumerate(row))


def _strip_html(value: str) -> str:
    if not value:
        return ""
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return _normalize_space(without_tags)


def _parse_env_overrides(values: Sequence[str]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --mineru-env value {value!r}; expected KEY=VALUE")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --mineru-env value {value!r}; empty key")
        env[key] = raw
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse PDFs with MinerU and create ingest chunks.")
    parser.add_argument("--pdf-dir", default="pdf_files", help="Folder containing PDF files.")
    parser.add_argument("--output-dir", default="ingested_data", help="Folder for markdown, chunks, CSVs, images.")
    parser.add_argument("--mineru-output-dir", default=None, help="Folder for raw MinerU outputs.")
    parser.add_argument("--mineru-binary", default="mineru", help="MinerU CLI command.")
    parser.add_argument("--backend", default="pipeline", help="MinerU backend. Use pipeline for CPU.")
    parser.add_argument("--method", default="auto", choices=["auto", "txt", "ocr"], help="MinerU parsing method.")
    parser.add_argument("--lang", default="el", help="MinerU OCR language.")
    parser.add_argument("--no-recursive", action="store_true", help="Do not recurse into subfolders.")
    parser.add_argument(
        "--no-skip-existing-mineru",
        action="store_true",
        help="Run MinerU even if raw outputs already exist.",
    )
    parser.add_argument("--table-row-threshold", type=int, default=25)
    parser.add_argument("--table-char-threshold", type=int, default=4000)
    parser.add_argument("--disable-formula", action="store_true")
    parser.add_argument("--disable-table", action="store_true")
    parser.add_argument(
        "--extra-mineru-arg",
        action="append",
        default=[],
        help="Extra argument passed through to MinerU. Repeat for multiple args.",
    )
    parser.add_argument(
        "--mineru-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable passed to MinerU. Repeat for multiple vars.",
    )
    args = parser.parse_args()

    ingest = INGEST(
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        mineru_output_dir=args.mineru_output_dir,
        mineru_binary=args.mineru_binary,
        backend=args.backend,
        method=args.method,
        lang=args.lang,
        recursive=not args.no_recursive,
        skip_existing_mineru=not args.no_skip_existing_mineru,
        table_row_threshold=args.table_row_threshold,
        table_char_threshold=args.table_char_threshold,
        formula=not args.disable_formula,
        table=not args.disable_table,
        extra_mineru_args=args.extra_mineru_arg,
        mineru_env=_parse_env_overrides(args.mineru_env),
    )
    results = ingest.run()
    summary = {
        "documents": len(results),
        "chunks_path": str(ingest.all_chunks_path),
        "manifest_path": str(ingest.manifest_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
