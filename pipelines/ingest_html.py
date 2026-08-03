"""HTML ingest pipeline powered by Playwright and custom density algorithms.

This module handles HTML ingestion:
- parse HTML to structured JSONL chunks while preserving CSS styling
- extract only the main article content via fallback selectors and density checks
- parse tables structurally, saving them to CSV and Markdown formats
- create retrieval-ready chunks without building a vector store
"""

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from playwright.sync_api import sync_playwright


@dataclass
class IngestedHTMLDocument:
    source_html: str
    chunks_path: str
    chunk_count: int


@dataclass
class HTMLIngestConfig:
    html_dir: Path = field(default_factory=lambda: Path("html_files"))
    output_dir: Path = field(default_factory=lambda: Path("ingested_data"))
    recursive: bool = True
    headless: bool = True


def _safe_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    if cleaned:
        return cleaned[:160]
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"document_{digest}"


class HTMLIngest:
    def __init__(
        self,
        html_dir: str = "html_files",
        output_dir: str = "ingested_data",
        recursive: bool = True,
        headless: bool = True,
    ) -> None:
        self.config = HTMLIngestConfig(
            html_dir=Path(html_dir),
            output_dir=Path(output_dir),
            recursive=recursive,
            headless=headless,
        )

        self.chunks_dir = self.config.output_dir / "chunks"
        self.tables_dir = self.config.output_dir / "tables"
        self.manifest_path = self.config.output_dir / "manifest.json"
        self.all_chunks_path = self.chunks_dir / "all_chunks.jsonl"

    def run(self) -> List[IngestedHTMLDocument]:
        self._prepare_output_dirs()
        html_files = self._discover_htmls()
        
        if not html_files:
            print(f"No html files found in {self.config.html_dir}")
            return []

        results: List[IngestedHTMLDocument] = []
        all_chunks: List[Dict[str, Any]] = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.config.headless)
            page = browser.new_page(bypass_csp=True)

            for html_path in html_files:
                print(f"-> Opened : {html_path.name}")
                try:
                    result, chunks = self.ingest_html(page, html_path)
                    if result:
                        results.append(result)
                        all_chunks.extend(chunks)
                except Exception as e:
                    print(f"   [Error] Failed in file {html_path.name}: {e}")

            browser.close()

        self._write_jsonl(self.all_chunks_path, all_chunks)
        self._write_manifest(results)
        return results

    def ingest_html(self, page: Any, html_path: Path) -> Tuple[Optional[IngestedHTMLDocument], List[Dict[str, Any]]]:
        doc_key = _safe_filename(html_path.stem)
        file_url = html_path.absolute().as_uri()
        
        page.goto(file_url, wait_until="load", timeout=60000)
        
        raw_blocks = []
        
        for frame in page.frames:
            try:
                blocks = frame.evaluate(r"""() => {
                    let data = [];
                    
                    let bestContainer = document.querySelector('article[role="article"]') 
                                     || document.querySelector('main[role="main"]') 
                                     || document.querySelector('.portlet-body');
                                     
                                        
                    if (!bestContainer) {
                        bestContainer = document.querySelector('article, main, #main, #content, .content');
                    }

                   
                    if (!bestContainer) {
                        let candidates = document.querySelectorAll('div, section, body');
                        let maxText = 0;
                        candidates.forEach(c => {
                            let textLen = c.textContent.trim().length;
                            if (textLen > maxText) {
                                maxText = textLen;
                                bestContainer = c;
                            }
                        });
                    }

                    if (!bestContainer || bestContainer.textContent.trim().length < 20) {
                        return [];
                    }

                    bestContainer.querySelectorAll('header, footer, aside, .table-of-contents').forEach(el => el.remove());
                    
                    bestContainer.querySelectorAll('nav').forEach(el => {
                        let pClass = el.getAttribute('class') || '';
                        if (!pClass.includes('related-links') && !pClass.includes('ullinks')) {
                            el.remove();
                        }
                    });

                    let elements = bestContainer.querySelectorAll('h1, h2, h3, h4, h5, h6, p, li, table');
                    let extractedTexts = new Set();

                    elements.forEach((el) => {
                        let tag = el.tagName.toLowerCase();
                        let text = el.textContent.trim().replace(/\s+/g, ' ');

                        if (text.length < 5) return;
                        if (tag !== 'table' && el.closest('table')) return;
                        if (tag === 'p' && el.closest('li')) return;
                        if (extractedTexts.has(text)) return;

                        if (tag !== 'table') {
                            let linkTextLen = 0;
                            el.querySelectorAll('a').forEach(a => {
                                linkTextLen += a.textContent.trim().replace(/\s+/g, ' ').length;
                            });
                            
                            let maxDensity = (tag === 'li') ? 0.85 : 0.50;
                            if ((linkTextLen / text.length) > maxDensity) return;
                        }

                        extractedTexts.add(text);

                        let styles = window.getComputedStyle(el);
                        let styling = {
                            color: styles.color,
                            fontSize: styles.fontSize,
                            fontWeight: styles.fontWeight,
                            fontFamily: styles.fontFamily,
                            backgroundColor: styles.backgroundColor,
                            border: styles.border,
                            margin: styles.margin
                        };

                        if (tag === 'table') {
                            let tableRows = [];
                            el.querySelectorAll('tr').forEach(row => {
                                let rowData = [];
                                row.querySelectorAll('th, td').forEach(cell => {
                                    rowData.push(cell.textContent.trim().replace(/\s+/g, ' '));
                                });
                                if (rowData.join('').trim() !== '') {
                                    tableRows.push(rowData);
                                }
                            });

                            if (tableRows.length > 0) {
                                data.push({
                                    tag: 'table',
                                    text: text,
                                    table_rows: tableRows,
                                    styling: styling
                                });
                            }
                        } else {
                            data.push({
                                tag: tag,
                                text: text,
                                styling: styling
                            });
                        }
                    });

                    return data;
                }""")
                        
                if blocks:
                    raw_blocks.extend(blocks)
                    
            except Exception as e:
                continue

        if not raw_blocks:
            print(f"   [Warning] The tool did not find content in {html_path.name}")
            return None, []

        chunks = self._build_chunks(html_path, doc_key, raw_blocks)
        doc_chunks_path = self.chunks_dir / f"{doc_key}.jsonl"
        self._write_jsonl(doc_chunks_path, chunks)

        result = IngestedHTMLDocument(
            source_html=str(html_path),
            chunks_path=str(doc_chunks_path),
            chunk_count=len(chunks),
        )
        return result, chunks

    def _prepare_output_dirs(self) -> None:
        for directory in (self.config.output_dir, self.chunks_dir, self.tables_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _discover_htmls(self) -> List[Path]:
        pattern = "**/*.html" if self.config.recursive else "*.html"
        return sorted(path for path in self.config.html_dir.glob(pattern) if path.is_file())

    def _write_table_files(self, doc_key: str, chunk_id: str, rows: List[List[str]]) -> Dict[str, str]:
        if not rows:
            return {}
            
        doc_tables_dir = self.tables_dir / doc_key
        doc_tables_dir.mkdir(parents=True, exist_ok=True)
        
        safe_chunk_id = chunk_id.replace(':', '_')
        base_path = doc_tables_dir / f"table_{safe_chunk_id}"
        files = {}

        csv_path = base_path.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerows(rows)
        files["csv_path"] = str(csv_path)

        md_path = base_path.with_suffix(".md")
        header = rows[0]
        separator = ["---"] * len(header)
        
        md_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(separator) + " |"
        ]
        for row in rows[1:]:
            padded_row = row + [""] * (len(header) - len(row))
            clean_row = [str(cell).replace("\n", " ") for cell in padded_row]
            md_lines.append("| " + " | ".join(clean_row) + " |")
            
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        files["markdown_path"] = str(md_path)

        return files

    def _build_chunks(self, html_path: Path, doc_key: str, raw_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        
        for index, block in enumerate(raw_blocks):
            chunk_id = f"{doc_key}:{len(chunks) + 1:05d}"
            tag = block["tag"]
            
            chunk_type = "table" if tag == "table" else ("heading" if tag.startswith("h") else "text")
            
            chunk_metadata = {
                "source_html": html_path.name,
                "source_html_path": str(html_path),
                "parser": "playwright_custom_density",
                "chunk_type": chunk_type,
                "html_tag": tag,
                "styling": block["styling"],
                "embed": True
            }

            if tag == "table":
                rows = block.get("table_rows", [])
                table_files = self._write_table_files(doc_key, chunk_id, rows)
                chunk_metadata["table_csv_path"] = table_files.get("csv_path")
                chunk_metadata["table_markdown_path"] = table_files.get("markdown_path")
            
            chunks.append({
                "id": chunk_id,
                "text": block["text"],
                "content": block["text"],
                "metadata": chunk_metadata,
            })
            
        return chunks

    @staticmethod
    def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_manifest(self, results: Sequence[IngestedHTMLDocument]) -> None:
        payload = {
            "html_dir": str(self.config.html_dir),
            "output_dir": str(self.config.output_dir),
            "all_chunks_path": str(self.all_chunks_path),
            "documents": [result.__dict__ for result in results],
        }
        self.manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse HTMLs with Playwright and create ingest chunks.")
    parser.add_argument("--html-dir", default="html_files", help="Folder containing HTML files.")
    parser.add_argument("--output-dir", default="ingested_data", help="Folder for JSONL chunks and manifest.")
    parser.add_argument("--no-recursive", action="store_true", help="Do not recurse into subfolders.")
    parser.add_argument("--show-browser", action="store_true", help="Run Playwright in non-headless mode.")
    args = parser.parse_args()

    ingest = HTMLIngest(
        html_dir=args.html_dir,
        output_dir=args.output_dir,
        recursive=not args.no_recursive,
        headless=not args.show_browser,
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