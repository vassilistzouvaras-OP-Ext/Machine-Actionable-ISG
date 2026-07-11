

 ## Machine-Actionable Interinstitutional Style Guide for Grounded AI and Semantic Knowledge Retrieval

The **ISG Access Interface** transforms the **Interinstitutional Style Guide (ISG)** into a machine-actionable, AI-ready knowledge base. It combines multimodal document processing, semantic technologies, knowledge graphs, Graph-RAG, and controlled vocabularies to provide trustworthy, explainable and interoperable access to editorial knowledge for both humans and AI systems.

Rather than treating the ISG as a collection of static PDF documents, the project represents its editorial knowledge as interconnected semantic entities that can be searched, validated, queried and consumed programmatically.

---

## Overview

The Interinstitutional Style Guide (ISG) is the authoritative reference for drafting publications across the European Union institutions. While comprehensive, it is primarily intended for human consultation.

The ISG Access Interface transforms the guide into structured semantic knowledge that can be consumed by AI systems, knowledge graphs, validation engines and authoring tools.

The platform combines:

- 📄 Multimodal document understanding
- 🧠 Semantic knowledge graphs
- 🔍 Graph Retrieval-Augmented Generation (Graph-RAG)
- 🤖 Grounded Large Language Models
- 📚 Controlled vocabularies
- ✅ Automated compliance checking

The result is an AI assistant that provides answers grounded exclusively in official ISG content.

---

# Features

## Machine-Actionable Style Guide

The ISG is transformed from human-readable documentation into structured semantic knowledge.

Editorial rules become machine-readable entities that can be:

- queried
- linked
- validated
- reused
- reasoned upon
- consumed by AI

---

## Multimodal Knowledge Extraction

Knowledge is extracted from multiple complementary evidence sources including:

- PDF documents
- document structure
- headings
- paragraphs
- tables
- figures
- captions
- images
- hyperlinks
- metadata
- references

Each modality contributes complementary evidence to the knowledge graph.

---

## Semantic Knowledge Graph

The extracted knowledge is represented using Semantic Web standards.

Supported technologies include:

- RDF
- SKOS
- OWL
- SHACL
- SPARQL

The graph represents:

- editorial concepts
- writing rules
- relationships
- terminology
- document structure
- examples
- exceptions
- provenance

---

## Controlled Vocabularies

The project integrates semantic reference data including:

- EuroVoc
- EU Authority Tables
- institutional taxonomies
- multilingual labels
- persistent URIs

This enables interoperability across EU semantic assets.

---

## Graph-RAG

Unlike traditional vector-only Retrieval-Augmented Generation systems, the ISG Access Interface combines multiple retrieval techniques.

Retrieval pipeline:

- Semantic graph traversal
- Vector similarity search
- BM25 lexical retrieval
- Metadata filtering
- Cross-encoder re-ranking
- Provenance-aware retrieval

Graph relationships provide contextual information that cannot be captured by embeddings alone.

---

## Grounded AI

The language model never answers using internal memory alone.

Every response is generated from retrieved ISG knowledge including:

- document fragments
- semantic entities
- graph relationships
- references
- provenance

This significantly reduces hallucinations while improving transparency and trustworthiness.

---

## Explainable Responses

Every generated answer can be traced back to:

- ISG sections
- supporting rules
- semantic entities
- related concepts
- original evidence

Users understand not only *what* the answer is, but also *why*.

---

## Automated Compliance

The semantic representation enables automatic validation against ISG rules.

Potential applications include:

- document review
- editorial assistance
- writing support
- publication workflows
- quality assurance
- style validation

---

# Architecture

```text
                    ISG Documents
         PDFs • HTML • Tables • Images
                      │
                      ▼
        Multimodal Knowledge Extraction
                      │
     ┌────────────────┴─────────────────┐
     │                                  │
 Text Processing                 Visual Processing
 Metadata                        Layout Analysis
 Tables                          Image Analysis
 OCR                             Captions
     │                                  │
     └────────────────┬─────────────────┘
                      ▼
           Semantic Knowledge Graph
          RDF • SKOS • OWL • SHACL
                      │
                      ▼
         Controlled Vocabularies
      EuroVoc • Authority Tables
                      │
                      ▼
               Graph-RAG Engine
        Graph + Vector Retrieval
                      │
              Cross-Encoder Ranking
                      │
                      ▼
              Grounded LLM Layer
                      │
                      ▼
          ISG Access Interface API
                      │
         Chat • Search • Validation
```

---

# Technology Stack

## Semantic Technologies

- RDF
- SKOS
- OWL
- SHACL
- SPARQL

## Artificial Intelligence

- Large Language Models
- Retrieval-Augmented Generation
- Graph-RAG
- Semantic Search
- Explainable AI

## Knowledge Retrieval

- Graph Traversal
- FAISS
- BM25
- Cross-Encoder Re-ranking
- Sentence Transformers

## Data Processing

- PDF Parsing
- OCR
- Table Extraction
- Metadata Extraction
- Image Processing

---

# Knowledge Pipeline

```text
ISG Documents
      │
      ▼
Multimodal Parsing
      │
      ▼
Semantic Entity Extraction
      │
      ▼
Knowledge Graph Construction
      │
      ▼
Controlled Vocabulary Linking
      │
      ▼
Graph + Vector Indexing
      │
      ▼
Grounded Retrieval
      │
      ▼
LLM Answer Generation
      │
      ▼
Explainable Response
```

---

# Example Questions

The ISG Access Interface can answer questions such as:

- How should EU legal acts be cited?
- Which abbreviations are permitted?
- How should multilingual publications be written?
- What typography rules apply to headings?
- When should italics be used?
- How should tables be formatted?
- Which ISG rule supports this recommendation?
- Which section of the ISG discusses this topic?

All answers are grounded exclusively in official ISG content.

---

# Applications

The platform can support:

- Editorial assistants
- AI writing assistants
- Publication workflows
- Semantic search
- Knowledge discovery
- Automated validation
- Document review
- Intelligent authoring
- Editorial training
- Institutional helpdesks

---

# Design Principles

The project is based on the following principles:

- Machine Actionability
- Explainable AI
- Grounded AI
- Semantic Interoperability
- FAIR Knowledge
- Provenance Preservation
- Open Standards
- Reusability
- Extensibility

---

# Roadmap

Future developments include:

- Machine-actionable editorial rules
- Additional Publications Office guidance
- Multilingual semantic representations
- MCP integration
- Workflow integration
- Advanced semantic reasoning
- Agentic AI support
- Integration with additional EU knowledge assets

---

# Repository Structure

```
.
├── data/                 # ISG source material
├── ingestion/            # Document ingestion pipelines
├── parsers/              # Multimodal extraction
├── graph/                # Knowledge graph generation
├── retrieval/            # Graph-RAG implementation
├── embeddings/           # Vector indexing
├── api/                  # FastAPI services
├── ui/                   # User interface
├── notebooks/            # Experiments
├── tests/
└── docs/
```

---

# License

This project demonstrates a machine-actionable implementation of the Interinstitutional Style Guide using Semantic Web technologies and grounded AI.

---

# Acknowledgements

Developed as part of ongoing work on semantic technologies, knowledge graphs, AI-assisted editorial workflows and machine-actionable documentation at the **Publications Office of the European Union**.
