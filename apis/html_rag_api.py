from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from ISG_Rule_Finder import Pipeline

app = FastAPI()

rag = Pipeline()
rag.valves.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

print("Loading FAISS, BM25 and CrossEncoder...")
rag._initialize()
print("RAG server is ready.")

class SearchRequest(BaseModel):
    query: str

@app.post("/search")
async def search_documents(request: SearchRequest):
    try:
        docs = rag._retrieve_and_rerank(
            query=request.query,
            top_k=int(rag.valves.TOP_K),
            candidates_k=int(rag.valves.CANDIDATES_K),
            alpha=float(rag.valves.FUSION_ALPHA)
        )
        
        if not docs:
            return {"results": "No files found."}

        context_text, _ = rag._build_context(docs, max_chars=int(rag.valves.CONTEXT_CHAR_LIMIT))
        
        return {"results": context_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
