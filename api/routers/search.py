from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np

from core.duckdb_manager import DuckDBManager
from core.context_lang import ContextLanguageEmbedder
from api.main import get_db_manager

router = APIRouter()
embedder = ContextLanguageEmbedder()

class SearchRequest(BaseModel):
    query_text: str
    k: int = 10
    source_type: Optional[str] = None

class SearchResult(BaseModel):
    id: str
    name: str
    source_type: str
    properties: Dict[str, Any]
    similarity: float

@router.post("/semantic", response_model=List[SearchResult])
async def semantic_search(request: SearchRequest, db: DuckDBManager = Depends(get_db_manager)):
    try:
        q_emb = embedder.embed_text(request.query_text)
        results = db.similarity_search(query_embedding=q_emb, k=request.k, source_type=request.source_type)
        return [SearchResult(**r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
