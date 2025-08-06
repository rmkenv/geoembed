from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import json
from core.duckdb_manager import DuckDBManager
from core.context_lang import ContextLanguageEmbedder
from api.main import get_db_manager

router = APIRouter()
embedder = ContextLanguageEmbedder()

class GeoJSONFeature(BaseModel):
    type: str
    geometry: Dict[str, Any]
    properties: Dict[str, Any]

class VectorEmbeddingRequest(BaseModel):
    features: List[GeoJSONFeature]
    context_template: Optional[str] = None
    include_topology: bool = True

class VectorEmbeddingResponse(BaseModel):
    feature_count: int
    embedding_ids: List[str]
    model_info: Dict[str, Any]

@router.post("/", response_model=VectorEmbeddingResponse)
async def embed_vectors(request: VectorEmbeddingRequest, db: DuckDBManager = Depends(get_db_manager)):
    try:
        embedding_ids = []
        for f in request.features:
            emb = embedder.embed_feature(f.dict(), request.context_template, request.include_topology)
            emb_id = db.insert_embedding(
                name=f.properties.get('name', 'Unknown'),
                source_type="vector",
                properties=f.properties,
                geometry=json.dumps(f.geometry) if f.geometry else None,
                embedding=emb,
                model=embedder.model_name
            )
            embedding_ids.append(emb_id)
        return VectorEmbeddingResponse(feature_count=len(embedding_ids), embedding_ids=embedding_ids,
                                       model_info={"model": embedder.model_name, "embedding_dim": embedder.embedding_dim})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
