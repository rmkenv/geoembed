from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from core.duckdb_manager import DuckDBManager
from routers import vector, search

db_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_manager
    db_manager = DuckDBManager("data/geo_embeddings.duckdb")
    yield
    if db_manager:
        db_manager.close()

app = FastAPI(
    title="Geospatial Embeddings Platform",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_manager() -> DuckDBManager:
    return db_manager

app.include_router(vector.router, prefix="/embed/vector", tags=["vector-embeddings"])
app.include_router(search.router, prefix="/search", tags=["search"])

@app.get("/")
async def root():
    return {"message": "Geospatial Embeddings Platform v2.0", "db_status": "connected"}

@app.get("/stats")
async def stats(db: DuckDBManager = Depends(get_db_manager)):
    return db.get_stats()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
