import duckdb
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class DuckDBManager:
    def __init__(self, db_path: str = "data/geo_embeddings.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_extensions()
        self._create_tables()

    def _init_extensions(self):
        for ext in ['spatial', 'json']:
            try:
                self.conn.execute(f"INSTALL {ext};")
                self.conn.execute(f"LOAD {ext};")
            except Exception:
                pass

    def _create_tables(self):
        sql = """
        CREATE TABLE IF NOT EXISTS geospatial_embeddings (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR NOT NULL,
            source_type VARCHAR NOT NULL,
            properties JSON,
            geometry GEOMETRY,
            embedding FLOAT[],
            embedding_model VARCHAR,
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_geometry ON geospatial_embeddings USING RTREE(geometry);
        CREATE OR REPLACE FUNCTION cosine_similarity(a FLOAT[], b FLOAT[]) AS (
            array_dot_product(a, b) / (sqrt(array_dot_product(a, a)) * sqrt(array_dot_product(b, b)))
        );
        """
        self.conn.execute(sql)

    def insert_embedding(self, name: str, source_type: str, properties: Dict[str, Any],
                         geometry: Optional[str], embedding: np.ndarray, model: str) -> str:
        embedding_list = embedding.tolist() if embedding is not None else []
        sql = """
        INSERT INTO geospatial_embeddings (name, source_type, properties, geometry, embedding, embedding_model)
        VALUES (?, ?, ?, ST_GeomFromGeoJSON(?), ?, ?)
        RETURNING id;
        """
        result = self.conn.execute(sql, (name, source_type, json.dumps(properties), geometry,
                                        embedding_list, model)).fetchone()
        return str(result[0])

    def get_stats(self) -> Dict[str, Any]:
        sql = "SELECT COUNT(*), COUNT(DISTINCT source_type), COUNT(DISTINCT embedding_model) FROM geospatial_embeddings"
        total, source_types, models = self.conn.execute(sql).fetchone()
        return {
            "total_embeddings": total,
            "source_types": source_types,
            "models": models
        }

    def close(self):
        self.conn.close()
