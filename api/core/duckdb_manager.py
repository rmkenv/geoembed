import duckdb
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DuckDBManager:
    def __init__(self, db_path: str = "data/geo_embeddings.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_extensions()
        self._create_tables()

    def _init_extensions(self):
        """Initialize required DuckDB extensions with proper error handling."""
        required_extensions = ['spatial', 'json']
        
        for ext in required_extensions:
            try:
                # First, try to install the extension
                logger.info(f"Installing {ext} extension...")
                self.conn.execute(f"INSTALL {ext};")
                logger.info(f"Successfully installed {ext} extension")
            except Exception as install_error:
                logger.warning(f"Install command for {ext} extension failed: {install_error}")
                # This might be expected if extension is already installed
            
            try:
                # Now try to load the extension
                logger.info(f"Loading {ext} extension...")
                self.conn.execute(f"LOAD {ext};")
                logger.info(f"Successfully loaded {ext} extension")
            except Exception as load_error:
                logger.error(f"Failed to load {ext} extension: {load_error}")
                # If loading fails, try installing first then loading again
                try:
                    logger.info(f"Retrying: Installing {ext} extension before loading...")
                    self.conn.execute(f"INSTALL {ext};")
                    self.conn.execute(f"LOAD {ext};")
                    logger.info(f"Successfully installed and loaded {ext} extension on retry")
                except Exception as retry_error:
                    logger.error(f"Failed to install and load {ext} extension on retry: {retry_error}")
                    raise RuntimeError(f"Required extension '{ext}' could not be installed and loaded. "
                                     f"This is needed for DuckDB functionality. "
                                     f"Install error: {install_error}, Load error: {load_error}, "
                                     f"Retry error: {retry_error}")

    def _create_tables(self):
        """Create the geospatial embeddings table and related objects."""
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
        CREATE OR REPLACE FUNCTION cosine_similarity(a, b) AS (
            array_dot_product(a, b) / (sqrt(array_dot_product(a, a)) * sqrt(array_dot_product(b, b)))
        );
        """
        try:
            self.conn.execute(sql)
            logger.info("Successfully created geospatial_embeddings table and related objects")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise RuntimeError(f"Database table creation failed. This might indicate missing "
                             f"extensions or incompatible SQL syntax. Error: {e}")

    def insert_embedding(self, name: str, source_type: str, properties: Dict[str, Any],
                         geometry: Optional[str], embedding: np.ndarray, model: str) -> str:
        """Insert a new embedding record into the database."""
        embedding_list = embedding.tolist() if embedding is not None else []
        sql = """
        INSERT INTO geospatial_embeddings (name, source_type, properties, geometry, embedding, embedding_model)
        VALUES (?, ?, ?, ST_GeomFromGeoJSON(?), ?, ?)
        RETURNING id;
        """
        try:
            result = self.conn.execute(sql, (name, source_type, json.dumps(properties), geometry,
                                            embedding_list, model)).fetchone()
            return str(result[0])
        except Exception as e:
            logger.error(f"Failed to insert embedding: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the embeddings in the database."""
        sql = "SELECT COUNT(*), COUNT(DISTINCT source_type), COUNT(DISTINCT embedding_model) FROM geospatial_embeddings"
        try:
            total, source_types, models = self.conn.execute(sql).fetchone()
            return {
                "total_embeddings": total,
                "source_types": source_types,
                "models": models
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
