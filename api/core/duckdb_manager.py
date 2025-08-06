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
        self.spatial_enabled = False  # Track if spatial extension is available
        self._init_extensions()
        self._create_tables()

    def _init_extensions(self):
        """Initialize DuckDB extensions with proper error handling."""
        # JSON extension is required for basic functionality
        self._load_extension('json', required=True)
        
        # Spatial extension is optional - application can work without it
        self.spatial_enabled = self._load_extension('spatial', required=False)
        
        if not self.spatial_enabled:
            logger.warning("Spatial extension is not available. Geospatial features will be disabled. "
                         "This is common on ARM64 systems where the spatial extension may not be available.")

    def _load_extension(self, ext_name: str, required: bool = True) -> bool:
        """Load a single extension with proper error handling."""
        install_error = None
        
        try:
            # First, try to install the extension
            logger.info(f"Installing {ext_name} extension...")
            self.conn.execute(f"INSTALL {ext_name};")
            logger.info(f"Successfully installed {ext_name} extension")
        except Exception as e:
            install_error = e
            logger.warning(f"Install command for {ext_name} extension failed: {install_error}")
            # This might be expected if extension is already installed
        
        try:
            # Now try to load the extension
            logger.info(f"Loading {ext_name} extension...")
            self.conn.execute(f"LOAD {ext_name};")
            logger.info(f"Successfully loaded {ext_name} extension")
            return True
        except Exception as load_error:
            logger.error(f"Failed to load {ext_name} extension: {load_error}")
            
            # If loading fails, try installing first then loading again
            try:
                logger.info(f"Retrying: Installing {ext_name} extension before loading...")
                self.conn.execute(f"INSTALL {ext_name};")
                self.conn.execute(f"LOAD {ext_name};")
                logger.info(f"Successfully installed and loaded {ext_name} extension on retry")
                return True
            except Exception as retry_error:
                logger.error(f"Failed to install and load {ext_name} extension on retry: {retry_error}")
                
                if required:
                    install_error_msg = f"Install error: {install_error}" if install_error else "No install error"
                    raise RuntimeError(f"Required extension '{ext_name}' could not be installed and loaded. "
                                     f"This is needed for DuckDB functionality. "
                                     f"{install_error_msg}, Load error: {load_error}, "
                                     f"Retry error: {retry_error}")
                else:
                    logger.warning(f"Optional extension '{ext_name}' could not be loaded. "
                                 f"Related functionality will be disabled.")
                    return False

    def _create_tables(self):
        """Create the embeddings table and related objects."""
        try:
            if self.spatial_enabled:
                # Full spatial table with geometry column and spatial index
                # Create table first
                self.conn.execute("""
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
                """)
                
                # Create spatial index separately
                try:
                    self.conn.execute("CREATE INDEX IF NOT EXISTS idx_geometry ON geospatial_embeddings USING RTREE(geometry);")
                except Exception as idx_error:
                    logger.warning(f"Could not create spatial index: {idx_error}")
                    
            else:
                # Basic table without spatial features
                self.conn.execute("""
                CREATE TABLE IF NOT EXISTS geospatial_embeddings (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR NOT NULL,
                    source_type VARCHAR NOT NULL,
                    properties JSON,
                    geometry_json VARCHAR,  -- Store geometry as JSON string instead of GEOMETRY type
                    embedding FLOAT[],
                    embedding_model VARCHAR,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                """)
            
            # Create cosine similarity macro
            self.conn.execute("""
            CREATE OR REPLACE MACRO cosine_similarity(a, b) AS (
                list_dot_product(a::FLOAT[], b::FLOAT[]) / 
                (sqrt(list_dot_product(a::FLOAT[], a::FLOAT[])) * sqrt(list_dot_product(b::FLOAT[], b::FLOAT[])))
            );
            """)
            
            table_type = "spatial" if self.spatial_enabled else "basic"
            logger.info(f"Successfully created geospatial_embeddings table ({table_type} mode) and related objects")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise RuntimeError(f"Database table creation failed. This might indicate missing "
                             f"extensions or incompatible SQL syntax. Error: {e}")

    def insert_embedding(self, name: str, source_type: str, properties: Dict[str, Any],
                         geometry: Optional[str], embedding: np.ndarray, model: str) -> str:
        """Insert a new embedding record into the database."""
        embedding_list = embedding.tolist() if embedding is not None else []
        
        if self.spatial_enabled:
            # Use spatial functions when available
            sql = """
            INSERT INTO geospatial_embeddings (name, source_type, properties, geometry, embedding, embedding_model)
            VALUES (?, ?, ?, ST_GeomFromGeoJSON(?), ?, ?)
            RETURNING id;
            """
            params = (name, source_type, json.dumps(properties), geometry, embedding_list, model)
        else:
            # Store geometry as JSON string when spatial extension is not available
            sql = """
            INSERT INTO geospatial_embeddings (name, source_type, properties, geometry_json, embedding, embedding_model)
            VALUES (?, ?, ?, ?, ?, ?)
            RETURNING id;
            """
            params = (name, source_type, json.dumps(properties), geometry, embedding_list, model)
        
        try:
            result = self.conn.execute(sql, params).fetchone()
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
                "models": models,
                "spatial_enabled": self.spatial_enabled
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise

    def search_similar_embeddings(self, query_embedding: np.ndarray, limit: int = 10, 
                                 similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar embeddings using cosine similarity."""
        query_list = query_embedding.tolist()
        
        if self.spatial_enabled:
            geometry_col = "ST_AsGeoJSON(geometry) as geometry"
        else:
            geometry_col = "geometry_json as geometry"
        
        sql = f"""
        SELECT id, name, source_type, properties, {geometry_col}, 
               cosine_similarity(embedding, ?) as similarity,
               embedding_model, created_at
        FROM geospatial_embeddings
        WHERE cosine_similarity(embedding, ?) >= ?
        ORDER BY similarity DESC
        LIMIT ?
        """
        
        try:
            results = self.conn.execute(sql, (query_list, query_list, similarity_threshold, limit)).fetchall()
            return [
                {
                    "id": str(row[0]),
                    "name": row[1],
                    "source_type": row[2],
                    "properties": json.loads(row[3]) if row[3] else {},
                    "geometry": row[4],
                    "similarity": row[5],
                    "embedding_model": row[6],
                    "created_at": row[7]
                }
                for row in results
            ]
        except Exception as e:
            logger.error(f"Failed to search similar embeddings: {e}")
            raise

    def similarity_search(self, query_embedding: np.ndarray, k: int = 10, 
                         source_type: Optional[str] = None, similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar embeddings using cosine similarity (compatibility method)."""
        query_list = query_embedding.tolist()
        
        if self.spatial_enabled:
            geometry_col = "ST_AsGeoJSON(geometry) as geometry"
        else:
            geometry_col = "geometry_json as geometry"
        
        # Build the WHERE clause
        where_clause = "WHERE cosine_similarity(embedding, ?) >= ?"
        params = [query_list, query_list, similarity_threshold]
        
        if source_type:
            where_clause += " AND source_type = ?"
            params.append(source_type)
        
        sql = f"""
        SELECT id, name, source_type, properties, {geometry_col}, 
               cosine_similarity(embedding, ?) as similarity,
               embedding_model, created_at
        FROM geospatial_embeddings
        {where_clause}
        ORDER BY similarity DESC
        LIMIT ?
        """
        params.append(k)
        
        try:
            results = self.conn.execute(sql, params).fetchall()
            return [
                {
                    "id": str(row[0]),
                    "name": row[1],
                    "source_type": row[2],
                    "properties": json.loads(row[3]) if row[3] else {},
                    "geometry": row[4],
                    "similarity": row[5],
                    "embedding_model": row[6],
                    "created_at": row[7]
                }
                for row in results
            ]
        except Exception as e:
            logger.error(f"Failed to search similar embeddings: {e}")
            raise

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
