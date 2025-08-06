import json
from pathlib import Path
import argparse
import sys
sys.path.append(str(Path(__file__).parent.parent))  # Add api to path

from api.core.duckdb_manager import DuckDBManager
from api.core.context_lang import ContextLanguageEmbedder

def load_sample_cities():
    return [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-74.006, 40.7128]},
            "properties": {"name": "New York", "country": "USA", "population": 8175133, "description": "Major metropolitan area"}
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-0.1278, 51.5074]},
            "properties": {"name": "London", "country": "UK", "population": 8982000, "description": "Capital of UK"}
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [139.6503,35.6762]},
            "properties": {"name": "Tokyo", "country": "Japan", "population": 13960000, "description": "Largest metro area in Japan"}
        }
    ]

def main():
    parser = argparse.ArgumentParser(description="Initialize geospatial embedding DB")
    parser.add_argument('--db-path', default='data/geo_embeddings.duckdb')
    args = parser.parse_args()

    db = DuckDBManager(args.db_path)
    embedder = ContextLanguageEmbedder()

    features = load_sample_cities()
    for f in features:
        emb = embedder.embed_feature(f)
        db.insert_embedding(
            name=f['properties']['name'],
            source_type="vector",
            properties=f['properties'],
            geometry=json.dumps(f['geometry']),
            embedding=emb,
            model=embedder.model_name
        )
    print(f"Inserted {len(features)} sample embeddings.")
    print("DB stats:", db.get_stats())
    db.close()

if __name__ == "__main__":
    main()
