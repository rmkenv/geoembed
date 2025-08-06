import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data

def test_stats():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_embeddings" in data

def test_vector_embedding_and_search():
    geojson_feature = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [-74.006, 40.7128]},
        "properties": {"name": "Test City", "country": "Testland"}
    }
    # Embed vector
    response = client.post("/embed/vector/", json={"features": [geojson_feature]})
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["feature_count"] == 1
    embedding_id = res_data["embedding_ids"][0]

    # Semantic search
    response = client.post("/search/semantic", json={"query_text": "Test City"})
    assert response.status_code == 200
    results = response.json()
    assert any(r["name"] == "Test City" for r in results)
