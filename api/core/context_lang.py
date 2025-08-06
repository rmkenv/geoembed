from sentence_transformers import SentenceTransformer
import h3
from typing import Dict, Any, Optional
import numpy as np

class ContextLanguageEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def _generate_context_text(self, feature: Dict[str, Any], template: Optional[str] = None,
                               include_topology: bool = True) -> str:
        props = feature.get('properties', {})
        geometry = feature.get('geometry', {})
        
        name = props.get('name', 'Unknown location')
        text_parts = [name]

        if include_topology and geometry and geometry.get('type') == 'Point':
            lon, lat = geometry['coordinates']
            h3_index = h3.geo_to_h3(lat, lon, resolution=9)
            text_parts.append(f"located at {lon:.4f}, {lat:.4f} in H3 cell {h3_index}")

        for k, v in props.items():
            if k != 'name' and isinstance(v, (str, int, float)):
                text_parts.append(f"{k}: {v}")

        if template:
            try:
                return template.format(name=name, properties=props, **geometry)
            except Exception:
                pass

        return ". ".join(text_parts)

    def embed_feature(self, feature: Dict[str, Any], template: Optional[str] = None,
                      include_topology: bool = True) -> np.ndarray:
        text = self._generate_context_text(feature, template, include_topology)
        return self.model.encode(text)

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text)
