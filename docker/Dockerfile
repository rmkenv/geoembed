FROM python:3.10-slim

RUN apt-get update && apt-get install -y gdal-bin libgdal-dev

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY data/ ./data/
COPY scripts/ ./scripts/

RUN python scripts/init_database.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
