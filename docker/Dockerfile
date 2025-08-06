FROM python:3.10-slim

RUN apt-get update && apt-get install -y gdal-bin libgdal-dev

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-install DuckDB extensions to avoid runtime network issues
# This is especially important for ARM64 containers that may have network connectivity issues
RUN python -c "\
import duckdb; \
conn = duckdb.connect(':memory:'); \
try: \
    conn.execute('INSTALL spatial'); \
    conn.execute('LOAD spatial'); \
    print('Successfully pre-installed spatial extension'); \
except Exception as e: \
    print(f'Warning: Could not pre-install spatial extension: {e}'); \
try: \
    conn.execute('INSTALL json'); \
    conn.execute('LOAD json'); \
    print('Successfully pre-installed json extension'); \
except Exception as e: \
    print(f'Warning: Could not pre-install json extension: {e}'); \
conn.close()"

COPY api/ ./api/
COPY data/ ./data/
COPY scripts/ ./scripts/

RUN python scripts/init_database.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
