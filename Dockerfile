FROM python:3.11-slim

# Evitar prompts interactivos durante instalación
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Directorio de trabajo
WORKDIR /app

# Dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements de producción primero para aprovechar cache de Docker
COPY requirements-prod.txt .

# Copiar main.py
COPY src/api/main.py ./main.py

# Instalar dependencias Python
# --no-cache-dir reduce el tamaño de la imagen
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# Copiar el código fuente
COPY src/ ./src/
COPY src/api/main.py ./main.py

# Crear directorio para la DB y el cache del modelo
RUN mkdir -p /tmp/model_cache /tmp/santiago_weather

# Puerto que expone Render
EXPOSE 8000

# Comando de arranque
# Render setea la variable PORT automáticamente
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
