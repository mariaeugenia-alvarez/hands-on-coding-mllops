# Usar imagen oficial de Python
FROM python:3.12-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo de la aplicación
COPY modulos_fastapi.py .

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pandas \
    transformers \
    torch \
    sentencepiece \
    protobuf

# Exponer el puerto
EXPOSE 8080

# Comando para ejecutar la aplicación usando la variable PORT de Cloud Run
CMD uvicorn modulos_fastapi:app --host 0.0.0.0 --port ${PORT:-8080}
