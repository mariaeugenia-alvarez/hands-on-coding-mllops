# 🐧 Masterclass Despliegue MLOps - API de Pingüinos

Proyecto completo de MLOps que incluye una API REST con FastAPI, procesamiento de texto con LLM, tracking con MLflow, interfaz de usuario con Streamlit y despliegue en Google Cloud Run.

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Tecnologías](#tecnologías)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación Local](#instalación-local)
- [Ejecución Local](#ejecución-local)
- [API Endpoints](#api-endpoints)
- [Interfaz de Usuario](#interfaz-de-usuario)
- [Docker](#docker)
- [Despliegue en Google Cloud Run](#despliegue-en-google-cloud-run)
- [MLflow Tracking](#mlflow-tracking)

## 📝 Descripción

Este proyecto implementa una API REST completa que proporciona:
- **Gestión de datos de pingüinos**: Consultas y filtros sobre dataset de pingüinos Palmer
- **Análisis con ML**: Clasificación Zero-Shot y Question Answering con transformers
- **Tracking de experimentos**: MLflow para seguimiento de experimentos con LLM
- **Interfaz de usuario**: UI interactiva construida con Streamlit
- **Despliegue cloud**: Contenedorizado y desplegado en Google Cloud Run

## 🛠 Tecnologías

- **Backend**: FastAPI, Uvicorn
- **ML/AI**: Transformers (Hugging Face), LangChain, Google Gemini
- **Data**: Pandas, NumPy
- **Tracking**: MLflow
- **Containerización**: Docker
- **Cloud**: Google Cloud Run
- **Registry**: Docker Hub

## 📁 Estructura del Proyecto

```
hands-on-coding-mllops/
├── modulos_fastapi.py          # API FastAPI con todos los endpoints
├── main.py                     # Script CLI para procesamiento de texto con LLM
├── funciones.py                # Funciones de procesamiento con MLflow tracing
├── ui_streamlit.py             # Interfaz de usuario con Streamlit
├── Dockerfile                  # Configuración de contenedor Docker
├── run_ui.sh                   # Script para ejecutar UI en puerto 80
├── .streamlit/
│   └── config.toml            # Configuración de Streamlit
├── mlruns/                    # Datos de MLflow tracking
├── mlartifacts/               # Artefactos de MLflow
└── README.md                  # Este archivo
```

## 💻 Instalación Local

### Prerequisitos

- Python 3.12+
- pip
- Docker (opcional, para containerización)
- Google Cloud SDK (opcional, para despliegue)

### 1. Clonar el repositorio

```bash
git clone <repository-url>
cd hands-on-coding-mllops
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

### 3. Instalar dependencias

#### Para la API FastAPI:
```bash
pip install fastapi uvicorn pandas transformers torch sentencepiece protobuf
```

#### Para la UI Streamlit:
```bash
pip install streamlit requests pandas
```

#### Para MLflow y procesamiento LLM:
```bash
pip install mlflow langchain-google-genai python-dotenv
```

### 4. Configurar variables de entorno

Crea un archivo `.env` con tu API key de Google Gemini:

```env
GEMINI_API_KEY=tu_api_key_aqui
```

## 🚀 Ejecución Local

### Opción 1: Ejecutar API FastAPI

```bash
uvicorn modulos_fastapi:app --reload --port 8000
```

Accede a la documentación interactiva:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Opción 2: Ejecutar UI de Streamlit

```bash
streamlit run ui_streamlit.py
```

Accede a la interfaz: http://localhost:8501

**Para puerto 80 con ruta /ui:**
```bash
./run_ui.sh  # Requiere sudo
```

### Opción 3: Ejecutar script de procesamiento de texto

```bash
python main.py --text "Tu texto aquí" --model "gemini-2.5-flash-lite" --temperature 0.7
```

### Opción 4: MLflow UI

```bash
mlflow ui --port 5000
```

Accede al dashboard: http://localhost:5000

## 🔌 API Endpoints

### Pingüinos

#### `GET /penguins`
Filtra y lista pingüinos
- **Parámetros**:
  - `sex` (opcional): "Male" o "Female"
  - `limit` (opcional): Número de resultados (default: 5)
- **Ejemplo**: `/penguins?sex=Male&limit=10`

#### `GET /penguins/{penguin_id}`
Obtiene un pingüino por ID
- **Parámetro**: `penguin_id` (int)
- **Ejemplo**: `/penguins/42`

#### `GET /species`
Estadísticas por especie o filtrar por especie
- **Parámetro opcional**: `specie` ("Adelie", "Chinstrap", "Gentoo")
- **Ejemplo**: `/species?specie=Adelie`

### Machine Learning

#### `GET /zero-shot-classification`
Clasificación de texto sin entrenamiento previo
- **Parámetros**:
  - `text`: Texto a clasificar
  - `candidate_labels`: Etiquetas separadas por comas
- **Ejemplo**: `/zero-shot-classification?text=I love this!&candidate_labels=positive,negative,neutral`

#### `GET /question-answering`
Sistema de preguntas y respuestas basado en contexto
- **Parámetros**:
  - `question`: Pregunta a responder
  - `context`: Contexto donde buscar la respuesta
- **Ejemplo**: `/question-answering?question=Where is it?&context=The cat is on the table`

## 🎨 Interfaz de Usuario

La UI de Streamlit proporciona una interfaz gráfica para interactuar con todos los endpoints de la API:

- ✅ Filtrado interactivo de pingüinos
- ✅ Búsqueda por ID
- ✅ Visualización de estadísticas por especie
- ✅ Clasificación de texto Zero-Shot
- ✅ Sistema de preguntas y respuestas
- ✅ Gráficos y visualizaciones de datos

## 🐳 Docker

### Imagen disponible en Docker Hub

La aplicación está disponible en Docker Hub:
```
docker.io/mariaeu/fastapi-penguins:latest
```

### Construir imagen localmente

```bash
docker build -t fastapi-penguins .
```

### Ejecutar contenedor localmente

```bash
docker run -p 8000:8080 mariaeu/fastapi-penguins:latest
```

Accede a la API en: http://localhost:8000

### Subir imagen a Docker Hub

```bash
# Login en Docker Hub
docker login

# Etiquetar imagen
docker tag fastapi-penguins:latest tu-usuario/fastapi-penguins:latest

# Push a Docker Hub
docker push tu-usuario/fastapi-penguins:latest
```

## ☁️ Despliegue en Google Cloud Run

### Prerequisitos

1. Tener instalado Google Cloud SDK:
```bash
gcloud --version
```

2. Autenticarse:
```bash
gcloud auth login
```

3. Configurar proyecto:
```bash
gcloud config set project TU_PROJECT_ID
```

### Habilitar servicios necesarios

```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### Opción 1: Deploy desde Docker Hub

```bash
gcloud run deploy fastapi-penguins \
  --image docker.io/mariaeu/fastapi-penguins:latest \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 600
```

### Opción 2: Build y Deploy desde código fuente

```bash
gcloud run deploy fastapi-penguins \
  --source . \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 600
```

### Verificar deployment

```bash
gcloud run services describe fastapi-penguins --region europe-west1
```

### Obtener URL del servicio

```bash
gcloud run services describe fastapi-penguins \
  --region europe-west1 \
  --format 'value(status.url)'
```

### Parámetros de configuración

- `--memory 2Gi`: Memoria asignada (necesario para modelos de transformers)
- `--cpu 2`: CPUs asignadas
- `--timeout 600`: Timeout de 10 minutos (para carga de modelos)
- `--allow-unauthenticated`: Permite acceso público
- `--port 8000`: Puerto expuesto por la aplicación

### Actualizar deployment

```bash
gcloud run deploy fastapi-penguins \
  --image docker.io/mariaeu/fastapi-penguins:latest \
  --platform managed \
  --region europe-west1
```

### Eliminar servicio

```bash
gcloud run services delete fastapi-penguins --region europe-west1
```

## 📊 MLflow Tracking

### Iniciar servidor MLflow

```bash
mlflow server --host 0.0.0.0 --port 5000
```

### Características

- **Tracking de experimentos**: Registro automático de ejecuciones
- **Tracing**: Seguimiento detallado de llamadas a LLM
- **Artifacts**: Almacenamiento de resúmenes y análisis de sentimiento
- **Métricas**: Longitud de textos, scores, etc.

### Experimentos disponibles

- `Text_Analysis`: Análisis de texto con LLM y generación de resúmenes

## 🔧 Desarrollo

### Formato de código

```bash
# Instalar herramientas de desarrollo
pip install black flake8 isort

# Formatear código
black .
isort .
flake8 .
```

### Testing

```bash
# Instalar pytest
pip install pytest pytest-cov

# Ejecutar tests
pytest
```

## 📝 Notas

- Los modelos de transformers se descargan automáticamente en la primera ejecución
- La primera petición a los endpoints de ML puede tardar varios segundos
- MLflow guarda los datos en `mlruns/` y `mlartifacts/`
- Para producción, considerar usar variables de entorno para configuración sensible

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es parte de una masterclass de MLOps.

## 👥 Autores

- Tu nombre/organización

## 🙏 Agradecimientos

- Dataset de pingüinos Palmer de [Seaborn](https://github.com/mwaskom/seaborn-data)
- Modelos de Hugging Face
- Google Gemini API
- FastAPI y Streamlit communities

---

**Nota**: Este proyecto es con fines educativos y de demostración de prácticas de MLOps.
