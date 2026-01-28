import os
from dotenv import load_dotenv
import mlflow
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()

# Configurar MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# Cargar API KEY:
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")


def initialize_llm(
    model: str = "gemini-2.5-flash-lite", temperature: float = 0.7
) -> ChatGoogleGenerativeAI:
    """
    Inicializa el modelo LLM de Google.

    Args:
        model: Nombre del modelo a usar
        temperature: Temperatura para la generación (0.0-1.0)
    """
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
        max_tokens=1024,
    )


@mlflow.trace
def generate_summary(
    text: str, max_words: int = 50, llm: ChatGoogleGenerativeAI = None
) -> str:
    """
    Genera un resumen del texto usando LLM.

    Args:
        text: Texto a resumir
        max_words: Máximo de palabras en el resumen
        llm: Instancia del modelo LLM
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"Resume el siguiente texto en máximo {max_words} palabras."),
            ("human", "{text}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})


@mlflow.trace
def analyze_sentiment(text: str, llm: ChatGoogleGenerativeAI = None) -> dict:
    """
    Analiza el sentimiento del texto.

    Args:
        text: Texto a analizar
        llm: Instancia del modelo LLM
    """
    messages = [
        SystemMessage(content="Eres un experto en análisis de sentimientos."),
        HumanMessage(
            content=f"Analiza el sentimiento del siguiente texto y responde con: Sentimiento (positivo/negativo/neutral), Confianza (0-100%)\n\nTexto: {text}"
        ),
    ]

    response = llm.invoke(messages)

    return {"text": text, "analysis": response.content}


def process_with_experiment(
    text: str, experiment_name: str, llm: ChatGoogleGenerativeAI = None
) -> dict:
    """
    Procesa texto con tracking en MLflow.

    Args:
        text: Texto a procesar
        experiment_name: Nombre del experimento en MLflow
        llm: Instancia del modelo LLM

    Returns:
        Resultados del procesamiento
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="análisis"):
        mlflow.log_param("text_length", len(text))
        mlflow.log_param("experiment", experiment_name)

        # Generar resumen
        summary = generate_summary(text, max_words=50, llm=llm)
        mlflow.log_text(summary, "summary.txt")

        # Analizar sentimiento
        sentiment = analyze_sentiment(text, llm=llm)
        mlflow.log_text(sentiment["analysis"], "sentiment_analysis.txt")

        mlflow.log_metric("summary_length", len(summary.split()))

        return {
            "summary": summary,
            "sentiment": sentiment["analysis"],
            "original_length": len(text),
        }
