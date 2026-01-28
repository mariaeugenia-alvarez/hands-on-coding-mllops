import argparse
import funciones


def main():
    parser = argparse.ArgumentParser(
        description="Procesa texto con LLM y MLflow Tracing"
    )

    parser.add_argument("--text", type=str, required=True, help="Texto a procesar")

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperatura del modelo (0.0-1.0)",
    )

    parser.add_argument(
        "--model", type=str, default="gemini-2.5-flash-lite", help="Modelo LLM a usar"
    )

    args = parser.parse_args()

    # Inicializar modelo
    llm = funciones.initialize_llm(model=args.model, temperature=args.temperature)

    # Procesar texto
    response = funciones.process_with_experiment(
        text=args.text, experiment_name="Text_Analysis", llm=llm
    )

    # Mostrar resultados

    print(f"\nTexto original ({response['original_length']} caracteres):")
    print(f"  {args.text[:100]}...")
    print(f"\nResumen:")
    print(f"  {response['summary']}")
    print(f"\nAnálisis de sentimiento:")
    print(f"  {response['sentiment']}")


if __name__ == "__main__":
    main()
