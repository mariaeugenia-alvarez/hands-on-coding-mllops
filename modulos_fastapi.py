from fastapi import FastAPI
import pandas as pd


# Cargar fastAPI
app = FastAPI()


# Cargar el DataFrame una sola vez
def load_penguins_data():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/refs/heads/master/penguins.csv"
    )
    # Agregar ID a cada fila
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    df = df.reset_index()
    df = df.rename(columns={"index": "id"})
    # Reemplazar NaN con null
    df = df.fillna("null")
    return df


# Crear DataFrame de estadísticas por especie
def load_species_stats():
    df = load_penguins_data()

    # Convertir body_mass_g a numérico
    df["body_mass_g"] = pd.to_numeric(df["body_mass_g"], errors="coerce")

    species_stats = (
        df.groupby("species")
        .agg(
            avg_body_mass_g=("body_mass_g", "mean"),
            min_body_mass_g=("body_mass_g", "min"),
            max_body_mass_g=("body_mass_g", "max"),
            penguin_ids=("id", lambda x: list(x)),
        )
        .reset_index()
    )

    return species_stats


# Cargar datos
penguins_df = load_penguins_data()
species_stats_df = load_species_stats()


@app.get("/penguins")
def filter_df(sex: str = None, limit: int = 5):
    global penguins_df

    # Hacer una copia para no modificar el original
    df = penguins_df.copy()

    # Filtrar por sexo si se proporciona
    if sex:
        df = df[df["sex"].str.strip().str.upper() == sex.upper()]

    # Limitar resultados si se proporciona
    if limit:
        df = df.head(limit)

    return df.to_dict(orient="records")


@app.get("/penguins/{penguin_id}")
def read_penguin_by_id(penguin_id: int):
    global penguins_df

    # Buscar el pingüino por ID
    df = penguins_df[penguins_df["id"] == penguin_id]

    if df.empty:
        return {"error": "Penguin not found"}

    return df.to_dict(orient="records")[0]


@app.get("/species")
def classify_by_species(specie: str = None):
    global penguins_df, species_stats_df

    # Si se proporciona una especie, filtrar del DataFrame original
    if specie:
        df = penguins_df[penguins_df["species"].str.upper() == specie.upper()]
        return df.to_dict(orient="records")

    # Si no, devolver las estadísticas globales
    return species_stats_df.to_dict(orient="records")


from transformers import pipeline


@app.get("/zero-shot-classification")
def zero_shot_classification(text: str, candidate_labels: str):
    pipe = pipeline("zero-shot-classification")
    labels = [label.strip() for label in candidate_labels.split(",")]
    response = pipe(text, labels)
    return {
        "Value": response["labels"][0],
        "scores": dict(zip(response["labels"], response["scores"])),
    }


@app.get("/question-answering")
def question_answering(question: str, context: str):
    pipe = pipeline(model="deepset/roberta-base-squad2")
    response = pipe(question=question, context=context)
    return {"answer": response["answer"], "score": response["score"]}
