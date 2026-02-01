from fastapi import FastAPI
import pandas as pd


app = FastAPI()


@app.get("/saluda")
def root(name):
    return {"Message": f"Hola,  mi nombre es {name}"}


@app.get("/read_df")
def read_df():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    )
    return df


@app.get("/read_sepal_lenght_positon")
def read_sepal_length(position: int):
    print(position)
    print(type(position))
    df = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    )
    value = df["sepal_length"][position]
    return {"Value": value}


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
