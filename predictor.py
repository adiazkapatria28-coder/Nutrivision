import joblib
import pandas as pd

stunting_model = joblib.load("stunting_model.pkl")
wasting_model = joblib.load("wasting_model.pkl")

def predict(gender, age, height, weight):

    X = pd.DataFrame(
        [[gender, age, height, weight]],
        columns=[
            "Jenis Kelamin",
            "Umur (bulan)",
            "Tinggi Badan (cm)",
            "Berat Badan (kg)"
        ]
    )

    stunting = stunting_model.predict(X)[0]
    wasting = wasting_model.predict(X)[0]

    return stunting, wasting