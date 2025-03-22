from rules import HeartDiseaseExpert
import pandas as pd
from experta import *

data = pd.read_csv("../data/cleaned_data.csv")

engine = HeartDiseaseExpert()
engine.reset()

patient = {
    "age": 58,
    "sex": 0,
    "cp": 0,
    "trestbps": 100,
    "chol": 248,
    "fbs": 0,
    "restecg": 0,
    "thalach": 122,
    "exang": 0,
    "oldpeak": 1,
    "slope": 1,
    "ca": 0,
    "thal": 2,
}

for key, value in patient.items():
    engine.declare(Fact(**{key: value}))

engine.run()

