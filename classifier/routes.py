from starlette.responses import JSONResponse
from model_io import write_model
from classifier import Classifier
from fastapi import FastAPI
from data_point_example import data_point_example

app = FastAPI()


@app.on_event("startup")
def load_models_on_startup():
    """load the model once when the server starts."""
    model_name = "phishing"
    write_model(model_name)
    print(f"{model_name} model loaded successfully at startup.")


@app.get("/")
def make_prediction(model="phishing", data_point=None):
    """return prediction and model accuracy for given data point."""
    if data_point is None:
        data_point = data_point_example  # set default data point

    classifier = Classifier(model)
    prediction = classifier.calculate_prediction(data_point)
    model_accuracy = classifier.get_model_accuracy()
    return JSONResponse(content={"model_accuracy": model_accuracy, "prediction": prediction})
