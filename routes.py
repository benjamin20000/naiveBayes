from data_preparation.load_data import LoadData
from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
from core_model.model import Model
from data_preparation.clean_data import CleanData
from core_model.testr import Tester
from type_converters import convert_to_native
from model_io import write_model
from core_model.classifier import Classifier
from data_point_example import data_point_example



app = FastAPI()


@app.post("/")
def load_model(model_name ="phishing", target_name="class", index_name = "Index"):
    try:
        df = LoadData.load_csv(model_name)
        df = CleanData.clean(df,index_name)
        train_df, test_df = Model.split_train_test(df)
        model = Model.create_model(target_name, train_df)
        tester = Tester(test_df, model_name, target_name)
        score = tester.test()
        model_and_score = convert_to_native({"model": model, "score": score})
        write_model(model_and_score, model_name)
        return {"status":200,"score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.get("/")
def make_prediction(model = "phishing", data_point = None):
    if data_point is None:
        data_point = data_point_example  # set default data point
    classifier = Classifier(model)
    prediction = classifier.calculate_prediction(data_point)
    model_accuracy = classifier.get_model_accuracy()
    return JSONResponse(content={"model_accuracy": model_accuracy,"prediction":prediction})


