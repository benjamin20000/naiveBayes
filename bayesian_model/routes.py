from data_preparation.load_data import LoadData
from fastapi import FastAPI
from core_model.model import Model
from fastapi.responses import JSONResponse
from data_preparation.clean_data import CleanData
from core_model.testr import Tester
from type_converters import convert_to_native

app = FastAPI()


@app.get("/")
def get_model(model_name="phishing", target_name="class", index_name="Index"):
    """train and evaluate model, return bayesian model and score."""
    df = LoadData.load_csv(model_name)
    df = CleanData.clean(df, index_name)
    train_df, test_df = Model.split_train_test(df)
    model = Model.create_model(target_name, train_df)
    tester = Tester(test_df, model, target_name)
    score = tester.test()
    model = convert_to_native(model)
    print(score)
    return JSONResponse(content={"bayesian_model": model, "score": score})
