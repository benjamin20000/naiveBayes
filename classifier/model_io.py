import json
import httpx
import os


def write_json(path, data):
    """write data as json to a file at the given path."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def req_model(model_name):
    """request model json from the model server using model_name."""
    url = f"http://model_server:8000?path=data/{model_name}.csv"
    try:
        res = httpx.get(url, timeout=100.0)
        return res.json()
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise RuntimeError("req not working") from e


def write_model(model_name):
    """fetch model data and write it as json to models directory."""
    model = req_model(model_name)
    path = f"models/{model_name}_model.json"
    write_json(path, model)


def read_model(model_name):
    """read model json file and return model and score, with error handling."""
    path = f"models/{model_name}_model.json"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file '{path}' does not exist.")

    try:
        with open(path, 'r') as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in model file '{path}': {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error reading model file '{path}': {e}")

    if 'model' not in data or 'score' not in data:
        raise KeyError(f"Missing 'model' or 'score' keys in model file '{path}'.")

    return data['model'], data['score']
