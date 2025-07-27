import json
import os
# from  type_converters import convert_to_numpy


# reads a model and its score from a json file
def write_json(path,data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def write_model(model_dict, model_name):
    path = f"models/{model_name}_model.json"
    write_json(path, model_dict)


def read_model(model_name):
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





