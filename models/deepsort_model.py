# app/models/deepsort_model.py

from deep_sort.deep_sort import DeepSort

def load_deepsort_model():
    config_path = "deep_sort/deep/checkpoint/ckpt.t7"
    model = DeepSort(model_path=config_path, max_age=35, n_init=4, nn_budget=110)
    return model
