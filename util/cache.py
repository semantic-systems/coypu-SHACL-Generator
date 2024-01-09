import os.path

import torch
from pykeen.models import Model
from pykeen.pipeline import PipelineResult

CACHE_DIR = '.cache'

if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)


class CacheMiss(Exception):
    pass


def _mk_path(dataset_name: str, model_name: str):
    file_name = dataset_name.replace(' ', '_') + '__' + model_name + '.pkl'
    file_path = os.path.join(CACHE_DIR, file_name)

    return file_path


def load(dataset_name: str, model_name: str) -> Model:
    file_path = _mk_path(dataset_name, model_name)

    if os.path.exists(file_path):
        model = torch.load(file_path)
        return model

    else:
        raise CacheMiss()


def store(dataset_name: str, model_name: str, result: PipelineResult):
    file_path = _mk_path(dataset_name, model_name)

    result.save_model(file_path)
