import numpy as np
import torch

PARAM_STATS = {
    "Omega0": {"min": 0.1, "max": 0.5},
    "sigma8": {"min": 0.6, "max": 1.0},
    "ASN1": {"min": 0.25, "max": 4.0},
    "AAGN1": {"min": 0.25, "max": 4.0},
    "ASN2": {"min": 0.5, "max": 2.0},
    "AAGN2": {"min": 0.5, "max": 2.0}
}

PARAM_ORDER = ["Omega0", "sigma8", "ASN1", "AAGN1", "ASN2", "AAGN2"]

def normalize_params(y_list: list[torch.tensor]) -> list[torch.tensor]:
    norm_y_list = []
    for y in y_list:
        norm_y = torch.zeros_like(y)
        for i, param in enumerate(PARAM_ORDER):
            norm_y[i] = (y[i] - PARAM_STATS[param]["min"]) / (PARAM_STATS[param]["max"] - PARAM_STATS[param]["min"])
        norm_y_list.append(norm_y)
    return norm_y_list


def denormalize_params(norm_y: np.ndarray) -> np.ndarray:
    y = np.zeros_like(norm_y)
    for i, param in enumerate(PARAM_ORDER):
        y[:, i] = norm_y[:, i] * (PARAM_STATS[param]["max"] - PARAM_STATS[param]["min"]) + PARAM_STATS[param]["min"]
    return y
