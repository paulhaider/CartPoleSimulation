#!/usr/bin/env python3


import sys
import json
from pathlib import Path

import numpy as np
import torch

from Control_Toolkit.Controllers import template_controller
from SI_Toolkit.computation_library import TensorType, PyTorchLibrary

PATH_TO_MINIMALE_REPO = Path("/home/timo/phd/Projects/gle/minimaLE")
PATH_TO_CARTPOLE_DIR = PATH_TO_MINIMALE_REPO / Path("experiments/cartpole")
PATH_TO_MODELS = PATH_TO_CARTPOLE_DIR / "models"
MODEL_PARAMS_FNAME = "params.json"
MODEL_STATE_DICT_FNAME = "model_59.torch"

sys.path.append(str(PATH_TO_MINIMALE_REPO / PATH_TO_CARTPOLE_DIR))

from cartpole import get_data, Net, get_phi_and_derivative


class controller_gle(template_controller):

    _computation_library = PyTorchLibrary

    def configure(self):

        # GLE network input:
        # ['angle', 'angleD', 'angle_cos', 'angle_sin', 'position', 'positionD']
        with open(MODEL_PARAMS_FNAME, "r") as f:
            self.params = json.load(
                f,
            )

        self.model = Net(
            tau_m=self.params["tau_m"],
            tau_r=self.params["tau_r"],
            dt=self.params["dt"],
            n_inputs=len(self.params["input_vars"]),
            n_hidden=self.params["n_hidden"],
            n_outputs=len(self.params["output_vars"]),
            params=self.params,
            phi=get_phi_and_derivative(self.params["phi"])[0],
            phi_prime=get_phi_and_derivative(self.params["phi"])[1],
            prospective_errors=self.params["prospective_errors"],
        )

        self.model.load_state_dict(MODEL_STATE_DICT_FNAME, weights_only=True)
        # TODO weights_only = True OR False???
        self.model.eval()

    def step(
        self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}
    ):

        # structure of s (hopefully...)
        # [angle, angleD, angle_cos, angle_sin, position, positionD]
        s = torch.from_numpy(s)
        s = s.view(1, -1)

        out = self.model(s, None, beta=0)

        return out.item()
