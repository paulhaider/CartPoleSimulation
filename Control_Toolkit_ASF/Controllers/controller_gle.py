#!/usr/bin/env python3


import sys
import json
from pathlib import Path

import numpy as np
import torch

from Control_Toolkit.Controllers import template_controller
from SI_Toolkit.computation_library import TensorType, PyTorchLibrary

PATH_TO_MINIMALE_REPO = Path("/home/paul/dev/minimaLE")
PATH_TO_CARTPOLE_DIR = PATH_TO_MINIMALE_REPO / Path("experiments/cartpole")
PATH_TO_MODELS = PATH_TO_CARTPOLE_DIR / "models"
# best so far
MODEL_PARAMS_FNAME = "run_218.json"
MODEL_STATE_DICT_FNAME = "run_218.torch"

sys.path.append(str(PATH_TO_MINIMALE_REPO / PATH_TO_CARTPOLE_DIR))

from cartpole import get_data, Net, get_phi_and_derivative


class controller_gle(template_controller):

    _computation_library = PyTorchLibrary

    def configure(self):

        # GLE network input:
        # ['angle', 'angleD', 'angle_cos', 'angle_sin', 'position', 'positionD']
        with open(PATH_TO_MODELS / MODEL_PARAMS_FNAME, "r") as f:
            self.params = json.load(
                f,
            )

        self.model = Net(params=self.params)

        self.model.load_state_dict(torch.load(PATH_TO_MODELS / MODEL_STATE_DICT_FNAME))
        self.model.eval()

    def step(
        self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}
    ):

        # structure of s (hopefully...)
        # [angle, angleD, angle_cos, angle_sin, position, positionD]
        # append state with target equilibrium and target position
        s = np.append(s, [1, 0])
        s = torch.from_numpy(s).to(torch.float)
        s = s.view(1, -1)

        for _ in range(9):
            out = self.model(s, None, beta=0)

        return out.to(torch.double).item()
