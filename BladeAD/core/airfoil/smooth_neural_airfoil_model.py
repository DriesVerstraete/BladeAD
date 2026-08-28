import json
from pathlib import Path

import csdl_alpha as csdl
import numpy as np
import torch
from torch import nn
from torch.func import jacrev, vmap


torch.set_default_dtype(torch.float64)
_MODEL_DIRECTORY = Path(__file__).parent / "data" / "mh117_smooth_neural_v1"


class _SmoothPolarNetwork(nn.Module):
    def __init__(self, hidden_width, hidden_layers):
        layers = []
        input_width = 2
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(input_width, hidden_width), nn.Tanh()))
            input_width = hidden_width
        layers.append(nn.Linear(input_width, 2))
        super().__init__()
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)


class _SmoothNeuralAirfoilOperation(csdl.CustomExplicitOperation):
    def __init__(self, model, metadata):
        self.model = model
        self.model_metadata = metadata
        super().__init__()

    def evaluate(self, alpha, Re, Ma):
        self.declare_input("alpha", alpha)
        self.declare_input("Re", Re)
        self.declare_input("Ma", Ma)
        shape = alpha.shape
        if shape != Re.shape or shape != Ma.shape:
            raise ValueError("alpha, Re, and Ma must have identical shapes.")
        if len(shape) not in (1, 2, 3):
            raise NotImplementedError("Only one-, two-, and three-dimensional inputs are supported.")
        indices = np.arange(np.prod(shape))
        cl = self.create_output("Cl", shape)
        cd = self.create_output("Cd", shape)
        for output in ("Cl", "Cd"):
            self.declare_derivative_parameters(output, "alpha", rows=indices, cols=indices)
            self.declare_derivative_parameters(output, "Re", rows=indices, cols=indices)
            self.declare_derivative_parameters(output, "Ma", dependent=False)
        return cl, cd

    def _inputs(self, alpha, reynolds):
        alpha = np.asarray(alpha, dtype=float)
        reynolds = np.asarray(reynolds, dtype=float)
        alpha_bounds = self.model_metadata["alpha_bounds_rad"]
        log_re_bounds = self.model_metadata["log_reynolds_bounds"]
        if np.any(alpha > alpha_bounds[1]):
            raise ValueError("alpha exceeds the trained MH117 neural-model domain.")
        alpha = np.maximum(alpha, alpha_bounds[0])
        log_re = np.log(reynolds)
        if np.any(log_re < log_re_bounds[0]) or np.any(log_re > log_re_bounds[1]):
            raise ValueError("Re is outside the trained MH117 neural-model domain.")
        normalized_alpha = 2 * (alpha.ravel() - alpha_bounds[0]) / (
            alpha_bounds[1] - alpha_bounds[0]
        ) - 1
        normalized_log_re = 2 * (log_re.ravel() - log_re_bounds[0]) / (
            log_re_bounds[1] - log_re_bounds[0]
        ) - 1
        return torch.tensor(np.column_stack((normalized_log_re, normalized_alpha)))

    def _predict(self, alpha, reynolds, derivatives=False):
        shape = np.asarray(alpha).shape
        below_physical_domain = np.asarray(alpha).ravel() < self.model_metadata[
            "alpha_bounds_rad"
        ][0]
        inputs = self._inputs(alpha, reynolds)
        raw = self.model(inputs)
        output_mean = torch.tensor(self.model_metadata["output_mean"])
        output_scale = torch.tensor(self.model_metadata["output_scale"])
        dimensional = raw * output_scale + output_mean
        cl = dimensional[:, 0]
        cd = torch.exp(dimensional[:, 1])
        if not derivatives:
            return cl.detach().numpy().reshape(shape), cd.detach().numpy().reshape(shape)

        jacobian = vmap(jacrev(self.model))(inputs).detach().numpy()
        raw_numpy = dimensional.detach().numpy()
        jacobian[:, 0, :] *= self.model_metadata["output_scale"][0]
        jacobian[:, 1, :] *= self.model_metadata["output_scale"][1] * np.exp(raw_numpy[:, 1, None])
        alpha_range = np.diff(self.model_metadata["alpha_bounds_rad"])[0]
        log_re_range = np.diff(self.model_metadata["log_reynolds_bounds"])[0]
        reynolds_flat = np.asarray(reynolds).ravel()
        dcl_dre = jacobian[:, 0, 0] * 2 / log_re_range / reynolds_flat
        dcd_dre = jacobian[:, 1, 0] * 2 / log_re_range / reynolds_flat
        dcl_da = jacobian[:, 0, 1] * 2 / alpha_range
        dcd_da = jacobian[:, 1, 1] * 2 / alpha_range
        dcl_da[below_physical_domain] = 0.0
        dcd_da[below_physical_domain] = 0.0
        return tuple(
            item.reshape(shape)
            for item in (cl.detach().numpy(), cd.detach().numpy(), dcl_da, dcd_da, dcl_dre, dcd_dre)
        )

    def compute(self, inputs, outputs):
        outputs["Cl"], outputs["Cd"] = self._predict(inputs["alpha"], inputs["Re"])

    def compute_derivatives(self, inputs, outputs, derivatives):
        _, _, dcl_da, dcd_da, dcl_dre, dcd_dre = self._predict(
            inputs["alpha"], inputs["Re"], derivatives=True
        )
        derivatives["Cl", "alpha"] = dcl_da.ravel()
        derivatives["Cd", "alpha"] = dcd_da.ravel()
        derivatives["Cl", "Re"] = dcl_dre.ravel()
        derivatives["Cd", "Re"] = dcd_dre.ravel()


class MH117SmoothNeuralAirfoilModel:
    """Smooth MH117 model with constant support below its trained domain."""

    def __init__(self, model_directory=None):
        model_directory = Path(model_directory or _MODEL_DIRECTORY)
        with (model_directory / "metadata.json").open() as stream:
            self.metadata = json.load(stream)
        self.model = _SmoothPolarNetwork(
            self.metadata["hidden_width"], self.metadata["hidden_layers"]
        )
        self.model.load_state_dict(
            torch.load(model_directory / "weights.pt", map_location="cpu", weights_only=True)
        )
        self.model.eval()

    def evaluate(self, alpha, Re, Ma):
        return _SmoothNeuralAirfoilOperation(self.model, self.metadata).evaluate(alpha, Re, Ma)

    def predict(self, alpha, Re, derivatives=False):
        operation = _SmoothNeuralAirfoilOperation(self.model, self.metadata)
        return operation._predict(alpha, Re, derivatives=derivatives)
