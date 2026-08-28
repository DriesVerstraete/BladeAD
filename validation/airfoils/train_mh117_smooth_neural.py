import json
from pathlib import Path

import numpy as np
import torch

from BladeAD.core.airfoil.smooth_neural_airfoil_model import _SmoothPolarNetwork


REPOSITORY = Path(__file__).resolve().parents[2]
DATA = REPOSITORY / "BladeAD/core/airfoil/data/mh117_completed_viterna_blend_3deg.csv"
OUTPUT = REPOSITORY / "BladeAD/core/airfoil/data/mh117_smooth_neural_v1"


def main():
    torch.manual_seed(117)
    np.random.seed(117)
    data = np.genfromtxt(DATA, delimiter=",", names=True, encoding=None)
    alpha = np.deg2rad(data["alpha"])
    log_re = np.log(data["reynolds"])
    alpha_bounds = [float(alpha.min()), float(alpha.max())]
    log_re_bounds = [float(log_re.min()), float(log_re.max())]
    inputs = np.column_stack(
        (
            2 * (log_re - log_re_bounds[0]) / np.diff(log_re_bounds)[0] - 1,
            2 * (alpha - alpha_bounds[0]) / np.diff(alpha_bounds)[0] - 1,
        )
    )
    targets = np.column_stack((data["CL"], np.log(data["CD"])))
    output_mean = targets.mean(axis=0)
    output_scale = targets.std(axis=0)
    normalized_targets = (targets - output_mean) / output_scale
    weights = np.ones(len(data))
    weights[(data["alpha"] >= -14) & (data["alpha"] < -4)] = 5
    weights[(data["alpha"] >= -4) & (data["alpha"] <= 10)] = 10
    weights[(data["alpha"] > 10) & (data["alpha"] <= data["alpha_stall"] + 3)] = 5

    input_tensor = torch.tensor(inputs)
    target_tensor = torch.tensor(normalized_targets)
    weight_tensor = torch.tensor(weights[:, None])
    model = _SmoothPolarNetwork(hidden_width=64, hidden_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=300, min_lr=2e-5
    )
    best_loss = np.inf
    best_state = None
    stale_epochs = 0
    for epoch in range(6001):
        optimizer.zero_grad()
        prediction = model(input_tensor)
        loss = torch.mean(weight_tensor * (prediction - target_tensor) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())
        value = loss.item()
        if value < best_loss - 1e-10:
            best_loss = value
            best_state = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch % 250 == 0:
            print(f"epoch={epoch} weighted_mse={value:.8e} lr={optimizer.param_groups[0]['lr']:.3e}")
        if stale_epochs >= 1000:
            break

    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, OUTPUT / "weights.pt")
    metadata = {
        "name": "mh117_smooth_neural_v1",
        "source_table": DATA.name,
        "seed": 117,
        "hidden_width": 64,
        "hidden_layers": 3,
        "activation": "tanh",
        "inputs": ["log_reynolds", "alpha_rad"],
        "outputs": ["CL", "log_CD"],
        "alpha_bounds_rad": alpha_bounds,
        "log_reynolds_bounds": log_re_bounds,
        "output_mean": output_mean.tolist(),
        "output_scale": output_scale.tolist(),
        "core_weight": 10,
        "extended_negative_weight": 5,
        "near_stall_weight": 5,
        "post_stall_weight": 1,
        "best_weighted_mse": best_loss,
        "epochs_completed": epoch,
    }
    with (OUTPUT / "metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
