import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "validation"
    / "acoustics"
    / "scripts"
    / "run_production_graph_verification.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_production_graph_verification", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forward_flight_production_graph_primal_and_gradient():
    result = _load_runner().verify_case("forward_flight", 1.0e-5)
    assert result["primal"]["absolute_error"] < 1.0e-12
    assert result["gradient"]["problem_vs_direct_relative_error"] < 1.0e-12
    assert result["gradient"]["problem_vs_fd_relative_error"] < 1.0e-5
