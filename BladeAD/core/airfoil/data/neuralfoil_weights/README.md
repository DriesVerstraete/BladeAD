# NeuralFoil trained weights (vendored)

Source: `neuralfoil` PyPI package v0.3.3, author Peter Sharpe <pds@mit.edu>, MIT license.
Copied verbatim from `neuralfoil/nn_weights_and_biases/` in the installed package
(`/opt/anaconda3/envs/spl-bricks/lib/python3.12/site-packages/neuralfoil/`), 2026-09-05.

Files: `nn-xxsmall.npz`, `nn-small.npz`, `nn-medium.npz`, `nn-large.npz`, `nn-xlarge.npz`,
`nn-xxlarge.npz` (six of the eight shipped model sizes -- `nn-xxxlarge.npz` skipped, 5.3 MB,
excluded to avoid repo bloat; add it later if a use case needs it), plus
`scaled_input_distribution.npz` (training-data mean/covariance in the 25-dim input latent
space, used for the analysis-confidence Mahalanobis-distance term).

Fixed numeric data only -- no retraining occurs against these files. See
`06-rotor-optimisation/neuralfoil-csdl-port/port-plan.md` for the port this supports.
