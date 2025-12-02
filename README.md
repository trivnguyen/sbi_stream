# sbi_stream

Simulation-Based Inference (SBI) for fitting stream-subhalo interaction models using neural posterior estimation.

## Overview

This repository implements neural density estimation methods to infer the properties of dark matter subhalo interactions with stellar streams. It uses PyTorch Lightning to train neural posterior estimators (NPE/SNPE) that can efficiently perform Bayesian inference on stream perturbation data.

## Project Structure

- `sbi_stream/` - Core module containing models, flows, and NPE implementation
- `datasets/` - Dataset utilities and I/O functions
- `configs/` - Configuration files for preprocessing and training
  - `preprocess/` - Preprocessing configurations for different dimensionalities (2D-6D)
  - `training/` - Training configurations and hyperparameter settings
  - `optuna/` - Hyperparameter optimization configurations
- `notebooks/` - Jupyter notebooks for paper figures

## Usage

### Preprocessing

Prepare simulation data for training:

```bash
python preprocess.py --config=configs/preprocess/6d-AAU-spline-sf1.py
```

### Training

Train a neural posterior estimator:

```bash
python train_npe.py --config=configs/training/npe-6d-large-nodensity.py
```

### Hyperparameter Optimization

Optimize model hyperparameters using Optuna:

```bash
python optuna_npe.py --config=configs/optuna/npe-6d-nodensity.py
```

### Inference

Run posterior inference on observed data:

```bash
python run_infer.py
```

## License

See [LICENSE](LICENSE) for details.
