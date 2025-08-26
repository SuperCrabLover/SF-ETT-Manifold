# SF-ETT Manifold: Shared Factor Extended Tensor Train Manifold Optimization

A Python library for manifold optimization with Shared Factor Extended Tensor Train (SF-ETT) format, 
providing efficient implementations for tensor decomposition and Riemannian optimization on structured tensor manifolds.

## Features

- **SF-ETT Tensor Format**: Efficient representation of high-dimensional tensors with shared factors
- **Riemannian Optimization**: Manifold-aware optimization algorithms including Riemannian LOCG
- **Flexible Backend**: Built on PyTorch for GPU acceleration and automatic differentiation
- **Experimental Suite**: Ready-to-run experiments for various applications including:
  - Grid function approximation
  - Discretized Laplace operator optimization
  - Quantum mechanics (Henon-Heiles potential)

## Repository Structure

```
SF-ETT-Manifold/
├── src/
│   └── sfett/                 # Main package
│       ├── __init__.py        # Package initialization
│       ├── manifolds.py       # TTSFTuckerManifold class & Riemannian operations
│       ├── tensors.py         # TTSFTucker and TTSFTuckerTangentVector classes
│       └── tools.py           # Utility functions for experiments
├── SFETT_LHH_exp.ipynb        # Laplace operator + Henon-Heiles experiments
├── TTSFTucker_Hilbert_exp.ipynb  # Grid function experiments
├── pyproject.toml            # Poetry configuration
├── poetry.lock               # Dependency lock file
├── LICENSE                   MIT License
└── README.md                 This file
```

## Installation

### Prerequisites
- Python 3.10.12+
- Poetry (recommended) or pip

### Using Poetry (Recommended)
```bash
git clone https://github.com/SuperCrabLover/SF-ETT-Manifold.git
cd SF-ETT-Manifold
poetry install  # Installs with all dependencies
```

### Using pip
```bash
git clone https://github.com/SuperCrabLover/SF-ETT-Manifold.git
cd SF-ETT-Manifold
pip install -e .  # Editable installation for development
```

## Quick Start

```python
from sfett.tensors import TTSFTuckerTangentVector, TTSFTuckerTensor
from sfett.manifolds import TTSFTuckerManifold
from functools import partial
import torch


def dummy_obj(X: TTSFTuckerTensor, dummy_target: torch.Tensor) -> torch.Tensor:
    return (dummy_target - X @ X) ** 2


# Create SF-ETT 8-tensor
tensor = TTSFTuckerTensor(shared_factors_amount=3, non_shared_factors_amount=5)
print("The mu-orthogonal with mu = ", tensor.orthogonalization)
print("Tensor SF-ETT rank:", tensor.tt_ranks, tensor.tucker_ranks)
tensor /= tensor.norm()
dummy_target = torch.randn(1)
print(dummy_target, dummy_obj(tensor, dummy_target))

# Get Riemannian gradient of dummy_obj
dummy_obj_grad = TTSFTuckerManifold().grad(
    partial(dummy_obj, dummy_target=dummy_target)
)
riem_grad_tangent = dummy_obj_grad(tensor)
print(riem_grad_tangent.norm())

# Represent gradient as 2r SF-ETT Tensor
riem_grad_sfett = riem_grad_tangent.construct()
print(
    "Tangent vector SF-ETT rank:",
    riem_grad_sfett.tt_ranks,
    riem_grad_sfett.tucker_ranks,
)
# Round ranks
riem_grad_sfett.round(
    tt_ranks=max(tensor.tt_ranks), sf_tucker_ranks=max(tensor.tucker_ranks)
)
print(
    "Tangent vector SF-ETT rank:",
    riem_grad_sfett.tt_ranks,
    riem_grad_sfett.tucker_ranks,
)
```

## Key Components

### `tensors.py`
- `TTSFTucker`: Core class for SF-ETT format tensor representation
- `TTSFTuckerTangentVector`: Tangent space elements for manifold operations

### `manifolds.py`
- `TTSFTuckerManifold`: Riemannian manifold operations including:
  - Tangent space operations
  - Riemannian gradients

### `tools.py`
- Utility functions for experiment setup and analysis
- Helper functions for tensor operations

## Experiments

### 1. Grid Function Approximation
`TTSFTucker_Hilbert_exp.ipynb` contains experiments for approximating functions on grids using the SF-ETT format.

### 2. Riemannian Optimization
`SFETT_LHH_exp.ipynb` implements Riemannian Locally Optimal Conjugate Gradient (LOCG) methods for:
- Discretized Laplace operator optimization
- Quantum systems with Henon-Heiles potential

## Dependencies

Core dependencies:
- torch (>=2.7.1,<3.0.0)",
- matplotlib (>=3.10.5,<4.0.0)",
- tqdm (>=4.67.1,<5.0.0)",
- tntorch (>=1.1.2,<2.0.0)",

Development tools:
- Jupyter Notebook
- Matplotlib (for visualization)
- Poetry (for dependency management)

## Citing This Work

If you use this library in your research, please cite (TBD)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Support

For questions and support, please open an issue on GitHub or contact the maintainers.
