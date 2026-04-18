# kingmaker

![Tests](https://github.com/USERNAME/kingmaker/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/USERNAME/kingmaker/branch/main/graph/badge.svg)](https://codecov.io/gh/USERNAME/kingmaker)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)

A Python library for working with King/Moffat distributions for modeling point spread functions (PSFs) in high-energy astrophysics.

## Overview

The **King distribution** (also known as the **Moffat distribution**, see [wikipedia](https://en.wikipedia.org/wiki/Moffat_distribution).) is a two-parameter probability distribution commonly used to describe the point spread function (PSF) of astronomical observations. The distribution has the form:

```
f(θ | α, β) ∝ [1 + (θ/α)² / (2β)]^(-β)
```

where:
- **θ** is the angular separation from the source
- **α** (alpha) is the scale parameter controlling the width of the distribution
- **β** (beta) is the shape parameter controlling the tail behavior

The distribution's flexibility in modeling both the core and tail regions makes it particularly well-suited for modeling extended tails to point spread functions. These are common in many instruments in astronomy. The King distribution offers significant advantages over simpler models (e.g., Gaussian or Rayleigh) by providing independent control over the core width (α) and tail weight (β), allowing more accurate modeling of real detector responses. 

Note that in the limit of β → ∞, this distribution reproduces the standard Rayleigh distribution. 

This package implements the King/Moffat distribution on a sphere. Normalizations are numerically calculated at initialization and linearly interpolated at runtime. The current implementation includes support for

- Direct KingPDF evaluations with normalizations calculated at runtime
- KingPDF evaluations with normalizations interpolated from tables calculated at intialization
- Numerically calculated "signal subtraction" PDFs
- Preliminary healpix map convolutions and template support using spherical convolutions

## Features

- **Efficient normalization**: Accurate integration over spherical caps with arbitrary cutoffs
- **Vectorized operations**: NumPy broadcasting for simultaneous evaluation at multiple points
- **Interpolation caching**: Pre-computed normalizations for 10-200× speedup
- **Multi-dimensional fitting**: Parameterize PSF as function of energy, declination, angular error, etc.
- **Signal subtraction**: Built-in RA marginalization for likelihood-based analyses
- **Template smearing**: Incorporate diffuse backgrounds and Galactic plane effects
- **Flexible binning**: Support for both equal-probability and explicit bin edges

## Installation

### Basic Installation

Install directly from the repository:

```bash
git clone git@github.com/mjlarson/kingmaker.git
pip install ./kingmaker
```

### Dependencies

Core dependencies:
- `numpy` - Array operations
- `scipy` - Optimization and interpolation
- `numba` - JIT compilation for performance
- `matplotlib` - Plotting and visualization used in example scripts
- `healpy` - Spherical harmonic operations

## Usage

### Basic PDF Evaluation

```python
import numpy as np
from kingmaker.pdf import KingPDF

# Create a King PDF with full-sphere coverage
king = KingPDF(angular_cutoff=np.pi)

# Define parameters
alpha = np.radians(1.0)  # 1 degree scale
beta = 2.0               # Moderate tail weight

# Evaluate PDF at various angular separations
angles = np.linspace(0, np.radians(10), 100)
pdf_values = king.pdf(angles, alpha, beta)

# Calculate containment radius (68% probability)
from scipy.optimize import brentq
containment_68 = brentq(
    lambda x: king.cdf(x, alpha, beta) - 0.68,
    0, np.pi
)
print(f"68% containment: {np.degrees(containment_68):.2f} degrees")
```

### Interpolated PDF for Speed

For applications requiring many PDF evaluations, use `InterpolatedKingPDF` which pre-computes normalizations on a grid:

```python
from kingmaker.pdf import InterpolatedKingPDF

# Create interpolated version (10-200x faster)
king_interp = InterpolatedKingPDF(angular_cutoff=np.pi)

# Use identical interface
pdf_values = king_interp.pdf(angles, alpha, beta)
```

The interpolated version provides **10-200× speedup** with **< 0.1% error** for most parameter values.

### Fitting PSF Parameters to Monte Carlo

Fit King distribution parameters to simulated signal events as a function of an arbitrary set of observables:

```python
from kingmaker.fitting import KingPSFFitter
import numpy as np

# Load your signal MC events (structured array)
# Must contain: 'ra', 'dec', 'trueRa', 'trueDec'
signal_events = np.load('signal_mc.npy')

# Define parameterization binning
parametrization_bins = {
    'logE': np.linspace(2, 6, 11),            # Energy bins
    'dec': np.arcsin(np.linspace(-1, 1, 11)), # Declination bins
    'angErr': 10                              # Equal-probability bins in angular error
}

# Initialize fitter
fitter = KingPSFFitter(
    signal_events=signal_events,
    parametrization_bins=parametrization_bins,
    dpsi_nbins=51,              # Angular error bins for fitting
    minimum_counts=100,         # Min events per bin
    spectral_indices=[2.0],     # Spectral indices for power-law weighting
    angular_cutoff=np.pi        # Maximum angular error to accept
)

# Run fitting
results = fitter.fit_all_bins(verbose=True)

# Access fitted parameters
alpha_fit = results['alpha']  # Shape: (n_gamma, n_energy, n_dec, n_sigma)
beta_fit = results['beta']

# Get interpolators for continuous evaluation
alpha_interp, beta_interp = fitter.get_interpolator(gamma_index=0)

# Evaluate at arbitrary points
test_point = np.array([[3.5, np.arcsin(0.0), np.radians(1.0)]])  # [logE, dec, sigma]
alpha_value = alpha_interp(test_point)
beta_value = beta_interp(test_point)
```

### Signal-Subtracted Likelihood

Compute the marginalized PDF for signal subtraction in likelihood analyses:

```python
# Marginalize over right ascension at a source declination
source_dec = np.radians(30)  # 30 degrees declination

sindec_bins, pdf_marginalized = king.marginalize(source_dec, alpha, beta)

# Use in likelihood calculation
# This represents the expected signal contribution in sin(dec) space
```

### Template Smearing

Apply Galactic or other template smearing to the PSF:

```python
from kingmaker.pdf import TemplateSmearedKingPDF

# Load a diffuse template (e.g., Galactic plane emission)
template = np.load('galactic_template.npy')  # HEALPix map

# Create smeared PSF
king_smeared = TemplateSmearedKingPDF(
    angular_cutoff=np.pi,
    template=template,
    nside=128
)

# Evaluate PDF including template effects
ra, dec = np.radians(120), np.radians(30)  # Source position
pdf_smeared = king_smeared.pdf(angles, ra, dec, alpha, beta)
```


## Examples

The `examples/` directory contains Jupyter notebooks demonstrating:

- `basic_demo.ipynb` - King PDF basics, parameter effects, and speed comparisons
- `fitting_demo.ipynb` - Fitting PSF parameters to Monte Carlo simulations
- `template_demo.ipynb` - Working with diffuse templates and template smearing

## References

1. King, I. (1962). " The structure of star clusters. I. an empirical density law" *The Astronomical Journal*, 67, 471.
2. Moffat, A. F. J. (1969). "A theoretical investigation of focal stellar images in the photographic emulsion and application to photographic photometry." *Astronomy and Astrophysics*, 3, 455.
3. Fermi-LAT Collaboration. "LAT Point Spread Function." Fermi Science Support Center. https://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_LAT_IRFs/IRF_PSF.html

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the terms specified in the repository.

## Author

Michael Larson (mlarson@icecube.wisc.edu)

## Citation

If you use `kingmaker` in your research, please cite the appropriate astronomical references above and acknowledge the software in your publications.
