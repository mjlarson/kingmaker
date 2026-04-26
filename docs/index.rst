kingmaker
=========

A Python library for working with King/Moffat distributions for modeling
point spread functions (PSFs) in high-energy astrophysics.

The **King distribution** is a two-parameter family commonly used to describe
PSFs with independent control over the core width (α) and tail weight (β):

.. math::

   f(\theta \mid \alpha, \beta) \propto \left[1 + \frac{(\theta/\alpha)^2}{2\beta}\right]^{-\beta}

This package implements the King distribution on a sphere, including
normalization over spherical caps, parameter fitting from simulation,
and spherical-harmonic convolution with HEALPix template maps.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   api

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples
