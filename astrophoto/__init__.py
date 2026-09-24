"""AstroPhoto Studio: an end-to-end astrophotography pipeline for ZWO Seestar raw FITS subs.

Stages
------
1. ``frames``      discovery, FITS/CFA handling, cosmetic (hot pixel) correction
2. ``analysis``    per-frame star metrics, registration, obstruction & cloud detection,
                   statistical + ML (IsolationForest) frame rejection and weighting
3. ``stacking``    streaming 3-pass integration: local normalization, weighted
                   sigma-clipping, optional CFA (Bayer) drizzle, split half-stacks
4. ``denoise``     self-supervised Noise2Noise CNN trained on the two half-stacks
5. ``postprocess`` gradient removal, colour calibration, deconvolution, star
                   separation, stretching, narrowband palettes, local contrast
"""

__version__ = "1.0.0"
