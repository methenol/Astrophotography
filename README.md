# AstroPhoto Studio — Seestar raw FITS → finished astrophoto

An end-to-end pipeline that turns the raw `.fit` subs saved by a ZWO Seestar
(S50 / S50 Pro / S30) into a finished image. It
handles the whole chain: frame grading, cloud/tree/obstruction rejection,
registration with alt-az field rotation, local normalisation, sigma-clipped
Bayer-drizzle integration, AI denoising, gradient removal, deconvolution,
star separation, narrowband palettes, stretching and export to a
full-resolution JPEG + 16-bit TIFF. It includes a web UI and a CLI.

Nothing is tuned for a particular object. Every decision comes from the
FITS headers (`BAYERPAT`, `BIAS`, `FILTER`, `EXPTIME`, …) and from statistics
of the data. The `LP` dual-band filter automatically gets an Ha/OIII (HOO)
workflow, and the `IRCUT` broadband filter gets a natural-colour RGB workflow.

## Quick start

```bash
source .venv/bin/activate
pip install -r requirements.txt          # or: uv pip install -r requirements.txt

# Web UI  →  http://127.0.0.1:8000
python -m webui.server                   # --images /path/to/Seestar/MyWorks  --port 8080  --host 0.0.0.0

# or headless, one command:
python -m astrophoto run "images/IC 5070_sub"
python -m astrophoto run DIR --palette hoo --saturation 1.8 --scale 1.5 --upscale 2 --device cuda
python -m astrophoto analyse DIR          # just the frame-quality report
python -m astrophoto devices              # show GPUs PyTorch can use
```

Point the UI or CLI at any folder of Seestar subs, for example the
`<Object>_sub` folders the Seestar writes. JPG thumbnails and non-light
frames are ignored. Results are cached in `output/<folder>-<hash>/`:
`stack.fits`, the two half stacks, `denoised.fits`, the coverage and
rejection maps, and `exports/`.

### NVIDIA GPUs (Windows / Linux)

The AI denoiser runs on NVIDIA CUDA, Apple Metal (MPS) or CPU. The default
PyTorch wheel on Windows/Linux is often CPU-only, so install the CUDA build:

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows  (Linux: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -m astrophoto devices          # should list your RTX card
```

On CUDA the denoiser uses mixed precision (bf16 on RTX 30/40/50, fp16 on
older cards). It sizes the training batch and the inference tiles to the
card's VRAM: 8 GB cards work fine, and 4–6 GB cards use smaller tiles. Pick a
specific GPU with `--device cuda:1`, or with `ASTROPHOTO_DEVICE=cuda:1`. The
UI also has a device selector. Everything outside the denoiser (NumPy,
OpenCV, SEP) runs on the CPU and is platform-independent.

## What happens to your data

| Stage | Technique |
|---|---|
| **Calibration** | Black level from the FITS `BIAS` header. There are no dark frames, so hot and warm pixels are found from the temporal median of unregistered subs: sky drifts between frames but sensor defects don't. A pixel is flagged when it is an isolated same-colour outlier, which protects star cores. |
| **Frame grading** | SEP star extraction on every sub measures star count, FWHM, elongation (wind or tracking trails), sky level and noise. |
| **Registration** | Asterism (triangle) matching, then RANSAC similarity refinement on every matched star. This handles the alt-az field rotation of the Seestar, which can reach about 100° over a long session. A robust **third-order polynomial distortion model** is then fitted per frame to hundreds of matched stars, which keeps edge stars round as the field rotates across the optics. |
| **Cloud / obstruction detection** | Reference-star photometry: each bright reference star that should appear in a frame is looked up, and the flux ratio is aggregated on a tile grid. Tiles whose stars dim or vanish (a tree, a roof, a passing cloud) become per-frame masks, so a partly blocked frame still contributes its clean area. |
| **Rejection & weighting** | Robust median/MAD tests on each metric, plus an unsupervised **Isolation Forest** over the multivariate metrics. Weights are signal²/noise² × sharpness. Sensitivity is adjustable, and each frame can be overridden in the UI. |
| **Integration** | Streaming three-pass integration with bounded memory (hundreds of subs fit in 16 GB of RAM). **Local normalisation** removes each frame's rotating gradient against the running mean. Weighted **sigma clipping** removes satellites, planes and cosmic rays. **Bayer drizzle** resamples each colour's samples directly, with no demosaic interpolation. Optional 1.5× or 2× output uses the dithering and rotation between frames. Frames alternate between two independent **half stacks**. |
| **AI denoise** | **Noise2Noise**: a U-Net is trained *on your own data* to map half-stack A to half-stack B. Because the noise in the two is independent, the network learns the expected clean signal for this exact sensor, sky and integration. It uses no pretrained weights, so it can't invent detail from other people's images. Training runs in a variance-stabilised (asinh) domain, inference averages 8 rotations/flips (self-ensemble), and bright star cores are handed back unchanged. |
| **Gradient removal** | Tile samples with stars masked. An iterative *lower-envelope* surface fit (polynomial or thin-plate RBF) rejects samples sitting on nebulosity or galaxies. When nebulosity dominates the field, the model order is reduced automatically. |
| **Crop** | Largest fully covered rectangle, found by an aspect-ratio search on the coverage map and centred on the deepest part of the stack. The minimum coverage is adjustable. |
| **Colour** | Background neutralisation and star-based white balance: aperture photometry of isolated, unsaturated stars, with the aperture sized to the *widest* colour channel. When AI deconvolution is available, the stars are measured on the deconvolved image, where each channel's halo light is back in the core. Refractors spread blue light into a wider halo, and small apertures miss it; that used to over-boost blue and gave galaxies a pink or magenta cast. Pixels clipped in any channel are rendered neutral, because white-balance gains would otherwise turn saturated cores blue or purple. |
| **AI deconvolution** (option) | A second network is trained on the same half-stack pairs to *undo the blur*. This is a Noise2Noise adaptation of ZS-DeconvNet. The network takes the denoised half A. Its output, blurred by **PSFs measured from your own stars (one per colour channel)**, must predict the raw half B (χ² with the measured per-pixel noise). The only way to lower that loss is to recover the true, sharper sky. A Hessian penalty and a physical **sky-floor prior** (no flux below the local sky) prevent the dark rings and noise that classic deconvolution produces. In held-out tests on M 27, IC 5070 and M 31, star FWHM fell by 2–2.5× with no ringing, against 1.1× for the old masked Richardson–Lucy (see `experiments/README.md`). Saturated stars are handed back to the denoised image. Richardson–Lucy with TV regularisation remains as the fallback when too few stars are available. |
| **Restoration: ImageMM** (default) | **ImageMM** (Sukurdeep et al. 2025, arXiv:2501.03002), as published. One non-negative, background-subtracted sky image is fitted to **every individual sub at once**. Each sub has its own PSF measured from its stars, photon-transfer variances and masks. The fit uses majorization-minimization, with Huber-robust weights (removes satellite trails) and the paper's update clipping and stopping rule. Biggs–Andrews acceleration is on, which gives the converged result in half the time. The sky background is driven to zero instead of being denoised. The subs are prepared by linear demosaicing and star-refined registration (RMS at the centroid-noise level), with per-colour photometric scales and background models. On held-out subs of M 27 it beats both the stack and the deconvolution networks in every channel, and sky noise falls by three orders of magnitude (`experiments/README.md`). Options: 2× super-resolution (the paper's Algorithm 2), Moffat PSFs, seeing groups, a Noise2Noise pass. The first run prepares the subs once (about 30 min for 270 subs); the full-field restoration time is measured below. |
| **Star separation** | Stars are detected on a background mesh scaled to the PSF, so stars on galaxy discs separate cleanly. A concentration index keeps galaxy nuclei, M32/M110-type companions and nebula knots out of the star layer. Mask radii come from each star's measured per-channel radial profile, and push-pull inpainting with matched grain fills the gaps. |
| **Star colour & halos** | Refractors bring blue/violet (and the OIII band) to a slightly different focus, so bright stars get coloured rings. Halo light above the local background is desaturated in linear data. The star layer uses a luminance-only stretch, true linear star colour and an "unscreen" recombination, which gives white cores with no coloured blooming, dark donuts or tints over bright backgrounds. |
| **Stretch** | **Generalized Hyperbolic Stretch**. Its strength is solved automatically so that the starless background lands on a target level. It is colour-preserving, with luminance-preserving gamut mapping so that saturated highlights never darken. |
| **Narrowband (LP filter)** | Ha comes from the red pixels and OIII from the green and blue pixels. Ha **leakage into OIII is estimated from the data** (lower envelope of OIII/Ha over high-SNR Ha pixels) and removed. OIII is then linearly fitted to Ha, both are stretched with one curve, and they are combined as **Foraxx** (dynamic), HOO or warm HOO. **Synthetic luminance** (LRGB-style) takes lightness from the best-SNR all-channel stretch, so red-dominant Ha regions keep their full brightness. |
| **Finishing** | Post-stretch starlet shrinkage on luminance, OKLab chroma noise reduction, wavelet local contrast, perceptual (OKLab) vibrance with background protection, SCNR, curves and masked sharpening. |

## Super-resolution

Every Seestar sub lands on the sky slightly shifted and rotated (tracking drift and alt-az
field rotation), so a stack samples the sky on a finer grid than any single frame. With
**Super-resolution 1.5× / 2×** (web UI: *Integration & compute options*; CLI: `--scale 2`),
the Bayer-drizzle integrator resamples every colour sample straight onto the finer grid.
There is no demosaic step and no invented detail: the extra resolution comes from the data.

Measured on 100 subs of M 27: star FWHM went from **7.7″ at 1× to 6.5″ at 2×**, about 16% sharper,
with better colour resolution because each colour channel is sampled directly. At 2× the pixel scale
(1.15″/px) already samples the seeing-limited stars properly, so going beyond 2× gains nothing.
The deconvolution stage then works on the finer grid.

Costs: about 4× the stacking time and 4× larger stack and export files. It needs roughly 50+ subs to
fill the finer grid evenly. Stacking parallelism is limited automatically to fit about 3 GB of RAM;
raise it with `ASTROPHOTO_STACK_RAM_GB=6` on bigger machines. The export **Upscale** option is only
interpolation. Use Super-resolution for real detail.

## Research & benchmarks

`experiments/` holds the harness used to choose the ML models. It scores every
denoiser and deconvolver on held-out data against the independent half-stack, so
no ground truth is needed. It covers U-Net variants, NAFNet, ZS-N2N,
non-local means, Richardson–Lucy, DPIR plug-and-play and the N2N deconvolution
network. The results and the arXiv papers behind each choice are in
[experiments/README.md](experiments/README.md).

## Tips

- **Presets** in the Process tab are starting points: *Balanced*, *Vivid nebula*,
  *Galaxy / broadband*, *Natural colour* and so on.
- Processing sliders re-render a fast preview in about a second. The heavy
  stages (analysis, stacking, denoise) only re-run when you press their buttons.
- If you change frame selection or sensitivity in the Frames tab, run
  **Register & integrate** again, then **AI denoise & deconvolve**.
- For very short sessions (fewer than about 15 subs) the stacker switches from
  drizzle to demosaic automatically. The denoiser still works, but has less to learn from.
- Export writes a JSON sidecar with every parameter, so a result can be reproduced.
