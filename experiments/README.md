# ML restoration experiments

This folder has the benchmark harness and the results behind the AI denoise and
deconvolution stage in `astrophoto/denoise.py`. Everything here is reproducible
from the cached half-stacks in `output/`.

## How to score a method without ground truth

The stacker writes two **half-stacks** (even and odd frames): the same sky with
independent noise. Each method sees only half A, and is trained only on two of every
three 256 px vertical bands. Its estimate x̂ is then compared with half B on the
held-out bands. Because B's noise is independent of everything the method saw,

    E|x̂ − B|² = E|x̂ − x|² + σ_B²

so subtracting B's measured noise variance gives the **true error** of x̂, with no
clean image needed. For deconvolution the same identity applies after re-blurring
by the measured PSF k: E|k∗x̂ − B|² − σ_B² (the fidelity score).

Every score is quoted in units of one half-stack's noise variance:
**1.0 = no better than the raw half, lower is better**. Each is computed in two ways:

* **lin**: linear, inverse-variance weighted. Dominated by stars and bright structure.
* **str**: in an asinh-stretched domain. This is what faint nebulosity looks like
  after stretching.

Deconvolution is also scored on:
* median **FWHM** of isolated stars
* **ringing**: azimuthal profile minimum / peak
* **moat**: undershoot below the local sky, in sky-noise σ
* **background noise** in faint star-free regions, relative to the denoised input

`viz.py` renders the same stretch of every method side by side for a visual check.

Test crops: the deepest 1536² region of M 27 (LP filter, planetary nebula) and
IC 5070 (LP filter, emission nebula), and the deepest 2048² region of M 31 (IRCUT
broadband, galaxy, 2× drizzle).

## Denoising (`exp_denoise.py`)

Noise2Noise training, A↔B (Lehtinen et al. 2018, arXiv:1803.04189), in the asinh
variance-stabilised domain, 2000 steps. Scores are dB of error reduction relative
to raw half A, M 27:

| Method | lin | str | Notes |
|---|---|---|---|
| Non-local means (classic) | +3.9 dB | +5.9 dB | tuned to the measured noise |
| **U-Net (production)** | **+6.9 dB** | **+7.3 dB** | 0.47 M params |
| U-Net + 8× self-ensemble (Timofte et al. 2016, arXiv:1511.02228) | **+7.0 dB** | **+7.4 dB** | +5 s of inference, **adopted** |
| U-Net, 2× training steps | +6.8 dB | +7.2 dB | overfits slightly |
| Wider 4-level residual U-Net | +6.2 dB | +7.2 dB | more capacity doesn't help |
| NAFNet (Chen et al. 2022, arXiv:2204.04676) + 8× self-ensemble | +5.5 dB | +7.4 dB | ties on faint signal, worse on stars, 4× slower to train |
| ZS-N2N 2-layer net (Mansour & Heckel 2023, arXiv:2303.11253) | +6.0 dB | +7.0 dB | only 2 min: most of the gain comes from training on the data itself |

The noise in one night's stack is simple enough that a compact U-Net is already
near the limit of what these data allow. Averaging the network over the 8
rotations and flips is the one free improvement, and it is on in production.
The recent astro papers point the same way: AstroSURE (arXiv:2604.16793) uses a
~1 M-parameter U-Net and reports that Noise2Noise nearly matches supervised
training when paired exposures exist, as they do here.

## Deconvolution (`exp_deconv.py`)

Every method gets the N2N-denoised half A and per-channel PSFs measured from the
stars. The PSFs differ a lot between colours: on IC 5070 the red (Hα) PSF covers
about twice the area of the green one, because refractors focus colours
differently.

| M 27 (FWHM 4.22 px) | fidelity lin / str | FWHM | ringing | moat | bg noise |
|---|---|---|---|---|---|
| Denoised only (not deconvolved) | 0.20 / 0.19 | 4.22 | 0 | – | ×1.0 |
| Old production RL + TV, masked | 1.70 / 0.38 | 4.08 | 0 | – | ×1.0 |
| Plain Richardson–Lucy, 30 it | 0.45 / 0.21 | 2.35 | −2.5% | – | ×2.9 |
| DPIR plug-and-play (arXiv:2008.13751), N2N-trained conditional prior | 2.9 / 0.35 | 3.22 | 0 | – | ×0.29 |
| N2N deconv net, 1-stage | 0.37 / 0.22 | 2.05 | −1.7% | – | ×4.3 |
| + Hessian 0.3 | 0.40 / 0.23 | 2.27 | −1.2% | – | ×1.8 |
| 2-stage (input = denoised half), Hessian 0.3 | 0.41 / 0.23 | 2.13 | −1.2% | – | ×1.8 |
| **+ sky-floor prior (3.0, margin 0.5σ)**, adopted | 0.51 / 0.24 | **2.29** | **0** | **+0.2σ** | **×0.97** |

| Adopted method vs alternatives | IC 5070 (FWHM 6.95 px) | M 31 (FWHM 5.98 px) |
|---|---|---|
| Old production RL | 6.38 px, fid 3.41 | 5.47 px, fid 0.82 |
| Plain RL 30 it | 3.76 px, **moat −56σ**, bg ×2.2 | 3.46 px, **moat −53σ**, bg ×1.7 |
| **N2N deconv (adopted)** | **2.83 px**, moat −0.6σ, bg ×0.54 | **2.35 px**, moat −0.9σ, bg ×0.72 |

![Sky-floor ablation on M 27](figures/m27_sky_floor_ablation.jpg)
*M 27, same stretch: denoised only, then N2N deconvolution without the sky floor
(dark moats), then with floor weight 1 and 10.*

![Methods on IC 5070](figures/ic5070_methods.jpg)
![Methods on M 31](figures/m31_methods.jpg)
*Raw half, denoised, old production RL, plain RL (ringing), adopted N2N deconvolution.*

**What worked.** The **Noise2Noise deconvolution network** is our adaptation of
ZS-DeconvNet (Qiao et al., Nat. Commun. 2024). ZS-DeconvNet trains on re-corrupted
copies of a single image; we train on the genuinely independent half-stacks. The
network receives the denoised half A and outputs x. Its loss is

    χ²(k ∗ x, raw half B)  +  0.3·|Hessian(asinh x)|²  +  3·|max(0, sky − 0.5σ − x)/σ|²

The χ² can only fall if x is closer to the true, sharper sky. The Hessian term
(from ZS-DeconvNet) and the **sky-floor prior** constrain what the PSF cannot see.
The sky-floor prior is our addition: real flux is never below the local sky.

Three observations led to the final design:

1. **χ² normalisation matters.** With an un-normalised weighted MSE the Hessian
   term was 10⁴× too small to do anything.
2. **Two-stage beats one-stage.** Feeding the denoised half, as ZS-DeconvNet
   does, gave sharper stars and less noise at the same regularisation.
3. **The sky floor removes dark moats.** Without it the network, like RL, dug
   dark moats around stars. Those are invisible in linear numbers but obvious
   after stretching.

**What didn't work.**

* **Plain Richardson–Lucy:** too noisy, with deep ringing.
* **Old masked RL:** safe but almost no effect.
* **SNR-gating any method:** cleaner background, but it gives back most of the sharpening.
* **DPIR:** a noise-level-conditional denoiser trained on one night's data isn't
  well enough calibrated across noise levels to act as a plug-and-play prior. It
  either over-smoothed or left stars blurred, with fidelity 2–6× worse.

**Found in full-pipeline testing.** Around bright stars and tight groups, where the
real PSF is wider than the median PSF, the network left a smooth undershoot of
about 0.6σ over a disk about 15 px wide. The benchmark missed it (it measures
isolated stars), but a hard stretch shows it as a dark disk. Production now clamps
the output at min(denoised, local median sky over 3 PSF widths) (`deconv_floor`),
which removes it and leaves dark lanes alone.

## Literature consulted

* Lehtinen et al. 2018, *Noise2Noise*, arXiv:1803.04189
* Qiao et al. 2024, *ZS-DeconvNet* (Nat. Commun.), zero-shot deconvolution with Hessian regularisation
* Zhang et al. 2021, *DPIR*, arXiv:2008.13751, plug-and-play restoration with a deep denoiser prior
* Chen et al. 2022, *NAFNet*, arXiv:2204.04676
* Mansour & Heckel 2023, *Zero-Shot Noise2Noise*, arXiv:2303.11253
* Timofte et al. 2016, *Seven ways to improve example-based SR* (self-ensemble), arXiv:1511.02228
* AstroSURE 2026, arXiv:2604.16793, and ASTERIS 2026, arXiv:2602.17205 (self-supervised astronomical denoising)
* Self-supervised single-image deconvolution with Siamese networks, arXiv:2308.09426

## Reproduce

```bash
cd experiments
python exp_denoise.py M27 IC5070 M31
python exp_deconv.py M27 --methods prod_rl,rl30,n2n_deconv2s_h30_f300_m50
python viz.py M27 rawA den prod_rl n2n_deconv2s_h30_f300_m50
```
Method names encode their settings: `_h30` = Hessian 0.30, `_f300` = floor 3.0,
`_m50` = margin 0.5σ, `2s` = two-stage.
