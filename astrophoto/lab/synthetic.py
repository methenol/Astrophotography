"""Synthetic Seestar-like datasets with an exactly known sky.

The sky is analytic: point stars and extended emission (Gaussian clouds, filaments, a
planetary-nebula shell, Sérsic galaxies), each with its own colour.  Every sub sees it
through its own geometry (dither and field rotation, as an alt-az mount gives), its own
elliptical Moffat PSF (seeing varies), transparency and sky background with a gradient;
the result is pixel-integrated, mosaicked through the Bayer pattern, converted to
electrons with Poisson shot noise and Gaussian read noise, digitised to ADU with a bias
and clipped at 16 bits.  Hot pixels, satellite trails and cosmic rays are added as the
outliers a robust method has to reject.  The subs are written as FITS files with the
header keys of a Seestar S50 (``frames.read_info``), so the whole pipeline runs on them
unchanged.

Truth.  ``render(spec, geometry, psf, ...)`` evaluates the same sky on any pixel grid
(for example the reference sub's grid at the stack scale) through any PSF, or none
(``psf=None``: the sky integrated over each pixel, the target of a restoration at r = 1).
Pixel integration uses midpoint quadrature on an ``os`` x ``os`` sub-grid for the extended
emission (convolved by FFT on that sub-grid) and exact per-star stamps: a star's PSF is
sampled at the sub-pixel points around its true position; with no PSF a star is a delta,
all its flux in the pixel that contains it.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta

import numpy as np
import torch

DEFAULT_SPEC = {
    "name": "synthetic",
    "seed": 1,
    "width": 1024, "height": 1024,           # sensor pixels (even: whole Bayer cells)
    "n_subs": 40, "exptime": 10.0,
    "bayer": "GRBG", "filter": "IRCUT",       # "LP" makes the pipeline treat it as dual-band
    "focallen": 250.0, "pixsize": 2.9,
    "ra": 180.0, "dec": 30.0,
    "e_per_adu": 0.5, "read_noise_e": 1.2, "bias_adu": 2064, "dark_e": 0.0,
    "sky_e": [180.0, 260.0, 160.0],          # sky background per pixel per sub (R, G, B), electrons
    "sky_gradient": 0.08,                    # fractional linear gradient across the frame
    "seeing_fwhm": 2.6, "seeing_spread": 0.15,   # median FWHM (px) and log-normal spread between subs
    "moffat_beta": 3.0, "ellipticity": 0.08,
    "dither_px": 12.0, "rotation_deg": 4.0,  # random dither (sigma) and field rotation over the session
    "transparency_spread": 0.04, "cloudy_fraction": 0.05,
    "star_density": 20000.0, "star_mag_bright": 7.0, "star_mag_faint": 18.0,   # stars per 10^6 px of sky, magnitude range
    "zero_point_e": 3.0e8,                   # electrons per sub from a magnitude-0 star (all channels)
    "nebula": "emission",                    # emission | reflection | none
    "nebula_peak_e": 120.0, "n_galaxies": 6, "galaxy_peak_e": 150.0,
    "hot_pixel_fraction": 3e-4, "trail_fraction": 0.1, "cosmic_rays": 4,
}


# ----------------------------------------------------------------------------- sky model
def make_sky(spec: dict) -> dict:
    """Random analytic sky in the sky plane (units: reference-sensor pixels, origin at the
    sensor centre).  Deterministic in ``spec['seed']``."""
    rng = np.random.default_rng(spec["seed"])
    W, H = spec["width"], spec["height"]
    R = 0.5 * math.hypot(W, H) + 4 * spec["dither_px"] + 20       # covers every dithered / rotated frame
    # stars: uniform positions; magnitudes from a power-law luminosity function (counts x10^0.35 per mag)
    n = int(round(spec["star_density"] * math.pi * R ** 2 / 1e6))
    r = R * np.sqrt(rng.random(n))
    t = 2 * np.pi * rng.random(n)
    m_lo, m_hi, a = spec["star_mag_bright"], spec["star_mag_faint"], 0.35 * math.log(10)
    u = rng.random(n)
    mag = m_lo + np.log1p(u * (np.exp(a * (m_hi - m_lo)) - 1)) / a
    flux = spec["zero_point_e"] * 10 ** (-0.4 * mag)
    teff = np.exp(rng.normal(math.log(5500), 0.35, n)).clip(2800, 30000)
    col = np.stack([blackbody_rgb(T) for T in teff])               # channel fractions, sum 3
    stars = {"x": r * np.cos(t), "y": r * np.sin(t), "flux": flux, "mag": mag, "teff": teff, "rgb": col}
    ext = []
    if spec["nebula"] != "none":
        # an emission complex: a few large clouds, many smaller knots, filaments, and a shell
        emi = spec["nebula"] == "emission"
        c_main = np.array([0.8, 0.25, 0.35]) if emi else np.array([0.35, 0.55, 1.0])
        c_oiii = np.array([0.1, 0.6, 0.7])
        for k in range(28):
            s = float(np.exp(rng.uniform(math.log(6), math.log(min(W, H) / 5))))
            amp = spec["nebula_peak_e"] * float(rng.uniform(0.15, 1.0)) * (s / (min(W, H) / 5)) ** 0.3
            c = c_main if rng.random() < 0.75 else c_oiii
            ext.append({"kind": "gauss", "x": float(rng.normal(0, W / 5)), "y": float(rng.normal(0, H / 5)),
                        "sx": s, "sy": s * float(rng.uniform(0.4, 1.0)), "theta": float(rng.uniform(0, np.pi)),
                        "amp": amp, "rgb": c.tolist()})
        for k in range(6):      # filaments: long thin Gaussians
            ext.append({"kind": "gauss", "x": float(rng.normal(0, W / 4)), "y": float(rng.normal(0, H / 4)),
                        "sx": float(rng.uniform(W / 10, W / 4)), "sy": float(rng.uniform(1.5, 4.0)),
                        "theta": float(rng.uniform(0, np.pi)), "amp": spec["nebula_peak_e"] * 0.3,
                        "rgb": c_main.tolist()})
        ext.append({"kind": "shell", "x": float(W * 0.18), "y": float(-H * 0.15), "r": 26.0, "w": 3.5,
                    "amp": spec["nebula_peak_e"] * 1.5, "rgb": c_oiii.tolist()})
        ext.append({"kind": "gauss", "x": float(W * 0.18), "y": float(-H * 0.15), "sx": 16.0, "sy": 16.0, "theta": 0.0,
                    "amp": spec["nebula_peak_e"] * 0.5, "rgb": [0.8, 0.3, 0.4]})
    for k in range(int(spec["n_galaxies"])):
        n_s = float(rng.choice([1.0, 1.0, 2.0, 4.0]))
        re = float(np.exp(rng.uniform(math.log(3), math.log(25))))
        ext.append({"kind": "sersic", "x": float(rng.uniform(-W / 2.3, W / 2.3)), "y": float(rng.uniform(-H / 2.3, H / 2.3)),
                    "re": re, "n": n_s, "q": float(rng.uniform(0.3, 1.0)), "theta": float(rng.uniform(0, np.pi)),
                    "amp": spec["galaxy_peak_e"] * float(rng.uniform(0.2, 1.0)), "rgb": [1.1, 1.0, 0.8]})
    return {"stars": stars, "extended": ext}


def blackbody_rgb(T: float) -> np.ndarray:
    """Relative R, G, B signal of a blackbody at T (effective wavelengths 610, 530, 460 nm),
    normalised to sum 3."""
    lam = np.array([610e-9, 530e-9, 460e-9])
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    b = lam ** -5 / np.expm1(h * c / (lam * k * T))
    return 3 * b / b.sum()


def _extended(xs: torch.Tensor, ys: torch.Tensor, comps: list) -> torch.Tensor:
    """Surface brightness (3, ...) of the extended components at sky-plane points."""
    out = torch.zeros((3,) + tuple(xs.shape), dtype=torch.float32, device=xs.device)
    for c in comps:
        dx, dy = xs - c["x"], ys - c["y"]
        rgb = torch.tensor(c["rgb"], dtype=torch.float32, device=xs.device).view(3, *([1] * xs.dim()))
        if c["kind"] == "gauss":
            ct, st = math.cos(c["theta"]), math.sin(c["theta"])
            u, v = dx * ct + dy * st, -dx * st + dy * ct
            f = c["amp"] * torch.exp(-0.5 * ((u / c["sx"]) ** 2 + (v / c["sy"]) ** 2))
        elif c["kind"] == "shell":
            rr = torch.sqrt(dx ** 2 + dy ** 2)
            f = c["amp"] * torch.exp(-0.5 * ((rr - c["r"]) / c["w"]) ** 2)
        elif c["kind"] == "sersic":
            ct, st = math.cos(c["theta"]), math.sin(c["theta"])
            u, v = dx * ct + dy * st, (-dx * st + dy * ct) / c["q"]
            n = c["n"]
            bn = 2 * n - 1 / 3 + 4 / (405 * n)                     # Ciotti & Bertin 1999
            rr = torch.sqrt(u ** 2 + v ** 2 + 0.25)                # softened by a quarter pixel at the centre
            f = c["amp"] * torch.exp(-bn * ((rr / c["re"]) ** (1 / n) - 1)) / math.exp(bn)
        else:
            continue
        out += rgb * f
    return out


# ----------------------------------------------------------------------------- geometry & PSFs
def frame_geometry(spec: dict) -> list[dict]:
    """Per-sub dither (px), rotation (rad), PSF and transparency.  Sensor pixel p (x = column,
    y = row, pixel centres at integers) of sub t sees sky-plane point
    q = R(-theta_t) (p - c - d_t), with c the sensor centre."""
    rng = np.random.default_rng(spec["seed"] + 1)
    n = int(spec["n_subs"])
    frames = []
    rot0 = -0.5 * math.radians(spec["rotation_deg"])
    for t in range(n):
        fw = spec["seeing_fwhm"] * float(np.exp(rng.normal(0, spec["seeing_spread"])))
        e = float(abs(rng.normal(0, spec["ellipticity"])))
        cloudy = rng.random() < spec["cloudy_fraction"]
        trans = float(np.clip(1 - abs(rng.normal(0, spec["transparency_spread"])), 0.5, 1.0)) * (0.55 if cloudy else 1.0)
        frames.append({"dx": float(rng.normal(0, spec["dither_px"])), "dy": float(rng.normal(0, spec["dither_px"])),
                       "theta": rot0 + math.radians(spec["rotation_deg"]) * t / max(n - 1, 1) + float(rng.normal(0, 0.002)),
                       "fwhm": fw, "beta": spec["moffat_beta"], "e": e, "psf_angle": float(rng.uniform(0, np.pi)),
                       "transparency": trans,
                       "grad_angle": float(rng.uniform(0, 2 * np.pi)), "trail": bool(rng.random() < spec["trail_fraction"])})
    return frames


class Moffat:
    """Elliptical Moffat profile (Moffat 1969), unit integral: FWHM along the major axis
    ``fwhm``, minor axis fwhm (1 - e), major axis at angle ``angle`` (radians, from +x)."""

    def __init__(self, fwhm: float, beta: float, e: float = 0.0, angle: float = 0.0):
        self.beta, self.angle = beta, angle
        k = 2 * math.sqrt(2 ** (1 / beta) - 1)
        self.ax, self.ay = fwhm / k, fwhm * (1 - e) / k
        self.norm = (beta - 1) / (math.pi * self.ax * self.ay)
        self.radius = fwhm * 5 + 4

    def __call__(self, dx, dy):
        c, s = math.cos(self.angle), math.sin(self.angle)
        u, v = dx * c + dy * s, -dx * s + dy * c
        return self.norm * (1 + (u / self.ax) ** 2 + (v / self.ay) ** 2) ** (-self.beta)


class Gaussian:
    def __init__(self, sigma: float):
        self.sigma = sigma
        self.radius = 5 * sigma + 2

    def __call__(self, dx, dy):
        return torch.exp(-0.5 * (dx ** 2 + dy ** 2) / self.sigma ** 2) / (2 * math.pi * self.sigma ** 2)


# ----------------------------------------------------------------------------- rendering
def render(sky: dict, geom: dict, shape: tuple[int, int], sensor: tuple[int, int], scale: float = 1.0,
           psf=None, os_: int = 3, device=None, stars: bool = True, extended: bool = True) -> np.ndarray:
    """The sky (electrons per sub, before transparency) on an output grid of ``shape`` (h, w)
    at ``scale`` output pixels per sensor pixel of the frame ``geom`` (output pixel u is centred
    on sensor coordinate (u + 0.5) / scale - 0.5), through ``psf`` (a callable of offsets in
    output pixels, unit integral) or pixel-integrated with no PSF.  Returns (h, w, 3)."""
    device = device or torch.device("cpu")
    h, w = shape
    W0, H0 = sensor
    cx, cy = (W0 - 1) / 2, (H0 - 1) / 2
    ct, st = math.cos(geom["theta"]), math.sin(geom["theta"])

    def to_sky(px, py):                       # output pixel coordinates -> sky plane
        sx = (px + 0.5) / scale - 0.5 - cx - geom["dx"]
        sy = (py + 0.5) / scale - 0.5 - cy - geom["dy"]
        return sx * ct + sy * st, -sx * st + sy * ct       # R(-theta)

    out = torch.zeros((3, h, w), dtype=torch.float32, device=device)
    area = 1.0 / scale ** 2                   # sensor-pixel area of one output pixel
    if extended and sky["extended"]:
        # midpoint quadrature on the os x os sub-grid, in row bands; FFT convolution needs a
        # margin of the PSF radius around each band
        rad = int(math.ceil(psf.radius)) if psf is not None else 0
        band = max(16, int(2 ** 22 // (w * os_ * os_)))
        for a in range(0, h, band):
            b = min(h, a + band)
            ya = torch.arange((a - rad) * os_, (b + rad) * os_, device=device, dtype=torch.float32)
            xa = torch.arange(-rad * os_, (w + rad) * os_, device=device, dtype=torch.float32)
            py, px = torch.meshgrid((ya + 0.5) / os_ - 0.5, (xa + 0.5) / os_ - 0.5, indexing="ij")
            qx, qy = to_sky(px, py)
            f = _extended(qx, qy, sky["extended"]) * (area / os_ ** 2)     # electrons per sub-cell
            if psf is not None:
                k = int(math.ceil(psf.radius * os_))
                o = torch.arange(-k, k + 1, device=device, dtype=torch.float32) / os_
                oy, ox = torch.meshgrid(o, o, indexing="ij")
                ker = psf(ox, oy)
                ker = ker / ker.sum()
                f = _fftconv(f, ker)
            f = f[:, rad * os_:(rad + b - a) * os_, rad * os_:(rad + w) * os_]
            out[:, a:b] += f.reshape(3, b - a, os_, w, os_).sum((2, 4))
    if stars:
        S = sky["stars"]
        # star positions in output pixels (inverse of to_sky)
        qx, qy = S["x"], S["y"]
        sx, sy = qx * ct - qy * st, qx * st + qy * ct      # R(theta)
        px = ((sx + cx + geom["dx"]) + 0.5) * scale - 0.5
        py = ((sy + cy + geom["dy"]) + 0.5) * scale - 0.5
        fl = S["flux"][:, None] * S["rgb"] / 3.0                           # electrons per channel
        if psf is None:
            ix, iy = np.floor(px + 0.5).astype(int), np.floor(py + 0.5).astype(int)
            ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            img = np.zeros((3, h, w), np.float64)
            for c in range(3):
                np.add.at(img[c], (iy[ok], ix[ok]), fl[ok, c])
            out += torch.from_numpy(img.astype(np.float32)).to(device)
        else:
            _stamp_stars(out, px, py, fl, psf, os_)
    return out.permute(1, 2, 0).cpu().numpy()


def _fftconv(f: torch.Tensor, ker: torch.Tensor) -> torch.Tensor:
    """'same' linear convolution of (C, H, W) with an odd (k, k) kernel via zero-padded FFT."""
    C, H, W = f.shape
    k = ker.shape[-1]
    Hp, Wp = H + k - 1, W + k - 1
    F_ = torch.fft.rfft2(f, s=(Hp, Wp))
    K_ = torch.fft.rfft2(ker, s=(Hp, Wp))
    g = torch.fft.irfft2(F_ * K_, s=(Hp, Wp))
    r = k // 2
    return g[:, r:r + H, r:r + W]


def _stamp_stars(out: torch.Tensor, px, py, fl, psf, os_: int, chunk: int = 512):
    """Add each star's PSF, integrated over output pixels by os x os midpoint quadrature
    around its exact position."""
    dev = out.device
    _, h, w = out.shape
    R = int(math.ceil(psf.radius))
    n = 2 * R + 1
    ok = (px > -R) & (px < w + R) & (py > -R) & (py < h + R)
    px, py, fl = px[ok], py[ok], fl[ok]
    sub = (torch.arange(os_, device=dev, dtype=torch.float32) + 0.5) / os_ - 0.5
    off = torch.arange(-R, R + 1, device=dev, dtype=torch.float32)
    for a in range(0, len(px), chunk):
        X = torch.as_tensor(px[a:a + chunk], dtype=torch.float32, device=dev)
        Y = torch.as_tensor(py[a:a + chunk], dtype=torch.float32, device=dev)
        Fc = torch.as_tensor(fl[a:a + chunk], dtype=torch.float32, device=dev)
        ix, iy = torch.round(X), torch.round(Y)
        # sample points: pixel (iy + oy, ix + ox) sub-point (sy, sx) relative to the star
        gx = (ix[:, None, None] + off[None, None, :] - X[:, None, None])[..., None] + sub     # (N,1,n,os)
        gy = (iy[:, None, None] + off[None, :, None] - Y[:, None, None])[..., None] + sub     # (N,n,1,os)
        val = psf(gx[:, :, :, None, :].expand(-1, n, -1, os_, -1), gy[:, :, :, :, None].expand(-1, -1, n, -1, os_))
        stamp = val.mean((-1, -2))                                    # (N, n, n) pixel integrals / pixel area
        stamp = stamp / stamp.sum((1, 2), keepdim=True).clamp_min(1e-30)   # flux-conserving (5 FWHM stamp)
        yy = (iy[:, None] + off[None, :]).long()                      # (N, n)
        xx = (ix[:, None] + off[None, :]).long()
        valid = ((yy >= 0) & (yy < h))[:, :, None] & ((xx >= 0) & (xx < w))[:, None, :]
        lin = (yy.clamp(0, h - 1)[:, :, None] * w + xx.clamp(0, w - 1)[:, None, :])
        for c in range(3):
            contrib = (stamp * Fc[:, c, None, None] * valid).reshape(-1)
            out[c].view(-1).index_add_(0, lin.reshape(-1), contrib)


def _psf_of(g: dict, scale: float = 1.0) -> Moffat:
    return Moffat(g["fwhm"] * scale, g["beta"], g["e"], g["psf_angle"])


# ----------------------------------------------------------------------------- dataset
def generate(spec: dict, out_dir: str, device=None, progress=None) -> dict:
    """Write the subs (``out_dir/subs/*.fit``) and the truth (``out_dir/truth.json``,
    ``out_dir/sky.npz``).  Returns the full spec with the per-sub geometry."""
    from astropy.io import fits
    from ..frames import cfa_channel_map
    spec = {**DEFAULT_SPEC, **spec}
    spec["width"] -= spec["width"] % 2
    spec["height"] -= spec["height"] % 2
    device = device or torch.device("cpu")
    os.makedirs(os.path.join(out_dir, "subs"), exist_ok=True)
    sky = make_sky(spec)
    geoms = frame_geometry(spec)
    W, H = spec["width"], spec["height"]
    rng = np.random.default_rng(spec["seed"] + 2)
    cmap = cfa_channel_map(spec["bayer"], (H, W))
    hot = rng.random((H, W)) < spec["hot_pixel_fraction"]
    hot_e = rng.uniform(2e3, 3e4, (H, W)) * hot
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t0 = datetime(2026, 1, 1, 22, 0, 0)
    sky_e = np.asarray(spec["sky_e"], np.float32)
    for t, g in enumerate(geoms):
        img = render(sky, g, (H, W), (W, H), 1.0, _psf_of(g), device=device) * g["transparency"]
        # sky background with a linear gradient
        gr = 1 + spec["sky_gradient"] * ((xx - W / 2) * math.cos(g["grad_angle"]) + (yy - H / 2) * math.sin(g["grad_angle"])) / max(W, H)
        img = img + sky_e[None, None, :] * gr[..., None]
        if g["trail"]:                                             # satellite trail: a bright line
            a = rng.uniform(0, np.pi)
            p0 = rng.uniform(0, W), rng.uniform(0, H)
            d = np.abs((xx - p0[0]) * math.sin(a) - (yy - p0[1]) * math.cos(a))
            img = img + (rng.uniform(300, 3000) * np.exp(-0.5 * (d / 1.0) ** 2))[..., None]
        mosaic = np.take_along_axis(img, cmap[..., None].astype(np.int64), 2)[..., 0]
        e = rng.poisson(np.maximum(mosaic + spec["dark_e"], 0)).astype(np.float32) + hot_e
        e += rng.normal(0, spec["read_noise_e"], e.shape).astype(np.float32)
        for _ in range(rng.poisson(spec["cosmic_rays"])):
            cy, cx = rng.integers(0, H), rng.integers(0, W)
            e[cy, cx] += rng.uniform(2e3, 2e4)
        adu = np.clip(np.round(e / spec["e_per_adu"] + spec["bias_adu"]), 0, 65535).astype(np.uint16)
        hdr = fits.Header()
        date = t0 + timedelta(seconds=t * (spec["exptime"] + 2.0))
        for k, v in {"CREATOR": "AstroPhoto synthetic", "IMAGETYP": "Light", "OBJECT": spec["name"],
                     "FILTER": spec["filter"], "EXPTIME": spec["exptime"], "EXPOSURE": spec["exptime"],
                     "GAIN": 80, "CCD-TEMP": 20.0, "DATE-OBS": date.isoformat(), "BAYERPAT": spec["bayer"],
                     "BIAS": spec["bias_adu"], "RA": spec["ra"], "DEC": spec["dec"], "FOCALLEN": spec["focallen"],
                     "XPIXSZ": spec["pixsize"], "YPIXSZ": spec["pixsize"]}.items():
            hdr[k] = v
        fits.PrimaryHDU(adu, header=hdr).writeto(
            os.path.join(out_dir, "subs", f"Light_{spec['name']}_{spec['exptime']:.1f}s_{t + 1:05d}.fit"), overwrite=True)
        if progress:
            progress(t + 1, len(geoms), f"Rendering sub {t + 1}/{len(geoms)}")
    np.savez_compressed(os.path.join(out_dir, "sky.npz"), **{f"star_{k}": v for k, v in sky["stars"].items()})
    truth = {"spec": spec, "frames": geoms, "extended": sky["extended"], "created": datetime.now().isoformat(timespec="seconds")}
    json.dump(truth, open(os.path.join(out_dir, "truth.json"), "w"), indent=1)
    return truth


def load_truth(d: str) -> tuple[dict, dict]:
    """(truth metadata, sky model) of a generated dataset."""
    truth = json.load(open(os.path.join(d, "truth.json")))
    z = np.load(os.path.join(d, "sky.npz"))
    sky = {"stars": {k[5:]: z[k] for k in z.files if k.startswith("star_")}, "extended": truth["extended"]}
    return truth, sky


def truth_image(d: str, ref_idx: int, shape, scale: float, psf=None, device=None,
                stars: bool = True, extended: bool = True) -> np.ndarray:
    """The true sky on the pipeline's reference grid (sub ``ref_idx`` at ``scale``) without
    sky background, in ADU at the pipeline's photometric convention: every sub is divided
    by its transparency relative to the median sub (``analysis.finalize_selection``), so the
    stack and the restorations are in ADU of a sub at the median transparency."""
    truth, sky = load_truth(d)
    spec, g = truth["spec"], truth["frames"][ref_idx]
    img = render(sky, g, shape, (spec["width"], spec["height"]), scale, psf, device=device, stars=stars,
                 extended=extended)
    tau = float(np.median([f["transparency"] for f in truth["frames"]]))
    return img * tau / spec["e_per_adu"]


def effective_psf(d: str, frame_idx: list[int], weights: list[float] | None, scale: float = 1.0):
    """The PSF of a weighted mean of registered subs: the weighted mean of their PSFs (seen on
    the reference grid, rotation relative to the reference ignored: small by construction)."""
    truth, _ = load_truth(d)
    ps = [_psf_of(truth["frames"][i], scale) for i in frame_idx]
    wt = np.ones(len(ps)) if weights is None else np.asarray(weights, float)
    wt = wt / wt.sum()

    class Mix:
        radius = max(p.radius for p in ps)

        def __call__(self, dx, dy):
            return sum(float(w_) * p(dx, dy) for w_, p in zip(wt, ps))
    return Mix()


def star_positions(d: str, ref_idx: int, scale: float = 1.0) -> dict:
    """True star positions on the reference grid (sub ``ref_idx`` at ``scale``: output pixel u
    centred on sensor coordinate (u + 0.5) / scale - 0.5) and fluxes per channel in the
    ADU convention of ``truth_image``."""
    truth, sky = load_truth(d)
    spec, g = truth["spec"], truth["frames"][ref_idx]
    W0, H0 = spec["width"], spec["height"]
    ct, st = math.cos(g["theta"]), math.sin(g["theta"])
    S = sky["stars"]
    sx, sy = S["x"] * ct - S["y"] * st, S["x"] * st + S["y"] * ct
    px = (sx + (W0 - 1) / 2 + g["dx"] + 0.5) * scale - 0.5
    py = (sy + (H0 - 1) / 2 + g["dy"] + 0.5) * scale - 0.5
    tau = float(np.median([f["transparency"] for f in truth["frames"]]))
    flux = S["flux"][:, None] * S["rgb"] / 3.0 * tau / spec["e_per_adu"]
    return {"x": px, "y": py, "flux": flux, "mag": S["mag"]}
