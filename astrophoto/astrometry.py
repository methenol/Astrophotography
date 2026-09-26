"""Plate solving and sky catalogues for the Explore view.

Astrometric solution of a stack (no WCS in Seestar headers, only the target's RA/Dec, the
focal length and the pixel size):

1. Sources of the stack (sep, SExtractor algorithms; windowed centroids).
2. Gaia DR3 (Gaia Collaboration 2023, A&A 674, A1) around the header position, via the
   VizieR TAP service (catalogue I/355/gaiadr3), positions propagated from epoch 2016.0 to
   the observation date with the Gaia proper motions.
3. Initial match: the catalogue is projected gnomonically about the header position at the
   header pixel scale; the brightest stars of both lists inside the circle inscribed in the
   frame (rotation invariant) are matched with triangle invariants and RANSAC
   (astroalign, Beroiz et al. 2020, Astron. Comput. 32, 100384).  Triangle invariants
   (side ratios) do not see mirror images and a similarity cannot represent one, so both
   parities are tried; the one with more inliers wins.
4. Refinement: every catalogue star inside the frame is predicted through the current
   solution and paired with the nearest detection; a TAN (+ SIP) WCS is fitted to the pairs
   by least squares (astropy.wcs.utils.fit_wcs_from_points), with the pairing radius
   shrinking and 3-sigma clipping of the residuals.

Objects in the field come from SIMBAD (Wenger et al. 2000, A&AS 143, 9) via its TAP
service, with the object-type definitions of its ``otypedef`` table, and from Gaia DR3.
Every catalogue response is cached in the session folder (``explore/``).
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime

import numpy as np
import sep

GAIA_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"
SIMBAD_TAP = "https://simbad.cds.unistra.fr/simbad/sim-tap"
GAIA_EPOCH = 2016.0
SOLAR_MG = 4.67            # absolute G magnitude of the Sun (Casagrande & VandenBerg 2018)
C_KMS = 299792.458
H0 = 70.0                  # km/s/Mpc, for the (clearly labelled) Hubble-law distance of galaxies
LY_PER_PC = 3.261563777


# ----------------------------------------------------------------------------- geometry
def pixel_scale_guess(focallen_mm: float, pixsize_um: float, stack_scale: float = 1.0) -> float:
    """Arcseconds per stack pixel from the optics (206.265 arcsec per radian x um / mm)."""
    return 206.264806 * pixsize_um / focallen_mm / stack_scale


def gnomonic(ra, dec, ra0, dec0):
    """TAN projection (standard coordinates xi east, eta north, in degrees) about (ra0, dec0)."""
    ra, dec, ra0, dec0 = (np.radians(np.asarray(v, float)) for v in (ra, dec, ra0, dec0))
    cosc = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    xi = np.cos(dec) * np.sin(ra - ra0) / cosc
    eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cosc
    return np.degrees(xi), np.degrees(eta)


def decimal_year(iso: str) -> float:
    t = datetime.fromisoformat(iso.replace("Z", ""))
    y0 = datetime(t.year, 1, 1)
    return t.year + (t - y0).total_seconds() / (datetime(t.year + 1, 1, 1) - y0).total_seconds()


# ----------------------------------------------------------------------------- sources
def detect_stars(img: np.ndarray, coverage: np.ndarray | None = None, thresh: float = 5.0) -> dict:
    """Point sources of a stack (luminance), brightest first: windowed centroids, flux, FWHM."""
    L = np.ascontiguousarray(img.mean(-1) if img.ndim == 3 else img, np.float32)
    mask = None
    if coverage is not None:
        c = coverage if coverage.ndim == 2 else coverage.mean(-1)
        mask = np.ascontiguousarray(c < 0.3 * np.percentile(c[c > 0], 90))
    bkg = sep.Background(L, mask=mask, bw=64, bh=64)
    sub = L - bkg.back()
    sep.set_extract_pixstack(5_000_000)
    sep.set_sub_object_limit(4096)
    obj = sep.extract(sub, thresh, err=bkg.globalrms, mask=mask, minarea=5)
    obj = obj[(obj["a"] > 0) & (obj["b"] / np.maximum(obj["a"], 1e-6) > 0.5)]
    fwhm = float(np.median(2.3548 * np.sqrt(0.5 * (obj["a"] ** 2 + obj["b"] ** 2)))) if len(obj) else 3.0
    x, y, flag = sep.winpos(sub, obj["x"], obj["y"], np.full(len(obj), fwhm / 2.3548))
    ok = flag == 0
    x, y, flux = np.where(ok, x, obj["x"]), np.where(ok, y, obj["y"]), obj["flux"]
    order = np.argsort(-flux)
    return {"x": x[order], "y": y[order], "flux": flux[order], "fwhm": fwhm, "rms": float(bkg.globalrms)}


# ----------------------------------------------------------------------------- catalogues
def _tap(url: str, query: str, maxrec: int = 500_000):
    import pyvo
    return pyvo.dal.TAPService(url).run_sync(query, maxrec=maxrec).to_table()


def gaia_cone(ra: float, dec: float, radius: float, gmax: float, epoch: float) -> dict:
    """Gaia DR3 sources brighter than G = gmax within ``radius`` degrees, at ``epoch``."""
    q = f"""SELECT Source, RA_ICRS, DE_ICRS, pmRA, pmDE, Gmag, "BP-RP", Plx, e_Plx, Teff, Dist, AG, "E(BP-RP)", RV, VarFlag, NSS
        FROM "I/355/gaiadr3"
        WHERE 1 = CONTAINS(POINT('ICRS', RA_ICRS, DE_ICRS), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius:.5f}))
        AND Gmag < {gmax:.2f}"""
    t = _tap(GAIA_TAP, q)

    def col(name, fill=np.nan):
        c = t[name]
        return np.asarray(c.filled(fill) if hasattr(c, "filled") else c, dtype=float)
    dt = epoch - GAIA_EPOCH
    d = col("DE_ICRS")
    pmra, pmde = col("pmRA", 0.0), col("pmDE", 0.0)        # mas/yr; pmRA includes cos(dec)
    out = {"source_id": np.asarray(t["Source"], dtype=np.int64),
           "ra": col("RA_ICRS") + pmra * dt / 3.6e6 / np.cos(np.radians(d)),
           "dec": d + pmde * dt / 3.6e6,
           "pmra": col("pmRA"), "pmdec": col("pmDE"), "g": col("Gmag"), "bp_rp": col("BP-RP"),
           "plx": col("Plx"), "plx_err": col("e_Plx"), "teff": col("Teff"), "dist_gspphot": col("Dist"),
           "ag": col("AG"), "ebr": col("E(BP-RP)"),
           "rv": col("RV"),
           "variable": np.asarray([str(v) == "VARIABLE" for v in t["VarFlag"]]),
           "nss": col("NSS", 0.0).astype(int)}
    order = np.argsort(out["g"])
    return {k: v[order] for k, v in out.items()}


def simbad_cone(ra: float, dec: float, radius: float) -> list[dict]:
    """Every SIMBAD object within ``radius`` degrees: identifiers, type, V magnitude,
    spectral / morphological type, parallax, redshift and angular size."""
    q = f"""SELECT b.oid, b.main_id, b.ra, b.dec, b.otype, b.sp_type, b.morph_type, b.plx_value, b.plx_err,
        b.rvz_redshift, b.rvz_radvel, b.galdim_majaxis, b.galdim_minaxis, b.galdim_angle, b.nbref,
        f.flux AS vmag, i.ids
        FROM basic AS b
        LEFT OUTER JOIN flux AS f ON (f.oidref = b.oid AND f.filter = 'V')
        LEFT OUTER JOIN ids AS i ON (i.oidref = b.oid)
        WHERE CONTAINS(POINT('ICRS', b.ra, b.dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius:.5f})) = 1"""
    t = _tap(SIMBAD_TAP, q)
    rows = []
    for r in t:
        def v(k):
            x = r[k]
            if np.ma.is_masked(x):
                return None
            if isinstance(x, (bytes, np.bytes_)):
                x = x.decode()
            if isinstance(x, (np.floating, float)):
                return float(x) if np.isfinite(x) else None
            if isinstance(x, np.integer):
                return int(x)
            return str(x).strip() or None
        if v("ra") is None or v("dec") is None:
            continue
        rows.append({k: v(k) for k in ("main_id", "ra", "dec", "otype", "sp_type", "morph_type", "plx_value",
                                       "plx_err", "rvz_redshift", "rvz_radvel", "galdim_majaxis", "galdim_minaxis",
                                       "galdim_angle", "nbref", "vmag", "ids")})
    return rows


def simbad_otypes() -> dict:
    """SIMBAD object-type definitions: code -> {label, description, path}."""
    t = _tap(SIMBAD_TAP, "SELECT otype, label, description, path FROM otypedef")
    out = {}
    for r in t:
        g = [str(r[k]).strip() if not np.ma.is_masked(r[k]) else "" for k in ("otype", "label", "description", "path")]
        out[g[0]] = {"label": g[1], "description": g[2], "path": g[3]}
    return out


# ----------------------------------------------------------------------------- solving
def _initial_match(det: dict, cat_xy: np.ndarray, shape, n_ctrl: int = 40):
    """Similarity transform (catalogue tangent-plane pixels -> image pixels) by triangle
    matching of the brightest stars inside the inscribed circle; both parities."""
    import astroalign
    h, w = shape
    c = np.array([w / 2, h / 2])
    r_in = 0.48 * min(h, w)
    dxy = np.stack([det["x"], det["y"]], 1)
    dsel = dxy[np.hypot(*(dxy - c).T) < r_in][:n_ctrl]
    best = None
    for parity in (1, -1):
        src = cat_xy * np.array([parity, 1.0])
        ssel = src[np.hypot(*src.T) < r_in][: int(n_ctrl * 1.5)]
        if len(ssel) < 5 or len(dsel) < 5:
            continue
        try:
            T, (s_m, d_m) = astroalign.find_transform(ssel, dsel, max_control_points=len(ssel))
        except Exception:
            continue
        cand = {"parity": parity, "T": T, "n": len(s_m)}
        if best is None or cand["n"] > best["n"]:
            best = cand
    return best


def solve(det: dict, gaia: dict, ra0: float, dec0: float, scale_arcsec: float, shape,
          sip_degree: int = 3) -> dict:
    """Astrometric solution of an image: a TAN-SIP astropy WCS (0-based numpy pixel
    coordinates, x = column, y = row) with its residual statistics."""
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import fit_wcs_from_points
    from scipy.spatial import cKDTree
    h, w = shape
    xi, eta = gnomonic(gaia["ra"], gaia["dec"], ra0, dec0)
    # tangent plane in (approximate) pixels: east to the left as on the sky seen from inside
    cat_xy = np.stack([-xi, eta], 1) * 3600.0 / scale_arcsec
    m = _initial_match(det, cat_xy, shape)
    if m is None or m["n"] < 6:
        raise RuntimeError("no consistent star pattern between the image and Gaia (is the header position right?)")
    T, parity = m["T"], m["parity"]
    pred = T(cat_xy * np.array([parity, 1.0]))
    dxy = np.stack([det["x"], det["y"]], 1)
    tree = cKDTree(dxy)
    world = SkyCoord(gaia["ra"], gaia["dec"], unit="deg")
    wcs = None
    tol = max(4.0, 2.0 * det["fwhm"])
    for it in range(6):
        inside = (pred[:, 0] > -5) & (pred[:, 0] < w + 5) & (pred[:, 1] > -5) & (pred[:, 1] < h + 5)
        d, j = tree.query(pred)
        ok = inside & (d < tol)
        # one catalogue star per detection: keep the closest
        jj, first = np.unique(j[ok][np.argsort(d[ok])], return_index=True)
        sel = np.flatnonzero(ok)[np.argsort(d[ok])][first]
        if len(sel) < 8:
            raise RuntimeError(f"only {len(sel)} stars matched Gaia")
        deg = None if (it < 2 or len(sel) < 60) else (2 if len(sel) < 200 else sip_degree)
        xy = (dxy[j[sel], 0], dxy[j[sel], 1])
        wcs = fit_wcs_from_points(xy, world[sel], proj_point="center", projection="TAN", sip_degree=deg)
        px = np.stack(wcs.world_to_pixel(world[sel]), 1)
        res = np.hypot(*(px - dxy[j[sel]]).T)
        rms = float(np.sqrt(np.mean(res ** 2)))
        clip = res < max(3 * rms, 0.5)
        if clip.sum() >= 8 and clip.sum() < len(sel):
            sel = sel[clip]
            xy = (dxy[j[sel], 0], dxy[j[sel], 1])
            wcs = fit_wcs_from_points(xy, world[sel], proj_point="center", projection="TAN", sip_degree=deg)
            px = np.stack(wcs.world_to_pixel(world[sel]), 1)
            res = np.hypot(*(px - dxy[j[sel]]).T)
            rms = float(np.sqrt(np.mean(res ** 2)))
        pred = np.stack(wcs.world_to_pixel(world), 1)
        tol = max(1.5, min(tol, 5 * rms))
    scales = np.abs(np.linalg.svd(wcs.pixel_scale_matrix, compute_uv=False)) * 3600
    return {"wcs": wcs, "n_matched": int(len(sel)), "rms_px": rms, "rms_arcsec": rms * float(np.mean(scales)),
            "matched_g": gaia["g"][sel], "scale_arcsec": float(np.mean(scales)),
            "parity": int(np.sign(np.linalg.det(wcs.pixel_scale_matrix))), "initial_inliers": int(m["n"])}


def completeness_limit(det: dict, gaia: dict, wcs, shape) -> float | None:
    """Faintest 0.25 mag bin of Gaia G in which at least half of the catalogued stars in the
    frame have a detection within 1.5 FWHM: the depth of the image in G."""
    from scipy.spatial import cKDTree
    h, w = shape
    px = np.stack(wcs.all_world2pix(gaia["ra"], gaia["dec"], 0), 1)
    inside = (px[:, 0] > 10) & (px[:, 0] < w - 10) & (px[:, 1] > 10) & (px[:, 1] < h - 10)
    d, _ = cKDTree(np.stack([det["x"], det["y"]], 1)).query(px[inside])
    g = gaia["g"][inside]
    found = d < 1.5 * det["fwhm"]
    lim = None
    for lo in np.arange(np.floor(np.nanmin(g)), np.nanmax(g), 0.25):
        s = (g >= lo) & (g < lo + 0.25)
        if s.sum() >= 10:
            if found[s].mean() >= 0.5:
                lim = float(lo + 0.25)
            else:
                break
    return lim


# ----------------------------------------------------------------------------- descriptions
def spectral_class_from_teff(teff: float | None) -> str | None:
    """Harvard class from effective temperature (boundaries of Pecaut & Mamajek 2013)."""
    if teff is None or not np.isfinite(teff):
        return None
    for lim, c in ((30000, "O"), (10000, "B"), (7300, "A"), (6000, "F"), (5300, "G"), (3900, "K")):
        if teff >= lim:
            return c
    return "M"


def common_names(ids: str | None) -> dict:
    """Useful designations from SIMBAD's ``ids`` list (pipe-separated)."""
    import re
    out = {"names": [], "messier": None, "ngc": None, "bright": None}
    if not ids:
        return out
    for p in (" ".join(q.split()) for q in ids.split("|")):
        if p.startswith("NAME "):
            out["names"].append(p[5:])
        elif re.fullmatch(r"M \d+", p):
            out["messier"] = p
        elif re.fullmatch(r"(NGC|IC) \d+[A-Za-z]?", p) and not out["ngc"]:
            out["ngc"] = p
        elif p.startswith(("* ", "V* ")) and not out["bright"]:
            out["bright"] = p.split(" ", 1)[1]
        elif p.startswith(("HD ", "HR ", "HIP ")) and not out["bright"]:
            out["bright"] = p
    return out


# broad categories for filtering, from the first element of the SIMBAD type hierarchy
CATEGORIES = {
    "star": "Stars", "variable": "Variable stars", "multiple": "Double & multiple stars",
    "galaxy": "Galaxies", "nebula": "Nebulae & clusters", "other": "Radio, IR, X-ray & other sources",
}


def category(otype: str | None, defs: dict) -> str:
    """Broad category of a SIMBAD object type, from its place in the type hierarchy."""
    d = defs.get(otype or "", {})
    parts = [q.strip() for q in d.get("path", otype or "").split(">")]
    top = parts[0] if parts else ""
    if otype == "PN" or top in ("ISM", "Cl*", "As*"):
        return "nebula"
    if top in ("G", "GrG", "ClG", "SCG", "PCG", "IG", "PaG", "PoG"):
        return "galaxy"
    if top == "*":
        if "**" in parts:
            return "multiple"
        if "V*" in parts or "Variable" in d.get("description", ""):
            return "variable"
        return "star"
    return "other"


# ----------------------------------------------------------------------------- explanations
# Short explanations of the common object types, shown with the SIMBAD definition.
TYPE_NOTES = {
    "PN": "A shell of gas thrown off by a Sun-like star at the end of its life. The hot core left behind "
          "(a future white dwarf) makes the gas glow; the phase lasts only some ten thousand years.",
    "HII": "A cloud of hydrogen ionised by the ultraviolet light of young, hot stars. It glows red in "
           "hydrogen-alpha as electrons recombine with protons.",
    "SNR": "The expanding debris of a star that exploded as a supernova, heating and shocking the gas around it.",
    "DNe": "A cloud of gas and dust dense enough to block the light of the stars behind it.",
    "MoC": "A cold, dense cloud of molecular hydrogen: the places where new stars form.",
    "RNe": "Dust that shines by reflecting the light of nearby stars, usually blue for the same reason the sky is.",
    "OpC": "A group of a few hundred to a few thousand stars born together from one cloud, loosely bound "
           "and gradually dispersing.",
    "GlC": "A dense ball of 10^5–10^6 very old stars orbiting a galaxy; many are over 10 billion years old.",
    "G": "A galaxy: a system of millions to trillions of stars, gas and dark matter, far beyond the Milky Way. "
         "Its light left it long before it reached your camera.",
    "QSO": "A quasar: an extremely luminous galactic nucleus powered by gas falling into a supermassive black "
           "hole, usually billions of light-years away.",
    "AGN": "An active galactic nucleus: a supermassive black hole accreting gas at the centre of a galaxy.",
    "EB*": "Two stars orbiting so that, seen from Earth, they pass in front of each other and the combined "
           "light dims at regular intervals.",
    "SB*": "A binary too close to separate in a telescope, revealed by the periodic Doppler shift of its spectrum.",
    "**": "Two or more stars that appear close together; many are physically bound and orbit each other.",
    "RR*": "An old pulsating giant that brightens and fades every few hours to a day; used as a standard candle.",
    "Ce*": "A pulsating supergiant whose period is tied to its luminosity (Leavitt's law), a rung of the "
           "cosmic distance ladder.",
    "LP*": "A cool giant whose brightness changes slowly over months to years as it pulsates.",
    "Mi*": "A pulsating red giant (like Mira) that can vary by a factor of 100 or more over about a year.",
    "dS*": "A pulsating star, somewhat hotter and bigger than the Sun, varying over hours.",
    "BY*": "A cool star whose brightness changes as starspots rotate in and out of view.",
    "RS*": "A close binary with a very active, spotted star, varying as it rotates.",
    "WD*": "The exposed core of a dead Sun-like star: about the size of Earth but with roughly the Sun's mass.",
    "RG*": "A star that has used up the hydrogen in its core and swollen into a cool, luminous giant.",
    "HB*": "An old giant burning helium in its core.",
    "C*": "A cool giant whose atmosphere has more carbon than oxygen, which gives it a deep red colour.",
    "Em*": "A star whose spectrum shows emission lines, from hot gas around it (a disc, wind or companion).",
    "Be*": "A fast-spinning hot star that sheds a disc of gas, which produces emission lines.",
    "Y*O": "A star still forming, often embedded in its birth cloud and surrounded by a disc.",
    "PM*": "A nearby star moving quickly across the sky (high proper motion).",
    "V*": "A star whose brightness changes.",
    "CV*": "A white dwarf pulling gas from a close companion star; the gas piles up in a disc and flares in outbursts.",
    "No*": "A white dwarf in a close binary whose surface layer of stolen hydrogen ignited in a thermonuclear flash.",
    "SN*": "A star that exploded; for a few weeks it can outshine its entire galaxy.",
    "LXB": "A neutron star or black hole feeding on a low-mass companion, glowing in X-rays.",
    "HXB": "A neutron star or black hole feeding on the wind of a massive companion, glowing in X-rays.",
    "Psr": "A rapidly spinning neutron star whose beams of radio waves sweep past Earth like a lighthouse.",
    "BH": "A black hole candidate.",
    "s*b": "A blue supergiant: a massive, very luminous and short-lived hot star.",
    "s*r": "A red supergiant: a massive star near the end of its life, large enough to engulf the orbit of Mars.",
    "Ae*": "A young star of a few solar masses, still surrounded by the disc it formed from.",
    "TT*": "A very young Sun-like star, still contracting, that varies irregularly.",
    "Sy1": "A Seyfert galaxy: a spiral galaxy with a bright, active nucleus.",
    "Sy2": "A Seyfert galaxy: a spiral galaxy with an active nucleus hidden behind dust.",
    "BLL": "A BL Lac object: an active galaxy whose jet points almost straight at Earth.",
    "GiC": "The central, dominant galaxy of a galaxy cluster.",
    "ClG": "A cluster of hundreds to thousands of galaxies held together by gravity.",
    "GrG": "A small group of galaxies bound by gravity.",
    "IG": "Galaxies that are interacting: their mutual gravity distorts their shapes.",
    "Cl*": "A group of stars born together.",
    "Rad": "A source detected at radio wavelengths; often a distant galaxy or quasar invisible in this image.",
    "X": "A source of X-rays: hot gas, an accreting compact object, or an active galaxy.",
    "IR": "A source detected in infrared surveys, often dusty or cool.",
    "FIR": "A far-infrared source: usually cold dust, often where stars are forming.",
}


def note_for(otype: str | None, defs: dict) -> str | None:
    if not otype:
        return None
    if otype in TYPE_NOTES:
        return TYPE_NOTES[otype]
    for p in reversed([q.strip() for q in defs.get(otype, {}).get("path", "").split(">")]):
        if p in TYPE_NOTES:
            return TYPE_NOTES[p]
    return None


# ----------------------------------------------------------------------------- session level
def _explore_dir(sess) -> str:
    d = sess._p("explore")
    os.makedirs(d, exist_ok=True)
    return d


def _stack_created(sess) -> str:
    return str(sess.meta.get("created", ""))


def load_solution(sess) -> dict | None:
    """The cached solution of the current stack (None if missing or stale)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    p = os.path.join(sess._p("explore"), "solution.json")
    if not os.path.exists(p):
        return None
    sol = json.load(open(p))
    if sol.get("stack_created") != _stack_created(sess):
        return None
    sol["wcs"] = WCS(fits.Header.fromstring(sol["wcs_header"], sep="\n"), relax=True)
    return sol


def solve_session(sess, gmax: float = 16.0, progress=None) -> dict:
    """Plate-solve the session's stack and cache the solution with the Gaia and SIMBAD
    catalogues of the field."""
    from .pipeline import _load_fits
    say = progress or (lambda i, n, m: None)
    if not os.path.exists(sess._p("stack.fits")):
        raise RuntimeError("stack the dataset first")
    info0 = sess.infos[0]
    if info0.ra is None or info0.dec is None:
        raise RuntimeError("the FITS headers have no RA/DEC to start from")
    d = _explore_dir(sess)
    st = _load_fits(sess._p("stack.fits"))
    cov = _load_fits(sess._p("coverage.fits")) if os.path.exists(sess._p("coverage.fits")) else None
    h, w = st.shape[:2]
    say(0, 5, "Detecting stars")
    det = detect_stars(st, cov)
    scale = pixel_scale_guess(info0.focallen, info0.pixsize, float(sess.meta.get("scale", 1.0)))
    radius = 0.5 * math.hypot(h, w) * scale / 3600 * 1.15 + 0.1
    epoch = decimal_year(info0.date_obs) if info0.date_obs else GAIA_EPOCH
    say(1, 5, f"Querying Gaia DR3 (G < {gmax:g}, {radius:.2f} deg)")
    gaia = gaia_cone(info0.ra, info0.dec, radius, gmax, epoch)
    say(2, 5, f"Matching {len(det['x'])} stars to {len(gaia['g'])} Gaia sources")
    sol = solve(det, gaia, info0.ra, info0.dec, scale, (h, w))
    wcs = sol["wcs"]
    wcs.pixel_shape = (w, h)
    lim = completeness_limit(det, gaia, wcs, (h, w))
    # the field actually covered, for the SIMBAD query and for trimming the Gaia list
    cra, cdec = (float(v) for v in wcs.all_pix2world([[w / 2 - 0.5, h / 2 - 0.5]], 0)[0])
    corners = wcs.all_pix2world([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], 0)
    from astropy.coordinates import SkyCoord
    c0 = SkyCoord(cra, cdec, unit="deg")
    rad_field = float(max(c0.separation(SkyCoord(corners[:, 0], corners[:, 1], unit="deg")).deg)) + 0.02
    say(3, 5, "Querying SIMBAD")
    simbad = simbad_cone(cra, cdec, rad_field)
    otp = os.path.join(os.path.dirname(sess.dir), "simbad_otypedef.json")
    if not os.path.exists(otp):
        json.dump(simbad_otypes(), open(otp, "w"))
    px = np.stack(wcs.all_world2pix(gaia["ra"], gaia["dec"], 0), 1)
    keep = (px[:, 0] > -20) & (px[:, 0] < w + 20) & (px[:, 1] > -20) & (px[:, 1] < h + 20)
    np.savez_compressed(os.path.join(d, "gaia.npz"), **{k: v[keep] for k, v in gaia.items()})
    json.dump(simbad, open(os.path.join(d, "simbad.json"), "w"))
    say(4, 5, "Saving")
    out = {k: v for k, v in sol.items() if k not in ("wcs", "matched_g")}
    out.update({"wcs_header": wcs.to_header(relax=True).tostring(sep="\n"), "stack_created": _stack_created(sess),
                "shape": [h, w], "center": [cra, cdec], "depth_g": lim, "gmax": gmax, "epoch": epoch,
                "n_detected": int(len(det["x"])), "fwhm_px": det["fwhm"],
                "solved": datetime.now().isoformat(timespec="seconds")})
    json.dump(out, open(os.path.join(d, "solution.json"), "w"), indent=1)
    return load_solution(sess)


def _nice_step(span_deg: float, steps_deg: list[float], n: int = 6) -> float:
    for s in steps_deg:
        if span_deg / s <= n:
            return s
    return steps_deg[-1]


def _fmt_ra(ra: float) -> str:
    h = (ra % 360) / 15
    hh, mm = int(h), (h - int(h)) * 60
    return f"{hh:02d}h{int(mm):02d}m{(mm - int(mm)) * 60:04.1f}s"


def _fmt_dec(dec: float) -> str:
    s = "-" if dec < 0 else "+"
    a = abs(dec)
    return f"{s}{int(a):02d}°{int((a % 1) * 60):02d}′{((a * 60) % 1) * 60:04.1f}″"


def annotate(sess, lin_info: dict, lin_shape) -> dict:
    """Everything the Explore view draws, in pixel coordinates of the processed (linear,
    cropped, full-resolution) image: stars, catalogued objects, a coordinate grid and a
    coarse RA/Dec lookup grid for the cursor."""
    from astropy.coordinates import SkyCoord, get_constellation
    sol = load_solution(sess)
    if sol is None:
        raise RuntimeError("not solved")
    wcs = sol["wcs"]
    d = sess._p("explore")
    gaia = dict(np.load(os.path.join(d, "gaia.npz")))
    simbad = json.load(open(os.path.join(d, "simbad.json")))
    otp = os.path.join(os.path.dirname(sess.dir), "simbad_otypedef.json")
    defs = json.load(open(otp)) if os.path.exists(otp) else {}
    up = float(lin_info.get("upscaled", 1.0))
    y0, y1, x0, x1 = lin_info.get("crop") or [0, lin_shape[0], 0, lin_shape[1]]
    W, H = x1 - x0, y1 - y0

    def to_disp(px, py):
        return (px + 0.5) * up - 0.5 - x0, (py + 0.5) * up - 0.5 - y0

    def to_stack(dx, dy):
        return (dx + x0 + 0.5) / up - 0.5, (dy + y0 + 0.5) / up - 0.5

    def inside(x, y, m=0):
        return (x >= -m) & (x < W + m) & (y >= -m) & (y < H + m)

    scale = sol["scale_arcsec"] / up                     # arcsec per display pixel
    # --- Gaia stars
    gx, gy = to_disp(*wcs.all_world2pix(gaia["ra"], gaia["dec"], 0))
    gi = np.flatnonzero(inside(gx, gy))
    plx, eplx = gaia["plx"][gi], gaia["plx_err"][gi]
    good_plx = np.isfinite(plx) & (plx > 0) & (plx / np.maximum(eplx, 1e-9) >= 5)
    dist_pc = np.where(good_plx, 1000.0 / np.where(plx > 0, plx, 1), gaia["dist_gspphot"][gi])
    dist_src = np.where(good_plx, 1, np.where(np.isfinite(gaia["dist_gspphot"][gi]), 2, 0))
    # absolute magnitude, corrected for the GSP-Phot extinction A_G where Gaia estimated it
    n_g = len(gaia["g"])
    ag = gaia.get("ag", np.full(n_g, np.nan))[gi]
    ebr = gaia.get("ebr", np.full(n_g, np.nan))[gi]
    abs_g = gaia["g"][gi] - np.nan_to_num(ag) - 5 * np.log10(np.where(np.isfinite(dist_pc) & (dist_pc > 0), dist_pc,
                                                                        np.nan)) + 5
    # --- SIMBAD objects, cross-matched to Gaia stars (2 arcsec)
    sb_ra = np.array([o["ra"] for o in simbad]) if simbad else np.zeros(0)
    sb_dec = np.array([o["dec"] for o in simbad]) if simbad else np.zeros(0)
    sx, sy = to_disp(*wcs.all_world2pix(sb_ra, sb_dec, 0)) if simbad else (np.zeros(0), np.zeros(0))
    star_simbad = np.full(len(gi), -1)
    if len(simbad) and len(gi):
        cg = SkyCoord(gaia["ra"][gi], gaia["dec"][gi], unit="deg")
        cs = SkyCoord(sb_ra, sb_dec, unit="deg")
        j, sep2d, _ = cs.match_to_catalog_sky(cg)
    objects = []
    for k, o in enumerate(simbad):
        if not inside(sx[k], sy[k], 50):
            continue
        cat = category(o["otype"], defs)
        names = common_names(o.get("ids"))
        gaia_idx = None
        if cat in ("star", "variable", "multiple") and len(gi) and sep2d[k].arcsec < 2.0:
            gaia_idx = int(j[k])
            if star_simbad[gaia_idx] < 0:
                star_simbad[gaia_idx] = len(objects)
        entry = {"x": float(sx[k]), "y": float(sy[k]), "id": o["main_id"], "otype": o["otype"],
                 "type": defs.get(o["otype"] or "", {}).get("description") or o["otype"],
                 "path": defs.get(o["otype"] or "", {}).get("path"), "category": cat,
                 "note": note_for(o["otype"], defs), "names": names["names"], "messier": names["messier"],
                 "ngc": names["ngc"], "designation": names["bright"], "vmag": o.get("vmag"),
                 "sp_type": o.get("sp_type"), "morph_type": o.get("morph_type"), "nbref": o.get("nbref"),
                 "gaia": gaia_idx, "ra": o["ra"], "dec": o["dec"]}
        # distance: parallax (stars), redshift (galaxies, Planck 2018 cosmology)
        if o.get("plx_value") and o["plx_value"] > 0 and o.get("plx_err") and o["plx_value"] / o["plx_err"] >= 5:
            entry["dist_ly"] = 1000.0 / o["plx_value"] * LY_PER_PC
            entry["dist_method"] = "parallax"
        z = o.get("rvz_redshift")
        if cat in ("galaxy", "other") and z is not None and z > 0.003:
            from astropy.cosmology import Planck18
            entry["z"] = z
            entry["dist_ly"] = float(Planck18.comoving_distance(z).to("lyr").value)
            entry["lookback_yr"] = float(Planck18.lookback_time(z).to("yr").value)
            entry["dist_method"] = "redshift (Planck 2018 cosmology, comoving)"
        if o.get("galdim_majaxis"):
            maj = o["galdim_majaxis"] * 60 / scale
            mnr = (o.get("galdim_minaxis") or o["galdim_majaxis"]) * 60 / scale
            pa = o.get("galdim_angle")
            # position angle (east of north) -> direction in display pixels
            e = 1.0 / 3600
            p0 = np.array(to_disp(*wcs.all_world2pix([[o["ra"], o["dec"]]], 0)[0]))
            pn = np.array(to_disp(*wcs.all_world2pix([[o["ra"], o["dec"] + e]], 0)[0])) - p0
            pe = np.array(to_disp(*wcs.all_world2pix([[o["ra"] + e / math.cos(math.radians(o["dec"])), o["dec"]]],
                                                      0)[0])) - p0
            pa_r = math.radians(pa if pa is not None else 0.0)
            v = pn / np.linalg.norm(pn) * math.cos(pa_r) + pe / np.linalg.norm(pe) * math.sin(pa_r)
            entry["ellipse"] = [maj / 2, mnr / 2, math.atan2(v[1], v[0])]
            entry["size_arcmin"] = [o["galdim_majaxis"], o.get("galdim_minaxis")]
            if entry.get("dist_ly"):
                entry["size_ly"] = entry["dist_ly"] * math.radians(o["galdim_majaxis"] / 60)
        objects.append(entry)
    def rnd(v, n):
        return np.round(np.asarray(v, float), n)
    stars = {"x": rnd(gx[gi], 1), "y": rnd(gy[gi], 1), "g": rnd(gaia["g"][gi], 2), "bp_rp": rnd(gaia["bp_rp"][gi], 2),
             "teff": rnd(gaia["teff"][gi], -1), "dist_ly": rnd(dist_pc * LY_PER_PC, 0), "dist_src": dist_src,
             "abs_g": rnd(abs_g, 2), "ag": rnd(ag, 2), "ebr": rnd(ebr, 2),
             "pmra": rnd(gaia["pmra"][gi], 2), "pmdec": rnd(gaia["pmdec"][gi], 2), "rv": rnd(gaia["rv"][gi], 1),
             "variable": gaia["variable"][gi].astype(int), "nss": gaia["nss"][gi],
             "source_id": [str(s) for s in gaia["source_id"][gi]], "simbad": star_simbad,
             "sp_class": [spectral_class_from_teff(t) for t in gaia["teff"][gi]]}
    # --- coordinate grid
    cra, cdec = sol["center"]
    span = max(W, H) * scale / 3600
    dstep = _nice_step(span, [1 / 60, 2 / 60, 5 / 60, 10 / 60, 15 / 60, 0.5, 1, 2, 5, 10])
    rstep = _nice_step(span / max(math.cos(math.radians(cdec)), 0.05),
                       [15 * s / 3600 for s in (30, 60, 120, 300, 600, 900, 1800, 3600, 7200)])
    corners = np.array([to_stack(0, 0), to_stack(W, 0), to_stack(0, H), to_stack(W, H), to_stack(W / 2, 0),
                        to_stack(W / 2, H), to_stack(0, H / 2), to_stack(W, H / 2)])
    cw = wcs.all_pix2world(corners, 0)
    ra_rel = (cw[:, 0] - cra + 180) % 360 - 180
    decs = np.arange(math.floor(cw[:, 1].min() / dstep) * dstep, cw[:, 1].max() + dstep, dstep)
    ras = np.arange(math.floor((cra + ra_rel.min()) / rstep) * rstep, cra + ra_rel.max() + rstep, rstep)
    dec_lo, dec_hi = max(cw[:, 1].min() - dstep, -89.99), min(cw[:, 1].max() + dstep, 89.99)
    ra_lo, ra_hi = cra + ra_rel.min() - rstep, cra + ra_rel.max() + rstep
    lines = []
    for dd in decs:
        ra_s = np.linspace(ra_lo, ra_hi, 80)
        px, py = to_disp(*wcs.all_world2pix(ra_s % 360, np.full_like(ra_s, dd), 0))
        lines.append({"kind": "dec", "label": _fmt_dec(dd)[:-6] + "′", "pts": np.round(np.stack([px, py], 1), 1)})
    for rr in ras:
        dec_s = np.linspace(dec_lo, dec_hi, 80)
        px, py = to_disp(*wcs.all_world2pix(np.full_like(dec_s, rr % 360), dec_s, 0))
        lines.append({"kind": "ra", "label": _fmt_ra(rr)[:-5] if rstep < 15 else _fmt_ra(rr)[:3],
                      "pts": np.round(np.stack([px, py], 1), 1)})
    # --- cursor lookup grid (bilinear interpolation in the browser)
    n = 24
    gxs, gys = np.meshgrid(np.linspace(0, W, n + 1), np.linspace(0, H, n + 1))
    sxs, sys_ = to_stack(gxs, gys)
    gw = wcs.all_pix2world(np.stack([sxs.ravel(), sys_.ravel()], 1), 0)
    look = {"n": n, "W": W, "H": H, "ra": ((gw[:, 0] - cra + 180) % 360 - 180 + cra).reshape(n + 1, n + 1),
            "dec": gw[:, 1].reshape(n + 1, n + 1)}
    # --- north / east arrows at the centre
    c = np.array(to_stack(W / 2, H / 2))
    cw0 = wcs.all_pix2world([c], 0)[0]
    pn = np.array(to_disp(*wcs.all_world2pix([[cw0[0], cw0[1] + 0.01]], 0)[0])) - np.array([W / 2, H / 2])
    pe = np.array(to_disp(*wcs.all_world2pix([[cw0[0] + 0.01 / math.cos(math.radians(cw0[1])), cw0[1]]], 0)[0])) - \
        np.array([W / 2, H / 2])
    centre = SkyCoord(cw0[0], cw0[1], unit="deg")
    field = {"center_ra": float(cw0[0]), "center_dec": float(cw0[1]), "center_text": f"{_fmt_ra(cw0[0])} {_fmt_dec(cw0[1])}",
             "constellation": get_constellation(centre), "width_deg": W * scale / 3600, "height_deg": H * scale / 3600,
             "scale_arcsec": scale, "north": (pn / np.linalg.norm(pn)).tolist(), "east": (pe / np.linalg.norm(pe)).tolist(),
             "north_angle_deg": float(math.degrees(math.atan2(pn[0], -pn[1]))),
             "rms_arcsec": sol["rms_arcsec"], "n_matched": sol["n_matched"], "depth_g": sol.get("depth_g"),
             "gmax": sol.get("gmax"), "galactic": [float(centre.galactic.l.deg), float(centre.galactic.b.deg)],
             "n_stars": int(len(gi)), "solar_abs_g": SOLAR_MG, "solved": sol.get("solved"),
             # on the sky, seen from Earth with north up, east is to the left
             "mirrored": bool(pn[0] * pe[1] - pn[1] * pe[0] > 0)}
    counts = {}
    for e in objects:
        if inside(e["x"], e["y"]):
            counts[e["category"]] = counts.get(e["category"], 0) + 1
    field["counts"] = counts
    return {"W": W, "H": H, "field": field, "stars": stars, "objects": objects, "grid": lines, "lookup": look,
            "categories": CATEGORIES}
