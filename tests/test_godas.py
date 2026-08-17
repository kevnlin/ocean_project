"""GODAS loader checks (mentor doc §4).

The doc's requirement is "checked GODAS loading, unit conversion, and
train-only normalization".  These tests pin the checks, because a silent unit
error or a missing month is exactly the kind of fault that produces
plausible-looking numbers.
"""
import numpy as np
import pytest

from ocean_tokenizer.godas import (check_contiguous_months, to_celsius,
                                   to_g_per_kg, check_alignment)


# --------------------------------------------------------------------------
# Contiguous, unique months
# --------------------------------------------------------------------------
def _months(start_year, n):
    return np.array([np.datetime64(f"{start_year + i // 12:04d}-"
                                   f"{i % 12 + 1:02d}-01") for i in range(n)])


def test_contiguous_months_pass():
    check_contiguous_months(_months(2000, 36))          # must not raise


def test_a_missing_month_raises():
    m = _months(2000, 12)
    gapped = np.delete(m, 5)                             # drop June 2000
    with pytest.raises(ValueError, match="gap|contiguous"):
        check_contiguous_months(gapped)


def test_a_duplicated_month_raises():
    """The duplicate sits in sorted position, so this tests duplication and
    not sortedness — appending it at the end would trip the order check
    first and the duplicate branch would never run."""
    m = _months(2000, 12)
    dup = np.sort(np.concatenate([m, m[3:4]]))
    with pytest.raises(ValueError, match="duplicate"):
        check_contiguous_months(dup)


def test_unsorted_months_raise():
    m = _months(2000, 12)[::-1]
    with pytest.raises(ValueError, match="sorted|order"):
        check_contiguous_months(m)


# --------------------------------------------------------------------------
# Unit conversion — converted only when the source says so
# --------------------------------------------------------------------------
def test_kelvin_becomes_celsius():
    out = to_celsius(np.array([273.15, 293.15]), "K")
    assert np.allclose(out, [0.0, 20.0])


def test_celsius_is_left_alone():
    a = np.array([0.0, 20.0])
    assert np.allclose(to_celsius(a, "degC"), a)


def test_an_unrecognised_temperature_unit_raises():
    """Guessing from magnitude would be how a silent unit bug gets in."""
    with pytest.raises(ValueError, match="unit"):
        to_celsius(np.array([1.0]), "furlongs")


def test_salinity_kg_per_kg_becomes_g_per_kg():
    out = to_g_per_kg(np.array([0.0359]), "kg/kg")
    assert np.allclose(out, [35.9])


def test_practical_salinity_is_left_alone():
    a = np.array([35.9])
    assert np.allclose(to_g_per_kg(a, "g/kg"), a)


def test_an_unrecognised_salinity_unit_raises():
    with pytest.raises(ValueError, match="unit"):
        to_g_per_kg(np.array([1.0]), "psu-ish")


# --------------------------------------------------------------------------
# Cross-variable alignment
# --------------------------------------------------------------------------
def _axes(nt=12, nlat=75, nlon=51):
    return dict(time=_months(2000, nt),
                lat=np.linspace(25, 50, nlat),
                lon=np.linspace(280, 331, nlon))


def test_aligned_variables_pass():
    check_alignment({"pottmp": _axes(), "salt": _axes(), "sshg": _axes()})


def test_a_different_time_axis_raises():
    bad = _axes(); bad["time"] = _months(2001, 12)
    with pytest.raises(ValueError, match="time"):
        check_alignment({"pottmp": _axes(), "salt": bad})


def test_a_different_grid_raises():
    bad = _axes(nlat=74)
    with pytest.raises(ValueError, match="lat|grid|shape"):
        check_alignment({"pottmp": _axes(), "salt": bad})


def test_a_shifted_grid_raises_even_at_the_same_shape():
    """Same number of cells, different coordinates — must not slip through."""
    bad = _axes(); bad["lon"] = bad["lon"] + 1.0
    with pytest.raises(ValueError, match="lon|grid"):
        check_alignment({"pottmp": _axes(), "salt": bad})


# --------------------------------------------------------------------------
# Train-only normalisation (doc §4: "fit on training months only")
# --------------------------------------------------------------------------
def _synthetic(nt=48, nz=4, ny=6, nx=5, seed=0):
    """T/S/SSH with a seasonal cycle plus noise, and a fixed land mask."""
    rng = np.random.default_rng(seed)
    months = _months(2000, nt)
    cal = np.array([int(str(m)[5:7]) for m in months])
    season = np.sin(2 * np.pi * cal / 12.0)[:, None, None, None]
    T = 10 + 5 * season + rng.normal(0, 0.5, (nt, nz, ny, nx))
    S = 35 + 0.2 * season + rng.normal(0, 0.05, (nt, nz, ny, nx))
    H = 0.1 * season[:, 0] + rng.normal(0, 0.02, (nt, ny, nx))
    T[:, :, 0, 0] = np.nan; S[:, :, 0, 0] = np.nan; H[:, 0, 0] = np.nan
    return dict(months=months, TEMP=T, SALT=S, SSH=H)


def test_norm_is_fitted_on_training_months_only():
    """Later months must not move the statistics — that would be leakage."""
    from ocean_tokenizer.godas import GodasNorm
    f = _synthetic(nt=48)
    train = slice(0, 24)
    a = GodasNorm(f, train)
    shifted = {**f, "TEMP": f["TEMP"].copy()}
    shifted["TEMP"][24:] += 100.0          # corrupt only the held-out months
    b = GodasNorm(shifted, train)
    assert np.allclose(a.clim["TEMP"], b.clim["TEMP"], equal_nan=True)
    assert np.allclose(a.scale["TEMP"], b.scale["TEMP"])


def test_zscore_of_training_data_is_about_unit_variance():
    from ocean_tokenizer.godas import GodasNorm
    f = _synthetic(nt=48)
    n = GodasNorm(f, slice(0, 24))
    z = n.z("TEMP", f["TEMP"][:24], f["months"][:24])
    assert abs(float(np.nanstd(z)) - 1.0) < 0.25


def test_temp_scale_is_per_depth_and_ssh_scale_is_scalar():
    """Doc §4: per-depth T/S anomaly scale, a global SSH anomaly scale."""
    from ocean_tokenizer.godas import GodasNorm
    f = _synthetic(nt=48, nz=4)
    n = GodasNorm(f, slice(0, 24))
    assert n.scale["TEMP"].shape == (4,)
    assert n.scale["SALT"].shape == (4,)
    assert np.ndim(n.scale["SSH"]) == 0


def test_roundtrip_recovers_the_field():
    from ocean_tokenizer.godas import GodasNorm
    f = _synthetic(nt=48)
    n = GodasNorm(f, slice(0, 24))
    x = f["TEMP"][:6]; m = f["months"][:6]
    assert np.allclose(n.unz("TEMP", n.z("TEMP", x, m), m), x, equal_nan=True)


def test_nan_cells_stay_nan_through_normalisation():
    from ocean_tokenizer.godas import GodasNorm
    f = _synthetic(nt=48)
    n = GodasNorm(f, slice(0, 24))
    z = n.z("TEMP", f["TEMP"][:6], f["months"][:6])
    assert np.isnan(z[:, :, 0, 0]).all()


# --------------------------------------------------------------------------
# load_godas — integration against the real downloaded subset
# --------------------------------------------------------------------------
import os

GODAS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "godas_gulfstream")
needs_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(GODAS_DIR, "manifest.json")),
    reason="GODAS subset not downloaded (experiments/13_download_godas.py)")


@needs_data
def test_load_godas_returns_the_documented_shape():
    """Doc §4: 16 levels 5-949 m, stride 2 -> 38 x 26, contiguous months."""
    from ocean_tokenizer.godas import load_godas
    f = load_godas(GODAS_DIR, years=(2000, 2002))
    assert f["TEMP"].shape == (36, 16, 38, 26)
    assert f["SALT"].shape == (36, 16, 38, 26)
    assert f["SSH"].shape == (36, 38, 26)
    assert f["depth"].shape == (16,)
    assert abs(f["depth"][0] - 5) < 1 and abs(f["depth"][-1] - 949) < 1


@needs_data
def test_load_godas_converts_units():
    """Kelvin -> degC and kg/kg -> g/kg, checked by physical plausibility."""
    from ocean_tokenizer.godas import load_godas
    f = load_godas(GODAS_DIR, years=(2000, 2000))
    t, s = f["TEMP"], f["SALT"]
    assert -3 < np.nanmin(t) and np.nanmax(t) < 40, "TEMP not in degC"
    assert 20 < np.nanmin(s) and np.nanmax(s) < 45, "SALT not in g/kg"


@needs_data
def test_load_godas_months_are_contiguous():
    from ocean_tokenizer.godas import load_godas, check_contiguous_months
    f = load_godas(GODAS_DIR, years=(2000, 2003))
    check_contiguous_months(f["months"])
    assert len(f["months"]) == 48


@needs_data
def test_load_godas_preserves_land_as_nan():
    from ocean_tokenizer.godas import load_godas
    f = load_godas(GODAS_DIR, years=(2000, 2000))
    frac = float(np.isnan(f["TEMP"]).mean())
    assert 0.05 < frac < 0.4, f"land fraction {frac:.3f} implausible"
