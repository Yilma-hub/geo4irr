"""
Crop library for Delaware irrigated cropping systems.

Stage lengths are FAO-56 Table 11 values shifted to Delmarva planting windows
and University of Delaware Cooperative Extension crop growth guides. Kcb values
are FAO-56 Table 17 for a sub-humid climate with moderate wind, which is a
reasonable starting prior for Sussex and Kent counties; the Bayesian module
replaces them with canopy-derived values as soon as imagery or in-canopy sensor
data arrive for the field.

MAD values follow the irrigation scheduling convention of allowing 50 percent
depletion outside the sensitive window and tightening during pollination, pod
fill or tuber bulking. On the droughty Evesboro and Rosedale sands that carry a
lot of Delaware's irrigated acreage, the scheduler additionally caps the applied
depth so a single set does not exceed what the profile can hold.
"""

from __future__ import annotations

from .fao56 import CropStage

# name -> CropStage
CROPS: dict[str, CropStage] = {
    "corn_grain": CropStage(
        name="Corn, grain",
        kcb_ini=0.15, kcb_mid=1.15, kcb_end=0.50,
        l_ini=25, l_dev=40, l_mid=45, l_late=30,
        root_min_in=6.0, root_max_in=36.0, height_max_ft=8.0,
        mad_default=0.50, mad_critical=0.40,
        critical_start_dap=58, critical_end_dap=95,   # V12 through R2
        yield_response_ky=1.25,
    ),
    "corn_silage": CropStage(
        name="Corn, silage",
        kcb_ini=0.15, kcb_mid=1.15, kcb_end=1.05,
        l_ini=25, l_dev=40, l_mid=40, l_late=15,
        root_min_in=6.0, root_max_in=34.0, height_max_ft=8.0,
        mad_default=0.50, mad_critical=0.40,
        critical_start_dap=58, critical_end_dap=92,
        yield_response_ky=1.25,
    ),
    "soybean_full": CropStage(
        name="Soybean, full season",
        kcb_ini=0.15, kcb_mid=1.10, kcb_end=0.30,
        l_ini=20, l_dev=35, l_mid=55, l_late=25,
        root_min_in=6.0, root_max_in=34.0, height_max_ft=3.3,
        mad_default=0.50, mad_critical=0.40,
        critical_start_dap=65, critical_end_dap=110,  # R3 through R6
        yield_response_ky=0.85,
    ),
    "soybean_double": CropStage(
        name="Soybean, double crop",
        kcb_ini=0.15, kcb_mid=1.10, kcb_end=0.30,
        l_ini=15, l_dev=30, l_mid=45, l_late=20,
        root_min_in=6.0, root_max_in=30.0, height_max_ft=2.8,
        mad_default=0.50, mad_critical=0.40,
        critical_start_dap=52, critical_end_dap=90,
        yield_response_ky=0.85,
    ),
    "sorghum_grain": CropStage(
        name="Grain sorghum",
        kcb_ini=0.15, kcb_mid=1.00, kcb_end=0.35,
        l_ini=20, l_dev=35, l_mid=40, l_late=30,
        root_min_in=6.0, root_max_in=38.0, height_max_ft=5.0,
        mad_default=0.55, mad_critical=0.45,
        critical_start_dap=55, critical_end_dap=85,
        yield_response_ky=0.90,
    ),
    "wheat": CropStage(
        name="Winter wheat",
        kcb_ini=0.15, kcb_mid=1.10, kcb_end=0.25,
        l_ini=30, l_dev=140, l_mid=40, l_late=30,
        root_min_in=6.0, root_max_in=40.0, height_max_ft=3.3,
        mad_default=0.55, mad_critical=0.45,
        critical_start_dap=180, critical_end_dap=215,
        yield_response_ky=1.05,
    ),
    "barley": CropStage(
        name="Winter barley",
        kcb_ini=0.15, kcb_mid=1.10, kcb_end=0.25,
        l_ini=30, l_dev=135, l_mid=35, l_late=25,
        root_min_in=6.0, root_max_in=38.0, height_max_ft=3.3,
        mad_default=0.55, mad_critical=0.45,
        critical_start_dap=170, critical_end_dap=205,
        yield_response_ky=1.00,
    ),
    "lima_bean": CropStage(
        name="Lima bean, processing",
        kcb_ini=0.15, kcb_mid=1.00, kcb_end=0.80,
        l_ini=20, l_dev=25, l_mid=30, l_late=15,
        root_min_in=5.0, root_max_in=24.0, height_max_ft=1.6,
        mad_default=0.40, mad_critical=0.35,
        critical_start_dap=45, critical_end_dap=80,
        yield_response_ky=1.15,
    ),
    "sweet_corn": CropStage(
        name="Sweet corn, processing",
        kcb_ini=0.15, kcb_mid=1.10, kcb_end=1.00,
        l_ini=20, l_dev=25, l_mid=30, l_late=10,
        root_min_in=6.0, root_max_in=28.0, height_max_ft=6.5,
        mad_default=0.40, mad_critical=0.35,
        critical_start_dap=42, critical_end_dap=72,
        yield_response_ky=1.25,
    ),
    "potato": CropStage(
        name="Potato",
        kcb_ini=0.15, kcb_mid=1.10, kcb_end=0.65,
        l_ini=25, l_dev=30, l_mid=40, l_late=30,
        root_min_in=5.0, root_max_in=22.0, height_max_ft=2.0,
        mad_default=0.35, mad_critical=0.30,
        critical_start_dap=50, critical_end_dap=100,
        yield_response_ky=1.10,
    ),
    "watermelon": CropStage(
        name="Watermelon",
        kcb_ini=0.15, kcb_mid=0.95, kcb_end=0.70,
        l_ini=20, l_dev=30, l_mid=30, l_late=20,
        root_min_in=6.0, root_max_in=30.0, height_max_ft=1.3,
        mad_default=0.40, mad_critical=0.35,
        critical_start_dap=48, critical_end_dap=82,
        yield_response_ky=1.10,
    ),
    "cover_crop": CropStage(
        name="Cover crop, small grain",
        kcb_ini=0.15, kcb_mid=0.90, kcb_end=0.30,
        l_ini=25, l_dev=60, l_mid=60, l_late=30,
        root_min_in=4.0, root_max_in=26.0, height_max_ft=2.0,
        mad_default=0.65, mad_critical=0.65,
        critical_start_dap=9999, critical_end_dap=9999,
        yield_response_ky=0.50,
    ),
}

# Typical Delaware planting windows, used to warn when an entered date is well
# outside what the growth-stage model was calibrated against.
PLANTING_WINDOWS: dict[str, tuple[str, str]] = {
    "corn_grain": ("04-10", "05-20"),
    "corn_silage": ("04-10", "05-25"),
    "soybean_full": ("05-01", "06-10"),
    "soybean_double": ("06-15", "07-10"),
    "sorghum_grain": ("05-10", "06-15"),
    "wheat": ("10-05", "11-05"),
    "barley": ("09-25", "10-25"),
    "lima_bean": ("05-20", "07-05"),
    "sweet_corn": ("04-20", "07-01"),
    "potato": ("03-20", "04-25"),
    "watermelon": ("05-01", "06-05"),
    "cover_crop": ("09-15", "11-15"),
}

# Statewide yield goals used when a grower has not entered field history.
# Irrigated yields from Delaware Agricultural Statistics; treat as a prior only.
DEFAULT_YIELD_GOALS: dict[str, tuple[float, str]] = {
    "corn_grain": (200.0, "bu/ac"),
    "corn_silage": (24.0, "ton/ac"),
    "soybean_full": (60.0, "bu/ac"),
    "soybean_double": (45.0, "bu/ac"),
    "sorghum_grain": (110.0, "bu/ac"),
    "wheat": (85.0, "bu/ac"),
    "barley": (100.0, "bu/ac"),
    "lima_bean": (3600.0, "lb/ac"),
    "sweet_corn": (8.0, "ton/ac"),
    "potato": (400.0, "cwt/ac"),
    "watermelon": (40000.0, "lb/ac"),
    "cover_crop": (0.0, "n/a"),
}


def get_crop(key: str) -> CropStage:
    try:
        return CROPS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown crop '{key}'. Available: {', '.join(sorted(CROPS))}"
        ) from exc


def crop_catalog() -> list[dict]:
    """Serializable list for the field setup screen."""
    out = []
    for key, c in CROPS.items():
        goal, unit = DEFAULT_YIELD_GOALS[key]
        out.append({
            "key": key,
            "name": c.name,
            "season_days": c.season_days,
            "max_root_in": c.root_max_in,
            "mad_default": c.mad_default,
            "mad_critical": c.mad_critical,
            "planting_window": PLANTING_WINDOWS.get(key),
            "default_yield_goal": goal,
            "yield_unit": unit,
        })
    return out
