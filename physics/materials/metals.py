"""
Metal work functions for Schottky contact calculations.
All values in eV.
"""

METAL_WORK_FUNCTIONS = {
    'Ni':  5.10,
    'Au':  5.10,
    'Pt':  5.65,
    'Ti':  4.33,
    'Al':  4.28,
    'Ag':  4.26,
    'Cr':  4.50,
    'W':   4.55,
    'Pd':  5.12,
    'Mo':  4.60,
}


def get_work_function(metal: str) -> float:
    """Return the work function of a metal in eV."""
    key = metal.strip()
    if key not in METAL_WORK_FUNCTIONS:
        available = list(METAL_WORK_FUNCTIONS.keys())
        raise ValueError(f"Unknown metal '{key}'. Available: {available}")
    return METAL_WORK_FUNCTIONS[key]
