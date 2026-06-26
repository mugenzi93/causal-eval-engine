from .psm          import run_psm
from .did          import run_did
from .drdid        import run_drdid
from .iv           import run_iv
from .rdd          import run_rdd
from .fixed_effects import run_fixed_effects
from .aipw         import run_aipw

REGISTRY = {
    "psm":           run_psm,
    "did":           run_did,
    "drdid":         run_drdid,
    "iv":            run_iv,
    "rdd":           run_rdd,
    "fixed_effects": run_fixed_effects,
    "aipw":          run_aipw,
}

__all__ = [
    "run_psm", "run_did", "run_drdid", "run_iv",
    "run_rdd", "run_fixed_effects", "run_aipw",
    "REGISTRY",
]
