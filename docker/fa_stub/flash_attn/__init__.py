"""Import-only stub of flash_attn for Blackwell (sm_120), where no FA2 build
exists. verl imports these symbols at module load; with VERL_ATTN_IMPL=sdpa and
use_remove_padding=False nothing here is ever CALLED. Every callable raises so
a misconfiguration fails loudly instead of silently computing garbage."""
__version__ = "2.7.4.post1"

def _unavailable(*a, **k):
    raise RuntimeError("flash_attn stub: FA kernels are unavailable on sm_120; "
                       "attention must run through sdpa (VERL_ATTN_IMPL=sdpa, use_remove_padding=False)")

flash_attn_func = _unavailable
flash_attn_varlen_func = _unavailable
flash_attn_qkvpacked_func = _unavailable
flash_attn_varlen_qkvpacked_func = _unavailable
flash_attn_with_kvcache = _unavailable

class _StubObj:
    """Any attribute access yields another stub; calling anything raises."""
    def __getattr__(self, name): return _StubObj()
    def __call__(self, *a, **k): _unavailable()

def __getattr__(name):          # module-level: satisfy ANY symbol lookup
    return _StubObj()
