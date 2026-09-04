from . import _unavailable
flash_attn_func = _unavailable
flash_attn_varlen_func = _unavailable
flash_attn_qkvpacked_func = _unavailable
flash_attn_kvpacked_func = _unavailable
flash_attn_varlen_qkvpacked_func = _unavailable
flash_attn_varlen_kvpacked_func = _unavailable
flash_attn_with_kvcache = _unavailable
_flash_attn_forward = _unavailable
_flash_attn_varlen_forward = _unavailable
_flash_attn_backward = _unavailable
_flash_attn_varlen_backward = _unavailable

from . import _StubObj
flash_attn_gpu = _StubObj()
flash_attn_cuda = _StubObj()

def __getattr__(name):
    return _StubObj()
