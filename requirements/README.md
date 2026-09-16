# WebShop dependency profiles

Use `scripts/setup_webshop.sh` from the repository root. It installs
`webshop-lock.txt`, adds the checksum-verified FlashAttention and spaCy model wheels
from `webshop-assets.json`, and validates the runtime. `webshop.txt` lists the direct
requirements used to resolve the lock for Linux x86_64 / Python 3.10.

## Migrated environment: observed conflicts

The existing training venv was preserved during provisioning on 2026-09-16. Its
`pip check` reports the following seven dependency declaration conflicts:

| Package | Declared requirement | Existing installation | Fresh setup |
| --- | --- | --- | --- |
| `verl 0.3.1.dev0` | `qwen-vl-utils` | Missing | Installs `qwen-vl-utils 0.0.10` |
| `verl 0.3.1.dev0` | `tensordict >=0.8, <=0.10, !=0.9` | `0.7.2` | Keeps tested `0.7.2`; imports pinned source through `.pth` |
| `cupy-cuda12x 14.2.0` | NumPy `>=2.0, <2.6` | `1.26.4` | Uses CuPy `13.4.1` with NumPy `1.26.4` |
| `opencv-python-headless 5.0.0.93` | NumPy `>=2` on Python 3.9+ | `1.26.4` | Uses OpenCV `4.11.0.86` |
| `gradio-client 0.15.1` | websockets `>=10, <12` | `16.1.1` | Omits the unused Gradio UI stack |
| `spacy 3.7.2` | Typer `>=0.3, <0.10` | `0.27.2` | Uses spaCy `3.7.5` |
| `weasel 0.3.4` | Typer `>=0.3, <0.10` | `0.27.2` | Uses weasel `0.4.1` |

The old environment completed M5 and passes the WebShop CPU checks, but these checks
do not establish compatibility of every installed package. Setup preserves that
venv to keep active and queued experiments on their original versions. Its report
records the conflicts instead of silently upgrading packages.

The pinned upstream `setup.py` declares a tensordict range that conflicts with the
working torch 2.6 / tensordict 0.7.2 stack. Fresh setup installs explicit dependencies
and exposes the pinned, patched source through a `.pth` file; it does not install
upstream `verl` distribution metadata. This is a deliberate source-install policy,
not a claim that tensordict 0.7.2 satisfies upstream's declared range. Validation
imports the trainer and actor and exercises `DataProto` with the installed tensordict.

The fresh dependency profile passed `pip check` and CPU runtime tests, including
index/catalogue matching, 6,910 synthetic goals, and perfect/partial purchases with
binary rewards. It has not been validated by a new full training run. GPU attention
execution is checked separately by the experiment launcher.
