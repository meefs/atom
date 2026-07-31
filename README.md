![tests](https://github.com/mprahboamey/atom/actions/workflows/tests.yml/badge.svg)
# ATOM
**Angular-Multiplexed Transformer Optical Model**

What if a neural network's weights didn't exist as floating points in memory but as phase structure in a holographic crystal?

This repo is a numerically verified simulation of that idea. The core result: when query and key vectors are encoded as optical waves with binary phase (0 or π), their interference scores are **algebraically identical** to scaled dot-product attention — term for term, verified to float precision on arbitrary tensors. A continuous-phase generalization (angular multiplexing) is also implemented and reduces to that exact case when relative positions are zero.

What is optical is the **score matrix**. Softmax, value aggregation, residuals, and norms stay digital. The practical block is hybrid: optical scores + digital remainder. See [`atom/hybrid.py`](atom/hybrid.py).

The math for the scores is proved. The hardware does not exist yet. That is the point of open sourcing this.

---

## What is proved vs what is projected

| Claim | Status |
|-------|--------|
| Optical interference = scaled dot-product **scores** (binary phase, exact) | Proved and verified to float precision |
| Continuous-phase path reduces to the above when positions match | Verified |
| ASM energy conservation, phase-mask intensity, reversible propagation | Verified |
| Gradients flow through the optical score path | Verified |
| Phase quantisation, phase noise, angular jitter, crosstalk models | Implemented (`atom/noise.py`) |
| M#-limited capacity for Fe:LiNbO₃ | Implemented (`atom/capacity.py`); geometric 90T is an upper bound only |
| Hybrid optical-QK + digital remainder module | Implemented (`atom/hybrid.py`) |
| Weight conversion from a local checkpoint | Implemented (`atom/convert.py`); you supply the model on disk |
| Full-model optical inference (everything optical) | **Not claimed** |
| Task accuracy on a real benchmark under noise | **Not claimed yet** — needs a real checkpoint + eval run |
| Measured hardware latency / energy | **Not claimed** |

The score equivalence does **not** require a trained model. It holds for any Q and K tensors. Task-level numbers require loading real weights and running the hybrid path under noise.

---

## Capacity (honest version)

Geometric ceiling under the simulation grid (1 cm³):

```
1,000 layers × 900 angular channels × 10⁸ pixels/layer = 9×10¹³  (90T)
```

That ignores dynamic range. Photorefractive media are limited by M#. For conservative Fe:LiNbO₃ parameters (M# ≈ 2, η_min ≈ 10⁻⁴) usable angular channels drop and the M#-aware estimate is lower (see `atom/capacity.py` and `docs/benchmarks.md`). Experimental net densities in the literature are lower still once scatter, coding, and readout are included.

Multi-crystal / rack-scale composition (many modules, each holding part of a large model) is an open systems-engineering problem, not a materials claim. See CONTRIBUTING.md.

---

## Phase precision

A sweep over phase bit-width (`examples/05_phase_quantization_sweep.py`) shows that around **8 bits** attention KL vs continuous phase is already ~1e-5 and top-1 agreement is ~99.5% on synthetic Q/K. That is the default write precision used by the weight converter unless you override it.

---

## Install and run

```bash
git clone https://github.com/mprahboamey/atom.git
cd atom
pip install -e .
```

```bash
python scripts/run_all.py
```

```bash
python examples/01_propagate_beam.py
python examples/02_train_phase_mask.py
python examples/03_optical_attention.py
python examples/04_validate_model.py
python examples/05_phase_quantization_sweep.py
python examples/06_capacity_fe_linbo3.py
python examples/07_hybrid_attention.py
python examples/08_convert_weights.py   # synthetic demo; point at your local model for real weights
```

---

## Project layout

```
atom/
├── propagation.py    Angular Spectrum Method
├── diffractive.py    Phase masks
├── attention.py      Optical scores (binary + continuous phase)
├── noise.py          Phase quant, phase noise, jitter, crosstalk
├── capacity.py       M#-aware capacity (Fe:LiNbO₃ defaults)
├── hybrid.py         Optical scores + digital remainder
└── convert.py        Local checkpoint → optical weight tensors

examples/             One concept per script
tests/
docs/model.md         Math derivation of the score proof
docs/benchmarks.md    Capacity, latency, energy assumptions
```

---

## Loading your own model

Do not commit multi-GB weights into this repo. Download a model on your machine (e.g. Midnight Bundle or any HF/safetensors folder), then:

```bash
python examples/08_convert_weights.py --model /path/to/model --out ./optical_weights
```

The converter maps attention projections into phase-encoded tensors (default 8-bit phase). You can then run them through `HybridOpticalAttention` under `NoiseConfig`.

---

## Where this needs to go next

See [CONTRIBUTING.md](CONTRIBUTING.md). Highest-leverage remaining work: end-to-end noisy eval on a real task with converted weights, richer noise (detector, measured Bragg curve), and multi-module system design.

---

## References

- Goodman, J. W. (2005). *Introduction to Fourier Optics*
- Psaltis, D., Brady, D., & Wagner, K. (1988). Adaptive optical networks using photorefractive crystals. *Applied Optics*, 27(9), 1752–1759.
- Psaltis, D., & Mok, F. (1995). Holographic memories. *Scientific American*, 273(5), 70–76.
- Lin, X., et al. (2018). All-optical machine learning using diffractive deep neural networks. *Science*, 361(6406), 1004–1008.
- Miller, D. A. B. (2017). Attojoule optoelectronics. *Journal of Lightwave Technology*, 35(3), 346–396.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*, 30.
