# Contributing to ATOM

The math is proved. The simulator works. Everything else is an open problem.

This document breaks down exactly what needs to happen next, by domain. Find your expertise, read your section, fork and build. If you're not sure where you fit, open an issue and say what you do — we'll find a place.

---

## Materials Science

**The core problem:** Photorefractive crystals like lithium niobate store holograms well, but reading erases them. The same light you use to read a hologram also partially overwrites it. Fix this and you've solved one of the oldest problems in holographic storage.

**Open problems:**
- Fixing holograms without destroying angular multiplexing fidelity (thermal fixing, two-photon gating)
- Reducing shrinkage during polymerization in holographic polymer media
- Characterizing M# (dynamic range figure of merit) for candidate materials under dense angular multiplexing
- Modelling SNR degradation as a function of hologram count for lithium niobate and barium titanate

**How to contribute:** The `atom.capacity` module already exposes M# and eta_min as parameters with Fe:LiNbO₃ defaults. Better measured values, or a calibrated SNR-vs-M curve, can be dropped in directly. A full materials noise model (erasure dynamics, scatter) is still open.

---

## Integrated Photonics

**The core problem:** Getting light into and out of the crystal fast enough to be useful. Spatial light modulators are slow. Photodetectors add noise. The optical interaction itself is femtojoule-level; everything around it is not.

**Open problems:**
- Input encoding: SLM vs direct waveguide coupling — speed vs precision tradeoff
- Detector array design for reading interference patterns with acceptable SNR
- Coherent source stabilisation: how much thermal drift can the Bragg condition tolerate before a channel becomes unreadable?
- ADC energy budget: analog-to-digital conversion per output channel is a known bottleneck — what's the minimum viable detector count for attention readout?

**How to contribute:** Add to `docs/benchmarks.md` with realistic peripheral energy figures. The current benchmark models only the optical interaction; the periphery is where the real system-level energy lives.

---

## Noise Modelling

**The core problem:** The simulator originally ran in a noiseless mathematical environment. Phase quantisation, Gaussian phase noise, angular jitter, and a simple crosstalk kernel are now present. Detector shot noise, a measured Bragg selectivity curve, and full thermal-drift models are still open.

**Open problems:**
- Replace the uniform crosstalk kernel with a measured angular selectivity profile
- Thermal drift: model Bragg angle shift as a function of temperature and crystal coefficient of thermal expansion
- Detector / readout noise on the intensity path
- End-to-end noisy inference on a real task with calibrated noise and reported accuracy

**How to contribute:** Extend `atom/noise.py` and the corresponding tests. The existing `NoiseConfig` is the intended place to hang new knobs.

---

## ML Systems

**The core problem:** ATOM proves the attention score computation can happen optically. Everything else — softmax, layer norm, residual addition, value aggregation — still runs digitally. Figuring out how to compose optical attention with a real transformer inference stack is unsolved.

**Open problems:**
- Weight conversion pipeline: given any HuggingFace model, produce the phase mask values for each layer ready for crystal writing
- Hybrid inference stack: optical QK scores + digital softmax/V aggregation — profile latency and energy for a real model
- Quantisation-aware weight encoding: what's the minimum phase precision needed to maintain attention accuracy within 1% of digital?
- Forward-pass-only training: can the optical path be trained without backpropagation? (See gradient estimation literature)
- Multi-crystal / rack-scale composition: treat each crystal (or crystal module) as an optical attention accelerator and partition a large model across many of them, the same way multi-GPU systems partition today. Interconnect, activation movement, and scheduling are open systems-engineering problems.

**How to contribute:** The weight conversion stub lives in `atom/attention.py`. A full pipeline that takes a `.safetensors` or `.gguf` checkpoint and outputs phase mask values would be the highest-leverage ML contribution possible right now. System-level multi-module design belongs here as well.

---

## Theory

**The core problem:** The attention–interference equivalence is proved for the linear QK score. Softmax is not optical. Value aggregation is not optical. Multi-head attention is not optical. Extending the theoretical framework to cover more of the transformer forward pass is wide open.

**Open problems:**
- Optical softmax approximations: is there a physical mechanism that produces softmax-like normalisation without digital conversion?
- Multi-head attention in a single crystal volume: can multiple heads be addressed by different angular sub-bands?
- Optical activation functions: what nonlinearities are physically achievable in photorefractive media?
- Training without backprop: forward-pass-only gradient estimation (SPSA, perturbation methods) for updating holographic weights without a full differentiable path

**How to contribute:** Open a discussion issue with a proof sketch. If you can prove or disprove any of the above, that is a paper, not just a PR.

---

## General

- **Issues** for questions, bugs, or pointing out something that doesn't add up
- **Discussions** for open-ended ideas, architecture questions, or "I work in X and I think I can help"
- **PRs** for actual code, models, or documentation

If you find an error in the math, open an issue. If the proof is wrong somewhere, that is the most important contribution possible and it will be credited as such.
