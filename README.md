# whitsmooth_rust

A **fast, compiled** (no JIT warmup) implementation of a **batched robust Whittaker smoother** with **divided differences**
in **Rust**, exposed to Python via **PyO3**.

It is designed for data shaped as many independent time series:

- `S` series (samples / pixels / bands / locations)
- `T` time points per series

and supports **time- and series-varying weights** (e.g. missing data, quality flags, IRLS robust weights).

This package provides both **float64** and **float32** implementations.

---

## Mathematical background

We solve a Whittaker-type penalized least squares problem for each series `s`:

$$
\min_{z_s}\; \sum_{t=1}^T w_{s,t}(y_{s,t}-z_{s,t})^2 + \lambda \lVert D z_s \rVert^2
$$

- `y_{s,t}`: observation
- `w_{s,t} >= 0`: weight (0 means missing/ignored)
- `D`: **divided-difference** operator of order `d` defined on a strictly increasing sampling grid `x`
- `λ` (lambda): smoothing strength

The normal equations are:

$$
(\operatorname{diag}(w_s) + \lambda D^\top D + \text{ridge} I) z_s = w_s \odot y_s
$$

`P = D^T D` is **symmetric positive semidefinite** and **banded** with half-bandwidth `k = 2d`.
We store only the **lower band** `Pb[j, i] = P[i, i-j]`.

We solve each system using an **SPD banded Cholesky** factorization:

$$
A = L L^\top
$$

which exploits symmetry and banded structure.

---

## Robust IRLS (iteratively reweighted least squares)

Robust smoothing down-weights outliers by iterating:

1. Solve Whittaker system with current weights
2. Compute residuals `r = y - z`
3. Estimate per-series scale (default: MAD)
4. Compute standardized residuals `u = r / scale`
5. Update robust weights `w_rob(u)` and repeat

Final weights per entry are:

$$
w_{total} = w_{base} \cdot w_{rob}
$$

Supported robust weight functions:

- **Tukey / Bisquare** (hard rejection outside `c`)
- **Huber** (soft clipping)
- **Cauchy**
- **Welsch**
- **Fair**
- **Hampel** (3-parameter redescending)

Supported scale estimators:

- `mad` (default): `scale = 1.4826 * median(|r|)`
- `huber`: iterative Huber M-scale (more stable for heavy tails)

---

## x normalization (recommended)

Because divided differences depend on the units/spacing of `x`, a fixed `λ` can mean different smoothness
when sampling density changes. You can normalize `x` to approximately unit spacing:

- `normalize="mean"`: average step becomes ~1
- `normalize="median"`: median step becomes ~1 (robust)

This makes `λ` more comparable across datasets.

---

## API overview

All functions use **series-first layout** for speed:

- `y_st`: shape `(S, T)`
- `w_base_st`: shape `(S, T)` or omitted

### Build penalty band

- `build_pb_f64(x, d, normalize=None, eps=1e-12) -> (pb, k)`
- `build_pb_f32(x, d, normalize=None, eps=1e-6) -> (pb, k)`

Returns:
- `pb`: array of shape `(k+1, T)` storing lower band of `P = D^T D`
- `k = 2d`

### Robust IRLS smoother

- `robust_whittaker_irls_f64(...) -> z_st` (or `(z_st, w_total_st)` if `return_weights=True`)
- `robust_whittaker_irls_f32(...) -> z_st` (or `(z_st, w_total_st)`)

Parameters (both dtypes):
- `x`: strictly increasing sampling positions (length `T`)
- `y_st`: `(S,T)` observations (NaNs allowed; they will be treated as missing)
- `w_base_st`: `(S,T)` base weights (optional; if provided, NaNs in y are forced to 0 weight)
- `lam`: smoothing parameter λ
- `d`: difference order (typical 1..6)
- `iterations`: IRLS iterations (typical 3..10)
- `weighting`: one of `"tukey"|"huber"|"cauchy"|"welsch"|"fair"|"hampel"`
- `scale`: `"mad"` or `"huber"`
- `ridge`: small diagonal stabilizer (e.g. 1e-10 for f64, 1e-6 for f32)
- `normalize`: None / `"mean"` / `"median"`
- `eps`: numerical floor for stability
- `parallel`: parallelize across series (Rayon). Recommended when S is large.
- `tuning`: optional tuple
  - for Tukey/Huber/Cauchy/Welsch/Fair: `(0,0,c)`
  - for Hampel: `(a,b,c)`
- `return_weights`: return final `w_total_st` as well

---

## Installation / compilation

You need Rust and maturin.

```bash
pip install maturin
cd whitsmooth_rust
maturin develop --release
```

This builds and installs the extension into the current Python environment.

---

## Usage examples

### Float64

```python
import numpy as np
import whitsmooth_rust

rng = np.random.default_rng(0)
T, S = 2000, 1000
x = np.cumsum(rng.uniform(0.8, 1.2, size=T)).astype(np.float64)

y = rng.normal(size=(S, T)).astype(np.float64)
y[2, T//2] += 25.0
y[4, 10:25] = np.nan

w = rng.uniform(0.2, 1.0, size=(S, T)).astype(np.float64)
w[~np.isfinite(y)] = 0.0

z, wtot = whitsmooth_rust.robust_whittaker_irls_f64(
    x, y, w,
    lam=50.0, d=2, iterations=3,
    weighting="tukey",
    scale="mad",
    ridge=1e-10,
    normalize="mean",
    eps=1e-12,
    parallel=True,
    tuning=None,
    return_weights=True
)
```

### Float32

```python
x32 = x.astype(np.float32)
y32 = y.astype(np.float32)
w32 = w.astype(np.float32)

z32 = whitsmooth_rust.robust_whittaker_irls_f32(
    x32, y32, w32,
    lam=50.0, d=2, iterations=6,
    weighting="tukey",
    scale="mad",
    ridge=1e-6,
    normalize="mean",
    eps=1e-6,
    parallel=True,
    tuning=None,
    return_weights=False
)
```

### If your data is (T,S)
Transpose when calling:

```python
z_st = whitsmooth_rust.robust_whittaker_irls_f64(x, y_ts.T, w_ts.T, ...)
z_ts = z_st.T
```

---

## Notes on performance

- The solver is **SPD banded Cholesky**, per series, with half-bandwidth `k=2d`.
- Parallelization is across series `S`.
- Float32 can be significantly faster and reduce memory footprint for very large `S`.

---

## License
MIT

---

## Single-pass (non-robust) Whittaker solver

If you don't need IRLS robustness, you can run a single solve (much cheaper):

- `whittaker_solve_f64(x, y_st, w_base_st, lam, d, ridge, normalize=None, eps=1e-12, parallel=True) -> z_st`
- `whittaker_solve_f32(x, y_st, w_base_st, lam, d, ridge, normalize=None, eps=1e-6, parallel=True) -> z_st`

These solve:
\[
(\operatorname{diag}(w_s) + \lambda D^\top D + \text{ridge} I) z_s = w_s \odot y_s
\]
with no robust reweighting.

---

## Benchmark script

This repo includes `benchmark.py` which times:

- single solve f64 / f32
- robust IRLS (Tukey) f64 / f32

Across multiple `S` values and plots runtime.

After installing the extension:

```bash
python benchmark.py
```

Optional: if you have SciPy installed, the benchmark can also compare against a sparse reference (looping over series).
