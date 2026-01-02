#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt

import whitsmooth_rust

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


def ddmat_sparse(x, d):
    import scipy.sparse as sp
    x = np.asarray(x, dtype=float)
    n = x.size
    if d == 0:
        return sp.eye(n, format="csc")
    Dprev = ddmat_sparse(x, d - 1)
    Ddiff = Dprev[1:, :] - Dprev[:-1, :]
    dx = x[d:] - x[:-d]
    V = sp.diags(1.0 / dx, 0, format="csc")
    return (V @ Ddiff).tocsc()


def whittaker_scipy_reference(x, y_st, w_st, lam, d, ridge=1e-10):
    # y_st, w_st: (S,T)
    T = x.size
    S = y_st.shape[0]
    D = ddmat_sparse(x, d)
    P = (D.T @ D).tocsc()
    Z = np.empty_like(y_st, dtype=float)
    for s in range(S):
        w = w_st[s]
        W = sp.diags(w, 0, shape=(T, T), format="csc")
        A = (W + lam * P + ridge * sp.eye(T, format="csc")).tocsc()
        b = w * np.nan_to_num(y_st[s], nan=0.0)
        Z[s] = spla.spsolve(A, b)
    return Z


def median_time(fn, warmup=1, repeat=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def make_data(rng, T, S, dtype):
    x = np.cumsum(rng.uniform(0.8, 1.2, size=T)).astype(dtype)
    y = rng.normal(size=(S, T)).astype(dtype)
    if S > 2:
        y[2, T//2] += dtype(25.0)
    if S > 4:
        y[4, 10:25] = np.nan

    w = rng.uniform(0.2, 1.0, size=(S, T)).astype(dtype)
    w[~np.isfinite(y)] = 0
    return x, y, w


def main():
    rng = np.random.default_rng(0)
    T = 2000
    d = 2
    lam = 50.0

    S_list = [1, 64, 256, 512, 1000, 5000, 10000]

    t_single64 = []
    t_single32 = []
    t_irls64 = []
    t_irls32 = []
    t_scipy = []

    for S in S_list:
        x64, y64, w64 = make_data(rng, T, S, np.float64)
        x32, y32, w32 = x64.astype(np.float32), y64.astype(np.float32), w64.astype(np.float32)

        def run_single64():
            _ = whitsmooth_rust.whittaker_solve_f64(
                x64, y64, w64,
                lam=lam, d=d,
                ridge=1e-10,
                normalize="mean",
                eps=1e-12,
                parallel=True,
            )

        def run_single32():
            _ = whitsmooth_rust.whittaker_solve_f32(
                x32, y32, w32,
                lam=np.float32(lam), d=d,
                ridge=np.float32(1e-6),
                normalize="mean",
                eps=np.float32(1e-6),
                parallel=True,
            )

        def run_irls64():
            _ = whitsmooth_rust.robust_whittaker_irls_f64(
                x64, y64, w64,
                lam=lam, d=d, iterations=6,
                weighting="tukey",
                scale="mad",
                ridge=1e-10,
                normalize="mean",
                eps=1e-12,
                parallel=True,
                tuning=None,
                return_weights=False,
            )

        def run_irls32():
            _ = whitsmooth_rust.robust_whittaker_irls_f32(
                x32, y32, w32,
                lam=np.float32(lam), d=d, iterations=6,
                weighting="tukey",
                scale="mad",
                ridge=np.float32(1e-6),
                normalize="mean",
                eps=np.float32(1e-6),
                parallel=True,
                tuning=None,
                return_weights=False,
            )

        single64 = median_time(run_single64, warmup=1, repeat=5) * 1e3
        single32 = median_time(run_single32, warmup=1, repeat=5) * 1e3
        irls64 = median_time(run_irls64, warmup=1, repeat=3) * 1e3
        irls32 = median_time(run_irls32, warmup=1, repeat=3) * 1e3

        t_single64.append(single64)
        t_single32.append(single32)
        t_irls64.append(irls64)
        t_irls32.append(irls32)

        print(f"[BENCH] S={S:6d}  single64={single64:8.2f}ms  single32={single32:8.2f}ms  "
              f"irls64={irls64:8.2f}ms  irls32={irls32:8.2f}ms")

        if SCIPY_OK and S <= 512:
            # SciPy ref gets very slow for huge S (loops per series)
            def run_scipy():
                _ = whittaker_scipy_reference(x64.astype(float), y64.astype(float), w64.astype(float), lam=lam, d=d, ridge=1e-10)
            sc = median_time(run_scipy, warmup=1, repeat=1) * 1e3
            t_scipy.append(sc)
            print(f"         SciPy(ref loop S)={sc:8.2f}ms")
        else:
            t_scipy.append(np.nan)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(S_list, t_single64, marker="o", label="single solve f64 (Rust)")
    plt.plot(S_list, t_single32, marker="o", label="single solve f32 (Rust)")
    plt.plot(S_list, t_irls64, marker="o", label="IRLS Tukey f64 (Rust)")
    plt.plot(S_list, t_irls32, marker="o", label="IRLS Tukey f32 (Rust)")
    if SCIPY_OK:
        plt.plot(S_list, t_scipy, marker="o", label="SciPy sparse ref (loop S)")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("S (number of series, log)")
    plt.ylabel("runtime ms (median, log)")
    plt.title(f"whitsmooth_rust benchmark (T={T}, d={d}, lam={lam})")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
