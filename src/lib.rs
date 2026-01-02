
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;
use ndarray::{Array2, ArrayView2};

const MAD_CONST_F64: f64 = 1.4826;
const MAD_CONST_F32: f32 = 1.4826_f32;

fn median_inplace_f64(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 { return 0.0; }
    let mid = n / 2;
    v.select_nth_unstable_by(mid, |a,b| a.partial_cmp(b).unwrap());
    let m2 = v[mid];
    if n % 2 == 1 { m2 } else {
        let mut max_lower = v[0];
        for &val in &v[..mid] { if val > max_lower { max_lower = val; } }
        0.5 * (max_lower + m2)
    }
}
fn median_inplace_f32(v: &mut [f32]) -> f32 {
    let n = v.len();
    if n == 0 { return 0.0; }
    let mid = n / 2;
    v.select_nth_unstable_by(mid, |a,b| a.partial_cmp(b).unwrap());
    let m2 = v[mid];
    if n % 2 == 1 { m2 } else {
        let mut max_lower = v[0];
        for &val in &v[..mid] { if val > max_lower { max_lower = val; } }
        0.5 * (max_lower + m2)
    }
}

fn normalize_x_unit_step_f64(x: &[f64], method: &str) -> Vec<f64> {
    let n = x.len();
    if n <= 1 { return x.to_vec(); }
    let x0 = x[0];
    let scale = match method {
        "mean" => {
            let span = x[n-1] - x[0];
            let denom = (n-1) as f64;
            if denom > 0.0 { span / denom } else { 1.0 }
        },
        "median" => {
            let mut dx: Vec<f64> = x.windows(2).map(|w| w[1]-w[0]).filter(|v| *v>0.0).collect();
            if dx.is_empty() { 1.0 } else { median_inplace_f64(&mut dx) }
        },
        _ => 1.0,
    };
    let s = if scale > 0.0 { scale } else { 1.0 };
    x.iter().map(|v| (v - x0)/s).collect()
}
fn normalize_x_unit_step_f32(x: &[f32], method: &str) -> Vec<f32> {
    let n = x.len();
    if n <= 1 { return x.to_vec(); }
    let x0 = x[0];
    let scale: f32 = match method {
        "mean" => {
            let span = x[n-1] - x[0];
            let denom = (n-1) as f32;
            if denom > 0.0 { span / denom } else { 1.0 }
        },
        "median" => {
            let mut dx: Vec<f32> = x.windows(2).map(|w| w[1]-w[0]).filter(|v| *v>0.0).collect();
            if dx.is_empty() { 1.0 } else { median_inplace_f32(&mut dx) }
        },
        _ => 1.0,
    };
    let s = if scale > 0.0 { scale } else { 1.0 };
    x.iter().map(|v| (v - x0)/s).collect()
}

// -------------------- build divided-difference stencils --------------------

fn build_divdiff_stencils_f64(x: &[f64], d: usize, eps: f64) -> Vec<Vec<f64>> {
    let n = x.len();
    if d == 0 { return (0..n).map(|_| vec![1.0]).collect(); }

    let mut coeffs: Vec<Vec<f64>> = Vec::with_capacity(n-1);
    for i in 0..(n-1) {
        let dx = (x[i+1]-x[i]).max(eps);
        coeffs.push(vec![-1.0/dx, 1.0/dx]);
    }

    for k in 2..=d {
        let prev = coeffs;
        let m_prev = prev.len();
        let width_prev = prev[0].len();
        let m_new = m_prev - 1;
        let width_new = width_prev + 1;

        let mut next: Vec<Vec<f64>> = vec![vec![0.0; width_new]; m_new];

        for i in 0..m_new {
            for j in 0..width_prev {
                next[i][j+1] += prev[i+1][j];
                next[i][j]   -= prev[i][j];
            }
        }

        for i in 0..m_new {
            let denom = (x[i+k] - x[i]).max(eps);
            let v = 1.0/denom;
            for j in 0..width_new { next[i][j] *= v; }
        }
        coeffs = next;
    }
    coeffs
}

fn build_divdiff_stencils_f32(x: &[f32], d: usize, eps: f32) -> Vec<Vec<f32>> {
    let n = x.len();
    if d == 0 { return (0..n).map(|_| vec![1.0f32]).collect(); }

    let mut coeffs: Vec<Vec<f32>> = Vec::with_capacity(n-1);
    for i in 0..(n-1) {
        let dx = (x[i+1]-x[i]).max(eps);
        coeffs.push(vec![-1.0f32/dx, 1.0f32/dx]);
    }

    for k in 2..=d {
        let prev = coeffs;
        let m_prev = prev.len();
        let width_prev = prev[0].len();
        let m_new = m_prev - 1;
        let width_new = width_prev + 1;

        let mut next: Vec<Vec<f32>> = vec![vec![0.0f32; width_new]; m_new];

        for i in 0..m_new {
            for j in 0..width_prev {
                next[i][j+1] += prev[i+1][j];
                next[i][j]   -= prev[i][j];
            }
        }

        for i in 0..m_new {
            let denom = (x[i+k] - x[i]).max(eps);
            let v = 1.0f32/denom;
            for j in 0..width_new { next[i][j] *= v; }
        }
        coeffs = next;
    }
    coeffs
}

// -------------------- build Pb band for P=D^T D --------------------

fn build_pb_from_x_f64(x: &[f64], d: usize, eps: f64) -> (Vec<f64>, usize, usize) {
    let n = x.len();
    let k = 2*d;
    let st = build_divdiff_stencils_f64(x, d, eps); // (n-d, d+1)
    let m = st.len();
    let mut pb = vec![0.0f64; (k+1)*n];

    for i in 0..m {
        let a = &st[i];
        for p in 0..=d {
            let cp = i+p;
            let ap = a[p];
            for q in 0..=p {
                let cq = i+q;
                let aq = a[q];
                let j = cp - cq;
                if j <= k {
                    pb[j*n + cp] += ap*aq;
                }
            }
        }
    }
    (pb, k, n)
}

fn build_pb_from_x_f32(x: &[f32], d: usize, eps: f32) -> (Vec<f32>, usize, usize) {
    let n = x.len();
    let k = 2*d;
    let st = build_divdiff_stencils_f32(x, d, eps); // (n-d, d+1)
    let m = st.len();
    let mut pb = vec![0.0f32; (k+1)*n];

    for i in 0..m {
        let a = &st[i];
        for p in 0..=d {
            let cp = i+p;
            let ap = a[p];
            for q in 0..=p {
                let cq = i+q;
                let aq = a[q];
                let j = cp - cq;
                if j <= k {
                    pb[j*n + cp] += ap*aq;
                }
            }
        }
    }
    (pb, k, n)
}

// -------------------- robust weights --------------------

#[derive(Clone, Copy)]
enum Weighting { Tukey, Huber, Cauchy, Welsch, Fair, Hampel }

fn parse_weighting(s: &str) -> Result<Weighting, String> {
    match s.to_lowercase().as_str() {
        "tukey" | "bisquare" => Ok(Weighting::Tukey),
        "huber" => Ok(Weighting::Huber),
        "cauchy" => Ok(Weighting::Cauchy),
        "welsch" => Ok(Weighting::Welsch),
        "fair" => Ok(Weighting::Fair),
        "hampel" => Ok(Weighting::Hampel),
        _ => Err(format!("Unknown weighting: {}", s)),
    }
}

fn default_tuning(w: Weighting) -> (f64,f64,f64) {
    match w {
        Weighting::Tukey => (0.0,0.0,4.685),
        Weighting::Huber => (0.0,0.0,1.345),
        Weighting::Cauchy => (0.0,0.0,2.385),
        Weighting::Welsch => (0.0,0.0,2.985),
        Weighting::Fair => (0.0,0.0,1.3998),
        Weighting::Hampel => (1.5,3.5,8.0),
    }
}

fn weight_value_f64(w: Weighting, u: f64, a: f64, b: f64, c: f64) -> f64 {
    let t = u.abs();
    match w {
        Weighting::Tukey => {
            let r = t / c;
            if r < 1.0 { let v = 1.0 - r*r; v*v } else { 0.0 }
        }
        Weighting::Huber => if t <= c { 1.0 } else { c / t.max(1e-12) },
        Weighting::Cauchy => { let r = u/c; 1.0/(1.0 + r*r) }
        Weighting::Welsch => { let r = u/c; (-r*r).exp() }
        Weighting::Fair => 1.0/(1.0 + t/c),
        Weighting::Hampel => {
            if t <= a { 1.0 }
            else if t <= b { a / t.max(1e-12) }
            else if t <= c { (a*(c-t)) / (t.max(1e-12) * (c-b).max(1e-12)) }
            else { 0.0 }
        }
    }
}
fn weight_value_f32(w: Weighting, u: f32, a: f32, b: f32, c: f32) -> f32 {
    let t = u.abs();
    match w {
        Weighting::Tukey => {
            let r = t / c;
            if r < 1.0 { let v = 1.0 - r*r; v*v } else { 0.0 }
        }
        Weighting::Huber => if t <= c { 1.0 } else { c / t.max(1e-12) },
        Weighting::Cauchy => { let r = u/c; 1.0/(1.0 + r*r) }
        Weighting::Welsch => { let r = u/c; (-r*r).exp() }
        Weighting::Fair => 1.0/(1.0 + t/c),
        Weighting::Hampel => {
            if t <= a { 1.0 }
            else if t <= b { a / t.max(1e-12) }
            else if t <= c { (a*(c-t)) / (t.max(1e-12) * (c-b).max(1e-12)) }
            else { 0.0 }
        }
    }
}

// -------------------- scale estimators --------------------

fn mad_scale_f64(res: &[f64], wmask: &[u8], eps: f64) -> f64 {
    let mut vals: Vec<f64> = Vec::new();
    vals.reserve(res.len());
    for i in 0..res.len() {
        if wmask[i] != 0 && res[i].is_finite() {
            vals.push(res[i].abs());
        }
    }
    if vals.is_empty() { return eps; }
    let med = median_inplace_f64(&mut vals);
    (MAD_CONST_F64 * med).max(eps)
}
fn mad_scale_f32(res: &[f32], wmask: &[u8], eps: f32) -> f32 {
    let mut vals: Vec<f32> = Vec::new();
    vals.reserve(res.len());
    for i in 0..res.len() {
        if wmask[i] != 0 && res[i].is_finite() {
            vals.push(res[i].abs());
        }
    }
    if vals.is_empty() { return eps; }
    let med = median_inplace_f32(&mut vals);
    (MAD_CONST_F32 * med).max(eps)
}

fn huber_mscale_f64(res: &[f64], wmask: &[u8], c_h: f64, eps: f64, iters: usize) -> f64 {
    let mut s = mad_scale_f64(res, wmask, eps);
    for _ in 0..iters {
        let cap = c_h * s;
        let cap2 = cap*cap;
        let mut sum = 0.0;
        let mut cnt = 0.0;
        for i in 0..res.len() {
            if wmask[i] != 0 && res[i].is_finite() {
                let r2 = res[i]*res[i];
                sum += if r2 < cap2 { r2 } else { cap2 };
                cnt += 1.0;
            }
        }
        if cnt == 0.0 { return eps; }
        let s_new = (sum/cnt).sqrt().max(eps);
        let rel = (s_new - s).abs() / s.max(eps);
        s = s_new;
        if rel < 1e-4 { break; }
    }
    s
}
fn huber_mscale_f32(res: &[f32], wmask: &[u8], c_h: f32, eps: f32, iters: usize) -> f32 {
    let mut s = mad_scale_f32(res, wmask, eps);
    for _ in 0..iters {
        let cap = c_h * s;
        let cap2 = cap*cap;
        let mut sum: f32 = 0.0;
        let mut cnt: f32 = 0.0;
        for i in 0..res.len() {
            if wmask[i] != 0 && res[i].is_finite() {
                let r2 = res[i]*res[i];
                sum += if r2 < cap2 { r2 } else { cap2 };
                cnt += 1.0;
            }
        }
        if cnt == 0.0 { return eps; }
        let s_new = (sum/cnt).sqrt().max(eps);
        let rel = (s_new - s).abs() / s.max(eps);
        s = s_new;
        if rel < 1e-4 { break; }
    }
    s
}

// -------------------- solver per series (SPD banded Cholesky) --------------------

fn solve_one_series_f64(
    pb: &[f64], n: usize, k: usize,
    w: &[f64], y: &[f64],
    lam: f64, ridge: f64, eps: f64,
    out: &mut [f64],
) {
    let mut lb = vec![0.0f64; (k+1)*n];
    let mut rhs = vec![0.0f64; n];
    for i in 0..n { rhs[i] = w[i]*y[i]; }

    for i in 0..n {
        let mut diag = w[i] + lam*pb[0*n + i] + ridge;
        let tmax = i.min(k);
        for t in 1..=tmax {
            let v = lb[t*n + i];
            diag -= v*v;
        }
        let lii = diag.max(eps).sqrt();
        lb[0*n + i] = lii;

        let rmax = (n-1-i).min(k);
        for r in 1..=rmax {
            let mut num = lam * pb[r*n + (i+r)];
            let tmax2 = tmax.min(k-r);
            for t in 1..=tmax2 {
                let a = lb[(r+t)*n + (i+r)];
                let b = lb[t*n + i];
                num -= a*b;
            }
            lb[r*n + (i+r)] = num / lii;
        }
    }

    for i in 0..n {
        rhs[i] /= lb[0*n + i].max(eps);
        let rmax = (n-1-i).min(k);
        for r in 1..=rmax {
            rhs[i+r] -= lb[r*n + (i+r)] * rhs[i];
        }
    }
    for ii in 0..n {
        let i = n-1-ii;
        rhs[i] /= lb[0*n + i].max(eps);
        let rmax = i.min(k);
        for r in 1..=rmax {
            rhs[i-r] -= lb[r*n + i] * rhs[i];
        }
    }

    out.copy_from_slice(&rhs);
}

fn solve_one_series_f32(
    pb: &[f32], n: usize, k: usize,
    w: &[f32], y: &[f32],
    lam: f32, ridge: f32, eps: f32,
    out: &mut [f32],
) {
    let mut lb = vec![0.0f32; (k+1)*n];
    let mut rhs = vec![0.0f32; n];
    for i in 0..n { rhs[i] = w[i]*y[i]; }

    for i in 0..n {
        let mut diag = w[i] + lam*pb[0*n + i] + ridge;
        let tmax = i.min(k);
        for t in 1..=tmax {
            let v = lb[t*n + i];
            diag -= v*v;
        }
        let lii = diag.max(eps).sqrt();
        lb[0*n + i] = lii;

        let rmax = (n-1-i).min(k);
        for r in 1..=rmax {
            let mut num = lam * pb[r*n + (i+r)];
            let tmax2 = tmax.min(k-r);
            for t in 1..=tmax2 {
                let a = lb[(r+t)*n + (i+r)];
                let b = lb[t*n + i];
                num -= a*b;
            }
            lb[r*n + (i+r)] = num / lii;
        }
    }

    for i in 0..n {
        rhs[i] /= lb[0*n + i].max(eps);
        let rmax = (n-1-i).min(k);
        for r in 1..=rmax {
            rhs[i+r] -= lb[r*n + (i+r)] * rhs[i];
        }
    }
    for ii in 0..n {
        let i = n-1-ii;
        rhs[i] /= lb[0*n + i].max(eps);
        let rmax = i.min(k);
        for r in 1..=rmax {
            rhs[i-r] -= lb[r*n + i] * rhs[i];
        }
    }

    out.copy_from_slice(&rhs);
}

// -------------------- robust IRLS (f64) --------------------

fn robust_irls_f64(
    pb: &[f64], n: usize, k: usize,
    y_st: ArrayView2<'_, f64>,      // (S,T)
    wbase_st: ArrayView2<'_, f64>,  // (S,T)
    lam: f64, ridge: f64,
    iterations: usize,
    weighting: Weighting,
    tuning: (f64,f64,f64),
    scale: &str,
    eps: f64,
    parallel: bool,
) -> (Array2<f64>, Array2<f64>) {
    let (s, t) = y_st.dim();
    assert_eq!(t, n);
    let mut z = Array2::<f64>::zeros((s, n));
    let mut wrob = Array2::<f64>::zeros((s, n));
    let mut wtot = Array2::<f64>::zeros((s, n));

    for si in 0..s {
        for i in 0..n {
            let yv = y_st[(si,i)];
            let wb = wbase_st[(si,i)];
            wrob[(si,i)] = if yv.is_finite() && wb > 0.0 { 1.0 } else { 0.0 };
        }
    }

    let (a,b,c) = tuning;

    let work = |si: usize, z_row: &mut [f64], wrob_row: &mut [f64], wtot_row: &mut [f64]| {
        let yrow = y_st.row(si);
        let wbase = wbase_st.row(si);

        let mut w = vec![0.0f64; n];
        let mut y = vec![0.0f64; n];
        let mut wmask = vec![0u8; n];

        for i in 0..n {
            let yv = yrow[i];
            let wb = wbase[i];
            let wr = wrob_row[i];
            let wt = wb * wr;
            w[i] = wt;
            wtot_row[i] = wt;

            if yv.is_finite() && wt > 0.0 {
                y[i] = yv;
                wmask[i] = 1;
            } else {
                y[i] = 0.0;
                wmask[i] = 0;
            }
        }

        solve_one_series_f64(pb, n, k, &w, &y, lam, ridge, eps, z_row);

        let mut res = vec![f64::NAN; n];
        for i in 0..n {
            if wmask[i] != 0 {
                res[i] = yrow[i] - z_row[i];
            }
        }

        let sscale = if scale.eq_ignore_ascii_case("huber") {
            huber_mscale_f64(&res, &wmask, 1.345, eps, 30)
        } else {
            mad_scale_f64(&res, &wmask, eps)
        };

        for i in 0..n {
            if wmask[i] != 0 {
                let u = res[i] / sscale;
                wrob_row[i] = weight_value_f64(weighting, u, a, b, c);
            } else {
                wrob_row[i] = 0.0;
            }
        }
    };

    for _ in 0..iterations {
        if parallel && s >= 2 {
            let z_slice = z.as_slice_mut().unwrap();
            let wrob_slice = wrob.as_slice_mut().unwrap();
            let wtot_slice = wtot.as_slice_mut().unwrap();

            z_slice
                .par_chunks_mut(n)
                .zip(wrob_slice.par_chunks_mut(n))
                .zip(wtot_slice.par_chunks_mut(n))
                .enumerate()
                .for_each(|(si, ((z_row, wrob_row), wtot_row))| {
                    work(si, z_row, wrob_row, wtot_row);
                });
        } else {
            for si in 0..s {
                let mut zr = z.row_mut(si);
                let mut wr = wrob.row_mut(si);
                let mut wt = wtot.row_mut(si);
                work(si, zr.as_slice_mut().unwrap(), wr.as_slice_mut().unwrap(), wt.as_slice_mut().unwrap());
            }
        }
    }

    (z, wtot)
}

// -------------------- robust IRLS (f32) --------------------

fn robust_irls_f32(
    pb: &[f32], n: usize, k: usize,
    y_st: ArrayView2<'_, f32>,      // (S,T)
    wbase_st: ArrayView2<'_, f32>,  // (S,T)
    lam: f32, ridge: f32,
    iterations: usize,
    weighting: Weighting,
    tuning: (f32,f32,f32),
    scale: &str,
    eps: f32,
    parallel: bool,
) -> (Array2<f32>, Array2<f32>) {
    let (s, t) = y_st.dim();
    assert_eq!(t, n);
    let mut z = Array2::<f32>::zeros((s, n));
    let mut wrob = Array2::<f32>::zeros((s, n));
    let mut wtot = Array2::<f32>::zeros((s, n));

    for si in 0..s {
        for i in 0..n {
            let yv = y_st[(si,i)];
            let wb = wbase_st[(si,i)];
            wrob[(si,i)] = if yv.is_finite() && wb > 0.0 { 1.0 } else { 0.0 };
        }
    }

    let (a,b,c) = tuning;

    let work = |si: usize, z_row: &mut [f32], wrob_row: &mut [f32], wtot_row: &mut [f32]| {
        let yrow = y_st.row(si);
        let wbase = wbase_st.row(si);

        let mut w = vec![0.0f32; n];
        let mut y = vec![0.0f32; n];
        let mut wmask = vec![0u8; n];

        for i in 0..n {
            let yv = yrow[i];
            let wb = wbase[i];
            let wr = wrob_row[i];
            let wt = wb * wr;
            w[i] = wt;
            wtot_row[i] = wt;

            if yv.is_finite() && wt > 0.0 {
                y[i] = yv;
                wmask[i] = 1;
            } else {
                y[i] = 0.0;
                wmask[i] = 0;
            }
        }

        solve_one_series_f32(pb, n, k, &w, &y, lam, ridge, eps, z_row);

        let mut res = vec![f32::NAN; n];
        for i in 0..n {
            if wmask[i] != 0 {
                res[i] = yrow[i] - z_row[i];
            }
        }

        let sscale = if scale.eq_ignore_ascii_case("huber") {
            huber_mscale_f32(&res, &wmask, 1.345_f32, eps, 30)
        } else {
            mad_scale_f32(&res, &wmask, eps)
        };

        for i in 0..n {
            if wmask[i] != 0 {
                let u = res[i] / sscale;
                wrob_row[i] = weight_value_f32(weighting, u, a, b, c);
            } else {
                wrob_row[i] = 0.0;
            }
        }
    };

    for _ in 0..iterations {
        if parallel && s >= 2 {
            let z_slice = z.as_slice_mut().unwrap();
            let wrob_slice = wrob.as_slice_mut().unwrap();
            let wtot_slice = wtot.as_slice_mut().unwrap();

            z_slice
                .par_chunks_mut(n)
                .zip(wrob_slice.par_chunks_mut(n))
                .zip(wtot_slice.par_chunks_mut(n))
                .enumerate()
                .for_each(|(si, ((z_row, wrob_row), wtot_row))| {
                    work(si, z_row, wrob_row, wtot_row);
                });
        } else {
            for si in 0..s {
                let mut zr = z.row_mut(si);
                let mut wr = wrob.row_mut(si);
                let mut wt = wtot.row_mut(si);
                work(si, zr.as_slice_mut().unwrap(), wr.as_slice_mut().unwrap(), wt.as_slice_mut().unwrap());
            }
        }
    }

    (z, wtot)
}

// ==================== PyO3 wrappers ====================

#[pyfunction]
fn build_pb_f64<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    d: usize,
    normalize: Option<&str>,
    eps: Option<f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, usize)> {
    let eps = eps.unwrap_or(1e-12);
    let x = x.as_slice()?;
    if x.len() < 2 { return Err(pyo3::exceptions::PyValueError::new_err("x must have length >=2")); }
    for i in 0..(x.len()-1) {
        if !(x[i+1] > x[i]) { return Err(pyo3::exceptions::PyValueError::new_err("x must be strictly increasing")); }
    }

    let x_use = if let Some(m) = normalize { normalize_x_unit_step_f64(x, m) } else { x.to_vec() };
    let (pb_vec, k, n) = build_pb_from_x_f64(&x_use, d, eps);

    let pb = PyArray2::<f64>::zeros(py, [k+1, n], false);
    let mut pbm = unsafe { pb.as_array_mut() };
    for j in 0..(k+1) {
        for i in 0..n {
            pbm[(j,i)] = pb_vec[j*n + i];
        }
    }
    Ok((pb, k))
}

#[pyfunction]
fn build_pb_f32<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    d: usize,
    normalize: Option<&str>,
    eps: Option<f32>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, usize)> {
    let eps = eps.unwrap_or(1e-6_f32);
    let x = x.as_slice()?;
    if x.len() < 2 { return Err(pyo3::exceptions::PyValueError::new_err("x must have length >=2")); }
    for i in 0..(x.len()-1) {
        if !(x[i+1] > x[i]) { return Err(pyo3::exceptions::PyValueError::new_err("x must be strictly increasing")); }
    }

    let x_use = if let Some(m) = normalize { normalize_x_unit_step_f32(x, m) } else { x.to_vec() };
    let (pb_vec, k, n) = build_pb_from_x_f32(&x_use, d, eps);

    let pb = PyArray2::<f32>::zeros(py, [k+1, n], false);
    let mut pbm = unsafe { pb.as_array_mut() };
    for j in 0..(k+1) {
        for i in 0..n {
            pbm[(j,i)] = pb_vec[j*n + i];
        }
    }
    Ok((pb, k))
}


#[pyfunction]
fn whittaker_solve_f64<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y_st: PyReadonlyArray2<'py, f64>,                 // (S,T)
    w_base_st: Option<PyReadonlyArray2<'py, f64>>,    // (S,T) or None
    lam: f64,
    d: usize,
    ridge: f64,
    normalize: Option<&str>,
    eps: f64,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x = x.as_slice()?;
    let y = y_st.as_array();
    let (s,t) = y.dim();
    if t != x.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("y_st shape must be (S,T) with T=len(x)"));
    }
    for i in 0..(x.len()-1) {
        if !(x[i+1] > x[i]) {
            return Err(pyo3::exceptions::PyValueError::new_err("x must be strictly increasing"));
        }
    }

    let x_use = if let Some(m) = normalize { normalize_x_unit_step_f64(x, m) } else { x.to_vec() };
    let (pb_vec, k, n) = build_pb_from_x_f64(&x_use, d, eps);

    // base weights (force NaNs to 0)
    let wbase: Array2<f64> = if let Some(wb) = w_base_st {
        let wb = wb.as_array();
        if wb.dim() != (s,t) {
            return Err(pyo3::exceptions::PyValueError::new_err("w_base_st must match y_st shape"));
        }
        let mut out = wb.to_owned();
        for si in 0..s {
            for i in 0..n {
                if !y[(si,i)].is_finite() { out[(si,i)] = 0.0; }
            }
        }
        out
    } else {
        let mut out = Array2::<f64>::zeros((s,n));
        for si in 0..s {
            for i in 0..n { out[(si,i)] = if y[(si,i)].is_finite() { 1.0 } else { 0.0 }; }
        }
        out
    };

    let z_out = PyArray2::<f64>::zeros(py, [s, n], false);
    let mut z = unsafe { z_out.as_array_mut() };

    let solve_series = |si: usize, z_row: &mut [f64]| {
        let yrow = y.row(si);
        let wrow = wbase.row(si);

        let mut wv = vec![0.0f64; n];
        let mut yv = vec![0.0f64; n];
        for i in 0..n {
            let wt = wrow[i];
            wv[i] = wt;
            yv[i] = if wt > 0.0 && yrow[i].is_finite() { yrow[i] } else { 0.0 };
        }
        solve_one_series_f64(&pb_vec, n, k, &wv, &yv, lam, ridge, eps, z_row);
    };

    if parallel && s >= 2 {
        let z_slice = z.as_slice_mut().unwrap();
        z_slice
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(si, z_row)| solve_series(si, z_row));
    } else {
        for si in 0..s {
            let mut zr = z.row_mut(si);
            solve_series(si, zr.as_slice_mut().unwrap());
        }
    }

    Ok(z_out)
}

#[pyfunction]
fn whittaker_solve_f32<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    y_st: PyReadonlyArray2<'py, f32>,                 // (S,T)
    w_base_st: Option<PyReadonlyArray2<'py, f32>>,    // (S,T) or None
    lam: f32,
    d: usize,
    ridge: f32,
    normalize: Option<&str>,
    eps: f32,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let x = x.as_slice()?;
    let y = y_st.as_array();
    let (s,t) = y.dim();
    if t != x.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("y_st shape must be (S,T) with T=len(x)"));
    }
    for i in 0..(x.len()-1) {
        if !(x[i+1] > x[i]) {
            return Err(pyo3::exceptions::PyValueError::new_err("x must be strictly increasing"));
        }
    }

    let x_use = if let Some(m) = normalize { normalize_x_unit_step_f32(x, m) } else { x.to_vec() };
    let (pb_vec, k, n) = build_pb_from_x_f32(&x_use, d, eps);

    // base weights (force NaNs to 0)
    let wbase: Array2<f32> = if let Some(wb) = w_base_st {
        let wb = wb.as_array();
        if wb.dim() != (s,t) {
            return Err(pyo3::exceptions::PyValueError::new_err("w_base_st must match y_st shape"));
        }
        let mut out = wb.to_owned();
        for si in 0..s {
            for i in 0..n {
                if !y[(si,i)].is_finite() { out[(si,i)] = 0.0; }
            }
        }
        out
    } else {
        let mut out = Array2::<f32>::zeros((s,n));
        for si in 0..s {
            for i in 0..n { out[(si,i)] = if y[(si,i)].is_finite() { 1.0 } else { 0.0 }; }
        }
        out
    };

    let z_out = PyArray2::<f32>::zeros(py, [s, n], false);
    let mut z = unsafe { z_out.as_array_mut() };

    let solve_series = |si: usize, z_row: &mut [f32]| {
        let yrow = y.row(si);
        let wrow = wbase.row(si);

        let mut wv = vec![0.0f32; n];
        let mut yv = vec![0.0f32; n];
        for i in 0..n {
            let wt = wrow[i];
            wv[i] = wt;
            yv[i] = if wt > 0.0 && yrow[i].is_finite() { yrow[i] } else { 0.0 };
        }
        solve_one_series_f32(&pb_vec, n, k, &wv, &yv, lam, ridge, eps, z_row);
    };

    if parallel && s >= 2 {
        let z_slice = z.as_slice_mut().unwrap();
        z_slice
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(si, z_row)| solve_series(si, z_row));
    } else {
        for si in 0..s {
            let mut zr = z.row_mut(si);
            solve_series(si, zr.as_slice_mut().unwrap());
        }
    }

    Ok(z_out)
}

#[pyfunction]
fn robust_whittaker_irls_f64<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y_st: PyReadonlyArray2<'py, f64>,                 // (S,T)
    w_base_st: Option<PyReadonlyArray2<'py, f64>>,    // (S,T) or None
    lam: f64,
    d: usize,
    iterations: usize,
    weighting: &str,
    scale: &str,
    ridge: f64,
    normalize: Option<&str>,
    eps: f64,
    parallel: bool,
    tuning: Option<(f64,f64,f64)>,
    return_weights: bool,
) -> PyResult<Py<PyAny>> {
    let x = x.as_slice()?;
    let y = y_st.as_array();
    let (s,t) = y.dim();
    if t != x.len() { return Err(pyo3::exceptions::PyValueError::new_err("y_st shape must be (S,T) with T=len(x)")); }
    for i in 0..(x.len()-1) {
        if !(x[i+1] > x[i]) { return Err(pyo3::exceptions::PyValueError::new_err("x must be strictly increasing")); }
    }

    let w_enum = parse_weighting(weighting).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let tun = tuning.unwrap_or_else(|| default_tuning(w_enum));

    let x_use = if let Some(m) = normalize { normalize_x_unit_step_f64(x, m) } else { x.to_vec() };
    let (pb_vec, k, n) = build_pb_from_x_f64(&x_use, d, eps);

    // base weights
    let wbase: Array2<f64> = if let Some(wb) = w_base_st {
        let wb = wb.as_array();
        if wb.dim() != (s,t) { return Err(pyo3::exceptions::PyValueError::new_err("w_base_st must match y_st shape")); }
        let mut out = wb.to_owned();
        for si in 0..s {
            for i in 0..n {
                if !y[(si,i)].is_finite() { out[(si,i)] = 0.0; }
            }
        }
        out
    } else {
        let mut out = Array2::<f64>::zeros((s,n));
        for si in 0..s {
            for i in 0..n { out[(si,i)] = if y[(si,i)].is_finite() { 1.0 } else { 0.0 }; }
        }
        out
    };

    let (z, wtot) = robust_irls_f64(&pb_vec, n, k, y, wbase.view(), lam, ridge, iterations, w_enum, tun, scale, eps, parallel);

    let z_out = PyArray2::<f64>::zeros(py, [s,n], false);
    unsafe { z_out.as_array_mut() }.assign(&z);

    if return_weights {
        let w_out = PyArray2::<f64>::zeros(py, [s,n], false);
        unsafe { w_out.as_array_mut() }.assign(&wtot);
        Ok(PyTuple::new(py, [z_out, w_out])?.into_any().unbind())
    } else {
        Ok(z_out.into_any().unbind())
    }
}

#[pyfunction]
fn robust_whittaker_irls_f32<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    y_st: PyReadonlyArray2<'py, f32>,                 // (S,T)
    w_base_st: Option<PyReadonlyArray2<'py, f32>>,    // (S,T) or None
    lam: f32,
    d: usize,
    iterations: usize,
    weighting: &str,
    scale: &str,
    ridge: f32,
    normalize: Option<&str>,
    eps: f32,
    parallel: bool,
    tuning: Option<(f32,f32,f32)>,
    return_weights: bool,
) -> PyResult<Py<PyAny>> {
    let x = x.as_slice()?;
    let y = y_st.as_array();
    let (s,t) = y.dim();
    if t != x.len() { return Err(pyo3::exceptions::PyValueError::new_err("y_st shape must be (S,T) with T=len(x)")); }
    for i in 0..(x.len()-1) {
        if !(x[i+1] > x[i]) { return Err(pyo3::exceptions::PyValueError::new_err("x must be strictly increasing")); }
    }

    let w_enum = parse_weighting(weighting).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let tun = tuning.unwrap_or_else(|| {
        let (a,b,c) = default_tuning(w_enum);
        (a as f32, b as f32, c as f32)
    });

    let x_use = if let Some(m) = normalize { normalize_x_unit_step_f32(x, m) } else { x.to_vec() };
    let (pb_vec, k, n) = build_pb_from_x_f32(&x_use, d, eps);

    // base weights
    let wbase: Array2<f32> = if let Some(wb) = w_base_st {
        let wb = wb.as_array();
        if wb.dim() != (s,t) { return Err(pyo3::exceptions::PyValueError::new_err("w_base_st must match y_st shape")); }
        let mut out = wb.to_owned();
        for si in 0..s {
            for i in 0..n {
                if !y[(si,i)].is_finite() { out[(si,i)] = 0.0; }
            }
        }
        out
    } else {
        let mut out = Array2::<f32>::zeros((s,n));
        for si in 0..s {
            for i in 0..n { out[(si,i)] = if y[(si,i)].is_finite() { 1.0 } else { 0.0 }; }
        }
        out
    };

    let (z, wtot) = robust_irls_f32(&pb_vec, n, k, y, wbase.view(), lam, ridge, iterations, w_enum, tun, scale, eps, parallel);

    let z_out = PyArray2::<f32>::zeros(py, [s,n], false);
    unsafe { z_out.as_array_mut() }.assign(&z);

    if return_weights {
        let w_out = PyArray2::<f32>::zeros(py, [s,n], false);
        unsafe { w_out.as_array_mut() }.assign(&wtot);
        Ok(PyTuple::new(py, [z_out, w_out])?.into_any().unbind())
    } else {
        Ok(z_out.into_any().unbind())
    }
}

#[pymodule]
fn whitsmooth_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_pb_f64, m)?)?;
    m.add_function(wrap_pyfunction!(build_pb_f32, m)?)?;
    m.add_function(wrap_pyfunction!(whittaker_solve_f64, m)?)?;
    m.add_function(wrap_pyfunction!(whittaker_solve_f32, m)?)?;
    m.add_function(wrap_pyfunction!(robust_whittaker_irls_f64, m)?)?;
    m.add_function(wrap_pyfunction!(robust_whittaker_irls_f32, m)?)?;
    Ok(())
}
