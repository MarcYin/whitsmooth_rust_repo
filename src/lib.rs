
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

fn compress_x_f64(x: &[f64], tol: f64) -> (Vec<f64>, Vec<(usize, usize)>) {
    let n = x.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if !(tol > 0.0) {
        let mut ranges = Vec::with_capacity(n);
        for i in 0..n {
            ranges.push((i, i + 1));
        }
        return (x.to_vec(), ranges);
    }

    let mut x_comp = Vec::new();
    let mut ranges = Vec::new();

    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && (x[end] - x[end - 1]) <= tol {
            end += 1;
        }

        let mut sum = 0.0;
        for v in &x[start..end] {
            sum += *v;
        }
        x_comp.push(sum / ((end - start) as f64));
        ranges.push((start, end));
        start = end;
    }

    (x_comp, ranges)
}

fn compress_x_f32(x: &[f32], tol: f32) -> (Vec<f32>, Vec<(usize, usize)>) {
    let n = x.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if !(tol > 0.0) {
        let mut ranges = Vec::with_capacity(n);
        for i in 0..n {
            ranges.push((i, i + 1));
        }
        return (x.to_vec(), ranges);
    }

    let mut x_comp = Vec::new();
    let mut ranges = Vec::new();

    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && (x[end] - x[end - 1]) <= tol {
            end += 1;
        }

        let mut sum = 0.0_f64;
        for v in &x[start..end] {
            sum += *v as f64;
        }
        x_comp.push((sum / ((end - start) as f64)) as f32);
        ranges.push((start, end));
        start = end;
    }

    (x_comp, ranges)
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

    if rhs.iter().all(|v| v.is_finite()) {
        out.copy_from_slice(&rhs);
        return;
    }

    // Fallback to mixed-precision (f64) solve when the f32 factorization becomes unstable.
    solve_one_series_f32_f64(pb, n, k, w, y, lam, ridge, eps, out);
}

fn solve_one_series_f32_f64(
    pb: &[f32], n: usize, k: usize,
    w: &[f32], y: &[f32],
    lam: f32, ridge: f32, eps: f32,
    out: &mut [f32],
) {
    let lam = lam as f64;
    let ridge = ridge as f64;
    let eps = (eps as f64).max(f64::EPSILON);

    let mut lb = vec![0.0f64; (k+1)*n];
    let mut rhs = vec![0.0f64; n];
    for i in 0..n { rhs[i] = (w[i] as f64) * (y[i] as f64); }

    for i in 0..n {
        let mut diag = (w[i] as f64) + lam*(pb[0*n + i] as f64) + ridge;
        let tmax = i.min(k);
        for t in 1..=tmax {
            let v = lb[t*n + i];
            diag -= v*v;
        }
        let lii = diag.max(eps).sqrt();
        lb[0*n + i] = lii;

        let rmax = (n-1-i).min(k);
        for r in 1..=rmax {
            let mut num = lam * (pb[r*n + (i+r)] as f64);
            let tmax2 = tmax.min(k-r);
            for t in 1..=tmax2 {
                let a = lb[(r+t)*n + (i+r)];
                let b = lb[t*n + i];
                num -= a*b;
            }
            lb[r*n + (i+r)] = num / lii.max(eps);
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

    for i in 0..n {
        out[i] = rhs[i] as f32;
    }
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

/// Build the lower-banded penalty matrix `Pb` for divided differences (`P = D.T @ D`).
///
/// Parameters
/// ----------
/// x : ndarray[float64], shape (T,)
///     Strictly increasing sampling positions.
/// d : int, default=2
///     Divided-difference order. The penalty bandwidth is `k = 2*d`.
/// normalize : {"mean","median"} or None, optional
///     Optionally normalize `x` to ~unit spacing before building `Pb`.
/// eps : float, optional
///     Small floor to avoid division by zero in divided differences.
///
/// Returns
/// -------
/// pb : ndarray[float64], shape (k+1, T)
///     Lower band of `P` stored as `pb[j, i] = P[i, i-j]`.
/// k : int
///     Half-bandwidth, `k = 2*d`.
#[pyfunction]
#[pyo3(signature = (x, d=2, normalize=None, eps=None))]
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

/// Build the lower-banded penalty matrix `Pb` for divided differences (`P = D.T @ D`).
///
/// Parameters
/// ----------
/// x : ndarray[float32], shape (T,)
///     Strictly increasing sampling positions.
/// d : int, default=2
///     Divided-difference order. The penalty bandwidth is `k = 2*d`.
/// normalize : {"mean","median"} or None, optional
///     Optionally normalize `x` to ~unit spacing before building `Pb`.
/// eps : float, optional
///     Small floor to avoid division by zero in divided differences.
///
/// Returns
/// -------
/// pb : ndarray[float32], shape (k+1, T)
///     Lower band of `P` stored as `pb[j, i] = P[i, i-j]`.
/// k : int
///     Half-bandwidth, `k = 2*d`.
#[pyfunction]
#[pyo3(signature = (x, d=2, normalize=None, eps=None))]
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

/// Solve a single-pass (non-robust) Whittaker smoother for each series.
///
/// Notes
/// -----
/// Solves, per series, the linear system:
/// `(diag(w) + lam * D.T @ D + ridge * I) z = w * y`.
///
/// NaNs in `y_st` are treated as missing (weight forced to 0).
///
/// Parameters
/// ----------
/// x : ndarray[float64], shape (T,)
/// y_st : ndarray[float64], shape (S, T)
/// w_base_st : ndarray[float64], shape (S, T) or None, optional
/// lam : float, default=10.0
/// d : int, default=2
/// ridge : float, default=1e-10
/// normalize : {"mean","median"} or None, default="mean"
/// eps : float, default=1e-10
/// parallel : bool, default=True
///
/// Returns
/// -------
/// z_st : ndarray[float64], shape (S, T)
#[pyfunction]
#[pyo3(signature = (x, y_st, w_base_st=None, lam=10.0, d=2, ridge=1e-10, normalize="mean", eps=1e-10, parallel=true))]
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

/// Solve a single-pass (non-robust) Whittaker smoother for each series.
///
/// Notes
/// -----
/// Solves, per series, the linear system:
/// `(diag(w) + lam * D.T @ D + ridge * I) z = w * y`.
///
/// NaNs in `y_st` are treated as missing (weight forced to 0).
///
/// Parameters
/// ----------
/// x : ndarray[float32], shape (T,)
/// y_st : ndarray[float32], shape (S, T)
/// w_base_st : ndarray[float32], shape (S, T) or None, optional
/// lam : float, default=10.0
/// d : int, default=2
/// ridge : float, default=1e-10
/// normalize : {"mean","median"} or None, default="mean"
/// eps : float, default=1e-10
/// parallel : bool, default=True
///
/// Returns
/// -------
/// z_st : ndarray[float32], shape (S, T)
#[pyfunction]
#[pyo3(signature = (x, y_st, w_base_st=None, lam=10.0_f32, d=2, ridge=1e-10_f32, normalize="mean", eps=1e-10_f32, parallel=true))]
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

/// Robust Whittaker smoother via IRLS (float64).
///
/// Notes
/// -----
/// Runs `iterations` steps of IRLS to down-weight outliers using the chosen `weighting`
/// function and scale estimator (`scale="mad"` or `"huber"`).
///
/// NaNs in `y_st` are treated as missing (weight forced to 0).
///
/// If `merge_x_tol > 0`, adjacent `x` values within that spacing are merged (in input-x units)
/// using a base-weighted mean of observations. The solve runs on the compressed grid and the
/// output is expanded back to the original `(S, T)` shape.
///
/// Parameters
/// ----------
/// x : ndarray[float64], shape (T,)
/// y_st : ndarray[float64], shape (S, T)
/// w_base_st : ndarray[float64], shape (S, T) or None, optional
/// lam : float, default=10.0
/// d : int, default=2
/// iterations : int, default=1
/// weighting : {"tukey","huber","cauchy","welsch","fair","hampel"}, default="tukey"
/// scale : {"mad","huber"}, default="mad"
/// ridge : float, default=1e-10
/// normalize : {"mean","median"} or None, default="mean"
/// eps : float, default=1e-10
/// parallel : bool, default=True
/// tuning : tuple or None, optional
/// return_weights : bool, default=False
/// merge_x_tol : float or None, default=1e-2
///     Merge adjacent `x` points when `x[i+1]-x[i] <= merge_x_tol`. Set to `None` (or `0.0`) to disable.
///
/// Returns
/// -------
/// z_st : ndarray[float64], shape (S, T)
/// or (z_st, w_total_st) if return_weights=True
#[pyfunction]
#[pyo3(signature = (x, y_st, w_base_st=None, lam=10.0, d=2, iterations=1, weighting="tukey", scale="mad", ridge=1e-10, normalize="mean", eps=1e-10, parallel=true, tuning=None, return_weights=false, merge_x_tol=1e-2))]
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
    merge_x_tol: Option<f64>,
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

    let merge_tol = merge_x_tol.unwrap_or(0.0);
    let n_orig = x.len();
    let (x_comp, ranges) = compress_x_f64(x, merge_tol);
    let n_work = x_comp.len();
    if n_work < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("x must have length >=2 after merging"));
    }
    if n_work <= d {
        return Err(pyo3::exceptions::PyValueError::new_err("x must have length > d after merging"));
    }
    for i in 0..(n_work - 1) {
        if !(x_comp[i + 1] > x_comp[i]) {
            return Err(pyo3::exceptions::PyValueError::new_err("x must be strictly increasing after merging"));
        }
    }

    let x_use = if let Some(m) = normalize { normalize_x_unit_step_f64(&x_comp, m) } else { x_comp };
    let (pb_vec, k, n) = build_pb_from_x_f64(&x_use, d, eps);

    // base weights
    let wbase: Array2<f64> = if let Some(wb) = w_base_st {
        let wb = wb.as_array();
        if wb.dim() != (s,t) { return Err(pyo3::exceptions::PyValueError::new_err("w_base_st must match y_st shape")); }
        let mut out = wb.to_owned();
        for si in 0..s {
            for i in 0..n_orig {
                if !y[(si,i)].is_finite() { out[(si,i)] = 0.0; }
            }
        }
        out
    } else {
        let mut out = Array2::<f64>::zeros((s,n_orig));
        for si in 0..s {
            for i in 0..n_orig { out[(si,i)] = if y[(si,i)].is_finite() { 1.0 } else { 0.0 }; }
        }
        out
    };

    let (z_work, wtot_work, wbase_work): (Array2<f64>, Array2<f64>, Option<Array2<f64>>) = if n == n_orig {
        let (z, wtot) = robust_irls_f64(&pb_vec, n, k, y, wbase.view(), lam, ridge, iterations, w_enum, tun, scale, eps, parallel);
        (z, wtot, None)
    } else {
        let mut y_merge = Array2::<f64>::zeros((s, n));
        let mut w_merge = Array2::<f64>::zeros((s, n));

        for si in 0..s {
            for (gi, &(start, end)) in ranges.iter().enumerate() {
                let mut sum_w = 0.0;
                let mut sum_wy = 0.0;
                for i in start..end {
                    let wb = wbase[(si, i)];
                    let yv = y[(si, i)];
                    if yv.is_finite() && wb.is_finite() && wb > 0.0 {
                        sum_w += wb;
                        sum_wy += wb * yv;
                    }
                }
                if sum_w > 0.0 {
                    y_merge[(si, gi)] = sum_wy / sum_w;
                    w_merge[(si, gi)] = sum_w;
                }
            }
        }

        let (z, wtot) = robust_irls_f64(
            &pb_vec,
            n,
            k,
            y_merge.view(),
            w_merge.view(),
            lam,
            ridge,
            iterations,
            w_enum,
            tun,
            scale,
            eps,
            parallel,
        );
        (z, wtot, Some(w_merge))
    };

    let (z_final, wtot_final) = if n == n_orig {
        (z_work, wtot_work)
    } else {
        let wbase_work = wbase_work.expect("wbase_work must exist when merging");
        let mut z_exp = Array2::<f64>::zeros((s, n_orig));
        let mut w_exp = Array2::<f64>::zeros((s, n_orig));

        for (gi, &(start, end)) in ranges.iter().enumerate() {
            for si in 0..s {
                let zg = z_work[(si, gi)];
                let wtg = wtot_work[(si, gi)];
                let wbg = wbase_work[(si, gi)];
                for i in start..end {
                    z_exp[(si, i)] = zg;
                    w_exp[(si, i)] = if wbg > 0.0 { wtg * (wbase[(si, i)] / wbg) } else { 0.0 };
                }
            }
        }

        (z_exp, w_exp)
    };

    let z_out = PyArray2::<f64>::zeros(py, [s,n_orig], false);
    unsafe { z_out.as_array_mut() }.assign(&z_final);

    if return_weights {
        let w_out = PyArray2::<f64>::zeros(py, [s,n_orig], false);
        unsafe { w_out.as_array_mut() }.assign(&wtot_final);
        Ok(PyTuple::new(py, [z_out, w_out])?.into_any().unbind())
    } else {
        Ok(z_out.into_any().unbind())
    }
}

/// Robust Whittaker smoother via IRLS (float32).
///
/// Notes
/// -----
/// Same API as `robust_whittaker_irls_f64`, but returns float32 output. For highly irregular or
/// near-duplicate `x`, the internal float32 solve can become numerically unstable; this function
/// automatically falls back to a float64 solve when needed (and casts back to float32).
///
/// If `merge_x_tol > 0`, adjacent `x` values within that spacing are merged (in input-x units)
/// using a base-weighted mean of observations. The solve runs on the compressed grid and the
/// output is expanded back to the original `(S, T)` shape.
///
/// Parameters
/// ----------
/// x : ndarray[float32], shape (T,)
/// y_st : ndarray[float32], shape (S, T)
/// w_base_st : ndarray[float32], shape (S, T) or None, optional
/// lam : float, default=10.0
/// d : int, default=2
/// iterations : int, default=1
/// weighting : {"tukey","huber","cauchy","welsch","fair","hampel"}, default="tukey"
/// scale : {"mad","huber"}, default="mad"
/// ridge : float, default=1e-10
/// normalize : {"mean","median"} or None, default="mean"
/// eps : float, default=1e-10
/// parallel : bool, default=True
/// tuning : tuple or None, optional
/// return_weights : bool, default=False
/// merge_x_tol : float or None, default=1e-2
///     Merge adjacent `x` points when `x[i+1]-x[i] <= merge_x_tol`. Set to `None` (or `0.0`) to disable.
///
/// Returns
/// -------
/// z_st : ndarray[float32], shape (S, T)
/// or (z_st, w_total_st) if return_weights=True
#[pyfunction]
#[pyo3(signature = (x, y_st, w_base_st=None, lam=10.0_f32, d=2, iterations=1, weighting="tukey", scale="mad", ridge=1e-10_f32, normalize="mean", eps=1e-10_f32, parallel=true, tuning=None, return_weights=false, merge_x_tol=1e-2_f32))]
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
    merge_x_tol: Option<f32>,
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

    let merge_tol = merge_x_tol.unwrap_or(0.0);
    let n_orig = x.len();
    let (x_comp, ranges) = compress_x_f32(x, merge_tol);
    let n_work = x_comp.len();
    if n_work < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("x must have length >=2 after merging"));
    }
    if n_work <= d {
        return Err(pyo3::exceptions::PyValueError::new_err("x must have length > d after merging"));
    }
    for i in 0..(n_work - 1) {
        if !(x_comp[i + 1] > x_comp[i]) {
            return Err(pyo3::exceptions::PyValueError::new_err("x must be strictly increasing after merging"));
        }
    }

    let x_use = if let Some(m) = normalize { normalize_x_unit_step_f32(&x_comp, m) } else { x_comp };
    let (pb_vec, k, n) = build_pb_from_x_f32(&x_use, d, eps);

    // base weights
    let wbase: Array2<f32> = if let Some(wb) = w_base_st {
        let wb = wb.as_array();
        if wb.dim() != (s,t) { return Err(pyo3::exceptions::PyValueError::new_err("w_base_st must match y_st shape")); }
        let mut out = wb.to_owned();
        for si in 0..s {
            for i in 0..n_orig {
                if !y[(si,i)].is_finite() { out[(si,i)] = 0.0; }
            }
        }
        out
    } else {
        let mut out = Array2::<f32>::zeros((s,n_orig));
        for si in 0..s {
            for i in 0..n_orig { out[(si,i)] = if y[(si,i)].is_finite() { 1.0 } else { 0.0 }; }
        }
        out
    };

    let mut y_merge: Option<Array2<f32>> = None;
    let mut w_merge: Option<Array2<f32>> = None;

    let (mut z_work, mut wtot_work) = if n == n_orig {
        robust_irls_f32(&pb_vec, n, k, y, wbase.view(), lam, ridge, iterations, w_enum, tun, scale, eps, parallel)
    } else {
        let mut y_m = Array2::<f32>::zeros((s, n));
        let mut w_m = Array2::<f32>::zeros((s, n));

        for si in 0..s {
            for (gi, &(start, end)) in ranges.iter().enumerate() {
                let mut sum_w = 0.0_f64;
                let mut sum_wy = 0.0_f64;
                for i in start..end {
                    let wb = wbase[(si, i)] as f64;
                    let yv = y[(si, i)] as f64;
                    if yv.is_finite() && wb.is_finite() && wb > 0.0 {
                        sum_w += wb;
                        sum_wy += wb * yv;
                    }
                }
                if sum_w > 0.0 {
                    y_m[(si, gi)] = (sum_wy / sum_w) as f32;
                    w_m[(si, gi)] = sum_w as f32;
                }
            }
        }

        y_merge = Some(y_m);
        w_merge = Some(w_m);

        let y_m = y_merge.as_ref().unwrap();
        let w_m = w_merge.as_ref().unwrap();
        robust_irls_f32(
            &pb_vec,
            n,
            k,
            y_m.view(),
            w_m.view(),
            lam,
            ridge,
            iterations,
            w_enum,
            tun,
            scale,
            eps,
            parallel,
        )
    };

    // f32 can become numerically unstable for very uneven/near-duplicate x spacing. If that happens,
    // fall back to an f64 solve (using an f64-built penalty) and cast back to f32.
    let need_f64 = !z_work.iter().all(|v| v.is_finite()) || !wtot_work.iter().all(|v| v.is_finite());
    if need_f64 {
        let x_use64: Vec<f64> = x_use.iter().map(|&v| v as f64).collect();
        let (pb_vec64, k64, n64) = build_pb_from_x_f64(&x_use64, d, eps as f64);

        let (y64, wbase64) = if n == n_orig {
            (y.mapv(|v| v as f64), wbase.mapv(|v| v as f64))
        } else {
            let y_m = y_merge.as_ref().unwrap();
            let w_m = w_merge.as_ref().unwrap();
            (y_m.mapv(|v| v as f64), w_m.mapv(|v| v as f64))
        };
        let tun64 = (tun.0 as f64, tun.1 as f64, tun.2 as f64);

        let (z64, wtot64) = robust_irls_f64(
            &pb_vec64,
            n64,
            k64,
            y64.view(),
            wbase64.view(),
            lam as f64,
            ridge as f64,
            iterations,
            w_enum,
            tun64,
            scale,
            eps as f64,
            parallel,
        );

        z_work = z64.mapv(|v| v as f32);
        wtot_work = wtot64.mapv(|v| v as f32);

        if n64 != n || k64 != k || !z_work.iter().all(|v| v.is_finite()) || !wtot_work.iter().all(|v| v.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Numerical instability in f32 solver; try robust_whittaker_irls_f64 or increase eps/ridge",
            ));
        }
    }

    let (z_final, wtot_final) = if n == n_orig {
        (z_work, wtot_work)
    } else {
        let w_merge = w_merge.as_ref().unwrap();
        let mut z_exp = Array2::<f32>::zeros((s, n_orig));
        let mut w_exp = Array2::<f32>::zeros((s, n_orig));

        for (gi, &(start, end)) in ranges.iter().enumerate() {
            for si in 0..s {
                let zg = z_work[(si, gi)];
                let wtg = wtot_work[(si, gi)];
                let wbg = w_merge[(si, gi)];
                for i in start..end {
                    z_exp[(si, i)] = zg;
                    w_exp[(si, i)] = if wbg > 0.0 { wtg * (wbase[(si, i)] / wbg) } else { 0.0 };
                }
            }
        }
        (z_exp, w_exp)
    };

    let z_out = PyArray2::<f32>::zeros(py, [s,n_orig], false);
    unsafe { z_out.as_array_mut() }.assign(&z_final);

    if return_weights {
        let w_out = PyArray2::<f32>::zeros(py, [s,n_orig], false);
        unsafe { w_out.as_array_mut() }.assign(&wtot_final);
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

#[cfg(test)]
mod tests {
    use super::{
        build_pb_from_x_f32,
        build_pb_from_x_f64,
        compress_x_f32,
        normalize_x_unit_step_f32,
        normalize_x_unit_step_f64,
        robust_irls_f32,
        robust_irls_f64,
        Weighting,
    };
    use ndarray::Array2;

    #[test]
    fn robust_irls_f32_fallback_no_nan_for_near_duplicate_x() {
        let x: [f32; 53] = [
            0.00000000e+00_f32, 2.63310187e-02_f32, 3.02673602e+00_f32, 4.02767372e+00_f32,
            1.50279627e+01_f32, 1.69971294e+01_f32, 1.70008335e+01_f32, 1.70262032e+01_f32,
            1.90286694e+01_f32, 2.70253944e+01_f32, 2.70323620e+01_f32, 3.10246525e+01_f32,
            3.20236130e+01_f32, 3.20236473e+01_f32, 3.20317345e+01_f32, 3.20317574e+01_f32,
            3.30048599e+01_f32, 3.40006714e+01_f32, 3.40007744e+01_f32, 3.50341187e+01_f32,
            3.60265388e+01_f32, 3.70327072e+01_f32, 3.70340500e+01_f32, 3.80329742e+01_f32,
            4.00268860e+01_f32, 4.00287170e+01_f32, 4.10296173e+01_f32, 4.10329170e+01_f32,
            5.80326042e+01_f32, 6.00260086e+01_f32, 6.70268250e+01_f32, 7.50326004e+01_f32,
            7.80259247e+01_f32, 7.80322571e+01_f32, 8.00325699e+01_f32, 8.20315170e+01_f32,
            1.09034401e+02_f32, 1.18034927e+02_f32, 1.21032341e+02_f32, 1.24026566e+02_f32,
            1.24027306e+02_f32, 1.25026375e+02_f32, 1.25027596e+02_f32, 1.31028046e+02_f32,
            1.32027206e+02_f32, 1.33027740e+02_f32, 1.34028137e+02_f32, 1.44033661e+02_f32,
            1.45013535e+02_f32, 1.46011627e+02_f32, 1.46025375e+02_f32, 1.46025406e+02_f32,
            1.46029343e+02_f32,
        ];
        let y: [f32; 53] = [
            0.6290387_f32, 0.6304045_f32, 0.6591430_f32, 0.64829564_f32, 0.6402782_f32,
            0.6406342_f32, 0.64828616_f32, 0.6303828_f32, 0.6483001_f32, 0.6462810_f32,
            0.6514024_f32, 0.61600894_f32, 0.6621996_f32, 0.6100803_f32, 0.6748406_f32,
            0.6165653_f32, 0.53618_f32, f32::NAN, 0.5469687_f32, 0.58047014_f32,
            0.60227007_f32, 0.3830874_f32, 0.5945248_f32, 0.46407568_f32, 0.3776188_f32,
            0.56584877_f32, 0.5451473_f32, 0.59275806_f32, 0.5730731_f32, 0.5221283_f32,
            0.46862745_f32, 0.65456235_f32, 0.6656672_f32, 0.68882096_f32, 0.67919797_f32,
            0.6992728_f32, 0.76031214_f32, 0.7839335_f32, 0.7466595_f32, 0.6589886_f32,
            0.66906095_f32, 0.6403982_f32, 0.6013683_f32, 0.7091882_f32, 0.6865953_f32,
            0.6635514_f32, 0.6495238_f32, 0.6129259_f32, f32::NAN, 0.7133333_f32,
            0.6148909_f32, 0.60245466_f32, 0.63146067_f32,
        ];

        let x_use = normalize_x_unit_step_f32(&x, "mean");
        let (pb_vec, k, n) = build_pb_from_x_f32(&x_use, 2, 1e-10_f32);

        let mut y_st = Array2::<f32>::zeros((1, n));
        let mut wbase_st = Array2::<f32>::zeros((1, n));
        for i in 0..n {
            y_st[(0, i)] = y[i];
            wbase_st[(0, i)] = if y[i].is_finite() { 1.0 } else { 0.0 };
        }

        let (z0, wtot0) = robust_irls_f32(
            &pb_vec,
            n,
            k,
            y_st.view(),
            wbase_st.view(),
            10.0_f32,
            1e-10_f32,
            1,
            Weighting::Tukey,
            (0.0_f32, 0.0_f32, 4.685_f32),
            "mad",
            1e-10_f32,
            false,
        );

        let need_f64 = !z0.iter().all(|v| v.is_finite()) || !wtot0.iter().all(|v| v.is_finite());
        let (z, wtot) = if !need_f64 {
            (z0, wtot0)
        } else {
            let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            let x_use64 = normalize_x_unit_step_f64(&x64, "mean");
            let (pb_vec64, k64, n64) = build_pb_from_x_f64(&x_use64, 2, 1e-10_f64);
            assert_eq!(k64, k);
            assert_eq!(n64, n);

            let y64 = y_st.mapv(|v| v as f64);
            let wbase64 = wbase_st.mapv(|v| v as f64);
            let (z64, wtot64) = robust_irls_f64(
                &pb_vec64,
                n64,
                k64,
                y64.view(),
                wbase64.view(),
                10.0_f64,
                1e-10_f64,
                1,
                Weighting::Tukey,
                (0.0_f64, 0.0_f64, 4.685_f64),
                "mad",
                1e-10_f64,
                false,
            );
            (z64.mapv(|v| v as f32), wtot64.mapv(|v| v as f32))
        };

        assert!(z.iter().all(|v| v.is_finite()), "z contains non-finite values");
        assert!(wtot.iter().all(|v| v.is_finite()), "weights contain non-finite values");
    }

    #[test]
    fn robust_irls_f32_merge_x_tol_no_nan() {
        let x: [f32; 53] = [
            0.00000000e+00_f32, 2.63310187e-02_f32, 3.02673602e+00_f32, 4.02767372e+00_f32,
            1.50279627e+01_f32, 1.69971294e+01_f32, 1.70008335e+01_f32, 1.70262032e+01_f32,
            1.90286694e+01_f32, 2.70253944e+01_f32, 2.70323620e+01_f32, 3.10246525e+01_f32,
            3.20236130e+01_f32, 3.20236473e+01_f32, 3.20317345e+01_f32, 3.20317574e+01_f32,
            3.30048599e+01_f32, 3.40006714e+01_f32, 3.40007744e+01_f32, 3.50341187e+01_f32,
            3.60265388e+01_f32, 3.70327072e+01_f32, 3.70340500e+01_f32, 3.80329742e+01_f32,
            4.00268860e+01_f32, 4.00287170e+01_f32, 4.10296173e+01_f32, 4.10329170e+01_f32,
            5.80326042e+01_f32, 6.00260086e+01_f32, 6.70268250e+01_f32, 7.50326004e+01_f32,
            7.80259247e+01_f32, 7.80322571e+01_f32, 8.00325699e+01_f32, 8.20315170e+01_f32,
            1.09034401e+02_f32, 1.18034927e+02_f32, 1.21032341e+02_f32, 1.24026566e+02_f32,
            1.24027306e+02_f32, 1.25026375e+02_f32, 1.25027596e+02_f32, 1.31028046e+02_f32,
            1.32027206e+02_f32, 1.33027740e+02_f32, 1.34028137e+02_f32, 1.44033661e+02_f32,
            1.45013535e+02_f32, 1.46011627e+02_f32, 1.46025375e+02_f32, 1.46025406e+02_f32,
            1.46029343e+02_f32,
        ];
        let y: [f32; 53] = [
            0.6290387_f32, 0.6304045_f32, 0.6591430_f32, 0.64829564_f32, 0.6402782_f32,
            0.6406342_f32, 0.64828616_f32, 0.6303828_f32, 0.6483001_f32, 0.6462810_f32,
            0.6514024_f32, 0.61600894_f32, 0.6621996_f32, 0.6100803_f32, 0.6748406_f32,
            0.6165653_f32, 0.53618_f32, f32::NAN, 0.5469687_f32, 0.58047014_f32,
            0.60227007_f32, 0.3830874_f32, 0.5945248_f32, 0.46407568_f32, 0.3776188_f32,
            0.56584877_f32, 0.5451473_f32, 0.59275806_f32, 0.5730731_f32, 0.5221283_f32,
            0.46862745_f32, 0.65456235_f32, 0.6656672_f32, 0.68882096_f32, 0.67919797_f32,
            0.6992728_f32, 0.76031214_f32, 0.7839335_f32, 0.7466595_f32, 0.6589886_f32,
            0.66906095_f32, 0.6403982_f32, 0.6013683_f32, 0.7091882_f32, 0.6865953_f32,
            0.6635514_f32, 0.6495238_f32, 0.6129259_f32, f32::NAN, 0.7133333_f32,
            0.6148909_f32, 0.60245466_f32, 0.63146067_f32,
        ];

        let (x_comp, ranges) = compress_x_f32(&x, 1e-2_f32);
        assert!(x_comp.len() >= 2);
        assert_eq!(ranges.len(), x_comp.len());

        let x_use = normalize_x_unit_step_f32(&x_comp, "mean");
        let (pb_vec, k, n) = build_pb_from_x_f32(&x_use, 2, 1e-10_f32);

        let mut y_merge = Array2::<f32>::zeros((1, n));
        let mut w_merge = Array2::<f32>::zeros((1, n));
        for (gi, &(start, end)) in ranges.iter().enumerate() {
            let mut sum_w = 0.0_f64;
            let mut sum_wy = 0.0_f64;
            for i in start..end {
                let yv = y[i] as f64;
                if yv.is_finite() {
                    sum_w += 1.0;
                    sum_wy += yv;
                }
            }
            if sum_w > 0.0 {
                y_merge[(0, gi)] = (sum_wy / sum_w) as f32;
                w_merge[(0, gi)] = sum_w as f32;
            }
        }

        let (z, wtot) = robust_irls_f32(
            &pb_vec,
            n,
            k,
            y_merge.view(),
            w_merge.view(),
            10.0_f32,
            1e-10_f32,
            1,
            Weighting::Tukey,
            (0.0_f32, 0.0_f32, 4.685_f32),
            "mad",
            1e-10_f32,
            false,
        );

        assert!(z.iter().all(|v| v.is_finite()), "z contains non-finite values");
        assert!(wtot.iter().all(|v| v.is_finite()), "weights contain non-finite values");
    }

    #[test]
    fn robust_irls_f32_merge_x_tol_expand_preserves_shape() {
        let x: [f32; 53] = [
            0.00000000e+00_f32, 2.63310187e-02_f32, 3.02673602e+00_f32, 4.02767372e+00_f32,
            1.50279627e+01_f32, 1.69971294e+01_f32, 1.70008335e+01_f32, 1.70262032e+01_f32,
            1.90286694e+01_f32, 2.70253944e+01_f32, 2.70323620e+01_f32, 3.10246525e+01_f32,
            3.20236130e+01_f32, 3.20236473e+01_f32, 3.20317345e+01_f32, 3.20317574e+01_f32,
            3.30048599e+01_f32, 3.40006714e+01_f32, 3.40007744e+01_f32, 3.50341187e+01_f32,
            3.60265388e+01_f32, 3.70327072e+01_f32, 3.70340500e+01_f32, 3.80329742e+01_f32,
            4.00268860e+01_f32, 4.00287170e+01_f32, 4.10296173e+01_f32, 4.10329170e+01_f32,
            5.80326042e+01_f32, 6.00260086e+01_f32, 6.70268250e+01_f32, 7.50326004e+01_f32,
            7.80259247e+01_f32, 7.80322571e+01_f32, 8.00325699e+01_f32, 8.20315170e+01_f32,
            1.09034401e+02_f32, 1.18034927e+02_f32, 1.21032341e+02_f32, 1.24026566e+02_f32,
            1.24027306e+02_f32, 1.25026375e+02_f32, 1.25027596e+02_f32, 1.31028046e+02_f32,
            1.32027206e+02_f32, 1.33027740e+02_f32, 1.34028137e+02_f32, 1.44033661e+02_f32,
            1.45013535e+02_f32, 1.46011627e+02_f32, 1.46025375e+02_f32, 1.46025406e+02_f32,
            1.46029343e+02_f32,
        ];
        let y: [f32; 53] = [
            0.6290387_f32, 0.6304045_f32, 0.6591430_f32, 0.64829564_f32, 0.6402782_f32,
            0.6406342_f32, 0.64828616_f32, 0.6303828_f32, 0.6483001_f32, 0.6462810_f32,
            0.6514024_f32, 0.61600894_f32, 0.6621996_f32, 0.6100803_f32, 0.6748406_f32,
            0.6165653_f32, 0.53618_f32, f32::NAN, 0.5469687_f32, 0.58047014_f32,
            0.60227007_f32, 0.3830874_f32, 0.5945248_f32, 0.46407568_f32, 0.3776188_f32,
            0.56584877_f32, 0.5451473_f32, 0.59275806_f32, 0.5730731_f32, 0.5221283_f32,
            0.46862745_f32, 0.65456235_f32, 0.6656672_f32, 0.68882096_f32, 0.67919797_f32,
            0.6992728_f32, 0.76031214_f32, 0.7839335_f32, 0.7466595_f32, 0.6589886_f32,
            0.66906095_f32, 0.6403982_f32, 0.6013683_f32, 0.7091882_f32, 0.6865953_f32,
            0.6635514_f32, 0.6495238_f32, 0.6129259_f32, f32::NAN, 0.7133333_f32,
            0.6148909_f32, 0.60245466_f32, 0.63146067_f32,
        ];

        let n_orig = x.len();
        let (x_comp, ranges) = compress_x_f32(&x, 1e-2_f32);
        let x_use = normalize_x_unit_step_f32(&x_comp, "mean");
        let (pb_vec, k, n) = build_pb_from_x_f32(&x_use, 2, 1e-10_f32);

        let mut y_merge = Array2::<f32>::zeros((1, n));
        let mut w_merge = Array2::<f32>::zeros((1, n));
        for (gi, &(start, end)) in ranges.iter().enumerate() {
            let mut sum_w = 0.0_f64;
            let mut sum_wy = 0.0_f64;
            for i in start..end {
                let yv = y[i] as f64;
                if yv.is_finite() {
                    sum_w += 1.0;
                    sum_wy += yv;
                }
            }
            if sum_w > 0.0 {
                y_merge[(0, gi)] = (sum_wy / sum_w) as f32;
                w_merge[(0, gi)] = sum_w as f32;
            }
        }

        let (mut z_work, mut wtot_work) = robust_irls_f32(
            &pb_vec,
            n,
            k,
            y_merge.view(),
            w_merge.view(),
            10.0_f32,
            1e-10_f32,
            1,
            Weighting::Tukey,
            (0.0_f32, 0.0_f32, 4.685_f32),
            "mad",
            1e-10_f32,
            false,
        );

        if !z_work.iter().all(|v| v.is_finite()) || !wtot_work.iter().all(|v| v.is_finite()) {
            let x_use64: Vec<f64> = x_use.iter().map(|&v| v as f64).collect();
            let (pb_vec64, k64, n64) = build_pb_from_x_f64(&x_use64, 2, 1e-10_f64);
            assert_eq!(k64, k);
            assert_eq!(n64, n);

            let y64 = y_merge.mapv(|v| v as f64);
            let w64 = w_merge.mapv(|v| v as f64);
            let (z64, wtot64) = robust_irls_f64(
                &pb_vec64,
                n64,
                k64,
                y64.view(),
                w64.view(),
                10.0_f64,
                1e-10_f64,
                1,
                Weighting::Tukey,
                (0.0_f64, 0.0_f64, 4.685_f64),
                "mad",
                1e-10_f64,
                false,
            );
            z_work = z64.mapv(|v| v as f32);
            wtot_work = wtot64.mapv(|v| v as f32);
        }

        let mut z_exp = Array2::<f32>::zeros((1, n_orig));
        let mut w_exp = Array2::<f32>::zeros((1, n_orig));
        for (gi, &(start, end)) in ranges.iter().enumerate() {
            let zg = z_work[(0, gi)];
            let wtg = wtot_work[(0, gi)];
            let wbg = w_merge[(0, gi)];
            let wrob = if wbg > 0.0 { wtg / wbg } else { 0.0_f32 };
            for i in start..end {
                z_exp[(0, i)] = zg;
                let wb = if y[i].is_finite() { 1.0_f32 } else { 0.0_f32 };
                w_exp[(0, i)] = wb * wrob;
            }
        }

        assert_eq!(z_exp.shape(), &[1, n_orig]);
        assert_eq!(w_exp.shape(), &[1, n_orig]);
        assert!(z_exp.iter().all(|v| v.is_finite()), "expanded z contains non-finite values");
        assert!(w_exp.iter().all(|v| v.is_finite()), "expanded weights contain non-finite values");
    }
}
