use rayon::prelude::*;
use std::alloc;
use std::alloc::{handle_alloc_error, Layout};
use std::arch::x86_64;
use std::f32::consts::PI;
use std::fmt::Formatter;
use std::mem::size_of;

struct MatrixImpl {
    ptr: *mut f32,
    pub row: usize,
    pub col: usize,
    pub real_col: usize,
}

impl MatrixImpl {
    pub unsafe fn resize_row(&mut self, x: usize) {
        let layout =
            Layout::from_size_align_unchecked(self.row * self.real_col * size_of::<f32>(), 32);
        let np = alloc::realloc(
            self.ptr as *mut u8,
            layout,
            x * self.real_col * size_of::<f32>(),
        ) as *mut f32;
        if np.is_null() {
            let layout =
                Layout::from_size_align_unchecked(x * self.real_col * size_of::<f32>(), 32);
            handle_alloc_error(layout);
        }
        self.ptr = np;
        self.row = x;
    }
    pub unsafe fn normal_init(&self) {
        let k = 1.0 / (2.0 * PI).sqrt();
        (0..self.row).into_par_iter().for_each(|index| {
            let dst = self.row_at(index as isize);
            let center = self.col / 2;
            for i in 0..self.col {
                let p = i as f32 - center as f32;
                let p = -0.5 * p * p;
                let val = k * p.exp();
                *dst.add(i) = val;
            }
        })
    }

    pub unsafe fn clamp(&self, lo: f32, hi: f32) {
        (0..self.row).into_par_iter().for_each(|index| {
            let dst = self.row_at(index as isize);
            let lo_clamp = x86_64::_mm256_set1_ps(lo);
            let hi_clamp = x86_64::_mm256_set1_ps(hi);
            for j in (0..self.col).step_by(32 / size_of::<f32>()) {
                let now = x86_64::_mm256_load_ps(dst.add(j));
                let now = x86_64::_mm256_max_ps(now, lo_clamp);
                let now = x86_64::_mm256_min_ps(now, hi_clamp);
                x86_64::_mm256_store_ps(dst.add(j), now);
            }
        })
    }

    pub unsafe fn row_at(&self, i: isize) -> *mut f32 {
        if i > self.row as isize {
            panic!("matrix visit row over bound");
        }
        self.ptr.offset(i * self.real_col as isize)
    }
    pub unsafe fn at(&self, i: isize, j: isize) -> f32 {
        if i > self.row as isize {
            panic!("matrix visit row over bound");
        }
        *self.ptr.offset(i * self.real_col as isize + j)
    }
    unsafe fn ops_with_matrix<T>(
        &self,
        rhs: &MatrixImpl,
        inplace: bool,
        f: T,
        expand: bool,
    ) -> Option<MatrixImpl>
    where
        T: std::ops::Fn(x86_64::__m256, x86_64::__m256) -> x86_64::__m256 + std::marker::Sync,
    {
        if !inplace {
            let ret = MatrixImpl::new(self.row, self.col);
            (0..self.row).into_par_iter().for_each(|index| {
                let index = index as isize;
                let fst = self.row_at(index);
                let snd = rhs.row_at(if expand { 0 } else { index });
                let dst = ret.row_at(index);
                for o in (0..self.real_col as isize).step_by(32 / size_of::<f32>()) {
                    let res = f(
                        x86_64::_mm256_load_ps(fst.offset(o)),
                        x86_64::_mm256_load_ps(snd.offset(o)),
                    );
                    x86_64::_mm256_store_ps(dst.offset(o as isize), res);
                }
            });
            Some(ret)
        } else {
            (0..self.row).into_par_iter().for_each(|index| {
                let index = index as isize;
                let dst = self.row_at(index);
                let src = rhs.row_at(if expand { 0 } else { index });
                for o in (0..self.real_col as isize).step_by(32 / size_of::<f32>()) {
                    let res = f(
                        x86_64::_mm256_load_ps(src.offset(o)),
                        x86_64::_mm256_load_ps(dst.offset(o)),
                    );
                    x86_64::_mm256_store_ps(dst.offset(o as isize), res);
                }
            });
            None
        }
    }

    pub unsafe fn add_with_vector(&self, rhs: &MatrixImpl, inplace: bool) -> Option<MatrixImpl> {
        if rhs.row != 1 || self.col != rhs.col {
            panic!("call add_with_vector with unmatched matrix shape");
        }
        self.ops_with_matrix(rhs, inplace, |a, b| x86_64::_mm256_add_ps(a, b), true)
    }

    pub unsafe fn add(&self, rhs: &MatrixImpl, inplace: bool) -> Option<MatrixImpl> {
        if self.row != rhs.row || self.col != rhs.col {
            panic!("call add with unmatched matrix shape");
        }
        self.ops_with_matrix(rhs, inplace, |a, b| x86_64::_mm256_add_ps(a, b), false)
    }

    pub unsafe fn dot(&self, rhs: &MatrixImpl, inplace: bool) -> Option<MatrixImpl> {
        if self.row != rhs.row || self.col != rhs.col {
            panic!("call add with unmatched matrix shape");
        }
        self.ops_with_matrix(rhs, inplace, |a, b| x86_64::_mm256_mul_ps(a, b), false)
    }

    unsafe fn ops_with_numeric<T>(&self, inplace: bool, rhs: f32, f: T) -> Option<MatrixImpl>
    where
        T: std::ops::Fn(x86_64::__m256, x86_64::__m256) -> x86_64::__m256 + std::marker::Sync,
    {
        if !inplace {
            let ret = MatrixImpl::new(self.row, self.col);
            (0..self.row).into_par_iter().for_each(|index| {
                let index = index as isize;
                let src = self.row_at(index);
                let dst = ret.row_at(index);
                let r = x86_64::_mm256_set1_ps(rhs);
                for o in (0..self.real_col as isize).step_by(32 / size_of::<f32>()) {
                    let val = x86_64::_mm256_load_ps(src.offset(o));
                    x86_64::_mm256_store_ps(dst.offset(o as isize), f(val, r));
                }
            });
            Some(ret)
        } else {
            (0..self.row).into_par_iter().for_each(|index| {
                let index = index as isize;
                let dst = self.row_at(index);
                let r = x86_64::_mm256_set1_ps(rhs);
                for o in (0..self.real_col as isize).step_by(32 / size_of::<f32>()) {
                    let val = x86_64::_mm256_load_ps(dst.offset(o));
                    x86_64::_mm256_store_ps(dst.offset(o as isize), f(val, r));
                }
            });
            None
        }
    }

    pub unsafe fn add_with_numeric(&self, rhs: f32, inplace: bool) -> Option<MatrixImpl> {
        self.ops_with_numeric(inplace, rhs, |a, b| x86_64::_mm256_add_ps(a, b))
    }
    pub unsafe fn mul_with_numeric(&self, rhs: f32, inplace: bool) -> Option<MatrixImpl> {
        self.ops_with_numeric(inplace, rhs, |a, b| x86_64::_mm256_mul_ps(a, b))
    }

    pub unsafe fn mul(&self, rhs: &MatrixImpl) -> MatrixImpl {
        if self.col != rhs.row {
            panic!("call mul with unmatched matrix shape");
        }
        let ret = MatrixImpl::new(self.row, rhs.col);
        (0..self.row as isize)
            .into_par_iter()
            .map(|x| {
                (0..rhs.col as isize)
                    .into_par_iter()
                    .step_by(32 / size_of::<f32>())
                    .map(move |y| (x, y))
            })
            .flatten()
            .for_each(|(i, j)| {
                let mut sum = x86_64::_mm256_set1_ps(0.0);
                for k in 0..self.col as isize {
                    let fst = x86_64::_mm256_broadcast_ss(&self.at(i, k));
                    let snd = x86_64::_mm256_load_ps(rhs.row_at(k).offset(j));
                    let res = x86_64::_mm256_mul_ps(fst, snd);
                    sum = x86_64::_mm256_add_ps(sum, res);
                }
                x86_64::_mm256_store_ps(ret.row_at(i).offset(j), sum);
            });
        ret
    }
    pub unsafe fn fill_(&self, val: f32) {
        let val = x86_64::_mm256_set1_ps(val);
        (0..self.row as isize).into_par_iter().for_each(move |idx| {
            let row = self.row_at(idx);
            for k in (0..self.col).step_by(32 / size_of::<f32>()) {
                x86_64::_mm256_store_ps(row.add(k), val);
            }
        });
    }

    unsafe fn Transpose4x4Block(src: *mut f32, dst: *mut f32, col_size1: usize, col_size2: usize) {
        let mut r1 = x86_64::_mm_load_ps(src);
        let mut r2 = x86_64::_mm_load_ps(src.add(col_size1));
        let mut r3 = x86_64::_mm_load_ps(src.add(col_size1 * 2));
        let mut r4 = x86_64::_mm_load_ps(src.add(col_size1 * 3));
        x86_64::_MM_TRANSPOSE4_PS(&mut r1, &mut r2, &mut r3, &mut r4);
        x86_64::_mm_store_ps(dst, r1);
        x86_64::_mm_store_ps(dst.add(col_size2), r2);
        x86_64::_mm_store_ps(dst.add(col_size2 * 2), r3);
        x86_64::_mm_store_ps(dst.add(col_size2 * 3), r4);
    }
    pub unsafe fn T(&self) -> MatrixImpl {
        let ret = MatrixImpl::new(self.col, self.row);
        if self.row < 4 || self.col < 4 {
            let meshgrid = (0..self.row as isize)
                .into_par_iter()
                .map(|x| (0..self.col as isize).into_par_iter().map(move |y| (x, y)))
                .flatten();
            meshgrid.for_each(|(x, y)| {
                *ret.row_at(y).offset(x) = self.at(x, y);
            });
        } else {
            let step = 16 / size_of::<f32>();
            let row_cut = self.row / step * step;
            let col_cut = self.col / step * step;
            let meshgrid = (0..row_cut as isize)
                .into_par_iter()
                .step_by(16 / size_of::<f32>())
                .map(|x| {
                    (0..col_cut as isize)
                        .into_par_iter()
                        .step_by(16 / size_of::<f32>())
                        .map(move |y| (x, y))
                })
                .flatten();
            meshgrid.for_each(|(x, y)| {
                let source = self.row_at(x).offset(y);
                let dst = ret.row_at(y).offset(x);
                MatrixImpl::Transpose4x4Block(source, dst, self.real_col, ret.real_col);
            });

            let meshgrid = (row_cut..self.row)
                .into_par_iter()
                .map(|x| (0..self.col as isize).into_par_iter().map(move |y| (x, y)))
                .flatten();
            meshgrid.for_each(|(x, y)| {
                let x = x as isize;
                let y = y as isize;
                *ret.row_at(y).offset(x) = self.at(x, y);
            });

            let meshgrid = (0..row_cut)
                .into_par_iter()
                .map(|x| (col_cut..self.col).into_par_iter().map(move |y| (x, y)))
                .flatten();
            meshgrid.for_each(|(x, y)| {
                let x = x as isize;
                let y = y as isize;
                *ret.row_at(y).offset(x) = self.at(x, y);
            });
        }
        ret
    }

    pub unsafe fn new(n: usize, m: usize) -> Self {
        let real_col = ((m >> 5) + (if m % 32 != 0 { 1 } else { 0 })) << 5;
        let layout = Layout::from_size_align_unchecked(n * real_col * size_of::<f32>(), 32);
        let ptr = alloc::alloc(layout) as *mut f32;
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        Self {
            ptr,
            row: n,
            col: m,
            real_col,
        }
    }
    pub unsafe fn free(&self) {
        let layout =
            Layout::from_size_align_unchecked(self.row * self.real_col * size_of::<f32>(), 32);
        alloc::dealloc(self.ptr as *mut u8, layout);
    }

    pub unsafe fn deep_copy(&self) -> MatrixImpl {
        let ret = MatrixImpl::new(self.row, self.col);
        (0..self.row).into_iter().for_each(|idx| {
            let src = self.row_at(idx as isize);
            let dst = ret.row_at(idx as isize);
            for j in (0..self.real_col).step_by(32 / size_of::<f32>()) {
                x86_64::_mm256_store_ps(dst.add(j), x86_64::_mm256_load_ps(src.add(j)));
            }
        });
        ret
    }
}

pub struct Matrix {
    inner: Option<Box<MatrixImpl>>,
}

impl Matrix {
    pub unsafe fn resize_row(&mut self, x: usize) {
        self.inner.as_mut().unwrap().resize_row(x);
    }
    pub unsafe fn new(n: usize, m: usize) -> Self {
        Self {
            inner: Some(Box::new(MatrixImpl::new(n, m))),
        }
    }
    pub unsafe fn null() -> Self {
        Self { inner: None }
    }

    pub unsafe fn is_null(&self) -> bool {
        self.inner.is_none()
    }

    pub fn shape(&self) -> (usize, usize) {
        (
            self.inner.as_ref().unwrap().row,
            self.inner.as_ref().unwrap().col,
        )
    }

    pub fn number_of_row(&self) -> usize {
        self.inner.as_ref().unwrap().row
    }

    pub fn number_of_col(&self) -> usize {
        self.inner.as_ref().unwrap().col
    }
    pub fn number_of_real_col(&self) -> usize {
        self.inner.as_ref().unwrap().real_col
    }

    pub unsafe fn free(&mut self) {
        if let Some(inner) = self.inner.as_ref() {
            inner.free();
            self.inner = None;
        } else {
            panic!("free a null mat");
        }
    }

    pub unsafe fn normal_init(&self) {
        self.inner.as_ref().unwrap().normal_init();
    }

    pub unsafe fn clamp(&self, lo: f32, hi: f32) {
        self.inner.as_ref().unwrap().clamp(lo, hi);
    }

    pub unsafe fn row_at(&self, i: isize) -> *mut f32 {
        self.inner.as_ref().unwrap().row_at(i)
    }

    pub unsafe fn at(&self, i: isize, j: isize) -> f32 {
        self.inner.as_ref().unwrap().at(i, j)
    }

    pub unsafe fn add_with_vector(&self, rhs: &Matrix, inplace: bool) -> Option<Matrix> {
        let rhs = rhs.inner.as_ref().unwrap().as_ref();
        let ret = self.inner.as_ref().unwrap().add_with_vector(rhs, inplace);
        if inplace {
            None
        } else {
            Some(Self {
                inner: Some(Box::new(ret.unwrap())),
            })
        }
    }

    pub unsafe fn add(&self, rhs: &Matrix, inplace: bool) -> Option<Matrix> {
        let rhs = rhs.inner.as_ref().unwrap().as_ref();
        let ret = self.inner.as_ref().unwrap().add(rhs, inplace);
        if inplace {
            None
        } else {
            Some(Self {
                inner: Some(Box::new(ret.unwrap())),
            })
        }
    }
    pub unsafe fn dot(&self, rhs: &Matrix, inplace: bool) -> Option<Matrix> {
        let rhs = rhs.inner.as_ref().unwrap().as_ref();
        let ret = self.inner.as_ref().unwrap().dot(rhs, inplace);
        if inplace {
            None
        } else {
            Some(Self {
                inner: Some(Box::new(ret.unwrap())),
            })
        }
    }

    pub unsafe fn add_with_numeric(&self, rhs: f32, inplace: bool) -> Option<Matrix> {
        let ret = self.inner.as_ref().unwrap().add_with_numeric(rhs, inplace);
        if inplace {
            None
        } else {
            Some(Self {
                inner: Some(Box::new(ret.unwrap())),
            })
        }
    }
    pub unsafe fn mul_with_numeric(&self, rhs: f32, inplace: bool) -> Option<Matrix> {
        let ret = self.inner.as_ref().unwrap().mul_with_numeric(rhs, inplace);
        if inplace {
            None
        } else {
            Some(Self {
                inner: Some(Box::new(ret.unwrap())),
            })
        }
    }

    pub unsafe fn mul(&self, rhs: &Matrix) -> Matrix {
        let ret = self
            .inner
            .as_ref()
            .unwrap()
            .mul(rhs.inner.as_ref().unwrap().as_ref());
        Self {
            inner: Some(Box::new(ret)),
        }
    }
    pub unsafe fn fill_(&self, val: f32) {
        self.inner.as_ref().unwrap().fill_(val);
    }

    pub unsafe fn T(&self) -> Matrix {
        Self {
            inner: Some(Box::new(self.inner.as_ref().unwrap().T())),
        }
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        unsafe {
            Self {
                inner: Some(Box::new(self.inner.as_ref().unwrap().deep_copy())),
            }
        }
    }
}

impl Drop for Matrix {
    fn drop(&mut self) {
        unsafe {
            if self.inner.is_some() {
                self.free()
            }
        }
    }
}

impl std::fmt::Display for MatrixImpl {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Matrix Sized [{}, {}]\n", self.row, self.col))?;
        for i in 0..self.row {
            for j in 0..self.col {
                unsafe {
                    f.write_fmt(format_args!("{:.2}", self.at(i as isize, j as isize)))?;
                }
                if j + 1 == self.col {
                    f.write_str("\n")?;
                } else {
                    f.write_str(" ")?;
                }
            }
        }
        std::fmt::Result::Ok(())
    }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if let Some(x) = self.inner.as_ref() {
            f.write_fmt(format_args!("{}", x))?;
        } else {
            f.write_fmt(format_args!("Null Matrix\n"))?;
        }
        std::fmt::Result::Ok(())
    }
}

unsafe impl core::marker::Sync for MatrixImpl {}

unsafe impl core::marker::Sync for Matrix {}
