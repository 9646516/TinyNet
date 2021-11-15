use rayon::prelude::*;
use std::alloc;
use std::alloc::{handle_alloc_error, Layout};
use std::arch::x86_64;
use std::f32::consts::PI;
use std::fmt::Formatter;
use std::mem::size_of;

#[derive(Debug)]
pub struct Matrix {
    ptr: *mut f32,
    row: usize,
    col: usize,
    real_col: usize,
}

unsafe impl core::marker::Sync for Matrix {}

impl Matrix {
    pub unsafe fn null() -> Self {
        Matrix {
            ptr: std::ptr::null::<f32>() as *mut f32,
            row: 0,
            col: 0,
            real_col: 0,
        }
    }
    pub unsafe fn normal_init(&self) {
        let k = 1.0 / (2.0 * PI).sqrt();
        (0..self.row).into_par_iter().for_each(|index| {
            let dst = self.row(index as isize);
            let center = self.col / 2;
            for i in 0..self.col {
                let p = i as f32 - center as f32;
                let p = -0.5 * p * p;
                let val = k * p.exp();
                *dst.add(i) = val;
            }
        })
    }

    pub unsafe fn new(n: usize, m: usize) -> Self {
        let real_col = ((m >> 5) + (if m % 32 != 0 { 1 } else { 0 })) << 5;
        let layout = Layout::from_size_align_unchecked(n * real_col * size_of::<f32>(), 32);
        let ptr = alloc::alloc(layout) as *mut f32;
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        Matrix {
            ptr,
            row: n,
            col: m,
            real_col,
        }
    }
    pub unsafe fn clamp(&self, lo: f32, hi: f32) {
        (0..self.row).into_par_iter().for_each(|index| {
            let dst = self.row(index as isize);
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
    pub fn is_null(&self) -> bool {
        self.ptr.is_null()
    }
    pub fn real_col(&self) -> usize {
        self.real_col
    }
    pub fn shape(&self) -> (usize, usize) {
        (self.row, self.col)
    }
    pub unsafe fn row(&self, i: isize) -> *mut f32 {
        if i > self.row as isize {
            panic!("matrix visit row over bound");
        }
        self.ptr.offset(i * self.real_col as isize)
    }
    pub unsafe fn at(&self, i: isize, j: isize) -> f32 {
        *self.ptr.offset(i * self.real_col as isize + j)
    }

    pub unsafe fn set(&self, i: isize, j: isize, val: f32) {
        *self.ptr.offset(i * self.real_col as isize + j) = val;
    }

    unsafe fn ops_with_matrix<T>(
        &self,
        rhs: &Matrix,
        inplace: bool,
        f: T,
        expand: bool,
    ) -> Option<Matrix>
    where
        T: std::ops::Fn(x86_64::__m256, x86_64::__m256) -> x86_64::__m256 + std::marker::Sync,
    {
        if !inplace {
            let ret = Matrix::new(self.row, self.col);
            (0..self.row).into_par_iter().for_each(|index| {
                let index = index as isize;
                let fst = self.row(index);
                let snd = rhs.row(if expand { 0 } else { index });
                let dst = ret.row(index);
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
                let dst = self.row(index);
                let src = rhs.row(if expand { 0 } else { index });
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

    pub unsafe fn add_with_vector(&self, rhs: &Matrix, inplace: bool) -> Option<Matrix> {
        if rhs.row != 1 || self.col != rhs.col {
            panic!("call add_with_vector with unmatched matrix shape");
        }
        self.ops_with_matrix(rhs, inplace, |a, b| x86_64::_mm256_add_ps(a, b), true)
    }

    pub unsafe fn add(&self, rhs: &Matrix, inplace: bool) -> Option<Matrix> {
        if self.row != rhs.row || self.col != rhs.col {
            panic!("call add with unmatched matrix shape");
        }
        self.ops_with_matrix(rhs, inplace, |a, b| x86_64::_mm256_add_ps(a, b), false)
    }

    unsafe fn ops_with_numeric<T>(&self, inplace: bool, rhs: f32, f: T) -> Option<Matrix>
    where
        T: std::ops::Fn(x86_64::__m256, x86_64::__m256) -> x86_64::__m256 + std::marker::Sync,
    {
        if !inplace {
            let ret = Matrix::new(self.row, self.col);
            (0..self.row).into_par_iter().for_each(|index| {
                let index = index as isize;
                let src = self.row(index);
                let dst = ret.row(index);
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
                let dst = self.row(index);
                let r = x86_64::_mm256_set1_ps(rhs);
                for o in (0..self.real_col as isize).step_by(32 / size_of::<f32>()) {
                    let val = x86_64::_mm256_load_ps(dst.offset(o));
                    x86_64::_mm256_store_ps(dst.offset(o as isize), f(val, r));
                }
            });
            None
        }
    }

    pub unsafe fn add_with_numeric(&self, rhs: f32, inplace: bool) -> Option<Matrix> {
        self.ops_with_numeric(inplace, rhs, |a, b| x86_64::_mm256_add_ps(a, b))
    }
    pub unsafe fn mul_with_numeric(&self, rhs: f32, inplace: bool) -> Option<Matrix> {
        self.ops_with_numeric(inplace, rhs, |a, b| x86_64::_mm256_mul_ps(a, b))
    }

    pub unsafe fn mul(&self, rhs: &Matrix) -> Matrix {
        if self.col != rhs.row {
            panic!("call mul with unmatched matrix shape");
        }
        let ret = Matrix::new(self.row, rhs.col);
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
                    let snd = x86_64::_mm256_load_ps(rhs.row(k).offset(j));
                    let res = x86_64::_mm256_mul_ps(fst, snd);
                    sum = x86_64::_mm256_add_ps(sum, res);
                }
                x86_64::_mm256_store_ps(ret.row(i).offset(j), sum);
            });
        ret
    }
    pub unsafe fn fill_(&self, val: f32) {
        let val = x86_64::_mm256_set1_ps(val);
        (0..self.row as isize).into_par_iter().for_each(move |idx| {
            let row = self.row(idx);
            for k in (0..self.col).step_by(32 / size_of::<f32>()) {
                x86_64::_mm256_store_ps(row.add(k), val);
            }
        });
    }

    pub unsafe fn Transpose4x4Block(
        src: *mut f32,
        dst: *mut f32,
        col_size1: usize,
        col_size2: usize,
    ) {
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
    pub unsafe fn T(&self) -> Matrix {
        let ret = Matrix::new(self.col, self.row);
        if self.row < 4 || self.col < 4 {
            let meshgrid = (0..self.row as isize)
                .into_par_iter()
                .map(|x| (0..self.col as isize).into_par_iter().map(move |y| (x, y)))
                .flatten();
            meshgrid.for_each(|(x, y)| {
                ret.set(y, x, self.at(x, y));
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
                let source = self.row(x).offset(y);
                let dst = ret.row(y).offset(x);
                Matrix::Transpose4x4Block(source, dst, self.real_col, ret.real_col);
            });

            let meshgrid = (row_cut..self.row)
                .into_par_iter()
                .map(|x| (0..self.col as isize).into_par_iter().map(move |y| (x, y)))
                .flatten();
            meshgrid.for_each(|(x, y)| {
                let x = x as isize;
                let y = y as isize;
                ret.set(y, x, self.at(x, y));
            });

            let meshgrid = (0..row_cut)
                .into_par_iter()
                .map(|x| (col_cut..self.col).into_par_iter().map(move |y| (x, y)))
                .flatten();
            meshgrid.for_each(|(x, y)| {
                let x = x as isize;
                let y = y as isize;
                ret.set(y, x, self.at(x, y));
            });
        }
        ret
    }

    pub unsafe fn move_copy(&mut self, mut rhs: Matrix) {
        let sptr = self.ptr;
        let srow = self.col;
        let scol = self.row;
        let sreal_col = self.real_col;

        self.ptr = rhs.ptr;
        self.col = rhs.col;
        self.row = rhs.row;
        self.real_col = rhs.real_col;

        rhs.col = scol;
        rhs.row = srow;
        rhs.real_col = sreal_col;
        rhs.ptr = sptr;
        std::mem::drop(rhs);
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        unsafe {
            let ret = Matrix::new(self.row, self.col);
            (0..self.row).into_par_iter().for_each(|index| {
                let index = index as isize;
                let src = self.row(index);
                let dst = ret.row(index);
                for o in (0..self.real_col as isize).step_by(32 / size_of::<f32>()) {
                    let val = x86_64::_mm256_load_ps(src.offset(o));
                    x86_64::_mm256_store_ps(dst.offset(o as isize), val);
                }
            });
            ret
        }
    }
}

impl Drop for Matrix {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                let layout = Layout::from_size_align_unchecked(
                    self.row * self.real_col * size_of::<f32>(),
                    32,
                );
                alloc::dealloc(self.ptr as *mut u8, layout);
            }
        }
    }
}

impl std::fmt::Display for Matrix {
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
