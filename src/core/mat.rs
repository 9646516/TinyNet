use std::{alloc, cmp, mem, slice};
use std::alloc::{handle_alloc_error, Layout};
use std::mem::ManuallyDrop;
use std::ops::Index;
use std::ptr::NonNull;
use num::traits::Float;

pub struct Matrix<T> {
    ptr: *mut T,
    row: usize,
    col: usize,
    real_col: usize,
}


impl<T> Matrix<T> where T: Float {
    pub unsafe fn new(n: usize, m: usize) -> Self {
        let real_col = ((m >> 5) + (if m >> 5 != 0 { 1 } else { 1 })) << 5;
        let layout = Layout::from_size_align_unchecked(n * real_col, 32);
        let ptr = alloc::alloc(layout);
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        Matrix {
            ptr: ptr as *mut T,
            row: n,
            col: m,
            real_col,
        }
    }

    pub fn inplace_neg(&mut self) {}

    pub fn inplace_transpose(&mut self) {}

    pub fn inverse(&mut self) -> Self {
        todo!()
    }
}

impl<T> Clone for Matrix<T> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<T> Drop for Matrix<T> {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::from_size_align_unchecked(self.row * self.real_col, 32);
            alloc::dealloc(self.ptr as *mut u8, layout);
        }
    }
}

impl<T> Index<usize> for Matrix<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        todo!()
    }
}

impl<T> std::ops::IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        todo!()
    }
}

impl<T> std::ops::Add<Matrix<T>> for Matrix<T> {
    type Output = ();

    fn add(self, rhs: Matrix<T>) -> Self::Output {
        todo!()
    }
}

impl<T> std::ops::Mul<Matrix<T>> for Matrix<T> {
    type Output = ();

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        todo!()
    }
}