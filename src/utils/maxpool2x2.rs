use crate::utils::mat::Matrix;
use crate::utils::nn_trait;
use rayon::prelude::*;
use std::arch::x86_64;
use std::mem::size_of;

pub struct MaxPool2x2 {
    pub in_channels: usize,
    pub im_row: usize,
    pub im_col: usize,
    pub max_mask: Matrix,
}

impl MaxPool2x2 {
    pub unsafe fn new(in_channels: usize, im_row: usize, im_col: usize) -> Self {
        Self {
            in_channels,
            im_row,
            im_col,
            max_mask: Matrix::null(),
        }
    }
    pub unsafe fn copy(
        src: *mut f32,
        a00: *mut f32,
        a01: *mut f32,
        a10: *mut f32,
        a11: *mut f32,
        mask: *mut f32,
        len: usize,
    ) {
        for i in 0..len {
            if (*mask.add(i)).abs() < 1e-5 {
                *a00.add(i) = *src.add(i);
            } else if (*mask.add(i) - 1.0).abs() < 1e-5 {
                *a01.add(i) = *src.add(i);
            } else if (*mask.add(i) - 2.0).abs() < 1e-5 {
                *a10.add(i) = *src.add(i);
            } else {
                *a11.add(i) = *src.add(i);
            }
        }
    }

    pub unsafe fn chmax(
        &self,
        src: *mut f32,
        dst: *mut f32,
        len: usize,
        label: f32,
        mask_base_ptr: *mut f32,
    ) {
        let step = 32 / size_of::<f32>();
        let cut = len / step * step;
        for i in (0..cut).step_by(step) {
            let lhs = x86_64::_mm256_load_ps(src.add(i));
            let rhs = x86_64::_mm256_load_ps(dst.add(i));
            let res = x86_64::_mm256_max_ps(lhs, rhs);
            x86_64::_mm256_store_ps(dst.add(i), res);
        }
        for i in cut..len {
            let a = *dst.add(i);
            let b = *src.add(i);
            *dst.add(i) = if a < b { b } else { a };
        }

        for i in 0..len {
            let a = *dst.add(i);
            let b = *src.add(i);
            if a < b {
                *mask_base_ptr.add(i) = label;
            }
        }
    }
}

impl nn_trait::Layer for MaxPool2x2 {
    fn forward(&mut self, input: Matrix) -> Matrix {
        unsafe {
            let im_row = self.im_row;
            let im_col = self.im_col;

            let in_channels = self.in_channels;
            let col_step = in_channels;
            let row_step = col_step * im_col;
            let h = input.number_of_row();
            let out_size = ((im_col + 1) / 2) * ((im_row + 1) / 2) * in_channels;
            let ret = Matrix::new(h, out_size);
            self.max_mask = Matrix::new(h, out_size);
            ret.fill_(-1e9);
            (0..h).into_par_iter().for_each(|batch_index| {
                for i in (0..im_row).step_by(2) {
                    for j in (0..im_col).step_by(2) {
                        let mask_base_ptr = self.max_mask.row_at(batch_index as isize);
                        let base_src_ptr = input.row_at(batch_index as isize);
                        let a00 = i * row_step + j * col_step;
                        let a01 = a00 + col_step;
                        let a10 = a00 + row_step;
                        let a11 = a10 + col_step;

                        let p_a00 = base_src_ptr.add(a00);
                        let p_a01 = base_src_ptr.add(a01);
                        let p_a10 = base_src_ptr.add(a10);
                        let p_a11 = base_src_ptr.add(a11);

                        let b00 = ((i / 2) * ((im_col + 1) / 2) + (j / 2)) * col_step;
                        let dst = ret.row_at(batch_index as isize).add(b00);
                        self.chmax(p_a00, dst, in_channels, 0.0, mask_base_ptr);
                        if j + 1 < im_col {
                            self.chmax(p_a01, dst, in_channels, 1.0, mask_base_ptr);
                        }
                        if i + 1 < im_row {
                            self.chmax(p_a10, dst, in_channels, 2.0, mask_base_ptr);
                        }
                        if j + 1 < im_col && i + 1 < im_row {
                            self.chmax(p_a11, dst, in_channels, 3.0, mask_base_ptr);
                        }
                    }
                }
            });
            ret
        }
    }
    fn backward(&mut self, dLoss: Matrix) -> Matrix {
        unsafe {
            let h = dLoss.number_of_row();
            let im_row = self.im_row;
            let im_col = self.im_col;
            let feat_col = (im_col + 1) >> 1;
            let feat_row = (im_row + 1) >> 1;
            let in_channels = self.in_channels;
            let ret = Matrix::new(h, im_col * im_row * in_channels);
            ret.fill_(0.0);
            (0..h).into_iter().for_each(|x| {
                for i in 0..feat_row {
                    for j in 0..feat_col {
                        let src_offset = (i * feat_col + j) * in_channels;
                        let d00 = (i * 2 * feat_col + j * 2) * in_channels;
                        let d01 = (i * 2 * feat_col + j * 2 + 1) * in_channels;
                        let d10 = ((i * 2 + 1) * feat_col + j * 2) * in_channels;
                        let d11 = ((i * 2 + 1) * feat_col + j * 2 + 1) * in_channels;
                        Self::copy(
                            dLoss.row_at(x as isize).add(src_offset),
                            ret.row_at(x as isize).add(d00),
                            ret.row_at(x as isize).add(d01),
                            ret.row_at(x as isize).add(d10),
                            ret.row_at(x as isize).add(d11),
                            self.max_mask.row_at(x as isize).add(src_offset),
                            in_channels,
                        );
                    }
                }
            });
            ret
        }
    }
    fn trainable(&self) -> bool {
        false
    }
}
