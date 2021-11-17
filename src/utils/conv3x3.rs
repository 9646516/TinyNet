use crate::utils::mat::Matrix;
use crate::utils::nn_trait;
use rayon::prelude::*;
use std::arch::x86_64;
use std::mem::size_of;

pub struct Conv3x3 {
    pub in_channels: usize,
    pub out_channels: usize,
    pub im_row: usize,
    pub im_col: usize,
    pub stride: usize,
    pub padding: usize,

    pub feat_row: usize,
    pub feat_col: usize,

    pub weight: Matrix,
    pub bias: Matrix,
    pub d_weight: Matrix,
    pub d_bias: Matrix,
    pub v_weight: Matrix,
    pub v_bias: Matrix,

    pub pinned_memory_for_im2col: Matrix,

    pub last_input_shape: (usize, usize),
}

impl Conv3x3 {
    pub unsafe fn new(
        in_channels: usize,
        out_channels: usize,
        im_row: usize,
        im_col: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let feat_row = (im_row + padding + padding - 3) / stride + 1;
        let feat_col = (im_col + padding + padding - 3) / stride + 1;
        let weight = Matrix::new(9 * in_channels, out_channels);
        let bias = Matrix::new(1, out_channels);
        weight.normal_init();
        bias.normal_init();
        Self {
            in_channels,
            out_channels,
            im_row,
            im_col,
            stride,
            padding,
            feat_row,
            feat_col,
            weight,
            bias,
            d_weight: Matrix::new(9 * in_channels, out_channels),
            d_bias: Matrix::new(1, out_channels),
            pinned_memory_for_im2col: Matrix::new(100, 9 * in_channels),
            last_input_shape: (0, 0),
            v_weight: Matrix::null(),
            v_bias: Matrix::null(),
        }
    }
    pub fn meshgrid(&self, h: usize) -> impl ParallelIterator<Item = (isize, isize, isize, isize)> {
        let padding = self.padding as isize;
        let im_row = self.im_row as isize;
        let im_col = self.im_col as isize;
        let feat_row = self.feat_row as isize;
        let feat_col = self.feat_col as isize;
        let stride = self.stride;
        (0..h as isize)
            .into_par_iter()
            .map(move |batch_index| {
                (-padding + 1..im_row + padding - 1)
                    .into_par_iter()
                    .step_by(stride)
                    .enumerate()
                    .map(move |(idx, row_index)| {
                        (-padding + 1..im_col + padding as isize - 1)
                            .into_par_iter()
                            .step_by(stride)
                            .enumerate()
                            .map(move |(idx2, col_index)| {
                                (
                                    batch_index,
                                    row_index,
                                    col_index,
                                    batch_index * feat_row * feat_col
                                        + idx as isize * feat_col
                                        + idx2 as isize,
                                )
                            })
                    })
            })
            .flatten()
            .flatten()
    }
    // B*HWC => BH'W'*9C
    pub fn im2col(&mut self, input: Matrix) {
        let (h, w) = input.shape();
        // println!("{} {}", h, w);
        let sz = h * self.feat_col * self.feat_row;
        unsafe {
            self.pinned_memory_for_im2col.resize_row(sz);
        }
        let in_channels = self.in_channels as isize;
        let im_col = self.im_col as isize;

        self.meshgrid(h)
            .into_par_iter()
            .for_each(|(batch, row, col, h_index)| unsafe {
                let row_offset = im_col * in_channels;
                let col_offset = in_channels;

                let src = input.row_at(batch as isize);
                let dst = self.pinned_memory_for_im2col.row_at(h_index as isize);

                let mut ptr = 0;
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let im_row_idx = row as isize + dx;
                        let im_col_idx = col as isize + dy;
                        let base_offset = im_row_idx * row_offset + im_col_idx * col_offset;
                        for dz in 0..in_channels {
                            let o = base_offset + dz;
                            *dst.offset(ptr) = if o < 0
                                || o >= w as isize
                                || im_row_idx < 0
                                || im_row_idx >= self.im_row as isize
                                || im_col_idx < 0
                                || im_col_idx >= self.im_col as isize
                            {
                                0.0
                            } else {
                                *src.offset(o)
                            };
                            ptr += 1;
                        }
                    }
                }
            });
    }
    pub fn add_bias_to_col(&self, x: &Matrix) {
        unsafe {
            let step = 32 / size_of::<f32>();
            let h = x.number_of_row();
            let w = x.number_of_real_col();
            (0..h).into_par_iter().for_each(|row_idx| {
                (0..w).into_par_iter().step_by(step).for_each(|idx| {
                    let src = x.row_at(row_idx as isize);
                    let src2 = self.bias.row_at(0);
                    let val = x86_64::_mm256_loadu_ps(src.add(idx));
                    let add = x86_64::_mm256_loadu_ps(src2.add(idx));
                    x86_64::_mm256_storeu_ps(src.add(idx), x86_64::_mm256_add_ps(val, add));
                });
            });
        }
    }
    // BH'W'*C => B*H'W'C
    pub fn col2im(&mut self, input: Matrix) -> Matrix {
        unsafe {
            let h = input.number_of_row();
            let sz = h / self.feat_col / self.feat_row;
            let ret = Matrix::new(sz, self.feat_col * self.feat_row * self.out_channels);
            let out_channels = self.out_channels;
            let feat_row = self.feat_row;
            let feat_col = self.feat_col;
            let block_size = feat_row * feat_col;
            (0..h)
                .into_par_iter()
                .map(|idx| {
                    let b = idx / (block_size);
                    let left = idx - b * block_size;
                    let h = left / feat_col;
                    let w = left - feat_col * h;
                    (idx, b, h, w)
                })
                .for_each(|(idx, b, h, w)| {
                    let src = input.row_at(idx as isize);
                    let offset = h * feat_col * out_channels + w * out_channels;
                    let dst = ret.row_at(b as isize).add(offset);

                    let step = 32 / size_of::<f32>();
                    let cut = out_channels / step * step;
                    (0..cut).into_iter().step_by(step).for_each(|idx| {
                        let val = x86_64::_mm256_loadu_ps(src.add(idx));
                        x86_64::_mm256_storeu_ps(dst.add(idx), val);
                    });
                    (cut..out_channels).into_iter().for_each(|idx| {
                        *dst.add(idx) = *src.add(idx);
                    })
                });
            ret
        }
    }

    // B*H'W'C => BH'W'*C
    pub fn split_loss(&mut self, input: &Matrix) -> Matrix {
        unsafe {
            let h = input.number_of_row();
            let sz = h * self.feat_row * self.feat_col;
            let ret = Matrix::new(sz, self.out_channels);
            let out_channels = self.out_channels;
            let feat_row = self.feat_row;
            let feat_col = self.feat_col;
            let block_size = feat_row * feat_col;
            (0..h).into_par_iter().for_each(|idx| {
                let base = idx * block_size;
                let base_src = input.row_at(idx as isize);
                for i in 0..block_size {
                    let dst = ret.row_at((base + i) as isize);
                    let src = base_src.add(i * out_channels);
                    for j in 0..out_channels {
                        *dst.add(j) = *src.add(j);
                    }
                }
            });
            ret
        }
    }
    // B*H'W'C => BH'W'*C
    pub fn merge_loss(&mut self, input: &Matrix) -> Matrix {
        unsafe {
            let (h, w) = self.last_input_shape;
            let ret = Matrix::new(h, w);
            ret.fill_(0.0);
            let in_channels = self.in_channels as isize;
            let row_offset = self.im_col as isize * in_channels;
            let col_offset = in_channels;

            for dx in -1..=1 {
                for dy in -1..=1 {
                    self.meshgrid(h)
                        .into_par_iter()
                        .for_each(|(batch, row, col, h_index)| {
                            let im_row_idx = row as isize + dx;
                            let im_col_idx = col as isize + dy;
                            let grid_idx = (dx + 1) * 3 + dy + 1;

                            let src = input
                                .row_at(h_index as isize)
                                .offset(grid_idx * in_channels);
                            let dst = ret.row_at(batch as isize);
                            let base_offset = im_row_idx * row_offset + im_col_idx * col_offset;

                            for dz in 0..in_channels {
                                let o = base_offset + dz;
                                if !(o < 0
                                    || o >= w as isize
                                    || im_row_idx < 0
                                    || im_row_idx >= self.im_row as isize
                                    || im_col_idx < 0
                                    || im_col_idx >= self.im_col as isize)
                                {
                                    *dst.offset(o) += *src.add(dz as usize)
                                };
                            }
                        })
                }
            }
            ret
        }
    }
}

impl nn_trait::Layer for Conv3x3 {
    fn forward(&mut self, input: Matrix) -> Matrix {
        unsafe {
            self.last_input_shape = input.shape();
            self.im2col(input);
            let res = self.pinned_memory_for_im2col.mul(&self.weight);
            self.add_bias_to_col(&res);
            self.col2im(res)
        }
    }
    fn backward(&mut self, dLoss: Matrix) -> Matrix {
        unsafe {
            (0..self.out_channels)
                .into_par_iter()
                .for_each(|channel_idx| {
                    let mut sum = 0.0;
                    for i in 0..dLoss.number_of_row() {
                        let row = dLoss.row_at(i as isize);
                        for j in 0..self.feat_col * self.feat_row {
                            sum += *row.add(j * self.out_channels + channel_idx);
                        }
                        *self.d_bias.row_at(0).add(channel_idx) = sum;
                    }
                });
            let split_loss = self.split_loss(&dLoss);
            self.d_weight = self.pinned_memory_for_im2col.T().mul(&split_loss);
            let wt = self.weight.T();
            let ret = split_loss.mul(&wt);
            self.merge_loss(&ret)
        }
    }

    fn update_parameters(&mut self, rate: f32, momentum: f32, decay: f32) {
        unsafe {
            let weight_decay = self.weight.mul_with_numeric(decay, false).unwrap();
            weight_decay.add(&self.d_weight, true);
            let weight_go = weight_decay.mul_with_numeric(-rate, false).unwrap();
            if self.v_weight.is_null() {
                self.v_weight = weight_go;
            } else {
                self.v_weight.mul_with_numeric(momentum, true);
                self.v_weight.add(&weight_go, true);
            }
            self.v_weight.clamp(-100.0, 100.0);
            self.weight.add(&self.v_weight, true);

            let bias_go = self.d_bias.mul_with_numeric(-rate, false).unwrap();
            if self.v_bias.is_null() {
                self.v_bias = bias_go;
            } else {
                self.v_bias.mul_with_numeric(momentum, true);
                self.v_bias.add(&bias_go, true);
            }
            self.v_bias.clamp(-100.0, 100.0);
            self.bias.add(&self.v_bias, true);
        }
    }
}
