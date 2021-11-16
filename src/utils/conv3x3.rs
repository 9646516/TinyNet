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

    weight: Matrix,
    bias: Matrix,
    d_weight: Matrix,
    d_bias: Matrix,

    pub pinned_memory_for_im2col: Matrix,
    pub pinned_memory_for_col2im: Matrix,
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
        Self {
            in_channels,
            out_channels,
            im_row,
            im_col,
            stride,
            padding,
            feat_row,
            feat_col,
            weight: Matrix::new(9 * in_channels, out_channels),
            bias: Matrix::new(1, out_channels),
            d_weight: Matrix::new(9 * in_channels, out_channels),
            d_bias: Matrix::new(1, out_channels),
            pinned_memory_for_im2col: Matrix::new(100, 9 * in_channels),
            pinned_memory_for_col2im: Matrix::new(100, feat_col * feat_row * out_channels),
        }
    }
    // B*HWC => BH'W'*9C
    pub fn im2col(&mut self, input: Matrix) {
        let (h, w) = input.shape();
        println!("{} {}", h, w);
        let sz = h * self.feat_col * self.feat_row;
        unsafe {
            self.pinned_memory_for_im2col.resize_row(sz);
        }
        let in_channels = self.in_channels as isize;
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
            .for_each(|(batch, row, col, h_index)| unsafe {
                let base_offset = row * im_col * in_channels + col * in_channels;

                let row_offset = im_col * in_channels;
                let col_offset = in_channels;

                let src = input.row_at(batch as isize);
                let dst = self.pinned_memory_for_im2col.row_at(h_index as isize);

                let mut ptr = 0;
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let corner = base_offset + dx * row_offset + dy * col_offset;
                        let feat_row_idx = row as isize + dx;
                        let feat_col_idx = col as isize + dy;
                        for dz in 0..in_channels {
                            let o = corner + dz;
                            *dst.offset(ptr) = if o < 0
                                || o >= w as isize
                                || feat_row_idx < 0
                                || feat_row_idx >= self.im_row as isize
                                || feat_col_idx < 0
                                || feat_col_idx >= self.im_col as isize
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
    // BH'W'*C => B*H'W'C
    pub fn add_bias_and_col2im(&mut self, input: Matrix) {
        let h = input.number_of_row();
        let sz = h / self.feat_col / self.feat_row;
        unsafe {
            self.pinned_memory_for_im2col.resize_row(sz);
        }
        let out_channels = self.out_channels;
        let feat_row = self.feat_row;
        let feat_col = self.feat_col;
        let block_size = feat_row * feat_col;
        (0..h)
            .into_iter()
            .map(|idx| {
                let b = idx / (block_size);
                let left = idx - b * block_size;
                let h = left / feat_col;
                let w = left - feat_col * h;
                (idx, b, h, w)
            })
            .for_each(|(idx, b, h, w)| unsafe {
                let bias_src = self.bias.row_at(0);
                let src = input.row_at(idx as isize);
                let offset = h * feat_col * out_channels + w * out_channels;
                let dst = self.pinned_memory_for_col2im.row_at(b as isize).add(offset);

                let step = 32 / size_of::<f32>();
                let cut = out_channels / step * step;
                (0..cut).into_iter().step_by(step).for_each(|idx| {
                    let val = x86_64::_mm256_load_ps(src.add(idx));
                    let add = x86_64::_mm256_load_ps(bias_src.add(idx));
                    x86_64::_mm256_store_ps(dst.add(idx), x86_64::_mm256_add_ps(val, add));
                });
                (cut..out_channels).into_iter().for_each(|idx| {
                    *dst.add(idx) = *src.add(idx) + *self.bias.row_at(0).add(idx);
                })
            });
    }
}

impl nn_trait::Layer for Conv3x3 {
    fn forward(&mut self, input: Matrix) -> Matrix {
        unsafe {
            self.im2col(input);
            let res = self.pinned_memory_for_im2col.mul(&self.weight);
            self.add_bias_and_col2im(res);
            self.pinned_memory_for_col2im.clone()
        }
    }
    fn backward(&mut self, dLoss: Matrix) -> Matrix {
        todo!()
    }

    fn update_parameters(&mut self, rate: f32, decay: f32) {
        unsafe {
            let weight_decay = self.weight.mul_with_numeric(decay, false).unwrap();
            weight_decay.add(&self.d_weight, true);
            let weight_go = weight_decay.mul_with_numeric(-rate, false).unwrap();
            weight_go.clamp(-100.0, 100.0);
            self.weight.add(&weight_go, true);

            let bias_go = self.d_bias.mul_with_numeric(-rate, false).unwrap();
            bias_go.clamp(-100.0, 100.0);
            self.bias.add(&bias_go, true);
        }
    }
}