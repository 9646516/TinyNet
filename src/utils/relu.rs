use crate::utils::mat::Matrix;
use crate::utils::nn_trait;
use rayon::prelude::*;
use std::arch::x86_64;
use std::arch::x86_64::_CMP_LE_OQ;
use std::mem::size_of;

pub struct ReluLayer {
    last_input: Matrix,
}

impl ReluLayer {
    pub unsafe fn new() -> Self {
        Self {
            last_input: Matrix::null(),
        }
    }
}

impl nn_trait::Layer for ReluLayer {
    fn forward(&mut self, input: Matrix) -> Matrix {
        unsafe {
            self.last_input = input.clone();
            let (h, _) = input.shape();
            let cmp = x86_64::_mm256_set1_ps(0.0);
            (0..h).into_par_iter().for_each(|idx| {
                let src = input.row_at(idx as isize);
                for i in (0..input.number_of_real_col()).step_by(256 / size_of::<f32>()) {
                    let val = x86_64::_mm256_max_ps(x86_64::_mm256_load_ps(src.add(i)), cmp);
                    x86_64::_mm256_store_ps(src.add(i), val);
                }
            });
            input
        }
    }
    fn backward(&mut self, dLoss: Matrix) -> Matrix {
        unsafe {
            let (h, _) = dLoss.shape();
            let cmp = x86_64::_mm256_set1_ps(0.0);
            (0..h).into_par_iter().for_each(|idx| {
                let src = self.last_input.row_at(idx as isize);
                let dst = dLoss.row_at(idx as isize);
                for i in (0..dLoss.number_of_real_col()).step_by(256 / size_of::<f32>()) {
                    let val = x86_64::_mm256_load_ps(src.add(i));
                    let mask = std::arch::x86_64::_mm256_cmp_ps::<_CMP_LE_OQ>(val, cmp);

                    let prev = x86_64::_mm256_load_ps(dst.add(i));
                    let filtered = x86_64::_mm256_blendv_ps(prev, cmp, mask);
                    x86_64::_mm256_store_ps(dst.add(i), filtered);
                }
            });
            dLoss
        }
    }
    fn update_parameters(&mut self, _: f32, _: f32) {}
}
