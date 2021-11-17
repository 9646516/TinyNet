use crate::utils::mat::Matrix;
use crate::utils::nn_trait;
use std::arch::x86_64;
use std::mem::size_of;

pub struct LinearLayer {
    last_input: Matrix,
    pub weight: Matrix,
    pub d_weight: Matrix,
    pub v_weight: Matrix,
    pub bias: Matrix,
    pub d_bias: Matrix,
    pub v_bias: Matrix,
}

impl LinearLayer {
    pub unsafe fn new(in_channels: usize, out_channels: usize) -> Self {
        let last_input = Matrix::null();
        let weight = Matrix::new(in_channels, out_channels);
        let d_weight = Matrix::new(in_channels, out_channels);
        let bias = Matrix::new(1, out_channels);
        let d_bias = Matrix::new(1, out_channels);
        bias.normal_init();
        weight.normal_init();
        Self {
            last_input,
            weight,
            d_weight,
            bias,
            d_bias,
            v_weight: Matrix::null(),
            v_bias: Matrix::null(),
        }
    }
}

impl nn_trait::Layer for LinearLayer {
    fn forward(&mut self, input: Matrix) -> Matrix {
        unsafe {
            let now = input.mul(&self.weight);
            now.add_with_vector(&self.bias, true);
            self.last_input = input;
            now
        }
    }
    fn backward(&mut self, dLoss: Matrix) -> Matrix {
        unsafe {
            let (h, w) = dLoss.shape();

            for j in (0..w).step_by(32 / size_of::<f32>()) {
                let mut sum = x86_64::_mm256_set1_ps(0.0);
                for i in 0..h {
                    let val = x86_64::_mm256_load_ps(dLoss.row_at(i as isize).add(j));
                    sum = x86_64::_mm256_add_ps(sum, val);
                }
                x86_64::_mm256_store_ps(self.d_bias.row_at(0).add(j), sum);
            }

            let xt = self.last_input.T();

            let xt = xt.mul(&dLoss);

            self.d_weight = xt;

            let wt = self.weight.T();

            dLoss.mul(&wt)
        }
    }
    fn trainable(&self) -> bool {
        true
    }

    fn parameters(
        &mut self,
    ) -> Option<(
        &mut Matrix,
        &mut Matrix,
        &mut Matrix,
        &mut Matrix,
        &mut Matrix,
        &mut Matrix,
    )> {
        Some((
            &mut self.weight,
            &mut self.d_weight,
            &mut self.v_weight,
            &mut self.bias,
            &mut self.d_bias,
            &mut self.v_bias,
        ))
    }
}
