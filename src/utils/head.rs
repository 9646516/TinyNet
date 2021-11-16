use crate::utils::mat::Matrix;
use crate::utils::nn_trait;
use rayon::prelude::*;

pub struct SoftMaxCrossEntropy {
    pub grad: Matrix,
}

impl SoftMaxCrossEntropy {
    pub unsafe fn new() -> Self {
        Self {
            grad: Matrix::null(),
        }
    }
}

impl nn_trait::Head for SoftMaxCrossEntropy {
    fn forward(&mut self, input: Matrix, target: Matrix) -> Matrix {
        unsafe {
            let (h, w) = input.shape();
            let ret = Matrix::new(h, 1);
            self.grad = Matrix::new(h, w);

            (0..h).into_par_iter().for_each(|idx| {
                let src_row = input.row_at(idx as isize);
                let grad_row = self.grad.row_at(idx as isize);
                let target_row = target.row_at(idx as isize);

                let max_val =
                    (1..w)
                        .map(|i| *src_row.add(i))
                        .fold(*src_row, |a, b| if a < b { b } else { a });

                let mut sum = 0.0;
                for i in 0..w {
                    let now = *src_row.add(i).to_owned() - max_val;
                    let now = now.exp();
                    *src_row.add(i) = now;
                    sum += now;
                }
                for i in 0..w {
                    *src_row.add(i) /= sum;
                }
                for i in 0..w {
                    *grad_row.add(i) = *src_row.add(i) - *target_row.add(i);
                }

                let mut loss = 0f32;
                for i in 0..w {
                    let mut pred = *src_row.add(i);
                    let target = *target_row.add(i);
                    if pred < 1e-7 {
                        pred = 1e-7;
                    }
                    loss -= target * pred.ln();
                }
                *ret.row_at(idx as isize).offset(0) = loss;
            });
            ret
        }
    }
    fn backward(&mut self, _: Matrix) -> Matrix {
        self.grad.clone()
    }

    fn eval_forward(&self, input: Matrix) -> Vec<usize> {
        unsafe {
            let (h, w) = input.shape();
            let mut ret = vec![0; h];
            let ptr = ret.as_mut_ptr();
            (0..h).into_iter().for_each(|idx| {
                let src_row = input.row_at(idx as isize);
                let arg_max = (1..w)
                    .map(|i| (i, *src_row.add(i)))
                    .fold((0, *src_row), |a, b| if a.1 < b.1 { b } else { a })
                    .0;
                *ptr.add(idx) = arg_max;
            });
            ret
        }
    }
}
