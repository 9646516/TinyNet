use crate::utils::mat::Matrix;
use crate::utils::nn_trait::Optimizer;

pub struct SGD {
    rate: f32,
    momentum: f32,
    decay: f32,
}

impl SGD {
    pub fn new(rate: f32, momentum: f32, decay: f32) -> Self {
        Self {
            rate,
            momentum,
            decay,
        }
    }
}

impl Optimizer for SGD {
    fn step(
        &self,
        weight: &mut Matrix,
        d_weight: &mut Matrix,
        v_weight: &mut Matrix,
        bias: &mut Matrix,
        d_bias: &mut Matrix,
        v_bias: &mut Matrix,
    ) {
        unsafe {
            let weight_decay = weight.mul_with_numeric(self.decay, false).unwrap();
            weight_decay.add(d_weight, true);
            let weight_go = weight_decay.mul_with_numeric(-self.rate, false).unwrap();
            if v_weight.is_null() {
                *v_weight = weight_go;
            } else {
                v_weight.mul_with_numeric(self.momentum, true);
                v_weight.add(&weight_go, true);
            }
            v_weight.clamp(-100.0, 100.0);
            weight.add(v_weight, true);

            let bias_go = d_bias.mul_with_numeric(-self.rate, false).unwrap();
            if v_bias.is_null() {
                *v_bias = bias_go;
            } else {
                v_bias.mul_with_numeric(self.momentum, true);
                v_bias.add(&bias_go, true);
            }
            v_bias.clamp(-100.0, 100.0);
            bias.add(v_bias, true);
        }
    }
}
