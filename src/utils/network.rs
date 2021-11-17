use crate::utils::mat::Matrix;
use crate::utils::nn_trait::{Head, Layer, Optimizer};

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
    pub head: Box<dyn Head>,
    pub opt: Box<dyn Optimizer>,
}

impl Network {
    pub fn new(layers: Vec<Box<dyn Layer>>, head: Box<dyn Head>, opt: Box<dyn Optimizer>) -> Self {
        Self { layers, head, opt }
    }
    pub fn forward(&mut self, mut x: Matrix) -> Matrix {
        for layer in self.layers.iter_mut() {
            x = layer.forward(x);
        }
        x
    }

    pub unsafe fn calc_loss(&mut self, pred: Matrix, target: Matrix) -> Matrix {
        self.head.forward(pred, target)
    }

    pub unsafe fn get_result(&self, pred: Matrix) -> Vec<usize> {
        self.head.eval_forward(pred)
    }

    pub fn backward(&mut self, x: Matrix) {
        let mut x = self.head.backward(x);
        for layer in self.layers.iter_mut().rev() {
            x = layer.backward(x);
        }
    }
    pub fn update_parameters(&mut self) {
        for layer in self.layers.iter_mut() {
            if layer.trainable() {
                let (weight, d_weight, v_weight, bias, d_bias, v_bias) =
                    layer.parameters().unwrap();
                self.opt
                    .step(weight, d_weight, v_weight, bias, d_bias, v_bias);
            }
        }
    }
}
