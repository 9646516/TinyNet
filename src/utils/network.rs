use crate::utils::mat::Matrix;
use crate::utils::nn_trait::{Head, Layer};

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
    pub head: Box<dyn Head>,
}

impl Network {
    pub fn new(layers: Vec<Box<dyn Layer>>, loss_fn: Box<dyn Head>) -> Self {
        Self {
            layers,
            head: loss_fn,
        }
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
    pub fn update_parameters(&mut self, rate: f32, momentum: f32, decay: f32) {
        for layer in self.layers.iter_mut() {
            layer.update_parameters(rate, momentum, decay);
        }
    }
}
