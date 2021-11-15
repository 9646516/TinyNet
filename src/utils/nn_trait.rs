use crate::utils::mat::Matrix;

pub trait Layer {
    fn forward(&mut self, input: Matrix) -> Matrix;
    fn backward(&mut self, dLoss: Matrix) -> Matrix;
    fn update_parameters(&mut self, rate: f32, decay: f32);
}

pub trait Head {
    fn forward(&mut self, input: Matrix, target: Matrix) -> Matrix;
    fn backward(&mut self, dLoss: Matrix) -> Matrix;
    fn eval_forward(&self, input: Matrix) -> Vec<usize>;
}

pub trait DataSet {
    fn dim(&self) -> usize;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    unsafe fn fetch_item(&self, idx: isize) -> (&[f32], u8);
}
