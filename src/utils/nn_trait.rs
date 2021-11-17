use crate::utils::mat::Matrix;

pub trait Layer {
    fn forward(&mut self, input: Matrix) -> Matrix;
    fn backward(&mut self, dLoss: Matrix) -> Matrix;
    fn trainable(&self) -> bool;
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
        None
    }
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

pub trait Optimizer {
    fn step(
        &self,
        weight: &mut Matrix,
        d_weight: &mut Matrix,
        v_weight: &mut Matrix,
        bias: &mut Matrix,
        d_bias: &mut Matrix,
        v_bias: &mut Matrix,
    );
}
