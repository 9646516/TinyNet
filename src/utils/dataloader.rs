use crate::utils::mat::Matrix;
use crate::utils::misc::random_shuffle;
use crate::utils::nn_trait::DataSet;
use rayon::prelude::*;
use std::arch::x86_64;
use std::cmp::min;
use std::mem::size_of;

pub struct DataLoader<'a, T>
where
    T: DataSet + std::marker::Sync,
{
    dataset: &'a T,
    batch_size: usize,
    count: usize,
    order: Vec<usize>,
}

impl<'a, T> DataLoader<'a, T>
where
    T: DataSet + std::marker::Sync,
{
    pub fn new(dataset: &'a T, batch_size: usize, seed: u32) -> Self {
        let mut order = (0..dataset.len()).collect::<Vec<_>>();
        random_shuffle(&mut order, seed);
        Self {
            dataset,
            batch_size,
            count: 0,
            order,
        }
    }

    pub unsafe fn fetch_batch(&self, start: usize, len: usize) -> (Matrix, Matrix) {
        let image = Matrix::new(len, self.dataset.dim());
        let gt = Matrix::new(len, 10);
        gt.fill_(0.0);
        (start..start + len)
            .into_par_iter()
            .enumerate()
            .map(|(batch_idx, idx)| {
                let idx = self.order.get(idx).unwrap().to_owned();
                (batch_idx, self.dataset.fetch_item(idx as isize))
            })
            .for_each(|(idx, (fetched_image, fetched_gt))| {
                let fetched_image = fetched_image.as_ptr();
                let to = image.row_at(idx as isize);
                for k in (0..image.number_of_real_col()).step_by(32 / size_of::<f32>()) {
                    let source = fetched_image.add(k);
                    let destination = to.add(k);
                    let val = x86_64::_mm256_loadu_ps(source);
                    x86_64::_mm256_storeu_ps(destination, val);
                }
                *gt.row_at(idx as isize).add(fetched_gt as usize) = 1.0;
            });

        (image, gt)
    }
}

impl<'a, T> Iterator for DataLoader<'a, T>
where
    T: DataSet + std::marker::Sync,
{
    type Item = (Matrix, Matrix);

    fn next(&mut self) -> Option<Self::Item> {
        if self.count >= self.dataset.len() {
            None
        } else {
            let left = self.dataset.len() - self.count;
            let sz = min(left, self.batch_size);
            unsafe {
                let ret = Some(self.fetch_batch(self.count, sz));
                self.count += sz;
                ret
            }
        }
    }
}
