#![allow(clippy::missing_safety_doc)]
#![allow(non_snake_case)]

use crate::utils::conv3x3::Conv3x3;
use crate::utils::dataloader::DataLoader;
use crate::utils::head::SoftMaxCrossEntropy;
use crate::utils::linear::LinearLayer;
use crate::utils::maxpool2x2::MaxPool2x2;
use crate::utils::mnist::MnistData;
use crate::utils::network::Network;
use crate::utils::nn_trait::{DataSet, Layer};
use crate::utils::relu::ReluLayer;

pub mod utils;

fn main() {
    unsafe {
        let mnist_train_path = r"C:\Users\Rinne\Desktop\mnist\train";
        let train_dataset = MnistData::new(mnist_train_path, -1);
        let mnist_test_path = r"C:\Users\Rinne\Desktop\mnist\test";
        let test_dataset = MnistData::new(mnist_test_path, -1);

        let layers: Vec<Box<dyn Layer>> = vec![
            Box::new(Conv3x3::new(1, 4, 28, 28, 1, 1)),
            Box::new(ReluLayer::new()),
            Box::new(MaxPool2x2::new(4, 28, 28)),
            Box::new(Conv3x3::new(4, 8, 14, 14, 1, 1)),
            Box::new(LinearLayer::new(14 * 14 * 8, 128)),
            Box::new(ReluLayer::new()),
            Box::new(LinearLayer::new(128, 10)),
        ];
        let loss_fn = Box::new(SoftMaxCrossEntropy::new());

        let mut network = Network::new(layers, loss_fn);

        let rate = 0.01f32;
        let momentum = 0.9f32;
        let decay = 0.0001f32;

        for i in 0..1 {
            let mut iter = 0;
            let dataloader = DataLoader::new(&train_dataset, 128, i << 10);
            for (image, gt) in dataloader {
                iter += 1;
                let pred = network.forward(image);
                let loss = network.calc_loss(pred, gt);

                let (h, _) = loss.shape();
                let sum = (0..h)
                    .into_iter()
                    .map(|idx| loss.at(idx as isize, 0))
                    .fold(0f32, |a, b| a + b);
                if iter % 10 == 0 {
                    println!("epoch {}, iter {}, loss {}", i, iter, sum / h as f32);
                }
                network.backward(loss);
                network.update_parameters(rate, momentum, decay);
            }
            println!("testing");
            let dataloader = DataLoader::new(&test_dataset, 1024, 0);
            let mut ok = 0;
            for (image, gt) in dataloader {
                let pred = network.forward(image);
                let result = network.get_result(pred);
                let sz = result.len();
                (0..sz).into_iter().for_each(|idx| {
                    let res = *result.get(idx).unwrap();
                    if (*gt.row_at(idx as isize).add(res) - 1.0).abs() < 1e-5 {
                        ok += 1;
                    }
                });
            }
            println!(
                "epoch {},acc [{}/{}],{:.4}%",
                i,
                ok,
                test_dataset.len(),
                ok as f32 / test_dataset.len() as f32 * 100.0
            )
        }
    }
}
