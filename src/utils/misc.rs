use crate::utils::mat::Matrix;
use std::ops::Shl;

pub fn check_abnormal(x: &Matrix) {
    let (h, w) = x.shape();
    unsafe {
        for i in 0..h {
            for j in 0..w {
                let cur = x.at(i as isize, j as isize);
                if cur.is_infinite() {
                    panic!("is_infinite")
                }
                if cur.is_nan() {
                    panic!("is_nan")
                }
            }
        }
    }
}

pub fn rand_next(seed: &mut u32) -> u32 {
    let lo = (std::num::Wrapping(16807) * std::num::Wrapping(*seed & 0xFFFF)).0;
    let hi = (std::num::Wrapping(16807) * std::num::Wrapping(*seed >> 16)).0;
    let val = (std::num::Wrapping(lo)
        + std::num::Wrapping(hi & 0x7FFF).shl(16)
        + std::num::Wrapping(hi >> 15))
    .0;

    *seed = if val > 0x7FFFFFFF {
        val - 0x7FFFFFFF
    } else {
        val
    };
    *seed
}

pub fn random_shuffle(x: &mut Vec<usize>, mut seed: u32) {
    let len = x.len();
    for i in 0..(len - 1) {
        let left = (len - i) as u32;
        let offset = rand_next(&mut seed) % left;
        if offset != 0 {
            let j = i + offset as usize;
            let lhs = *x.get(i).unwrap();
            let rhs = *x.get(j).unwrap();
            *x.get_mut(i).unwrap() = rhs;
            *x.get_mut(j).unwrap() = lhs;
        }
    }
}
