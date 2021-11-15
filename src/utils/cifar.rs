use crate::utils::mat::Matrix;
use crate::utils::nn_trait::DataSet;
use image::io::Reader as ImageReader;
use rayon::prelude::*;

pub struct CifarData {
    image: Matrix,
    gt: Vec<u8>,
    len: usize,
}

impl CifarData {
    pub unsafe fn new(source_dir: &str, size: i32) -> Self {
        let mut x = std::fs::read_dir(source_dir)
            .unwrap()
            .into_iter()
            .map(|x| x.unwrap().file_name().into_string().unwrap())
            .collect::<Vec<_>>();
        if size > 0 && size < x.len() as i32 {
            x.resize(size as usize, String::new());
        }
        let L = x.len();
        let mut gt = Vec::new();
        let image = Matrix::new(L, 32 * 32 * 3);
        for x in x.iter() {
            let t = x.as_str().split('_').nth(1).unwrap();
            let t2 = &t[0..t.len() - 4];
            let g = Self::generate_lable(t2);
            gt.push(g);
        }
        x.into_par_iter().enumerate().for_each(|(idx, x)| {
            let source = format!("{}/{}", source_dir, x);
            let img = ImageReader::open(source)
                .unwrap()
                .decode()
                .unwrap()
                .to_rgb8();
            let buf = img.as_raw();
            let row = image.row(idx as isize);
            for i in 0..buf.len() {
                *row.add(i) = buf.get(i as usize).unwrap().to_owned() as f32 / 255.0;
            }
        });
        Self { image, gt, len: L }
    }
    pub fn generate_lable(x: &str) -> u8 {
        match x {
            "airplane" => 0,
            "automobile" => 1,
            "bird" => 2,
            "cat" => 3,
            "deer" => 4,
            "dog" => 5,
            "frog" => 6,
            "horse" => 7,
            "ship" => 8,
            "truck" => 9,
            _ => {
                panic!("bad label");
            }
        }
    }
}

impl DataSet for CifarData {
    fn dim(&self) -> usize {
        32 * 32 * 3
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len != 0
    }

    unsafe fn fetch_item(&self, idx: isize) -> (&[f32], u8) {
        if idx >= self.len as isize || idx < 0 {
            panic!("fetch item with incorrect idx");
        }
        let image = self.image.row(idx);
        let image = std::slice::from_raw_parts(image, self.dim());
        let gt = self.gt.get(idx as usize).unwrap().to_owned();

        (image, gt)
    }
}
