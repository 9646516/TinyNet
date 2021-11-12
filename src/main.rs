use std::mem::ManuallyDrop;

pub mod core;

struct A {}

impl A {
    pub unsafe fn g(self) -> ManuallyDrop<A> {
        ManuallyDrop::new(self)
    }
}


impl Drop for A {
    fn drop(&mut self) {
        println!("drop");
    }
}

unsafe fn gao(mut a: A) {
    let mut p = ManuallyDrop::new(a);
    ;
}

fn main() {
    unsafe {
        let a = A {};
        println!("111");
        let b = a.g();
        println!("111");
    }
}