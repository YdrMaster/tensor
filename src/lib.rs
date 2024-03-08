//! 具有不连续存储结构的张量类型。

mod logical;
mod strides;

use logical::Logical;
use strides::Strides;

/// 具有不连续存储结构的张量类型。
#[derive(Clone, Debug)]
pub struct Tensor<Physical> {
    logical: Logical,
    strides: Strides,
    physical: Physical,
}

impl<Physical> Tensor<Physical> {
    pub fn new(shape: &[usize], unit: usize, physical: Physical) -> Self {
        Self {
            logical: Logical::new(shape),
            strides: Strides::new(shape, unit),
            physical,
        }
    }
}

#[test]
fn test() {
    let frame = Tensor::new(&[20, 30, 40], 2, ());
    assert_eq!(frame.logical.shape_vec(), &[20, 30, 40]);
    assert_eq!(frame.logical.tile(0).unwrap().shape_vec(), &[20]);
    assert_eq!(frame.logical.tile(1).unwrap().shape_vec(), &[30]);
    assert_eq!(frame.logical.tile(2).unwrap().shape_vec(), &[40]);
}
