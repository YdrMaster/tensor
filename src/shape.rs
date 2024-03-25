use nalgebra::DVector;
use std::{
    iter::zip,
    ops::{Deref, DerefMut},
};

#[derive(Clone, Debug)]
#[repr(transparent)]
pub(crate) struct Shape(pub DVector<isize>);

impl Deref for Shape {
    type Target = [isize];
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

impl DerefMut for Shape {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut_slice()
    }
}

impl Shape {
    /// 维度切分变换。
    ///
    /// 将形状的 `axis` 维度依 `tiles` 指定的方式切分。
    pub fn split_dim(mut self, axis: usize, tiles: &[usize]) -> Self {
        self.0 = self.0.insert_rows(axis, tiles.len() - 1, 1);
        match &mut self[axis..][..tiles.len()] {
            [.., 1] => {}
            body => {
                for (dst, src) in zip(body, tiles) {
                    *dst = *src as _;
                }
            }
        }
        self
    }
}

#[test]
fn test() {
    let shape = Shape(DVector::from_vec(vec![2, 3, 4, 1]));
    assert_eq!(&*shape.clone().split_dim(2, &[2, 2]), &[2, 3, 2, 2, 1]);
    assert_eq!(&*shape.split_dim(3, &[2, 3, 4]), &[2, 3, 4, 1, 1, 1]);
}
