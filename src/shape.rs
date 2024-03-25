use nalgebra::DVector;
use std::{
    iter::zip,
    ops::{Deref, DerefMut},
};

#[derive(Clone, Debug)]
#[repr(transparent)]
pub(crate) struct Shape(pub DVector<isize>);

impl Shape {
    /// 形状切分变换。
    ///
    /// 将第 `axis` 维度的形状依 `tiles` 指定的方式切分。
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
