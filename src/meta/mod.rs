mod tile_split;
mod transpose;

use fixedbitset::FixedBitSet;
use nalgebra::{DMatrixView, Scalar};

/// 元信息张量。
///
/// 储存[张量](crate::Tensor)的元信息，由形状和数据组成。
/// 元信息张量的形状将广播到[张量](crate::Tensor)的分块。
#[derive(Clone, Debug)]
pub(crate) struct MetaTensor<T> {
    /// 元信息张量的形状，广播到[张量](crate::Tensor)的分块。
    pub shape: MetaShape,
    /// [张量](crate::Tensor)元信息数据。
    pub value: Vec<T>,
}

impl<T: Scalar> MetaTensor<T> {
    #[inline]
    pub fn as_matrix(&self) -> DMatrixView<T> {
        let nrows: usize = self.shape.0.len();
        let ncols = self.value.len() / nrows;
        DMatrixView::from_slice(&self.value, nrows, ncols)
    }
}

#[derive(Clone, Debug)]
#[repr(transparent)]
pub(crate) struct MetaShape(FixedBitSet);

impl ToString for MetaShape {
    #[inline]
    fn to_string(&self) -> String {
        format!("{:b}", self.0)
    }
}

impl MetaShape {
    #[inline]
    pub fn new(rank: usize) -> Self {
        Self(FixedBitSet::with_capacity(rank + 1))
    }
}
