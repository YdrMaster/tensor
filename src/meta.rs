use crate::MetaTensor;
use std::iter::repeat;

impl<T> MetaTensor<T> {
    /// 元信息张量的阶数。
    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// 形状切分变换。
    ///
    /// 将第 `axis` 维度的形状依 `tiles` 指定的方式切分。
    pub fn shape_split(mut self, axis: usize, tiles: &[usize]) -> Self {
        let tail = self.shape.split_off(axis);
        if tail[0] == 1 {
            self.shape.extend(repeat(1).take(tiles.len()));
        } else {
            self.shape.extend_from_slice(&tiles);
        }
        self.shape.extend_from_slice(&tail[1..]);
        self
    }
}
