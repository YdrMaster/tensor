use super::MetaTensor;

impl<T> MetaTensor<T> {
    /// 元信息张量再切分变换。
    ///
    /// 再切分运算不改变数据的数量和排布，只改变元信息。
    #[inline]
    pub fn tile_split(self, axis: usize, tiles: &[usize]) -> Self {
        Self {
            shape: self.shape.split(axis, tiles.len() - 1),
            ..self
        }
    }
}
