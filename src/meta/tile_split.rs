use super::MetaTensor;

impl<T> MetaTensor<T> {
    /// 元信息张量再切分变换。
    ///
    /// 再切分运算不改变数据的数量和排布，只改变元信息。
    #[inline]
    pub fn tile_split(mut self, axis: usize, tiles: &[usize]) -> Self {
        let insert = tiles.len() - 1;
        let bits = &mut self.shape.0;
        bits.grow(bits.len() + insert);
        for i in (axis + insert..bits.len()).rev() {
            bits.copy_bit(i - insert, i);
        }
        bits.set_range(axis..axis + insert, bits[axis + insert]);

        self
    }
}

#[test]
fn test() {
    let mut meta = MetaTensor {
        shape: super::MetaShape::new(4),
        value: vec![()],
    };
    meta.shape.0.insert(2);
    meta.shape.0.insert(3);
    assert_eq!(meta.shape.to_string(), "00110");
    {
        let meta = meta.clone().tile_split(2, &[2, 2]);
        assert_eq!(meta.shape.to_string(), "001110");
    }
    {
        let meta = meta.clone().tile_split(1, &[2, 2]);
        assert_eq!(meta.shape.to_string(), "000110");
    }
}
