use super::{MetaShape, MetaTensor};
use std::collections::BTreeMap;

impl<T> MetaTensor<T> {
    /// 元信息张量转置。
    #[inline]
    pub fn transpose(self, tiles: &[usize], btree: &BTreeMap<usize, usize>) -> Self {
        Self {
            shape: {
                let mut shape = self.shape.0.clone();
                for (&dst, &src) in btree {
                    shape.set(dst, self.shape.0[src]);
                }
                MetaShape(shape)
            },
            ..self
        }
    }
}

#[test]
fn test_transpose() {
    let &tiles = &[1, 3, 4, 4];
    let mut meta = MetaTensor {
        shape: super::MetaShape::new(tiles.len()),
        value: vec![],
    };
    meta.shape.0.insert(1);
    meta.shape.0.insert(2);
    let size = (tiles.len() + 1) * meta.shape.0.ones().map(|i| tiles[i]).product::<usize>();
    meta.value = (0..size).collect();

    assert_eq!(meta.shape.to_string(), "01100");
    println!("{}", meta.as_matrix());
}
