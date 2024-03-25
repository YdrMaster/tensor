use crate::Tensor;
use nalgebra::{DMatrix, DMatrixView};
use std::{collections::BTreeMap, iter::zip};

impl<Storage> Tensor<Storage> {
    /// 块转置变换。
    #[inline]
    pub fn tile_transpose(mut self, perm: &[usize]) -> Self {
        if perm.is_empty() {
            return self;
        }
        let n = self.tiles.len();
        assert!(perm.len() < n);
        // 构造置换维度映射。
        let mut btree = perm.iter().map(|&p| (p, p)).collect::<BTreeMap<_, _>>();
        assert_eq!(btree.len(), perm.len());
        assert!(*btree.keys().last().unwrap() < n - 1);
        zip(&mut btree, perm).for_each(|((_, v), &p)| *v = p);
        // 若实际上不需要置换，返回原张量。
        if btree.iter().all(|(k, v)| k == v) {
            return self;
        }

        // 变换块形状。
        self.tiles = self.tiles.transpose(&btree);
        // 变换元信息张量形状。
        self.pattern.shape = self.pattern.shape.transpose(&btree);
        self.storage.shape = self.storage.shape.transpose(&btree);
        // 变换访存模式。
        let affine = DMatrix::from_fn(n, n, |r, c| {
            if c == btree.get(&r).copied().unwrap_or(r) {
                1
            } else {
                0
            }
        });
        let pattern = DMatrixView::from_slice(&self.pattern.value, n, self.pattern.value.len() / n);
        self.pattern.value = (affine * pattern).data.into();

        self
    }
}

#[test]
fn test() {
    let tensor = Tensor::new(&[2, 3, 4, 5], ());
    assert_eq!(tensor.shape(), &[2, 3, 4, 5]);
    assert_eq!(tensor.tiles(), &[2, 3, 4, 5, 1]);
    assert_eq!(tensor.pattern.shape.to_string(), "00000");
    assert_eq!(tensor.pattern.value, &[60, 20, 5, 1, 0]);
    assert_eq!(tensor.storage.shape.to_string(), "00000");
    assert_eq!(tensor.storage.value.len(), 1);
    assert_eq!(tensor.size(), 120);

    let tensor = tensor.tile_transpose(&[3, 1, 2]);
    assert_eq!(tensor.shape(), &[2, 5, 3, 4]);
    assert_eq!(tensor.tiles(), &[2, 5, 3, 4, 1]);
    assert_eq!(tensor.pattern.shape.to_string(), "00000");
    assert_eq!(tensor.pattern.value, &[60, 1, 20, 5, 0]);
    assert_eq!(tensor.storage.shape.to_string(), "00000");
    assert_eq!(tensor.storage.value.len(), 1);
    assert_eq!(tensor.size(), 120);
}
