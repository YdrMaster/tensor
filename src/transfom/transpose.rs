use crate::Tensor;
use nalgebra::DMatrix;
use std::{collections::BTreeMap, iter::zip, ops::Range};

impl<Storage> Tensor<Storage> {
    /// 块转置变换。
    pub fn tile_transpose(self, perm: &[usize]) -> Self {
        if perm.is_empty() {
            return self;
        }
        assert!(perm.len() < self.tiles.len());
        // 构造置换维度映射
        let btree = btree_from_perm(perm);
        assert_eq!(btree.len(), perm.len());
        assert!(*btree.keys().last().unwrap() < self.tiles.len() - 1);
        if btree.iter().all(|(k, v)| k == v) {
            return self;
        }
        // 执行变换
        self._transpose(btree)
    }

    pub fn transpose(mut self, perm: &[usize]) -> Self {
        if perm.is_empty() {
            return self;
        }
        assert!(perm.len() < self.shape_groups.len());
        // 构造置换维度映射
        let btree = btree_from_perm(perm);
        assert_eq!(btree.len(), perm.len());
        assert!(*btree.keys().last().unwrap() < self.tiles.len() - 1);
        if btree.iter().all(|(k, v)| k == v) {
            return self;
        }
        // 转置形状组
        let shape_groups = (0..self.shape_groups.len())
            .map(|i| self.shape_groups[*btree.get(&i).unwrap_or(&i)])
            .collect::<Vec<_>>();
        // 转置前后的块组
        let range_in = group_tiles_range(&self.shape_groups);
        let range_out = group_tiles_range(&shape_groups);
        let mut tiles_btree = BTreeMap::new();
        for (dst, src) in btree {
            tiles_btree.extend(zip(range_out[dst].clone(), range_in[src].clone()));
        }
        // 执行变换
        self.shape_groups = shape_groups;
        self._transpose(tiles_btree)
    }

    fn _transpose(mut self, btree: BTreeMap<usize, usize>) -> Self {
        // 变换块形状
        self.tiles = self.tiles.transpose(&btree);
        // 变换访存模式
        let n = self.tiles.len();
        let affine = DMatrix::from_fn(n, n, |r, c| {
            if c == btree.get(&r).copied().unwrap_or(r) {
                1
            } else {
                0
            }
        });
        self.pattern.value = (affine * self.pattern.as_matrix()).data.into();
        // 变换元信息张量形状
        self.pattern.shape = self.pattern.shape.transpose(&btree);
        self.storage.shape = self.storage.shape.transpose(&btree);

        self
    }
}

/// 从形状组计算分块的轴范围。
fn group_tiles_range(shape_groups: &[usize]) -> Vec<Range<usize>> {
    shape_groups
        .iter()
        .scan(0, |acc, &d| {
            let range = *acc..*acc + d;
            *acc = range.end;
            Some(range)
        })
        .collect()
}

/// 使用 B 树，从 `perm` 生成有序的轴映射关系。
#[inline]
fn btree_from_perm(perm: &[usize]) -> BTreeMap<usize, usize> {
    let mut btree = perm.iter().map(|&p| (p, p)).collect::<BTreeMap<_, _>>();
    zip(&mut btree, perm).for_each(|((_, v), &p)| *v = p);
    btree
}

#[test]
fn test_tiles() {
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

#[test]
fn test_shape() {
    let tensor = Tensor::new(&[4, 6, 8, 9], ())
        .tile_split(1, &[2, 3])
        .tile_split(3, &[2, 2, 2]);
    assert_eq!(tensor.shape(), &[4, 6, 8, 9]);
    assert_eq!(tensor.tiles(), &[4, 2, 3, 2, 2, 2, 9, 1]);
    assert_eq!(tensor.pattern.shape.to_string(), "00000000");
    assert_eq!(tensor.pattern.value, &[432, 216, 72, 36, 18, 9, 1, 0]);
    assert_eq!(tensor.storage.shape.to_string(), "00000000");
    assert_eq!(tensor.storage.value.len(), 1);
    assert_eq!(tensor.size(), 1728);

    let tensor = tensor.transpose(&[3, 1, 2]);
    assert_eq!(tensor.shape(), &[4, 9, 6, 8]);
    assert_eq!(tensor.tiles(), &[4, 9, 2, 3, 2, 2, 2, 1]);
    assert_eq!(tensor.pattern.shape.to_string(), "00000000");
    assert_eq!(tensor.pattern.value, &[432, 1, 216, 72, 36, 18, 9, 0]);
    assert_eq!(tensor.storage.shape.to_string(), "00000000");
    assert_eq!(tensor.storage.value.len(), 1);
    assert_eq!(tensor.size(), 1728);
}
