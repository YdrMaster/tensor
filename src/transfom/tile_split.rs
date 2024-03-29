﻿use crate::Tensor;

impl<Storage> Tensor<Storage> {
    /// 块再切分变换。
    ///
    /// 将第 `axis` 维度的块依 `tiles` 指定的方式切分。
    pub fn tile_split(mut self, axis: usize, tiles: &[usize]) -> Self {
        match tiles {
            [] => panic!(),
            [d] => {
                assert_eq!(*d, self.tiles[axis] as _);
                return self;
            }
            _ => {
                // 增广形状不可分
                assert!(axis < self.tiles.len() - 1);
                // 分块数不匹配
                assert_eq!(tiles.iter().product::<usize>(), self.tiles[axis] as _);
            }
        }
        // 增加的维度数量
        let insert = tiles.len() - 1;
        // 找到 axis 块对应的形状维度
        let i = self
            .shape_groups
            .iter()
            .scan(axis, |rest, &len| {
                rest.checked_sub(len).map(|val| *rest = val)
            })
            .count();
        self.shape_groups[i] += insert;
        // 插入分块
        self.tiles = self.tiles.tile_split(axis, tiles);
        // 变换访存模式
        let pattern_mat = self.pattern.as_matrix();
        let nrows = pattern_mat.nrows();
        let ncols = pattern_mat.ncols();
        let pattern = &mut self.pattern.value;
        pattern.reserve(ncols * insert);
        #[allow(clippy::uninit_vec)]
        unsafe {
            pattern.set_len(ncols * (nrows + insert))
        };
        for i in (0..ncols).rev() {
            let src = i * nrows;
            let dst = src + i * insert;
            pattern.copy_within(src..src + axis, dst);
            pattern.copy_within(src + axis..src + nrows, dst + axis + insert);
            for i in (dst + axis..dst + axis + insert).rev() {
                pattern[i] = pattern[i + 1] * self.tiles[i + 1] as isize;
            }
        }
        // 变换元信息张量
        self.pattern = self.pattern.tile_split(axis, tiles);
        self.storage = self.storage.tile_split(axis, tiles);

        self
    }
}

#[test]
fn test() {
    let tensor = Tensor::new(&[6, 10], ());
    assert_eq!(tensor.shape(), &[6, 10]);
    assert_eq!(tensor.tiles(), &[6, 10, 1]);
    assert_eq!(tensor.pattern.shape.to_string(), "000");
    assert_eq!(tensor.pattern.value, &[10, 1, 0]);
    assert_eq!(tensor.storage.shape.to_string(), "000");
    assert_eq!(tensor.storage.value.len(), 1);
    assert_eq!(tensor.size(), 60);

    let tensor = tensor.tile_split(1, &[2, 5]);
    assert_eq!(tensor.shape(), &[6, 10]);
    assert_eq!(tensor.tiles(), &[6, 2, 5, 1]);
    assert_eq!(tensor.pattern.shape.to_string(), "0000");
    assert_eq!(tensor.pattern.value, &[10, 5, 1, 0]);
    assert_eq!(tensor.storage.shape.to_string(), "0000");
    assert_eq!(tensor.storage.value.len(), 1);
    assert_eq!(tensor.size(), 60);
}
