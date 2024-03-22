use crate::Tensor;

pub trait Splitable {
    fn split(&self) -> Self;
}

impl<T: Clone> Splitable for T {
    #[inline]
    fn split(&self) -> Self {
        self.clone()
    }
}

impl<Storage> Tensor<Storage> {
    /// 块再切分。
    #[inline]
    pub fn tile_split(mut self, axis: usize, tiles: &[usize]) -> Self {
        // 增广形状不可分
        assert!(axis < self.tiles.len() - 1);
        // 分块数不匹配
        assert_eq!(tiles.iter().product::<usize>(), self.tiles[axis] as _);

        let i = self
            .shape_groups
            .iter()
            .scan(0, |acc, &len| {
                *acc += len;
                Some(*acc)
            })
            .take_while(|&acc| acc < axis)
            .count();
        self.shape_groups[i] += tiles.len() - 1;

        let tail = self.tiles.split_off(axis);
        self.tiles.extend(tiles.iter().map(|&t| t as isize));
        self.tiles.extend_from_slice(&tail[1..]);

        todo!("split pattern and storage");

        self
    }
}

#[test]
fn test() {
    let tensor = Tensor::new(&[6, 10], ());
    assert_eq!(tensor.shape(), &[6, 10]);
    assert_eq!(tensor.tiles(), &[6, 10, 1]);
    assert_eq!(tensor.pattern.shape, &[1, 1, 1]);
    assert_eq!(tensor.pattern.value, &[10, 1, 0]);
    assert_eq!(tensor.storage.shape, &[1, 1, 1]);
    assert_eq!(tensor.storage.value.len(), 1);
    assert_eq!(tensor.size(), 60);

    let tensor = tensor.tile_split(0, &[2, 3]);
    assert_eq!(tensor.shape(), &[6, 10]);
    assert_eq!(tensor.tiles(), &[2, 3, 10, 1]);
    assert_eq!(tensor.pattern.shape, &[1, 1, 1, 1]);
    assert_eq!(tensor.pattern.value, &[30, 10, 1, 0]);
    assert_eq!(tensor.storage.shape, &[1, 1, 1, 1]);
    assert_eq!(tensor.storage.value.len(), 1);
    assert_eq!(tensor.size(), 60);
}
