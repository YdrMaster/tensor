use crate::Tensor;

impl<Storage> Tensor<Storage> {
    /// 重组张量形状的块分组。
    #[inline]
    pub fn tile_transpose(self, perm: &[usize]) -> Self {
        assert!(perm.len() <= self.tiles.len() - 1);
        todo!("")
    }
}

#[test]
fn test() {
    let tensor = Tensor::new(&[2, 3, 4, 5], ());
    assert_eq!(tensor.shape(), &[2, 3, 4, 5]);
    assert_eq!(tensor.tiles(), &[2, 3, 4, 5, 1]);
    assert_eq!(tensor.pattern.shape, &[1, 1, 1, 1, 1]);
    assert_eq!(tensor.pattern.value, &[60, 20, 5, 1, 0]);
    assert_eq!(tensor.storage.shape, &[1, 1, 1, 1, 1]);
    assert_eq!(tensor.storage.value.len(), 1);
    assert_eq!(tensor.size(), 120);

    let tensor = tensor.tile_transpose(&[3, 1, 2]);
    assert_eq!(tensor.shape(), &[2, 5, 3, 4]);
    assert_eq!(tensor.tiles(), &[2, 5, 3, 4, 1]);
    assert_eq!(tensor.pattern.shape, &[1, 1, 1, 1, 1]);
    assert_eq!(tensor.pattern.value, &[60, 1, 20, 5, 0]);
    assert_eq!(tensor.pattern.shape, &[1, 1, 1, 1, 1]);
    assert_eq!(tensor.storage.value.len(), 1);
    assert_eq!(tensor.size(), 120);
}
