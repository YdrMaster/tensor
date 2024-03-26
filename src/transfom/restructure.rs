use crate::Tensor;

impl<Storage> Tensor<Storage> {
    /// 重组张量形状的块分组。
    #[inline]
    pub fn restructure(mut self, shape_groups: Vec<usize>) -> Self {
        assert_eq!(
            shape_groups.iter().sum::<usize>(),
            self.shape_groups.iter().sum()
        );
        self.shape_groups = shape_groups;
        self
    }
}

#[test]
fn test() {
    let tensor = Tensor::new(&[2, 3, 4], ());
    assert_eq!(tensor.shape(), &[2, 3, 4]);

    let tensor = tensor.restructure(vec![3]);
    assert_eq!(tensor.shape(), &[24]);

    let tensor = tensor.restructure(vec![2, 1]);
    assert_eq!(tensor.shape(), &[6, 4]);
}
