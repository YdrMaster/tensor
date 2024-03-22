//! 具有不连续存储结构的张量类型。

use std::iter::once;

/// 支持分块、不连续存储结构的张量类型。
///
/// # 注意
///
/// 张量仅关心元素的存储和访问，不关心元素的具体类型。
#[derive(Clone, Debug)]
pub struct Tensor<Storage> {
    /// 存储每个逻辑形状被分为几阶分块。
    shape_group: Vec<usize>,
    /// 按分块表示的张量形状。
    ///
    /// 为了对齐模式中的偏移项，增广一个维度，值固定为 1。
    tiles: Vec<isize>,
    /// 存储元素的所有对象组成的[元信息张量](MetaTensor)。
    storage: MetaTensor<Storage>,
    /// 每个元素的访问模式组成的[元信息张量](MetaTensor)。
    /// 这个设计方便用一次矩阵乘变换所有元素的访问模式。
    pattern: MetaTensor<isize>,
}

impl<S> Tensor<S> {
    /// 创建一个标准的张量。
    ///
    /// - 所有元素连续存储在一个存储空间；
    /// - 形状中维度按从范围从大到小排列（行优先）；
    /// - 所有维度不分块；
    pub fn new(shape: &[usize], storage: S) -> Self {
        let shape = shape.iter().chain(once(&1)).copied().collect::<Vec<_>>();
        Self {
            shape_group: vec![1; shape.len() - 1],
            tiles: shape.iter().map(|&len| len as _).collect(),
            storage: MetaTensor {
                shape: shape.clone(),
                value: vec![storage],
            },
            pattern: MetaTensor {
                shape: shape.clone(),
                value: vec![0; shape.iter().product()],
            },
        }
    }

    /// 获取张量逻辑形状。
    pub fn shape(&self) -> Vec<usize> {
        self.shape_group
            .iter()
            .scan(self.tiles.as_slice(), |tiles, &len| {
                let (head, tail) = tiles.split_at(len);
                *tiles = tail;
                Some(head.iter().product::<isize>() as usize)
            })
            .collect()
    }

    /// 获取张量中的元素个数。
    #[inline]
    pub fn size(&self) -> usize {
        self.tiles.iter().product::<isize>() as _
    }
}

#[test]
fn test() {
    let tensor = Tensor::new(&[2, 3, 4], vec![0.0f32; 2 * 3 * 4]);
    assert_eq!(tensor.shape(), &[2, 3, 4]);
    assert_eq!(tensor.size(), 24);

    let tensor = Tensor::new(&[], 0.0f32);
    assert_eq!(tensor.shape(), &[]);
    assert_eq!(tensor.size(), 1);
}

/// 元信息张量。
///
/// 储存[张量](crate::Tensor)的元信息，由形状和数据组成。
/// 元信息张量的形状将广播到[张量](crate::Tensor)的分块。
#[derive(Clone, Debug)]
struct MetaTensor<T> {
    /// 元信息张量的形状，广播到[张量](crate::Tensor)的分块。
    shape: Vec<usize>,
    /// [张量](crate::Tensor)元信息数据。
    value: Vec<T>,
}
