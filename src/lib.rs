//! 具有不连续存储结构的张量类型。

mod meta;
mod tiles;
mod transfom;

use meta::{MetaShape, MetaTensor};
use std::iter::once;
use tiles::Tiles;

pub use transfom::SliceDim;

/// 支持分块、不连续存储结构的张量类型。
///
/// # 注意
///
/// 张量仅关心元素的存储和访问，不关心元素的具体类型。
#[derive(Clone, Debug)]
pub struct Tensor<Storage> {
    /// 存储每个逻辑形状被分为几阶分块。
    shape_groups: Vec<usize>,
    /// 按分块表示的张量形状。
    ///
    /// 为了对齐模式中的偏移项，增广一个维度，值固定为 1。
    tiles: Tiles,
    /// 每个元素的访问模式组成的[元信息张量](MetaTensor)。
    /// 这个设计方便用一次矩阵乘变换所有元素的访问模式。
    pattern: MetaTensor<isize>,
    /// 存储元素的所有对象组成的[元信息张量](MetaTensor)。
    storage: MetaTensor<Storage>,
}

impl<S> Tensor<S> {
    /// 创建一个标准的张量。
    ///
    /// - 所有元素连续存储在一个存储空间；
    /// - 形状中维度按从范围从大到小排列（行优先）；
    /// - 所有维度不分块；
    pub fn new(shape: &[usize], storage: S) -> Self {
        Self {
            shape_groups: vec![1; shape.len()],
            tiles: Tiles::from_iter(shape.iter().copied().chain(once(1))),
            pattern: MetaTensor {
                shape: MetaShape::new(shape.len()),
                value: {
                    let mut strides = once(0)
                        .chain(shape.iter().rev().scan(1, |acc, &d| {
                            let f = *acc;
                            *acc *= d;
                            Some(f as _)
                        }))
                        .collect::<Vec<_>>();
                    strides.reverse();
                    strides
                },
            },
            storage: MetaTensor {
                shape: MetaShape::new(shape.len()),
                value: vec![storage],
            },
        }
    }

    /// 获取张量逻辑形状。
    pub fn shape(&self) -> Vec<usize> {
        self.shape_groups
            .iter()
            .scan(&*self.tiles, |tiles, &len| {
                let (head, tail) = tiles.split_at(len);
                *tiles = tail;
                Some(head.iter().product())
            })
            .collect()
    }

    /// 获取张量逻辑分块结构。
    #[inline]
    pub fn tiles(&self) -> &[usize] {
        &self.tiles
    }

    /// 获取张量中的元素个数。
    #[inline]
    pub fn size(&self) -> usize {
        self.tiles.iter().product()
    }
}

#[test]
fn test() {
    let tensor = Tensor::new(&[2, 3, 4], vec![0.0f32; 2 * 3 * 4]);
    assert_eq!(tensor.shape(), &[2, 3, 4]);
    assert_eq!(tensor.tiles(), &[2, 3, 4, 1]);
    assert_eq!(tensor.pattern.shape.to_string(), "0000");
    assert_eq!(tensor.pattern.value, &[12, 4, 1, 0]);
    assert_eq!(tensor.storage.shape.to_string(), "0000");
    assert_eq!(tensor.storage.value.len(), 1);
    assert_eq!(tensor.size(), 24);

    let tensor = Tensor::new(&[], 0.0f32);
    assert_eq!(tensor.shape(), &[]);
    assert_eq!(tensor.tiles(), &[1]);
    assert_eq!(tensor.pattern.shape.to_string(), "0");
    assert_eq!(tensor.pattern.value, &[0]);
    assert_eq!(tensor.storage.shape.to_string(), "0");
    assert_eq!(tensor.storage.value.len(), 1);
    assert_eq!(tensor.size(), 1);
}
