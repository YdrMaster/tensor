//! 具有不连续存储结构的张量类型。

mod logical;

use logical::Logical;
use std::{collections::HashMap, ops::RangeInclusive};

/// 具有不连续存储结构的张量类型。
#[derive(Clone, Debug)]
pub struct Tensor<Physical> {
    logical: Logical,
    physical: HashMap<Physical, Ranges>,
    layouts: HashMap<Strides, Ranges>,
}

/// 增广一个偏移维度的坐标跳步。
///
/// 表示一个张量分块中，使用坐标计算数据位置的方式。
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
struct Strides(Vec<isize>);

/// 分块坐标判据。
///
/// 根据分块坐标计算是否命中物理空间或坐标跳步。
#[derive(Clone, Debug)]
#[repr(transparent)]
struct Ranges(Vec<RangeInclusive<usize>>);
