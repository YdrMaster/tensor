//! 具有不连续存储结构的张量类型。

/// 支持分块、不连续存储结构的张量类型。
#[derive(Clone, Debug)]
pub struct Tensor<Storage, Extra> {
    /// 存储每个逻辑形状被分为几阶分块。
    shape_group: Vec<usize>,
    /// 按分块表示的张量形状。
    tiles: Vec<isize>,
    /// 存储元素的所有对象组成的[元信息张量](MetaTensor)。
    storage: MetaTensor<Storage>,
    /// 每个元素的访问模式组成的[元信息张量](MetaTensor)。
    pattern: MetaTensor<isize>,
    /// 张量附加信息。
    extra: Extra,
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
    data: Vec<T>,
}
