use std::usize;

/// 增广一个偏移维度的坐标跳步。
///
/// 表示一个张量分块中，使用坐标计算数据位置的方式。
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub(crate) struct Strides(Vec<isize>);

impl Strides {
    #[inline]
    pub fn new(shape: &[usize], mut unit: usize) -> Self {
        let mut ans = vec![0; shape.len()];
        for (x, &d) in ans.iter_mut().zip(shape).rev() {
            *x = unit as isize;
            unit *= d;
        }
        Self(ans)
    }
}
