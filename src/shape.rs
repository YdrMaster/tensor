use crate::udim;
use smallvec::SmallVec;
use std::ops::Deref;

/// 张量或张量分块的逻辑形状。
// 保存为张量的总元素数和除去最后一维的坐标跳步的数组，以免反复计算跳步。
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct Shape(SmallVec<[udim; 4]>);

impl Shape {
    /// 从形状构造逻辑形状对象。
    #[inline]
    pub fn new(shape: &[usize]) -> Self {
        Self(
            shape
                .iter()
                .rev()
                .scan(1, |mul, &d| {
                    *mul *= d as udim;
                    Some(*mul)
                })
                .collect(),
        )
    }

    /// 张量形状的维度数。
    #[inline]
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// 张量的总元素数。
    #[inline]
    pub fn size(&self) -> usize {
        self.0.first().map(|&d| d as _).unwrap_or(1)
    }

    /// 获取形状第 `i` 维的长度。
    #[inline]
    pub fn get(&self, i: usize) -> Option<usize> {
        if i >= self.rank() {
            None
        } else {
            let i = i + 1;
            Some(if i == self.rank() {
                1
            } else {
                (self.0[i - 1] / self.0[i]) as _
            })
        }
    }

    /// 计算在铺平的张量存储中第 `index` 个元素在张量中的坐标。
    /// 返回逐维度计算坐标的迭代器。
    #[inline]
    pub fn flat_locate(&self, index: usize) -> FlatIndices {
        FlatIndices {
            strides: &*self.0,
            rem: index,
        }
    }
}

/// 张量形状迭代器。
pub struct Iter<'a> {
    inner: &'a Shape,
    index: usize,
}

impl<'a> IntoIterator for &'a Shape {
    type Item = <Iter<'a> as Iterator>::Item;
    type IntoIter = Iter<'a>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Iter {
            inner: &self,
            index: 0,
        }
    }
}

impl Iterator for Iter<'_> {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let Self { inner, index } = self;
        if *index >= inner.rank() {
            None
        } else {
            *index += 1;
            Some(if *index == inner.rank() {
                1
            } else {
                (inner.0[*index - 1] / inner.0[*index]) as _
            })
        }
    }

    #[inline]
    fn count(self) -> usize {
        let Self { inner, index } = self;
        if index >= inner.rank() {
            0
        } else {
            inner.rank() - index
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        let Self { inner, index } = self;
        if index >= inner.rank() {
            None
        } else {
            Some(1)
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let Self { inner, index } = self;
        *index += n;
        if *index >= inner.rank() {
            None
        } else {
            Some(if *index == inner.rank() {
                1
            } else {
                (inner.0[*index - 1] / inner.0[*index]) as _
            })
        }
    }
}

/// 元素坐标迭代器。
pub struct FlatIndices<'a> {
    strides: &'a [udim],
    rem: usize,
}

impl Iterator for FlatIndices<'_> {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.rem == 0 {
            None
        } else {
            Some(match self.strides {
                [s, tail @ ..] => {
                    let div = *s as usize;
                    self.strides = tail;

                    let ans = self.rem / div;
                    self.rem %= div;
                    ans
                }
                [] => {
                    let ans = self.rem;
                    self.rem = 0;
                    ans
                }
            })
        }
    }
}
