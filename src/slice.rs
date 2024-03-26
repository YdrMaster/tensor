use crate::Tensor;
use nalgebra::DMatrix;
use std::{
    cmp::Ordering,
    iter::{once, zip},
};

impl<Storage> Tensor<Storage> {
    /// 对块级别切片。
    #[inline]
    pub fn tile_slice(mut self, meta: &[SliceDim]) -> Self {
        assert_eq!(self.tiles.len(), meta.len() + 1);
        // 根据具体分块调整变换元信息
        let meta = zip(meta, &*self.tiles)
            .map(|(d, &len)| d.normalize(len))
            .collect::<Vec<_>>();
        // 更新分块
        self.tiles.0 = meta.iter().map(|d| d.len).chain(once(1)).collect();
        // 变换访存模式
        let n = self.tiles.len();
        let affine = DMatrix::from_fn(n, n, |r, c| {
            if r == n - 1 {
                meta.get(c).map_or(1, |d| d.start as _)
            } else if r == c {
                meta.get(c).map(|d| d.step).unwrap()
            } else {
                0
            }
        });
        self.pattern.value = (affine * self.pattern.as_matrix()).data.into();

        self
    }
}

#[test]
fn test_tile() {
    let tensor = Tensor::new(&[2, 3, 4, 5], ());
    assert_eq!(tensor.shape(), &[2, 3, 4, 5]);
    assert_eq!(tensor.tiles(), &[2, 3, 4, 5, 1]);
    assert_eq!(tensor.pattern.shape.to_string(), "00000");
    assert_eq!(tensor.pattern.value, &[60, 20, 5, 1, 0]);
    assert_eq!(tensor.storage.shape.to_string(), "00000");
    assert_eq!(tensor.storage.value.len(), 1);
    assert_eq!(tensor.size(), 120);

    use crate::slice;
    let tensor = tensor.tile_slice(&[
        slice![all],
        slice![from 1, take 2],
        slice![from 1, take 2, per 2],
        slice![rev],
    ]);
    assert_eq!(tensor.shape(), &[2, 2, 2, 5]);
    assert_eq!(tensor.tiles(), &[2, 2, 2, 5, 1]);
    assert_eq!(tensor.pattern.shape.to_string(), "00000");
    assert_eq!(tensor.pattern.value, &[60, 20, 10, -1, 29]);
    assert_eq!(tensor.storage.shape.to_string(), "00000");
    assert_eq!(tensor.storage.value.len(), 1);
    assert_eq!(tensor.size(), 40);
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SliceDim {
    pub start: usize,
    pub step: isize,
    pub len: usize,
}

impl SliceDim {
    #[inline]
    pub fn normalize(&self, len: usize) -> Self {
        match self.step.cmp(&0) {
            Ordering::Greater => {
                assert!(self.start < len);
                Self {
                    start: self.start,
                    step: self.step,
                    len: {
                        let step = self.step as usize;
                        ((len - self.start + step - 1) / step).min(len)
                    },
                }
            }
            Ordering::Equal => {
                assert!(self.start < len);
                Self {
                    start: self.start,
                    step: self.step,
                    len: self.len,
                }
            }
            Ordering::Less => {
                let start = self.start.min(len - 1);
                Self {
                    start,
                    step: self.step,
                    len: {
                        let step = self.step.unsigned_abs();
                        ((start + 1 + step - 1) / step).min(len)
                    },
                }
            }
        }
    }
}

#[macro_export]
macro_rules! slice {
    [all] => {
        slice![0; 1; usize::MAX]
    };
    [rev] => {
        slice![usize::MAX; -1; usize::MAX]
    };
    [take $len:expr] => {
        slice![0; 1; $len]
    };
    [from $start:expr, until $end:expr] => {
        slice![$start; 1; $end - $start]
    };
    [from $start:expr, take $len:expr] => {
        slice![$start; 1; $len]
    };
    [from $start:expr, take $len:expr, per $step:expr] => {
        slice![$start; $step; $len]
    };
    [$start:expr; $step:expr; $len:expr] => {
        $crate::SliceDim {
            start: $start as _,
            step : $step  as _,
            len  : $len   as _,
        }
    };
}

#[test]
fn test_macro() {
    assert_eq!(
        slice![5; -3; 2],
        SliceDim {
            start: 5,
            step: -3,
            len: 2,
        }
    );
    assert_eq!(slice![all], slice![0; 1; usize::MAX]);
    assert_eq!(slice![rev], slice![usize::MAX; -1; usize::MAX]);
    assert_eq!(slice![take 5], slice![0; 1; 5]);
    assert_eq!(slice![from 3, until 5], slice![3; 1; 2]);
    assert_eq!(slice![from 3, take 5], slice![3; 1; 5]);
    assert_eq!(slice![from 3, take 5, per 2], slice![3; 2; 5]);
}
