#![deny(warnings)]

use digit_layout::DigitLayout;
use ndarray_layout::{ArrayLayout, Endian::BigEndian};
use std::{
    borrow::Cow,
    ops::{Deref, DerefMut},
};

pub extern crate digit_layout;
pub extern crate ndarray_layout;

// TODO: 现在 digit_layout 要求无符号类型中单个元素的宽度是 2 的幂且不小于 8，没有必要。

/// 张量是一种数据的容器，代表在均质的数据上附加了数据类型、形状和数据布局的动态信息。
///
/// 作为一个容器，`Tensor<T, N>` 类似于 [`Option<T>`]、[`Result<T, _>`]，可通过一系列方法变换其信息或数据：
///
/// - [`clone`](Tensor::clone)：复制张量的信息，也复制张量的数据；
/// - [`transform`](Tensor::transform)：在不改变数据的情况下变换张量的布局；
/// - [`map`](Tensor::map)：替换张量的数据；
/// - [`as_ref`](Tensor::as_ref)：返回引用原始数据的新张量；
/// - [`as_mut`](Tensor::as_mut)：返回可变引用原始数据的新张量；
/// - ...
#[derive(Clone)]
pub struct Tensor<T, const N: usize> {
    /// 数据类型。
    dt: DigitLayout,
    /// 形状和布局。
    layout: ArrayLayout<N>,
    /// 数据成员。
    item: T,
}

impl<const N: usize> Tensor<usize, N> {
    /// 创建使用指定数据类型 `dt` 和形状 `shape` 的张量，张量的“数据”是连续存储其数据占用的字节数。
    ///
    /// 传入的 `shape` 应为张量中的数值的数量。
    /// 在底层存储中，可能将多个数值捆绑为一个数据组。
    /// 获取张量的形状时，将返回作为 N 维数组的形状，其连续维度除去了组的规模。
    ///
    /// 例如，对于将 32 个数字绑定为一组的数据类型，`shape` 为 `[7, 1024]` 时，产生的张量的形状是 `[7, 32]`。
    ///
    /// ```rust
    /// # use tensor::Tensor;
    /// // 定义一个数据类型，以 32 个 8 位无符号数为一组。
    /// digit_layout::layout!(GROUP u(8); 32);
    ///
    /// let tensor = Tensor::new(GROUP, [7, 1024]);
    /// assert_eq!(tensor.dt(), GROUP);
    /// assert_eq!(tensor.shape(), [7, 32]);
    /// assert_eq!(tensor.take(), 7 * 32 * 32);
    /// ```
    pub fn new(dt: DigitLayout, shape: [usize; N]) -> Self {
        Self::from_dim_slice(dt, shape)
    }

    pub fn from_dim_slice(dt: DigitLayout, shape: impl AsRef<[usize]>) -> Self {
        let shape = shape.as_ref();

        let shape = match dt.group_size() {
            1 => Cow::Borrowed(shape),
            g => {
                let mut shape = shape.to_vec();
                let last = shape.last_mut().unwrap();
                assert_eq!(*last % g, 0);
                *last /= g;
                Cow::Owned(shape)
            }
        };

        let element_size = dt.nbytes();
        let layout = ArrayLayout::new_contiguous(&shape, BigEndian, element_size);
        let size = layout.num_elements() * element_size;
        Self {
            dt,
            layout,
            item: size,
        }
    }
}

impl<T, const N: usize> Tensor<T, N> {
    pub const fn dt(&self) -> DigitLayout {
        self.dt
    }

    pub const fn layout(&self) -> &ArrayLayout<N> {
        &self.layout
    }

    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    pub fn offset(&self) -> isize {
        self.layout.offset()
    }

    pub const fn get(&self) -> &T {
        &self.item
    }

    pub fn get_mut(&mut self) -> &mut T {
        &mut self.item
    }

    pub fn take(self) -> T {
        self.item
    }

    pub const fn from_raw_parts(dt: DigitLayout, layout: ArrayLayout<N>, item: T) -> Self {
        Self { dt, layout, item }
    }

    pub fn into_raw_parts(self) -> (DigitLayout, ArrayLayout<N>, T) {
        let Self { dt, layout, item } = self;
        (dt, layout, item)
    }

    pub fn use_info(&self) -> Tensor<usize, N> {
        let dt = self.dt;
        let element_size = dt.nbytes();
        let layout = ArrayLayout::new_contiguous(self.layout.shape(), BigEndian, element_size);
        let size = layout.num_elements() * element_size;
        Tensor {
            dt,
            layout,
            item: size,
        }
    }

    pub fn is_contiguous(&self) -> bool {
        match self.layout.merge_be(0, self.layout.ndim()) {
            Some(layout) => {
                let &[s] = layout.strides() else {
                    unreachable!()
                };
                s == self.dt.nbytes() as isize
            }
            None => false,
        }
    }
}

impl<T, const N: usize> Tensor<T, N> {
    pub fn as_ref(&self) -> Tensor<&T, N> {
        Tensor {
            dt: self.dt,
            layout: self.layout.clone(),
            item: &self.item,
        }
    }

    pub fn as_mut(&mut self) -> Tensor<&mut T, N> {
        Tensor {
            dt: self.dt,
            layout: self.layout.clone(),
            item: &mut self.item,
        }
    }

    pub fn transform(self, f: impl FnOnce(ArrayLayout<N>) -> ArrayLayout<N>) -> Self {
        let Self { dt, layout, item } = self;
        Self {
            dt,
            layout: f(layout),
            item,
        }
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Tensor<U, N> {
        let Self { dt, layout, item } = self;
        Tensor {
            dt,
            layout,
            item: f(item),
        }
    }
}

impl<T: Deref, const N: usize> Tensor<T, N> {
    pub fn as_deref(&self) -> Tensor<&<T as Deref>::Target, N> {
        Tensor {
            dt: self.dt,
            layout: self.layout.clone(),
            item: self.item.deref(),
        }
    }
}

impl<T: DerefMut, const N: usize> Tensor<T, N> {
    pub fn as_deref_mut(&mut self) -> Tensor<&mut <T as Deref>::Target, N> {
        Tensor {
            dt: self.dt,
            layout: self.layout.clone(),
            item: self.item.deref_mut(),
        }
    }
}

#[test]
fn test_basic_functions() {
    digit_layout::layout!(GROUP u(8); 32);
    let t1 = Tensor::new(GROUP, [7, 1024]);
    assert_eq!(t1.dt(), GROUP);

    let l1 = t1.layout();
    let l2: ArrayLayout<2> = ArrayLayout::new_contiguous(&[7, 1024 / 32], BigEndian, 32);
    assert_eq!(l1.shape(), l2.shape());
    assert_eq!(l1.strides(), l2.strides());
    assert_eq!(l1.offset(), l2.offset());

    assert_eq!(t1.shape(), [7, 1024 / 32]);
    assert_eq!(t1.strides(), &[1024, 32]);
    assert_eq!(t1.offset(), 0);

    assert_eq!(*t1.get(), 7 * 1024);
    let mut t2 = t1.clone();
    *(t2.get_mut()) += 1;
    assert_eq!(*t2.get(), 7 * 1024 + 1);
    assert_eq!(t1.take(), 7 * 1024);
}

#[test]
fn test_extra_functions() {
    digit_layout::layout!(GROUP u(8); 32);
    let shape = [7, 1024];
    let element_size = 32;
    let layout: ArrayLayout<2> = ArrayLayout::new_contiguous(&shape, BigEndian, element_size);
    let item = 7 * 1024;

    let tensor = Tensor::from_raw_parts(GROUP, layout.clone(), item);
    assert_eq!(tensor.dt(), GROUP);

    assert_eq!(tensor.layout().shape(), layout.shape());
    assert_eq!(tensor.layout().strides(), layout.strides());
    assert_eq!(tensor.layout().offset(), layout.offset());
    assert_eq!(*tensor.get(), item);

    let (dt, layout_from_parts, item_from_parts) = tensor.into_raw_parts();
    assert_eq!(dt, GROUP);

    assert_eq!(layout_from_parts.shape(), layout.shape());
    assert_eq!(layout_from_parts.strides(), layout.strides());
    assert_eq!(layout_from_parts.offset(), layout.offset());
    assert_eq!(item_from_parts, item);

    let tensor = Tensor::from_raw_parts(GROUP, layout, item);

    let info_tensor = tensor.use_info();
    assert_eq!(info_tensor.dt(), GROUP);
    assert_eq!(info_tensor.shape(), tensor.shape());
    let expected_size = info_tensor.layout().num_elements() * GROUP.nbytes();
    assert_eq!(info_tensor.take(), expected_size);

    let is_contig = tensor.is_contiguous();
    assert!(is_contig);

    let non_contig_layout: ArrayLayout<2> = ArrayLayout::new(&[7, 1024], &[1, 7], 0);
    let non_contig_tensor = Tensor::from_raw_parts(GROUP, non_contig_layout, item);
    assert!(!non_contig_tensor.is_contiguous());
}

#[test]
fn test_as_ref() {
    digit_layout::layout!(GROUP u(8); 32);
    let t1 = Tensor::new(GROUP, [7, 1024]);

    let ref_tensor = t1.as_ref();
    assert_eq!(*ref_tensor.item, 7168);
}

#[test]
fn test_as_mut() {
    digit_layout::layout!(GROUP u(8); 32);
    let mut t1 = Tensor::new(GROUP, [7, 1024]);

    let mut ref_tensor = t1.as_mut();
    (**ref_tensor.get_mut()) += 1;
    assert_eq!(*t1.get(), 7169);
}

#[test]
fn test_transform() {
    digit_layout::layout!(GROUP u(8); 32);
    let t1 = Tensor::new(GROUP, [7, 1024]);

    fn trans(layout: ArrayLayout<2>) -> ArrayLayout<2> {
        layout.transpose(&[0, 1])
    }

    let t2 = t1.transform(trans);
    assert_eq!(t2.shape(), [7, 32]);
    assert_eq!(t2.strides(), &[1024, 32]);
    assert_eq!(t2.offset(), 0);
}

#[test]
fn test_map() {
    digit_layout::layout!(GROUP u(8); 32);
    let t1 = Tensor::new(GROUP, [7, 1024]);

    fn trans(n: usize) -> isize {
        n as isize
    }

    let t2 = t1.map(trans);
    assert_eq!((*t2.get()), (7 * 1024) as isize);
}

#[test]
fn test_as_deref() {
    struct TestDeref<T>(T);
    impl<T> Deref for TestDeref<T> {
        type Target = T;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    digit_layout::layout!(GROUP u(8); 32);
    let tensor = Tensor {
        dt: GROUP,
        layout: ArrayLayout::<2>::new_contiguous(&[7, 1024], BigEndian, 32),
        item: TestDeref(42),
    };
    let tensor_ref = tensor.as_deref();
    assert_eq!(*tensor_ref.item, 42);
}

#[test]
fn test_as_deref_mut() {
    struct TestDeref<T>(T);
    impl<T> Deref for TestDeref<T> {
        type Target = T;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<T> DerefMut for TestDeref<T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    digit_layout::layout!(GROUP u(8); 32);
    let mut tensor = Tensor {
        dt: GROUP,
        layout: ArrayLayout::<2>::new_contiguous(&[7, 1024], BigEndian, 32),
        item: TestDeref(42),
    };
    let mut tensor_ref = tensor.as_deref_mut();
    *(*tensor_ref.get_mut()) += 1;
    assert_eq!(**tensor.get(), 43);
}

#[test]
fn test_borrow() {
    digit_layout::layout!(GROUP u(8); 1);
    let _tensor = Tensor::new(GROUP, [7, 1024]);
}
