# 张量

[![CI](https://github.com/YdrMaster/tensor/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/YdrMaster/tensor/actions)
[![Latest version](https://img.shields.io/crates/v/tensor.svg)](https://crates.io/crates/tensor)
[![Documentation](https://docs.rs/tensor/badge.svg)](https://docs.rs/tensor)
[![license](https://img.shields.io/github/license/YdrMaster/tensor)](https://mit-license.org/)
[![codecov](https://codecov.io/gh/YdrMaster/tensor/branch/main/graph/badge.svg)](https://codecov.io/gh/YdrMaster/tensor/)

[![GitHub Issues](https://img.shields.io/github/issues/YdrMaster/tensor)](https://github.com/YdrMaster/tensor/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/YdrMaster/tensor)](https://github.com/YdrMaster/tensor/pulls)
![GitHub repo size](https://img.shields.io/github/repo-size/YdrMaster/tensor)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/YdrMaster/tensor)
![GitHub contributors](https://img.shields.io/github/contributors/YdrMaster/tensor)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/YdrMaster/tensor)

张量是一种数据的容器，代表在均质的数据上附加了数据类型、形状和数据布局的动态信息。

本项目提供 `Tensor<T, N>` 结构体，用于灵活管理多维数据，它能通过一系列方法对数据信息和内容进行变换，同时提供了类型安全和高效的操作接口。

`Tensor<T, N>` 具备以下特性：

- **数据类型和布局管理**：能精确记录数据类型和多维布局信息，方便对不同维度、不同类型的数据进行处理；
- **方法丰富**：提供多种实用方法，如 `clone` 用于复制张量信息和数据，`transform` 用于在不改变数据的情况下变换张量布局，`map` 用于替换张量数据等；
- **引用操作**：支持 `as_ref` 和 `as_mut` 方法，可分别返回引用和可变引用原始数据的新张量，便于对数据进行安全的读写操作；

## 使用示例

```rust
use tensor::Tensor;
use digit_layout::DigitLayout;
use ndarray_layout::{ArrayLayout, Endian::BigEndian};

// 定义一个数据类型，以 32 个 8 位无符号数为一组。
digit_layout::layout!(GROUP u(8); 32);
let shape = [7, 1024];
let element_size = 32;
let layout = ArrayLayout::<2>::new_contiguous(&shape, BigEndian, element_size);
let item = 7 * 1024;

// 创建张量。
let tensor = Tensor::from_raw_parts(GROUP, layout.clone(), item);

// 获取张量数据类型。
assert_eq!(tensor.dt(), GROUP);

// 获取张量布局信息。
assert_eq!(tensor.layout().shape(), layout.shape());

// 复制张量。
let mut cloned_tensor = tensor.clone();
*(cloned_tensor.get_mut()) += 1;
assert_eq!(*cloned_tensor.get(), item + 1);
```
