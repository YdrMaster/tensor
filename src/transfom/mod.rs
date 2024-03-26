//! 张量变换。
//!
//! 张量变换分为 2 个级别：张量级变换和分块级变换。
//!
//! 通常，分块级变换经过 3 步完成：
//!
//! 1. 变换分块结构；
//! 2. 变换访存模式；
//! 3. 变换元信息张量；
//!
//! 张量级变换除了对应的块级变换外，还需要额外变换块分组。

mod concat;
mod restructure;
mod slice;
mod tile_split;
mod transpose;

pub use slice::SliceDim;
