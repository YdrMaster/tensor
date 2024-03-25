use std::{
    collections::BTreeMap,
    iter::once,
    ops::{Deref, DerefMut},
};

#[derive(Clone, Debug)]
#[repr(transparent)]
pub(crate) struct Tiles(pub Vec<usize>);

impl Deref for Tiles {
    type Target = [usize];
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

impl DerefMut for Tiles {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut_slice()
    }
}

impl Tiles {
    #[inline]
    pub fn new(shape: &[usize]) -> Self {
        Self(shape.iter().map(|&d| d as _).chain(once(1)).collect())
    }

    /// 维度切分变换。
    ///
    /// 将形状的 `axis` 维度依 `tiles` 指定的方式切分。
    pub fn split(mut self, axis: usize, tiles: &[usize]) -> Self {
        let len = self.0.len();
        self.0.reserve(tiles.len() - 1);
        unsafe { self.0.set_len(len + tiles.len() - 1) };
        self.0.copy_within(axis + 1..len, axis + tiles.len());
        self.0[axis..axis + tiles.len()].copy_from_slice(tiles);
        self
    }

    /// 维度切分变换。
    ///
    /// 将形状的 `axis` 维度依 `tiles` 指定的方式切分。
    pub fn transpose(self, btree: &BTreeMap<usize, usize>) -> Self {
        let mut ans = self.0.clone();
        for (&dst, &src) in btree {
            ans[dst] = self.0[src];
        }
        Self(ans)
    }
}

#[test]
fn test() {
    let shape = Tiles::new(&[2, 3, 4]);
    assert_eq!(&*shape.clone().split(2, &[2, 2]), &[2, 3, 2, 2, 1]);
}
