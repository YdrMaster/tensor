use std::{collections::BTreeMap, ops::Deref};

#[derive(Clone, Debug)]
#[repr(transparent)]
pub(crate) struct Tiles(Vec<usize>);

impl Deref for Tiles {
    type Target = [usize];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Tiles {
    #[inline]
    pub fn from_iter(iter: impl IntoIterator<Item = usize>) -> Self {
        Self(iter.into_iter().collect())
    }

    #[inline]
    pub fn tile_split(mut self, axis: usize, tiles: &[usize]) -> Self {
        let len = self.0.len();
        self.0.reserve(tiles.len() - 1);
        #[allow(clippy::uninit_vec)]
        unsafe {
            self.0.set_len(len + tiles.len() - 1)
        };
        self.0.copy_within(axis + 1..len, axis + tiles.len());
        self.0[axis..axis + tiles.len()].copy_from_slice(tiles);
        self
    }

    #[inline]
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
    let shape = Tiles::from_iter([2, 3, 4, 1]);
    assert_eq!(&*shape.clone().tile_split(2, &[2, 2]), &[2, 3, 2, 2, 1]);
}
