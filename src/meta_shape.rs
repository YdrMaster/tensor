use fixedbitset::FixedBitSet;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
#[repr(transparent)]
pub(crate) struct MetaShape(FixedBitSet);

impl ToString for MetaShape {
    #[inline]
    fn to_string(&self) -> String {
        format!("{:b}", self.0)
    }
}

impl MetaShape {
    #[inline]
    pub fn new(bits: usize) -> Self {
        Self(FixedBitSet::with_capacity(bits))
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn split(mut self, axis: usize, insert: usize) -> Self {
        self.0.grow(self.0.len() + insert);
        for i in (axis + insert..self.0.len()).rev() {
            self.0.copy_bit(i - insert, i);
        }
        self.0.set_range(axis..axis + insert, self.0[axis + insert]);
        self
    }

    #[inline]
    pub fn transpose(self, btree: &BTreeMap<usize, usize>) -> Self {
        let mut ans = self.0.clone();
        for (&dst, &src) in btree {
            ans.set(dst, self.0[src]);
        }
        Self(ans)
    }
}

#[test]
fn test_split() {
    let mut shape = MetaShape::new(5);
    shape.0.insert(2);
    shape.0.insert(3);

    assert_eq!(shape.clone().split(2, 2).to_string(), "0011110");
    assert_eq!(shape.clone().split(1, 2).to_string(), "0000110");
}

#[test]
fn test_transpose() {
    let mut shape = MetaShape::new(5);
    shape.0.insert(2);
    shape.0.insert(3);

    let btree = BTreeMap::from([(1, 2), (2, 3), (3, 1)]);
    assert_eq!(shape.transpose(&btree).to_string(), "01100");
}
