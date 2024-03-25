use crate::{tiles::Tiles, MetaTensor, Tensor};
use std::iter::once;

impl<Storage> Tensor<Storage> {
    /// 张量拼接。
    #[inline]
    pub fn concat(axis: usize, inputs: Vec<Self>) -> Self {
        assert!(!inputs.is_empty());
        let head = &inputs[0].tiles[..axis];
        let tail = &inputs[0].tiles[axis + 1..];
        assert!(inputs
            .iter()
            .skip(1)
            .all(|x| &x.tiles[..axis] == head && &x.tiles[axis + 1..] == tail));
        Self {
            shape_groups: vec![inputs[0].tiles.len() - 1],
            tiles: Tiles(
                head.iter()
                    .chain(once(&inputs.iter().map(|t| t.tiles[axis]).sum::<usize>()))
                    .chain(tail)
                    .copied()
                    .collect(),
            ),
            pattern: MetaTensor {
                shape: todo!(),
                value: todo!(),
            },
            storage: MetaTensor {
                shape: todo!(),
                value: todo!(),
            },
        }
    }
}

#[test]
fn test() {
    let a = Tensor::new(&[2, 3, 3], ()).tile_split(0, &[2, 1]);
    let b = Tensor::new(&[2, 3, 3], ()).tile_split(1, &[1, 3]);
    let c = Tensor::new(&[2, 6, 3], ()).tile_split(1, &[2, 3]);
    let d = Tensor::concat(1, vec![a, b, c]);
    assert_eq!(d.shape(), &[72]);
    assert_eq!(d.tiles(), &[2, 4, 3, 3, 1]);
    assert_eq!(d.size(), 72);
}
