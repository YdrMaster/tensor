/// 张量的逻辑形状和分块方案。
///
/// # 张量分块的概念
///
/// 张量分块是关联张量逻辑形状和张量物理存储的桥梁。
/// 同一个分块上的所有数据存储在同一个物理地址空间中；
/// 不同的的块可能存储在不同的物理地址空间。
/// 这些地址空间可能是相互隔离的，包括不同的进程、异构硬件、非易失存储器甚至是分布式节点。
///
/// 张量的分块在逻辑上将张量形状的每个维度视作由一些不连续的维度揉合成的。
/// 例如将一个形状为 `[A, B]` 的矩阵在两个维度上各分为两块，
/// 意味着维度 `A` 是由 `[2, A/2]` 两个维度揉合；
/// 维度 `B` 是由 `[2, B/2]` 两个维度揉合。
/// 现在具有 2 个维度的张量 `[A, B]` 可以
/// 如同具有 4 个维度的张量 `[2, A/2, 2, B/2]` 一样进行变换物理存储布局的操作。
///
/// 借助块选择函数，还可以为每个块都选择不同的存储布局，
/// 从而承载更复杂的变换计算。
///
/// # 数据结构实现
///
/// 存储结构：`[rank, [rank tile shape], [[dim tile shape]]]`
///
/// - `rank` 是张量的维度数，也是 `[rank tile shape]` 和 `[[dim tile shape]]` 的维度数。
/// - `[rank tile shape]` 张量的每个形状维度的分块的维度数量，以前序和数组的形式存储，方便随机访问。
/// - `[[dim tile shape]]` 张量的每个形状维度的分块的形状，以后序积（strides）的形式存储，方便计算坐标。
///
/// 例如，一个形状为 `[20, 30, 40]` 的张量，
/// 将长度为 30 的维度平均分为 3 块，
/// 长度为 40 的维度平均分为 2 块，
/// 则保存为：`[3, [0, 1, 3], [[20], [30, 10], [40, 20]]]`。
///
/// 其中，
///
/// - `3` 是张量的维度数；
/// - `[0, 1, 3]` 是每个维度的分块的维度数量的前序和；
///   - `0` 表示第 0 维（长度为 20）的分块信息位于 `[[dim tile shape]]` 的位置 0；
///   - `1` 表示第 1 维（长度为 30）的分块信息位于 `[[dim tile shape]]` 的位置 1，因为第 0 维没有分块，只需要一个数字存储长度；
///   - `3` 表示第 2 维（长度为 40）的分块信息位于 `[[dim tile shape]]` 的位置 3，因为第 1 维平均切分一次，需要两个数字存储后序积；
/// - `[[20], [30, 10], [40, 20]]` 是每个维度的分块的形状；
///   - `[[20]]` 表示第 0 维长度为 20，没有分块；
///   - `[[30, 10]]` 表示第 1 维长度为 30，分为 3 块，每块长度为 10；
///   - `[[40, 20]]` 表示第 2 维长度为 40，分为 2 块，每块长度为 20；
#[derive(Clone, Debug)]
#[repr(transparent)]
pub(crate) struct Logical(Vec<usize>);

impl Logical {
    pub fn new(shape: &[usize]) -> Self {
        let rank = shape.len();
        let mut logical = vec![0; 1 + 1 + rank + rank];
        logical[0] = rank;
        logical[1..][..1 + rank]
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = i);
        logical[2 + rank..]
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = shape[i]);
        Self(logical)
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.0[0]
    }

    #[inline]
    pub fn shape(&self, i: usize) -> Option<usize> {
        let (&rank, slice) = self.0.split_first().unwrap();
        if i < rank {
            Some(slice[1 + rank..][slice[i]])
        } else {
            None
        }
    }

    #[inline]
    pub fn shape_vec(&self) -> Vec<usize> {
        let (&rank, body) = self.0.split_first().unwrap();
        let (pos, tiles) = body.split_at(1 + rank);
        (0..rank).map(|i| tiles[pos[i]]).collect()
    }

    #[inline]
    pub fn tile(&self, i: usize) -> Option<Tile> {
        let (&rank, slice) = self.0.split_first().unwrap();
        if i < rank {
            Some(Tile(&slice[1 + rank..][slice[i]..slice[i + 1]]))
        } else {
            None
        }
    }

    #[inline]
    pub fn partition(&mut self, i: usize, shape: &[usize]) {
        let mut strides = shape
            .iter()
            .rev()
            .scan(1, |mul, &d| {
                *mul *= d;
                Some(*mul)
            })
            .collect::<Vec<_>>();
        strides.reverse();

        let (&mut rank, body) = self.0.split_first_mut().unwrap();
        let (pos, tiles) = body.split_at_mut(1 + rank);
        assert_eq!(tiles[pos[i]], strides[0]);

        let current_len = pos[i + 1] - pos[i];
        let new_len = strides.len();
        let diff = new_len as isize - current_len as isize;
        if diff < 0 {
            tiles.copy_within(pos[i + 1].., pos[i] + new_len);
            self.0.truncate(self.0.len() - (-diff as usize));
        } else if 0 < diff {
            self.0.resize(self.0.len() + diff as usize, 0);
            let (_, body) = self.0.split_first_mut().unwrap();
            let (pos, tiles) = body.split_at_mut(1 + rank);
            tiles.copy_within(pos[i + 1]..pos[rank], pos[i] + new_len);
        }

        let (_, body) = self.0.split_first_mut().unwrap();
        let (pos, tiles) = body.split_at_mut(1 + rank);
        tiles[pos[i]..][..strides.len()].copy_from_slice(&strides);
        for x in &mut pos[i + 1..] {
            *x = (*x as isize + diff) as usize;
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Tile<'a>(&'a [usize]);

impl Tile<'_> {
    #[inline]
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.0[0]
    }

    #[inline]
    pub fn shape(&self, i: usize) -> Option<usize> {
        self.0
            .get(i)
            .copied()
            .map(|x| x / self.0.get(i + 1).copied().unwrap_or(1))
    }

    #[inline]
    pub fn shape_vec(&self) -> Vec<usize> {
        let n = self.0.len();
        let mut ans = vec![0; n];
        for (pair, x) in self.0.windows(2).zip(&mut ans) {
            *x = pair[0] / pair[1];
        }
        ans[n - 1] = self.0[n - 1];
        ans
    }
}

#[test]
fn test() {
    let mut logical = Logical::new(&[20, 30, 40]);

    assert_eq!(logical.0, &[3, 0, 1, 2, 3, 20, 30, 40]);
    assert_eq!(logical.rank(), 3);
    assert_eq!(logical.shape(0), Some(20));
    assert_eq!(logical.shape(1), Some(30));
    assert_eq!(logical.shape(2), Some(40));
    assert_eq!(logical.shape(3), None);
    assert_eq!(logical.tile(0).map(|t| t.0), Some(&[20][..]));
    assert_eq!(logical.tile(1).map(|t| t.0), Some(&[30][..]));
    assert_eq!(logical.tile(2).map(|t| t.0), Some(&[40][..]));
    assert_eq!(logical.tile(3).map(|t| t.0), None);

    logical.partition(1, &[3, 10]);
    assert_eq!(logical.0, &[3, 0, 1, 3, 4, 20, 30, 10, 40]);
    assert_eq!(logical.shape(0), Some(20));
    assert_eq!(logical.shape(1), Some(30));
    assert_eq!(logical.shape(2), Some(40));
    assert_eq!(logical.shape(3), None);
    assert_eq!(logical.tile(0).map(|t| t.0), Some(&[20][..]));
    assert_eq!(logical.tile(1).map(|t| t.0), Some(&[30, 10][..]));
    assert_eq!(logical.tile(2).map(|t| t.0), Some(&[40][..]));
    assert_eq!(logical.tile(3).map(|t| t.0), None);

    logical.partition(2, &[2, 20]);
    assert_eq!(logical.0, &[3, 0, 1, 3, 5, 20, 30, 10, 40, 20]);
    assert_eq!(logical.shape(0), Some(20));
    assert_eq!(logical.shape(1), Some(30));
    assert_eq!(logical.shape(2), Some(40));
    assert_eq!(logical.shape(3), None);
    assert_eq!(logical.tile(0).map(|t| t.0), Some(&[20][..]));
    assert_eq!(logical.tile(1).map(|t| t.0), Some(&[30, 10][..]));
    assert_eq!(logical.tile(2).map(|t| t.0), Some(&[40, 20][..]));
    assert_eq!(logical.tile(3).map(|t| t.0), None);

    logical.partition(1, &[5, 6]);
    assert_eq!(logical.0, &[3, 0, 1, 3, 5, 20, 30, 6, 40, 20]);
    assert_eq!(logical.shape(0), Some(20));
    assert_eq!(logical.shape(1), Some(30));
    assert_eq!(logical.shape(2), Some(40));
    assert_eq!(logical.shape(3), None);
    assert_eq!(logical.tile(0).map(|t| t.0), Some(&[20][..]));
    assert_eq!(logical.tile(1).map(|t| t.0), Some(&[30, 6][..]));
    assert_eq!(logical.tile(2).map(|t| t.0), Some(&[40, 20][..]));
    assert_eq!(logical.tile(3).map(|t| t.0), None);

    logical.partition(1, &[2, 3, 5]);
    assert_eq!(logical.0, &[3, 0, 1, 4, 6, 20, 30, 15, 5, 40, 20]);
    let tile = logical.tile(1).unwrap();
    assert_eq!(tile.rank(), 3);
    assert_eq!(tile.shape(0), Some(2));
    assert_eq!(tile.shape(1), Some(3));
    assert_eq!(tile.shape(2), Some(5));
    assert_eq!(tile.shape(3), None);
}
