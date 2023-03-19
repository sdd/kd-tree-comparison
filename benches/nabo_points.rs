use std::ops::{Add, Sub};
use std::fmt::Debug;
use std::ops::{AddAssign, SubAssign};
use nabo::{NotNan, Point};
use num_traits::{Bounded, Float, Zero};
use rand::distributions::{Distribution, Standard};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct P<A: Float, const K: usize>(pub [NotNan<A>; K]);

impl<A: Float + Debug + Default + AddAssign + SubAssign, const K: usize> P<A, K> {
    pub fn new(data: [NotNan<A>; K]) -> P<A, K> {
        P(data)
    }
}

impl<A: Float + Debug + Default + AddAssign + SubAssign, const K: usize> Bounded for P<A, K> {
    fn min_value() -> P<A, K> {
        P([NotNan::<A>::min_value(); K])
    }
    fn max_value() -> P<A, K> {
        P([NotNan::<A>::max_value(); K])
    }
}

impl<A: Float, const K: usize> Default for P<A, K> {
    fn default() -> Self {
        P([NotNan::<A>::zero(); K])
    }
}

impl<A: Float + Debug + Default + AddAssign + SubAssign, const K: usize> Point<A> for P<A, K> {
    const DIM: u32 = K as u32;
    fn set(&mut self, index: u32, value: NotNan<A>) {
        self.0[index as usize] = value;
    }
    fn get(&self, index: u32) -> NotNan<A> {
        self.0[index as usize]
    }
}

impl<A: Float + Debug + Default + AddAssign, const K: usize> Add for P<A, K> {
    type Output = P<A, K>;

    fn add(self, rhs: P<A, K>) -> Self::Output {
        let mut res: [NotNan<A>; K] = self.0;
        for i in 0..K {
            res[i] += rhs.0[i];
        }

        P(res)
    }
}
impl<A: Float + Debug + Default + AddAssign + SubAssign, const K: usize> Sub for P<A, K> {
    type Output = P<A, K>;

    fn sub(self, rhs: P<A, K>) -> Self::Output {
        let mut res: [NotNan<A>; K] = self.0;
        for i in 0..K {
            res[i] -= rhs.0[i];
        }

        P(res)
    }
}

/// Creates a random point whose coordinate are in the interval [-100:100].
pub fn random_point<A: Float + Debug + Default + AddAssign + SubAssign, const K: usize>() -> P<A, K>
where Standard: Distribution<[A; K]>,
{
    let raw_point: [A; K] = rand::random::<[A; K]>();

    let mut res: [NotNan<A>; K] = [NotNan::new(A::zero()).unwrap(); K];
    for i in 0..K {
        res[i] = NotNan::new(raw_point[i]).unwrap();
    }

    P(res)
}

/// Creates a random cloud of count points using [random_point()] for each.
pub fn random_point_cloud<A: Float + Debug + Default + AddAssign + SubAssign, const K: usize>(count: u32) -> Vec<P<A, K>>
    where Standard: Distribution<[A; K]>,
{
    (0..count).map(|_| random_point()).collect()
}
