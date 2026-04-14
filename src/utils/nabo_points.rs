use nabo::{NotNan, Point};
use num_traits::{Bounded, Float, Zero};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Sub, SubAssign};

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
