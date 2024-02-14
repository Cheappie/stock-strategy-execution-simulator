/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::compute::generator::exponential_moving_average::ExponentialMovingAverageGenerator;
use crate::sim::compute::generator::true_range::TrueRangeGenerator;

pub struct AverageTrueRangeGenerator<'a> {
    atr: ExponentialMovingAverageGenerator,
    true_range_generator: TrueRangeGenerator<'a>,
}

impl<'a> AverageTrueRangeGenerator<'a> {
    pub fn next(&mut self, frame_index: usize) -> f64 {
        let tr = self.true_range_generator.next(frame_index);
        self.atr.next(tr)
    }
}
