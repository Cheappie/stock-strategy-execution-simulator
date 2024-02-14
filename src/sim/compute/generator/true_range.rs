/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::spatial_buffer::SpatialReader;

pub struct TrueRangeGenerator<'a> {
    high: SpatialReader<'a>,
    low: SpatialReader<'a>,
    close: SpatialReader<'a>,
}

impl TrueRangeGenerator<'_> {
    pub fn next(&self, frame_index: usize) -> f64 {
        let prev_close = self.close.get(frame_index - 1);
        let high = self.high.get(frame_index);
        let low = self.low.get(frame_index);
        calculate_true_range(prev_close, high, low)
    }
}

fn calculate_true_range(prev_close: &f64, high: &f64, low: &f64) -> f64 {
    let high_low = high - low;
    let high_prev_close = (high - prev_close).abs();
    let low_prev_close = (low - prev_close).abs();
    high_low.max(high_prev_close).max(low_prev_close)
}
