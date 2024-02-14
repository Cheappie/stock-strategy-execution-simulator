/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::collections::circular_buffer::CircularBuffer;
use crate::sim::compute::generator::simple_moving_average::SimpleMovingAverageGenerator;
use crate::sim::compute::guard;

pub struct StandardDeviationGenerator {
    span: usize,
    span_f64: f64,
    raw_data_buffer: CircularBuffer<f64>,
    simple_moving_average_gen: SimpleMovingAverageGenerator,
}

impl StandardDeviationGenerator {
    #[inline]
    pub fn next(&mut self, prev_last: f64, next: f64) -> f64 {
        self.raw_data_buffer.push(next);

        let mean = self.simple_moving_average_gen.next(prev_last, next);
        self.raw_data_buffer
            .iter_top(self.span)
            .map(|n| (n - mean).abs().powi(2))
            .sum::<f64>()
            .sqrt()
    }

    pub fn mean(&self) -> f64 {
        self.simple_moving_average_gen.peek()
    }
}
