/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

pub struct SimpleMovingAverageGenerator {
    simple_moving_average: f64,
    span: usize,
    sum_of_interval: f64,
    span_f64: f64,
}

impl SimpleMovingAverageGenerator {
    pub fn new(span: usize, sum_of_interval: f64) -> SimpleMovingAverageGenerator {
        let span_f64 = span as f64;

        Self {
            simple_moving_average: sum_of_interval / span_f64,
            span,
            sum_of_interval,
            span_f64,
        }
    }

    #[inline]
    pub fn next(&mut self, prev_last: f64, next: f64) -> f64 {
        self.sum_of_interval -= prev_last;
        self.sum_of_interval += next;

        self.simple_moving_average = self.sum_of_interval / self.span_f64;
        self.simple_moving_average
    }

    pub fn peek(&self) -> f64 {
        self.simple_moving_average
    }
}
