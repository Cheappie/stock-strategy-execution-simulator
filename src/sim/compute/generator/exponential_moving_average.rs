/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

pub struct ExponentialMovingAverageGenerator {
    ema: f64,
    multiplier: f64,
}

impl ExponentialMovingAverageGenerator {
    #[inline]
    pub fn next(&mut self, next: f64) -> f64 {
        self.ema = (next - self.ema) * self.multiplier + self.ema;
        self.ema
    }
}
