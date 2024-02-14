/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::compute::generator::standard_deviation::StandardDeviationGenerator;
use crate::sim::spatial_buffer::SpatialReader;

pub struct BollingerBandsGenerator {
    span: usize,
    band_multiplier: f64,
    standard_deviation_gen: StandardDeviationGenerator,
}

impl BollingerBandsGenerator {
    #[inline]
    pub fn next(&mut self, prev_last: f64, next: f64) -> BollingerBandsEntry {
        let sd = self.standard_deviation_gen.next(prev_last, next);
        let mean = self.standard_deviation_gen.mean();

        BollingerBandsEntry {
            upper_band: mean + sd * self.band_multiplier,
            average: mean,
            lower_band: mean - sd * self.band_multiplier,
        }
    }
}

pub struct BollingerBandsEntry {
    upper_band: f64,
    average: f64,
    lower_band: f64,
}
