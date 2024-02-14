/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::cursor::{CursorExpansion, Epoch};
use crate::sim::bb::time_frame::{TimeFrame, SECOND_1, SECOND_10, SECOND_15, SECOND_5};
use crate::sim::compute::guard::epoch_matches;
use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
use crate::sim::compute::wide::indicator::WideProcessor;
use crate::sim::context::SpanContext;
use crate::sim::spatial_buffer::SpatialBuffer;

pub struct CumQuotes {
    open: SpatialBuffer,
    high: SpatialBuffer,
    low: SpatialBuffer,
    close: SpatialBuffer,
}

///
/// Time frames grouped by multiplicity and sorted in ascending order.
/// Example:
/// * [TF(1s), TF(2s), TF(10s)]
/// * [TF(1s), TF(3s) TF(15s)]
///
///
/// Initial: [TF(1s), TF(2s), TF(3s), TF(5s), TF(10s), TF(15s)]
/// Groups:
/// * [TF(1s), TF(2s), TF(10s)]
/// * [TF(1s), TF(3s)]
/// * [TF(1s), TF(5s), TF(15s)]
///
///
pub struct TimeStore(Vec<Vec<TimeFrame>>);

impl WideProcessor for CumQuotes {
    fn eval(
        &mut self,
        ctx: &SpanContext,
        cursor_expansion: CursorExpansion,
        store: &DynamicStore,
    ) -> Result<(), anyhow::Error> {
        todo!()
    }

    fn output_buffer(&self, output_selector: usize) -> Result<&SpatialBuffer, anyhow::Error> {
        todo!()
    }

    fn epoch(&self) -> &Epoch {
        epoch_matches!(self.open, self.high, self.low, self.close)
    }
}

struct FrameQuote {
    open: f64,
    high: f64,
    low: f64,
    close: f64,
}

impl FrameQuote {
    pub fn new() -> Self {
        Self {
            open: 0f64,
            high: f64::MIN,
            low: f64::MAX,
            close: 0f64,
        }
    }

    pub fn open(&self) -> f64 {
        self.open
    }

    pub fn high(&self) -> f64 {
        self.high
    }

    pub fn low(&self) -> f64 {
        self.low
    }

    pub fn close(&self) -> f64 {
        self.close
    }

    pub fn mark(&mut self, price: f64) {
        self.open = price;
        self.high = price;
        self.low = price;
        self.close = price;
    }

    pub fn update(&mut self, price: f64) {
        self.high = self.high.max(price);
        self.low = self.low.min(price);
        self.close = price;
    }
}
