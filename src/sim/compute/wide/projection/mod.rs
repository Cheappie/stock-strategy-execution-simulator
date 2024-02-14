/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use anyhow::Error;

use crate::sim::bb::cursor::{CursorExpansion, Epoch};
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
use crate::sim::compute::wide::indicator::WideProcessor;
use crate::sim::compute::wide::projection::simple_moving_average::SimpleMovingAverageProjection;
use crate::sim::context::SpanContext;
use crate::sim::spatial_buffer::SpatialBuffer;

///
/// Projects `computable` of higher time frame into base quotes.
/// Lanes will be emitted in same time frame as base quotes.
///
pub mod simple_moving_average;

pub enum ProjectedIndicator {
    SimpleMovingAverageProjection(SimpleMovingAverageProjection),
}

impl WideProcessor for ProjectedIndicator {
    fn eval(
        &mut self,
        ctx: &SpanContext,
        cursor_expansion: CursorExpansion,
        store: &DynamicStore,
    ) -> Result<(), Error> {
        match self {
            ProjectedIndicator::SimpleMovingAverageProjection(simple_moving_average) => {
                simple_moving_average.eval(ctx, cursor_expansion, store)
            }
        }
    }

    fn output_buffer(&self, output_selector: usize) -> Result<&SpatialBuffer, Error> {
        match self {
            ProjectedIndicator::SimpleMovingAverageProjection(simple_moving_average) => {
                simple_moving_average.output_buffer(output_selector)
            }
        }
    }

    fn epoch(&self) -> &Epoch {
        match self {
            ProjectedIndicator::SimpleMovingAverageProjection(simple_moving_average) => {
                simple_moving_average.epoch()
            }
        }
    }
}
