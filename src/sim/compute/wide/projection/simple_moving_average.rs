/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::cursor::{CursorExpansion, Epoch};
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
use crate::sim::compute::wide::indicator::WideProcessor;
use crate::sim::context::SpanContext;
use crate::sim::spatial_buffer::SpatialBuffer;

pub struct SimpleMovingAverageProjection {
    span: usize,
    output: SpatialBuffer,
}

impl WideProcessor for SimpleMovingAverageProjection {
    fn eval(
        &mut self,
        ctx: &SpanContext,
        cursor_expansion: CursorExpansion,
        store: &DynamicStore,
    ) -> Result<(), anyhow::Error> {
        debug_assert_eq!(self.output.time_frame(), ctx.cursor().time_frame());
        todo!()
    }

    fn output_buffer(&self, output_selector: usize) -> Result<&SpatialBuffer, anyhow::Error> {
        todo!()
    }

    fn epoch(&self) -> &Epoch {
        todo!()
    }
}
