/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::cursor;
use crate::sim::bb::cursor::{CursorExpansion, Epoch};
use crate::sim::compute::guard;
use crate::sim::context::SpanContext;
use crate::sim::error::SetupError;
use crate::sim::spatial_buffer::{SpatialBuffer, SpatialReader, SpatialWriter};

pub fn create_reader<'a>(
    writer: &SpatialWriter,
    look_back_period: usize,
    input: &'a SpatialBuffer,
) -> Result<SpatialReader<'a>, anyhow::Error> {
    let epoch_read_with_look_back = cursor::epoch_look_back(&writer.epoch(), look_back_period)?;
    input.create_reader(&epoch_read_with_look_back)
}

pub fn create_writer<'a>(
    ctx: &SpanContext,
    cursor_expansion: CursorExpansion,
    output: &'a mut SpatialBuffer,
) -> Result<SpatialWriter<'a>, anyhow::Error> {
    let translation_buffer = ctx.session().translation_buffer(output.time_frame())?;
    let epoch_write =
        cursor::cursor_to_frame_indices(ctx.cursor(), cursor_expansion, &translation_buffer)?;
    Ok(output.create_writer(&epoch_write)?.unstable_writer())
}
