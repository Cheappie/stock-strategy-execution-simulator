/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use std::collections::VecDeque;
use std::str::FromStr;

use crate::sim::bb::cursor;
use anyhow::anyhow;
use smallvec::SmallVec;

use crate::sim::bb::cursor::{CursorExpansion, Epoch, TimeShift};
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::compute::generator::simple_moving_average::SimpleMovingAverageGenerator;
use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
use crate::sim::compute::wide::indicator::{SingleInputSelector, WideProcessor};
use crate::sim::compute::wide::pipeline_assembly;
use crate::sim::context::SpanContext;
use crate::sim::error::DefinitionError;
use crate::sim::selector::Selector;
use crate::sim::spatial_buffer::SpatialBuffer;

pub struct SimpleMovingAverage {
    span: usize,
    output: SpatialBuffer,
}

impl WideProcessor for SimpleMovingAverage {
    fn eval(
        &mut self,
        ctx: &SpanContext,
        cursor_expansion: CursorExpansion,
        store: &DynamicStore,
    ) -> Result<(), anyhow::Error> {
        let input = store.read_handle(SingleInputSelector)?;
        // match store.read_handle(SingleInputSelector)? {
        //     ReadHandle::Indicator { .. } => {}
        //     ReadHandle::MathExpr { .. } => {}
        //     ReadHandle::Constant(_) => {}
        // }

        // let (reader, mut writer) =
        //     assembly::channel(ctx, cursor_expansion, self.span, input, &mut self.output)?;
        //
        // let historical_epoch = cursor::epoch_look_back(writer.epoch(), self.span)?;
        // let sum_of_interval = reader.iter_epoch(&historical_epoch).sum::<f64>();
        // let mut generator = SimpleMovingAverageGenerator::new(self.span, sum_of_interval);
        //
        // let prev_epoch_read = writer.epoch().shift_by(TimeShift::Past(self.span))?;
        // let writer_epoch_read = writer.epoch();
        //
        // reader
        //     .iter_epoch(&prev_epoch_read)
        //     .zip(reader.iter_epoch(writer_epoch_read))
        //     .for_each(|(prev_last, curr)| {
        //         let sma = generator.next(*prev_last, *curr);
        //         writer.write(sma);
        //     });

        Ok(())
    }

    fn output_buffer(&self, output_selector: usize) -> Result<&SpatialBuffer, anyhow::Error> {
        match SimpleMovingAverageOutputSelector::from_ordinal(output_selector)? {
            SimpleMovingAverageOutputSelector::SimpleMovingAverage => Ok(&self.output),
        }
    }

    fn epoch(&self) -> &Epoch {
        self.output.epoch()
    }
}

pub enum SimpleMovingAverageOutputSelector {
    SimpleMovingAverage,
}

impl FromStr for SimpleMovingAverageOutputSelector {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "simple_moving_average" => Ok(SimpleMovingAverageOutputSelector::SimpleMovingAverage),
            _ => Err(DefinitionError::UnrecognizedSelector {
                component: "SimpleMovingAverage",
                unrecognized: String::from(s),
            }
            .into()),
        }
    }
}

impl Selector for SimpleMovingAverageOutputSelector {
    fn from_ordinal(ordinal: usize) -> Result<Self, anyhow::Error> {
        match ordinal {
            0 => Ok(SimpleMovingAverageOutputSelector::SimpleMovingAverage),
            _ => Err(DefinitionError::UnrecognizedSelectorOrdinal {
                component: "SimpleMovingAverage",
                unrecognized: ordinal,
            }
            .into()),
        }
    }

    fn ordinal(&self) -> usize {
        match self {
            SimpleMovingAverageOutputSelector::SimpleMovingAverage => 0,
        }
    }
}
