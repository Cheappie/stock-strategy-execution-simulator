/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use std::str::FromStr;

use crate::sim::bb::cursor::{Cursor, CursorExpansion, Epoch};
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::collections::circular_buffer::CircularBuffer;
use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
use crate::sim::compute::wide::indicator::{SingleInputSelector, WideProcessor};
use crate::sim::context::{SessionContext, SessionContextRef, SpanContext};
use crate::sim::error::DefinitionError;
use crate::sim::selector::{LaneSelector, Selector};
use crate::sim::spatial_buffer::SpatialBuffer;

pub struct ExponentialMovingAverage {
    span: usize,
    output: SpatialBuffer,
}

impl WideProcessor for ExponentialMovingAverage {
    fn eval(
        &mut self,
        ctx: &SpanContext,
        cursor_expansion: CursorExpansion,
        store: &DynamicStore,
    ) -> Result<(), anyhow::Error> {
        let _input_buffer = store.read_handle(SingleInputSelector);
        todo!()
    }

    fn output_buffer(&self, output_selector: usize) -> Result<&SpatialBuffer, anyhow::Error> {
        match ExponentialMovingAverageOutputSelector::from_ordinal(output_selector)? {
            ExponentialMovingAverageOutputSelector::ExponentialMovingAverage => Ok(&self.output),
        }
    }

    fn epoch(&self) -> &Epoch {
        self.output.epoch()
    }
}

pub enum ExponentialMovingAverageOutputSelector {
    ExponentialMovingAverage,
}

impl FromStr for ExponentialMovingAverageOutputSelector {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "exponential_moving_average" => {
                Ok(ExponentialMovingAverageOutputSelector::ExponentialMovingAverage)
            }
            _ => Err(DefinitionError::UnrecognizedSelector {
                component: "ExponentialMovingAverageOutputSelector",
                unrecognized: String::from(s),
            }
            .into()),
        }
    }
}

impl Selector for ExponentialMovingAverageOutputSelector {
    fn from_ordinal(ordinal: usize) -> Result<Self, anyhow::Error> {
        match ordinal {
            0 => Ok(ExponentialMovingAverageOutputSelector::ExponentialMovingAverage),
            _ => Err(DefinitionError::UnrecognizedSelectorOrdinal {
                component: "ExponentialMovingAverageOutputSelector",
                unrecognized: ordinal,
            }
            .into()),
        }
    }

    fn ordinal(&self) -> usize {
        match self {
            ExponentialMovingAverageOutputSelector::ExponentialMovingAverage => 0,
        }
    }
}
