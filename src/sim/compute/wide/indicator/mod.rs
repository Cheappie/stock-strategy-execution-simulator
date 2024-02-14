/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use std::str::FromStr;

use dynamic_store::DynamicStore;

use crate::sim::bb::cursor::{CursorExpansion, Epoch};
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::compute::wide::indicator::exponential_moving_average::ExponentialMovingAverage;
use crate::sim::compute::wide::indicator::simple_moving_average::SimpleMovingAverage;
use crate::sim::context::SpanContext;
use crate::sim::error::DefinitionError;
use crate::sim::selector::{LaneSelector, Selector};
use crate::sim::spatial_buffer::SpatialBuffer;

pub mod average_true_range;
pub mod bollinger_bands;
mod cumulative_quotes;
pub mod dynamic_store;
pub mod exponential_moving_average;
pub mod simple_moving_average;

pub trait WideProcessor {
    fn eval(
        &mut self,
        ctx: &SpanContext,
        cursor_expansion: CursorExpansion,
        store: &DynamicStore,
    ) -> Result<(), anyhow::Error>;

    fn output_buffer(&self, output_selector: usize) -> Result<&SpatialBuffer, anyhow::Error>;

    fn epoch(&self) -> &Epoch;
}

pub struct SingleInputSelector;

impl FromStr for SingleInputSelector {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "single_input_selector" => Ok(SingleInputSelector),
            _ => Err(DefinitionError::UnrecognizedSelector {
                component: "SingleInputSelector",
                unrecognized: String::from(s),
            }
            .into()),
        }
    }
}

impl Selector for SingleInputSelector {
    fn from_ordinal(ordinal: usize) -> Result<Self, anyhow::Error> {
        match ordinal {
            0 => Ok(SingleInputSelector),
            _ => Err(DefinitionError::UnrecognizedSelectorOrdinal {
                component: "SingleInputSelector",
                unrecognized: ordinal,
            }
            .into()),
        }
    }

    fn ordinal(&self) -> usize {
        0
    }
}

pub struct SingleOutputSelector;

impl FromStr for SingleOutputSelector {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "single_output_selector" => Ok(SingleOutputSelector),
            _ => Err(DefinitionError::UnrecognizedSelector {
                component: "SingleOutputSelector",
                unrecognized: String::from(s),
            }
            .into()),
        }
    }
}

impl Selector for SingleOutputSelector {
    fn from_ordinal(ordinal: usize) -> Result<Self, anyhow::Error> {
        match ordinal {
            0 => Ok(SingleOutputSelector),
            _ => Err(DefinitionError::UnrecognizedSelectorOrdinal {
                component: "SingleOutputSelector",
                unrecognized: ordinal,
            }
            .into()),
        }
    }

    fn ordinal(&self) -> usize {
        0
    }
}
