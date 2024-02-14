/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use std::str::FromStr;

use crate::sim::bb::cursor::{CursorExpansion, Epoch};
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
use crate::sim::compute::wide::indicator::WideProcessor;
use crate::sim::context::SpanContext;
use crate::sim::error::DefinitionError;
use crate::sim::selector::Selector;
use crate::sim::spatial_buffer::SpatialBuffer;

pub struct AverageTrueRange {
    output: SpatialBuffer,
}

impl WideProcessor for AverageTrueRange {
    fn eval(
        &mut self,
        ctx: &SpanContext,
        cursor_expansion: CursorExpansion,
        store: &DynamicStore,
    ) -> Result<(), anyhow::Error> {
        let _high = store.read_handle(AverageTrueRangeInputSelector::High)?;
        let _low = store.read_handle(AverageTrueRangeInputSelector::Low)?;
        let _close = store.read_handle(AverageTrueRangeInputSelector::Close)?;
        todo!("Impl ATR")
    }

    fn output_buffer(&self, output_selector: usize) -> Result<&SpatialBuffer, anyhow::Error> {
        todo!()
    }

    fn epoch(&self) -> &Epoch {
        self.output.epoch()
    }
}

pub enum AverageTrueRangeInputSelector {
    High,
    Low,
    Close,
}

impl FromStr for AverageTrueRangeInputSelector {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "atr_high" => Ok(AverageTrueRangeInputSelector::High),
            "atr_low" => Ok(AverageTrueRangeInputSelector::Low),
            "atr_close" => Ok(AverageTrueRangeInputSelector::Close),
            _ => Err(DefinitionError::UnrecognizedSelector {
                component: "AverageTrueRangeInputSelector",
                unrecognized: String::from(s),
            }
            .into()),
        }
    }
}

impl Selector for AverageTrueRangeInputSelector {
    fn from_ordinal(ordinal: usize) -> Result<Self, anyhow::Error> {
        match ordinal {
            0 => Ok(AverageTrueRangeInputSelector::High),
            1 => Ok(AverageTrueRangeInputSelector::Low),
            2 => Ok(AverageTrueRangeInputSelector::Close),
            _ => Err(DefinitionError::UnrecognizedSelectorOrdinal {
                component: "AverageTrueRangeInputSelector",
                unrecognized: ordinal,
            }
            .into()),
        }
    }

    fn ordinal(&self) -> usize {
        match self {
            AverageTrueRangeInputSelector::High => 0,
            AverageTrueRangeInputSelector::Low => 1,
            AverageTrueRangeInputSelector::Close => 2,
        }
    }
}

pub enum AverageTrueRangeOutputSelector {
    AverageTrueRange,
}

impl FromStr for AverageTrueRangeOutputSelector {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "average_true_range" => Ok(AverageTrueRangeOutputSelector::AverageTrueRange),
            _ => Err(DefinitionError::UnrecognizedSelector {
                component: "AverageTrueRangeOutputSelector",
                unrecognized: String::from(s),
            }
            .into()),
        }
    }
}

impl Selector for AverageTrueRangeOutputSelector {
    fn from_ordinal(ordinal: usize) -> Result<Self, anyhow::Error> {
        match ordinal {
            0 => Ok(AverageTrueRangeOutputSelector::AverageTrueRange),
            _ => Err(DefinitionError::UnrecognizedSelectorOrdinal {
                component: "AverageTrueRangeOutputSelector",
                unrecognized: ordinal,
            }
            .into()),
        }
    }

    fn ordinal(&self) -> usize {
        match self {
            AverageTrueRangeOutputSelector::AverageTrueRange => 0,
        }
    }
}
