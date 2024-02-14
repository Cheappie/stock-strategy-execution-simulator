/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use std::str::FromStr;

use crate::sim::bb::cursor::{CursorExpansion, Epoch};
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::compute::guard::epoch_matches;
use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
use crate::sim::compute::wide::indicator::{SingleInputSelector, WideProcessor};
use crate::sim::context::SpanContext;
use crate::sim::error::DefinitionError;
use crate::sim::selector::Selector;
use crate::sim::spatial_buffer::SpatialBuffer;

pub struct BollingerBands {
    span: usize,
    multiplier: f64,
    output: BollingerBandsOutputLanes,
}

impl WideProcessor for BollingerBands {
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
        match BollingerBandsOutputSelector::from_ordinal(output_selector)? {
            BollingerBandsOutputSelector::UpperBound => Ok(&self.output.upper_bound),
            BollingerBandsOutputSelector::Average => Ok(&self.output.average),
            BollingerBandsOutputSelector::LowerBound => Ok(&self.output.lower_bound),
        }
    }

    fn epoch(&self) -> &Epoch {
        epoch_matches!(
            &self.output.upper_bound,
            &self.output.average,
            &self.output.lower_bound
        )
    }
}

struct BollingerBandsOutputLanes {
    upper_bound: SpatialBuffer,
    average: SpatialBuffer,
    lower_bound: SpatialBuffer,
}

pub enum BollingerBandsOutputSelector {
    UpperBound,
    Average,
    LowerBound,
}

impl FromStr for BollingerBandsOutputSelector {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "upper_bound" => Ok(BollingerBandsOutputSelector::UpperBound),
            "average" => Ok(BollingerBandsOutputSelector::Average),
            "lower_bound" => Ok(BollingerBandsOutputSelector::LowerBound),
            _ => Err(DefinitionError::UnrecognizedSelector {
                component: "BollingerBandsOutputSelector",
                unrecognized: String::from(s),
            }
            .into()),
        }
    }
}

impl Selector for BollingerBandsOutputSelector {
    fn from_ordinal(ordinal: usize) -> Result<Self, anyhow::Error> {
        match ordinal {
            0 => Ok(BollingerBandsOutputSelector::UpperBound),
            1 => Ok(BollingerBandsOutputSelector::Average),
            2 => Ok(BollingerBandsOutputSelector::LowerBound),
            _ => Err(DefinitionError::UnrecognizedSelectorOrdinal {
                component: "BollingerBandsOutputSelector",
                unrecognized: ordinal,
            }
            .into()),
        }
    }

    fn ordinal(&self) -> usize {
        match self {
            BollingerBandsOutputSelector::UpperBound => 0,
            BollingerBandsOutputSelector::Average => 1,
            BollingerBandsOutputSelector::LowerBound => 2,
        }
    }
}
