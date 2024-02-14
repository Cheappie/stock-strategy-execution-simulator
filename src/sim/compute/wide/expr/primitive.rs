/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::cursor::Epoch;
use crate::sim::bb::quotes::FrameQuotes;
use crate::sim::compute::wide::expr::PrimitiveNode;
use crate::sim::error::{ContractError, DefinitionError};
use crate::sim::primitive_buffer::PrimitiveBuffer;
use crate::sim::reader::{DataReader, Reader, ReaderFactory};
use crate::sim::selector::Selector;
use std::ops::Range;
use std::str::FromStr;
use std::sync::Arc;

impl PrimitiveNode {
    pub fn new(frame_quotes: Arc<FrameQuotes>) -> Self {
        Self { frame_quotes }
    }

    pub fn output_buffer(
        &self,
        output_selector: usize,
    ) -> Result<PrimitiveBuffer<'_>, anyhow::Error> {
        let epoch = Epoch::new(0..self.frame_quotes.len(), self.frame_quotes.time_frame());

        match PrimitiveOutputSelector::from_ordinal(output_selector)? {
            PrimitiveOutputSelector::Open => {
                Ok(PrimitiveBuffer::new(self.frame_quotes.open_array(), epoch))
            }
            PrimitiveOutputSelector::High => {
                Ok(PrimitiveBuffer::new(self.frame_quotes.high_array(), epoch))
            }
            PrimitiveOutputSelector::Low => {
                Ok(PrimitiveBuffer::new(self.frame_quotes.low_array(), epoch))
            }
            PrimitiveOutputSelector::Close => {
                Ok(PrimitiveBuffer::new(self.frame_quotes.close_array(), epoch))
            }
        }
    }
}

pub enum PrimitiveOutputSelector {
    Open,
    High,
    Low,
    Close,
}

impl FromStr for PrimitiveOutputSelector {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "open" => Ok(PrimitiveOutputSelector::Open),
            "high" => Ok(PrimitiveOutputSelector::High),
            "low" => Ok(PrimitiveOutputSelector::Low),
            "close" => Ok(PrimitiveOutputSelector::Close),
            _ => Err(DefinitionError::UnrecognizedSelector {
                component: "PrimitiveOutputSelector",
                unrecognized: String::from(s),
            }
            .into()),
        }
    }
}

impl Selector for PrimitiveOutputSelector {
    fn from_ordinal(ordinal: usize) -> Result<Self, anyhow::Error> {
        match ordinal {
            0 => Ok(PrimitiveOutputSelector::Open),
            1 => Ok(PrimitiveOutputSelector::High),
            2 => Ok(PrimitiveOutputSelector::Low),
            3 => Ok(PrimitiveOutputSelector::Close),
            _ => Err(DefinitionError::UnrecognizedSelectorOrdinal {
                component: "PrimitiveOutputSelector",
                unrecognized: ordinal,
            }
            .into()),
        }
    }

    fn ordinal(&self) -> usize {
        match self {
            PrimitiveOutputSelector::Open => 0,
            PrimitiveOutputSelector::High => 1,
            PrimitiveOutputSelector::Low => 2,
            PrimitiveOutputSelector::Close => 3,
        }
    }
}
