/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use std::any::{Any, TypeId};
use std::marker::PhantomData;
use std::str::FromStr;
use std::sync::{Arc, Mutex, RwLock};

use crate::sim::bb::cursor::Epoch;
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::bb::ComputeId;
use crate::sim::context::SpanContext;

pub trait Selector: FromStr {
    fn from_ordinal(ordinal: usize) -> Result<Self, anyhow::Error>;

    fn ordinal(&self) -> usize;
}

pub struct LaneSelector {
    compute_id: ComputeId,
    selector: usize,
}

impl LaneSelector {
    pub fn new(compute_id: ComputeId, selector: impl Selector) -> LaneSelector {
        Self {
            compute_id,
            selector: selector.ordinal(),
        }
    }

    pub fn compute_id(&self) -> ComputeId {
        self.compute_id
    }

    pub fn selector(&self) -> usize {
        self.selector
    }
}
