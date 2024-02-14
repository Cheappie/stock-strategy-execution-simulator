/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use std::any::TypeId;
use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use smallvec::SmallVec;

use crate::sim::bb::ComputeId;
use crate::sim::compute::wide::expr::ReadHandle;
use crate::sim::context::SpanContext;
use crate::sim::error::SetupError;
use crate::sim::flow_descriptor::FlowContract;
use crate::sim::selector::{LaneSelector, Selector};
use crate::sim::spatial_buffer::{SpatialBuffer, SpatialReader};

pub struct DynamicStore<'a> {
    compute_id: ComputeId,
    store: SmallVec<[ReadHandle<'a>; 8]>,
}

impl<'a> DynamicStore<'a> {
    pub fn new(compute_id: ComputeId) -> Self {
        Self {
            compute_id,
            store: SmallVec::new(),
        }
    }
}

impl<'a> DynamicStore<'a> {
    pub fn read_handle(&self, input_selector: impl Selector) -> Result<&ReadHandle, anyhow::Error> {
        self.store.get(input_selector.ordinal()).ok_or_else(|| {
            SetupError::LaneReadHandleNotCreated {
                receiver_compute_id: self.compute_id,
                selector: input_selector.ordinal(),
            }
            .into()
        })
    }

    pub fn insert<'b: 'a>(&mut self, handle: ReadHandle<'b>) {
        self.store.push(handle);
    }
}
