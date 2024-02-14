/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::cursor::{CursorExpansion, TimeShift};
use crate::sim::bb::ComputeId;
use crate::sim::compute::wide::expr::{IdentifiedProcessorNode, ProcessorHandle};
use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
use crate::sim::context::SpanContext;
use crate::sim::error::SetupError;
use crate::sim::selector::LaneSelector;
use std::collections::HashMap;
use std::rc::Rc;

pub struct FlowDescriptor {
    consumer: ComputeId,
    contracts: Vec<FlowContract>,
}

impl FlowDescriptor {
    pub fn new(consumer: ComputeId, contracts: Vec<FlowContract>) -> Self {
        assert!(
            contracts.as_slice().windows(2).all(|arr| {
                0 == arr[1]
                    .consumer_lane_selector()
                    .wrapping_sub(arr[0].consumer_lane_selector())
            }),
            "Flows must be sorted by consumer selector's and there cannot be gaps"
        );
        assert_eq!(
            0,
            contracts
                .get(0)
                .map(|f| f.consumer_lane_selector())
                .unwrap_or(0),
            "Sorted flows by consumer selector's must start from selector equal to 0"
        );
        assert!(contracts.iter().all(|fc| fc.consumer_id() == consumer));

        Self {
            consumer,
            contracts,
        }
    }

    pub fn create_aligned_dependencies(
        &self,
        registry: &HashMap<ComputeId, Rc<IdentifiedProcessorNode>>,
    ) -> Result<Dependencies, anyhow::Error> {
        let mut handles = Vec::with_capacity(self.contracts.len());

        for contract in &self.contracts {
            let processor = registry
                .get(&contract.producer_id())
                .ok_or_else(|| SetupError::IndicatorNotCreated)?;

            handles.push(ProcessorHandle::new(
                Rc::clone(&processor),
                contract.producer_lane_selector(),
                contract.time_shift(),
            ));
        }

        Ok(Dependencies {
            consumer: self.consumer,
            dependencies: handles,
        })
    }
}

pub struct Dependencies {
    consumer: ComputeId,
    dependencies: Vec<ProcessorHandle>,
}

impl Dependencies {
    pub fn eval(
        &self,
        ctx: &SpanContext,
        cursor_expansion: CursorExpansion,
    ) -> Result<DynamicStore, anyhow::Error> {
        for dep in &self.dependencies {
            let _: () = dep.eval(ctx, cursor_expansion)?;
        }

        let mut store = DynamicStore::new(self.consumer);
        for dep in &self.dependencies {
            store.insert(dep.take()?);
        }

        Ok(store)
    }
}

///
/// Producer describes what lane we are interested in from data source.
/// Where consumer means receiver, to what input we will assign producer.
///
/// For example:
/// BollingerBands producers(outputs) are [upper_band, average, lower_band],
/// and AverageTrueRange consumers(inputs) are [high, low, close].
///
/// Then we could create three selectors to assign BollingerBands producers(outputs) to AverageTrueRange consumers(inputs):
/// * FlowContract(producer: 0, consumer: 0), [upper_band -> high]
/// * FlowContract(producer: 2, consumer: 1), [lower_band -> low]
/// * FlowContract(producer: 1, consumer: 2), [average    -> close]
///
pub struct FlowContract {
    producer: LaneSelector,
    consumer: LaneSelector,
    time_shift: Option<TimeShift>,
}

impl FlowContract {
    pub fn new(
        producer: LaneSelector,
        consumer: LaneSelector,
        time_shift: Option<TimeShift>,
    ) -> Self {
        Self {
            producer,
            consumer,
            time_shift,
        }
    }

    pub fn consumer_id(&self) -> ComputeId {
        self.consumer.compute_id()
    }

    pub fn producer_id(&self) -> ComputeId {
        self.producer.compute_id()
    }

    pub fn producer_lane_selector(&self) -> usize {
        self.producer.selector()
    }

    pub fn consumer_lane_selector(&self) -> usize {
        self.consumer.selector()
    }

    pub fn time_shift(&self) -> Option<TimeShift> {
        self.time_shift
    }
}
