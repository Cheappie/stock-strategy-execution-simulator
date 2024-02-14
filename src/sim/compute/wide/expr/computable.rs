/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use std::cell::Ref;
use std::ops::Range;
use std::rc::Rc;

use crate::sim::bb::cursor::{CursorExpansion, Epoch, TimeShift};
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::bb::{cursor, ComputeId};
use crate::sim::collections::bitmap::Bitmap;
use crate::sim::compute::wide::expr::{
    ConstantHandle, IdentifiedProcessorNode, LaneProcessorHandle, MathExprHandle, PredicateNode,
    PrimitiveNode, ProcessorHandle, ProcessorNode, WideMathExprNode, WideProcessorHandle,
};
use crate::sim::compute::wide::expr::{IndicatorNode, ReadHandle};
use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
use crate::sim::compute::wide::indicator::exponential_moving_average::ExponentialMovingAverage;
use crate::sim::compute::wide::indicator::simple_moving_average::SimpleMovingAverage;
use crate::sim::compute::wide::indicator::WideProcessor;
use crate::sim::context::SpanContext;
use crate::sim::error::ContractError;
use crate::sim::mediator::Mediator;
use crate::sim::reader::{DataReader, Reader, ReaderFactory};
use crate::sim::spatial_buffer::SpatialBuffer;

impl ProcessorHandle {
    pub fn new(
        processor: Rc<IdentifiedProcessorNode>,
        output_selector: usize,
        time_shift: Option<TimeShift>,
    ) -> Self {
        Self {
            processor,
            output_selector,
            time_shift,
        }
    }

    pub fn eval(
        &self,
        ctx: &SpanContext,
        cursor_expansion: CursorExpansion,
    ) -> Result<(), anyhow::Error> {
        self.processor.eval(ctx, cursor_expansion)
    }

    pub fn take(&self) -> Result<ReadHandle, anyhow::Error> {
        self.processor.read_handle(self.output_selector)
    }
}

impl IdentifiedProcessorNode {
    pub fn new(compute_id: ComputeId, processor: ProcessorNode) -> Self {
        Self {
            compute_id,
            processor,
        }
    }

    pub fn compute_id(&self) -> ComputeId {
        self.compute_id
    }

    pub fn eval(
        &self,
        ctx: &SpanContext,
        cursor_expansion: CursorExpansion,
    ) -> Result<(), anyhow::Error> {
        match &self.processor {
            ProcessorNode::Indicator {
                indicator,
                dependencies,
            } => {
                let mut indicator = indicator.borrow_mut();

                let available = indicator.epoch();
                let requested = cursor::cursor_to_frame_indices(
                    ctx.cursor(),
                    cursor_expansion,
                    &*ctx.session().translation_buffer(available.time_frame())?,
                )?;

                if Epoch::is_superset(available, &requested)? {
                    Ok(())
                } else {
                    let store = dependencies.eval(ctx, cursor_expansion)?;
                    indicator.eval(ctx, cursor_expansion, &store)
                }
            }
            ProcessorNode::WideMathExpr(math_expr) => {
                let mut math_expr = math_expr.borrow_mut();

                let available = math_expr.epoch();
                let requested = cursor::cursor_to_frame_indices(
                    ctx.cursor(),
                    cursor_expansion,
                    &*ctx.session().translation_buffer(available.time_frame())?,
                )?;

                if Epoch::is_superset(available, &requested)? {
                    Ok(())
                } else {
                    let empty = DynamicStore::new(self.compute_id);
                    math_expr.eval(ctx, cursor_expansion, &empty)
                }
            }
            ProcessorNode::MathExpr(math_expr) => {
                math_expr.borrow_mut().eval(ctx, cursor_expansion)
            }
            ProcessorNode::Primitive(_) => Ok(()),
            ProcessorNode::Constant(_constant) => Ok(()),
        }
    }

    pub fn read_handle(&self, output_selector: usize) -> Result<ReadHandle, anyhow::Error> {
        match &self.processor {
            ProcessorNode::Indicator { indicator, .. } => Ok(ReadHandle::LaneProcessor(
                LaneProcessorHandle::WideProcessor(WideProcessorHandle {
                    wide_processor_ref: indicator.borrow(),
                    output_selector,
                }),
            )),
            ProcessorNode::WideMathExpr(wide_math_expr) => Ok(ReadHandle::LaneProcessor(
                LaneProcessorHandle::WideProcessor(WideProcessorHandle {
                    wide_processor_ref: wide_math_expr.borrow(),
                    output_selector,
                }),
            )),
            ProcessorNode::MathExpr(math_expr) => Ok(ReadHandle::Constant(
                ConstantHandle::MathExpr(MathExprHandle {
                    math_expr_ref: math_expr.borrow(),
                    output_selector,
                }),
            )),
            ProcessorNode::Primitive(primitive) => Ok(ReadHandle::LaneProcessor(
                LaneProcessorHandle::Primitive(primitive.output_buffer(output_selector)?),
            )),
            ProcessorNode::Constant(constant) => {
                debug_assert_eq!(
                    0, output_selector,
                    "Output selector for constant should be equal to 0"
                );

                Ok(ReadHandle::Constant(ConstantHandle::Constant(constant)))
            }
        }
    }
}

impl WideProcessor for IndicatorNode {
    fn eval(
        &mut self,
        ctx: &SpanContext,
        cursor_expansion: CursorExpansion,
        store: &DynamicStore,
    ) -> Result<(), anyhow::Error> {
        match self {
            IndicatorNode::SimpleMovingAverage(simple_moving_average) => {
                simple_moving_average.eval(ctx, cursor_expansion, store)
            }
            IndicatorNode::ExponentialMovingAverage(exponential_moving_average) => {
                exponential_moving_average.eval(ctx, cursor_expansion, store)
            }
        }
    }

    fn output_buffer(&self, output_selector: usize) -> Result<&SpatialBuffer, anyhow::Error> {
        match self {
            IndicatorNode::SimpleMovingAverage(simple_moving_average) => {
                simple_moving_average.output_buffer(output_selector)
            }
            IndicatorNode::ExponentialMovingAverage(exponential_moving_average) => {
                exponential_moving_average.output_buffer(output_selector)
            }
        }
    }

    fn epoch(&self) -> &Epoch {
        match self {
            IndicatorNode::SimpleMovingAverage(simple_moving_average) => {
                simple_moving_average.epoch()
            }
            IndicatorNode::ExponentialMovingAverage(exponential_moving_average) => {
                exponential_moving_average.epoch()
            }
        }
    }
}
