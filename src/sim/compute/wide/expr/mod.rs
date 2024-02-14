/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use std::cell::{Ref, RefCell};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use crate::sim::bb::cursor::{Epoch, TimeShift};
use crate::sim::bb::quotes::FrameQuotes;
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::bb::ComputeId;
use crate::sim::compute::operators::{BinaryMathOperator, Ordering, UnaryMathOperator};
use crate::sim::compute::translation_strategy::TranslationStrategy;
use crate::sim::compute::wide::indicator::exponential_moving_average::ExponentialMovingAverage;
use crate::sim::compute::wide::indicator::simple_moving_average::SimpleMovingAverage;
use crate::sim::compute::wide::indicator::WideProcessor;
use crate::sim::flow_descriptor::Dependencies;
use crate::sim::primitive_buffer::PrimitiveBuffer;
use crate::sim::reader::{Reader, ReaderFactory};
use crate::sim::spatial_buffer::SpatialBuffer;

pub mod computable;
pub mod logical;
pub mod math;
mod math_tests;
pub mod predicate;
mod predicate_tests;
pub mod primitive;

pub enum LogicalNode {
    And(Box<LogicalNode>, Box<LogicalNode>),
    Or(Box<LogicalNode>, Box<LogicalNode>),
    Not(Box<LogicalNode>),
    Predicate(Box<PredicateNode>),
}

pub struct PredicateNode {
    left: Box<ProcessorHandle>,
    ordering: Ordering,
    right: Box<ProcessorHandle>,
    strategy: TranslationStrategy,
}

pub struct ProcessorHandle {
    processor: Rc<IdentifiedProcessorNode>,
    output_selector: usize,
    time_shift: Option<TimeShift>,
}

pub struct IdentifiedProcessorNode {
    compute_id: ComputeId,
    processor: ProcessorNode,
}

pub enum ProcessorNode {
    Indicator {
        indicator: RefCell<IndicatorNode>,
        dependencies: Dependencies,
    },
    WideMathExpr(RefCell<WideMathExprNode>),
    MathExpr(RefCell<MathExprNode>),
    Primitive(PrimitiveNode),
    Constant(Constant),
}

pub enum IndicatorNode {
    SimpleMovingAverage(SimpleMovingAverage),
    ExponentialMovingAverage(ExponentialMovingAverage),
}

pub enum WideMathExprNode {
    Unary(WideUnaryMathExpr),
    Binary(WideBinaryMathExpr),
}

pub struct WideBinaryMathExpr {
    left: Box<ProcessorHandle>,
    operator: BinaryMathOperator,
    right: Box<ProcessorHandle>,
    output: WideBinaryExprOutput,
}

pub struct WideBinaryExprOutput {
    strategy: TranslationStrategy,
    buffer: SpatialBuffer,
}

impl WideBinaryExprOutput {
    pub fn new(strategy: TranslationStrategy, buffer: SpatialBuffer) -> Self {
        Self { strategy, buffer }
    }
}

pub struct WideUnaryMathExpr {
    expr: Box<ProcessorHandle>,
    operator: UnaryMathOperator,
    output: SpatialBuffer,
}

pub enum MathExprNode {
    Unary(UnaryMathExpr),
    Binary(BinaryMathExpr),
}

pub struct UnaryMathExpr {
    expr: Box<ProcessorHandle>,
    operator: UnaryMathOperator,
    output: Option<Constant>,
}

pub struct BinaryMathExpr {
    left: Box<ProcessorHandle>,
    operator: BinaryMathOperator,
    right: Box<ProcessorHandle>,
    output: Option<Constant>,
}

pub struct PrimitiveNode {
    frame_quotes: Arc<FrameQuotes>,
}

pub struct Constant(Scalar);

impl Constant {
    pub fn new(value: Scalar) -> Self {
        Self(value)
    }

    pub fn value(&self) -> &Scalar {
        &self.0
    }
}

pub enum Scalar {
    F64(f64),
}

pub enum ReadHandle<'a> {
    LaneProcessor(LaneProcessorHandle<'a>),
    Constant(ConstantHandle<'a>),
}

pub enum LaneProcessorHandle<'a> {
    WideProcessor(WideProcessorHandle<'a>),
    Primitive(PrimitiveBuffer<'a>),
}

impl LaneProcessorHandle<'_> {
    pub fn try_read(&self) -> Result<&dyn ReaderFactory, anyhow::Error> {
        match self {
            LaneProcessorHandle::WideProcessor(wide) => wide.try_read(),
            LaneProcessorHandle::Primitive(primitive) => Ok(primitive as &dyn ReaderFactory),
        }
    }

    pub fn time_frame(&self) -> TimeFrame {
        match self {
            LaneProcessorHandle::WideProcessor(wide) => wide.time_frame(),
            LaneProcessorHandle::Primitive(primitive) => primitive.time_frame(),
        }
    }
}

pub struct WideProcessorHandle<'a> {
    wide_processor_ref: Ref<'a, dyn WideProcessor>,
    output_selector: usize,
}

impl WideProcessorHandle<'_> {
    pub fn try_read(&self) -> Result<&dyn ReaderFactory, anyhow::Error> {
        self.wide_processor_ref
            .output_buffer(self.output_selector)
            .map(|buf| buf as &dyn ReaderFactory)
    }

    pub fn time_frame(&self) -> TimeFrame {
        self.wide_processor_ref.epoch().time_frame()
    }
}

pub enum ConstantHandle<'a> {
    MathExpr(MathExprHandle<'a>),
    Constant(&'a Constant),
}

impl ConstantHandle<'_> {
    pub fn try_read(&self) -> Result<&Constant, anyhow::Error> {
        match self {
            ConstantHandle::MathExpr(math_expr_handle) => math_expr_handle
                .math_expr_ref
                .output(math_expr_handle.output_selector),
            ConstantHandle::Constant(constant) => Ok(constant),
        }
    }
}

pub struct MathExprHandle<'a> {
    math_expr_ref: Ref<'a, MathExprNode>,
    output_selector: usize,
}

pub struct Lane<'a> {
    reader_factory: &'a dyn ReaderFactory,
    time_shift: Option<TimeShift>,
}

impl<'a> Lane<'a> {
    pub fn new(reader_factory: &'a dyn ReaderFactory, time_shift: Option<TimeShift>) -> Self {
        Self {
            reader_factory,
            time_shift,
        }
    }

    pub fn create_reader(&self, epoch_read: &Epoch) -> Result<Reader<'_>, anyhow::Error> {
        self.reader_factory.create_reader(epoch_read)
    }

    pub fn time_shift(&self) -> Option<TimeShift> {
        self.time_shift
    }

    pub fn time_frame(&self) -> TimeFrame {
        self.reader_factory.time_frame()
    }
}
