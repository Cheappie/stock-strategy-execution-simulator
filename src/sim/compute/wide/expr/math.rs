/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::cursor;
use crate::sim::bb::cursor::{CursorExpansion, Epoch, TimeShift};
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::compute::operators::math::{
    AbsOperator, AddOperator, BinaryOperator, DivideOperator, MaxOperator, MinOperator,
    MultiplyOperator, PowOperator, SqrtOperator, SubtractOperator, UnaryOperator,
};
use crate::sim::compute::operators::{BinaryMathOperator, UnaryMathOperator};
use crate::sim::compute::translation_strategy::{select_translation_strategy, TranslationStrategy};
use crate::sim::compute::wide::expr::{
    BinaryMathExpr, Constant, Lane, MathExprNode, ProcessorHandle, ReadHandle, Scalar,
    UnaryMathExpr, WideBinaryExprOutput, WideBinaryMathExpr, WideMathExprNode, WideUnaryMathExpr,
};
use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
use crate::sim::compute::wide::indicator::{SingleOutputSelector, WideProcessor};
use crate::sim::compute::wide::pipeline_assembly;
use crate::sim::context::SpanContext;
use crate::sim::error::{ASTError, ExecutionError, SetupError};
use crate::sim::mediator::Mediator;
use crate::sim::reader::{DataReader, Reader, ReaderFactory};
use crate::sim::selector::Selector;
use crate::sim::spatial_buffer::{SpatialBuffer, SpatialReader, SpatialWriter};
use crate::sim::tlb::{
    DirectBaseLifetimeTranslation, DirectBaseLifetimeTranslationBuffer, DirectLifetimeTranslation,
    DirectLifetimeTranslationBuffer, DirectTranslationBuffer, FrameTranslationBuffer,
    InlinedReverseTranslation, InlinedTranslation, TranslationUnitDescriptor,
    WaterfallBaseLifetimeTranslationBuffer, WaterfallLifetimeTranslation,
    WaterfallLifetimeTranslationBuffer,
};
use anyhow::anyhow;

impl WideProcessor for WideMathExprNode {
    fn eval(
        &mut self,
        ctx: &SpanContext,
        cursor_expansion: CursorExpansion,
        store: &DynamicStore,
    ) -> Result<(), anyhow::Error> {
        match self {
            WideMathExprNode::Unary(unary) => unary.eval(ctx, cursor_expansion, store),
            WideMathExprNode::Binary(binary) => binary.eval(ctx, cursor_expansion, store),
        }
    }

    fn output_buffer(&self, output_selector: usize) -> Result<&SpatialBuffer, anyhow::Error> {
        match self {
            WideMathExprNode::Unary(unary) => unary.output_buffer(output_selector),
            WideMathExprNode::Binary(binary) => binary.output_buffer(output_selector),
        }
    }

    fn epoch(&self) -> &Epoch {
        match self {
            WideMathExprNode::Unary(unary) => unary.epoch(),
            WideMathExprNode::Binary(binary) => binary.epoch(),
        }
    }
}

impl WideBinaryMathExpr {
    pub fn new(
        left: Box<ProcessorHandle>,
        operator: BinaryMathOperator,
        right: Box<ProcessorHandle>,
        output: WideBinaryExprOutput,
    ) -> Self {
        Self {
            left,
            operator,
            right,
            output,
        }
    }
}

impl WideProcessor for WideBinaryMathExpr {
    fn eval(
        &mut self,
        ctx: &SpanContext,
        cursor_expansion: CursorExpansion,
        _: &DynamicStore,
    ) -> Result<(), anyhow::Error> {
        let _: () = self.left.eval(
            ctx,
            TimeShift::max_expansion(cursor_expansion, self.left.time_shift),
        )?;
        let _: () = self.right.eval(
            ctx,
            TimeShift::max_expansion(cursor_expansion, self.right.time_shift),
        )?;

        let left = self.left.take()?;
        let right = self.right.take()?;

        match (left, right) {
            (ReadHandle::LaneProcessor(left), ReadHandle::LaneProcessor(right)) => {
                let left = Lane::new(left.try_read()?, self.left.time_shift);
                let right = Lane::new(right.try_read()?, self.right.time_shift);

                match &self.output.strategy {
                    TranslationStrategy::Direct(_) => eval_d2d_direct(
                        ctx,
                        cursor_expansion,
                        self.operator,
                        left,
                        right,
                        &mut self.output.buffer,
                    ),
                    TranslationStrategy::DirectBaseLifetime(_) => eval_d2d_direct_base_lifetime(
                        ctx,
                        cursor_expansion,
                        self.operator,
                        left,
                        right,
                        &mut self.output.buffer,
                    ),
                    TranslationStrategy::DirectLifetime(_) => eval_d2d_direct_lifetime(
                        ctx,
                        cursor_expansion,
                        self.operator,
                        left,
                        right,
                        &mut self.output.buffer,
                    ),
                    TranslationStrategy::WaterfallBaseLifetime(_) => {
                        eval_d2d_waterfall_base_lifetime(
                            ctx,
                            cursor_expansion,
                            self.operator,
                            left,
                            right,
                            &mut self.output.buffer,
                        )
                    }
                    TranslationStrategy::WaterfallLifetime(_) => eval_d2d_waterfall_lifetime(
                        ctx,
                        cursor_expansion,
                        self.operator,
                        left,
                        right,
                        &mut self.output.buffer,
                    ),
                }
            }
            (ReadHandle::LaneProcessor(left), ReadHandle::Constant(right)) => {
                let left = Lane::new(left.try_read()?, self.left.time_shift);
                let right = right.try_read()?;

                eval_d2c(
                    ctx,
                    cursor_expansion,
                    self.operator,
                    left,
                    right,
                    &mut self.output.buffer,
                )
            }
            (ReadHandle::Constant(left), ReadHandle::LaneProcessor(right)) => {
                let left = left.try_read()?;
                let right = Lane::new(right.try_read()?, self.right.time_shift);

                eval_c2d(
                    ctx,
                    cursor_expansion,
                    self.operator,
                    left,
                    right,
                    &mut self.output.buffer,
                )
            }
            (ReadHandle::Constant(_), ReadHandle::Constant(_)) => {
                Err(ASTError::WideMathExprNodeError(
                    "Scalar binary math expr should be used for constant input's",
                )
                .into())
            }
        }
    }

    fn output_buffer(&self, output_selector: usize) -> Result<&SpatialBuffer, anyhow::Error> {
        match SingleOutputSelector::from_ordinal(output_selector)? {
            SingleOutputSelector => Ok(&self.output.buffer),
        }
    }

    fn epoch(&self) -> &Epoch {
        self.output.buffer.epoch()
    }
}

fn eval_d2d_direct(
    ctx: &SpanContext,
    cursor_expansion: CursorExpansion,
    operator: BinaryMathOperator,
    left: Lane,
    right: Lane,
    output: &mut SpatialBuffer,
) -> Result<(), anyhow::Error> {
    let output = pipeline_assembly::create_writer(ctx, cursor_expansion, output)?;

    let left_read = TimeShift::shift_epoch(output.epoch(), left.time_shift())?;
    let left = left.create_reader(&left_read)?;

    let right_read = TimeShift::shift_epoch(output.epoch(), right.time_shift())?;
    let right = right.create_reader(&right_read)?;

    match operator {
        BinaryMathOperator::Add => eval_d2d_direct_inlined_reader::<AddOperator>(
            left, left_read, right, right_read, output,
        ),
        BinaryMathOperator::Sub => eval_d2d_direct_inlined_reader::<SubtractOperator>(
            left, left_read, right, right_read, output,
        ),
        BinaryMathOperator::Mul => eval_d2d_direct_inlined_reader::<MultiplyOperator>(
            left, left_read, right, right_read, output,
        ),
        BinaryMathOperator::Div => eval_d2d_direct_inlined_reader::<DivideOperator>(
            left, left_read, right, right_read, output,
        ),
        BinaryMathOperator::Pow => eval_d2d_direct_inlined_reader::<PowOperator>(
            left, left_read, right, right_read, output,
        ),
        BinaryMathOperator::Max => eval_d2d_direct_inlined_reader::<MaxOperator>(
            left, left_read, right, right_read, output,
        ),
        BinaryMathOperator::Min => eval_d2d_direct_inlined_reader::<MinOperator>(
            left, left_read, right, right_read, output,
        ),
    }
}

fn eval_d2d_direct_inlined_reader<OP: BinaryOperator>(
    left: Reader,
    left_read: Epoch,
    right: Reader,
    right_read: Epoch,
    output: SpatialWriter,
) -> Result<(), anyhow::Error> {
    match (left, right) {
        (Reader::SpatialReader(left), Reader::SpatialReader(right)) => {
            eval_d2d_direct_inlined_op::<_, _, OP>(left, left_read, right, right_read, output)
        }
        (Reader::SpatialReader(left), Reader::StaticReader(right)) => {
            eval_d2d_direct_inlined_op::<_, _, OP>(left, left_read, right, right_read, output)
        }
        (Reader::StaticReader(left), Reader::SpatialReader(right)) => {
            eval_d2d_direct_inlined_op::<_, _, OP>(left, left_read, right, right_read, output)
        }
        (Reader::StaticReader(left), Reader::StaticReader(right)) => {
            eval_d2d_direct_inlined_op::<_, _, OP>(left, left_read, right, right_read, output)
        }
    }
}

fn eval_d2d_direct_inlined_op<LDR, RDR, OP>(
    left: LDR,
    left_read: Epoch,
    right: RDR,
    right_read: Epoch,
    mut output: SpatialWriter,
) -> Result<(), anyhow::Error>
where
    LDR: DataReader,
    RDR: DataReader,
    OP: BinaryOperator,
{
    left.iter(left_read.as_range())
        .zip(right.iter(right_read.as_range()))
        .for_each(|(left, right)| output.write(OP::apply(*left, *right)));

    output.finish()
}

fn eval_d2d_direct_base_lifetime(
    ctx: &SpanContext,
    cursor_expansion: CursorExpansion,
    operator: BinaryMathOperator,
    left: Lane,
    right: Lane,
    output: &mut SpatialBuffer,
) -> Result<(), anyhow::Error> {
    debug_assert_eq!(ctx.cursor().time_frame(), output.time_frame());
    let output = pipeline_assembly::create_writer(ctx, cursor_expansion, output)?;
    let out_tlb = &*ctx.session().translation_buffer(output.time_frame())?;

    let lt_tlb = &*ctx.session().translation_buffer(left.time_frame())?;
    let left_reader = {
        let lt_read = cursor::convert_source_to_output_epoch(output.epoch(), out_tlb, &lt_tlb)?;
        let lt_read = TimeShift::shift_epoch(&lt_read, left.time_shift())?;
        left.create_reader(&lt_read)
    }?;

    let rt_tlb = &*ctx.session().translation_buffer(right.time_frame())?;
    let right_reader = {
        let rt_read = cursor::convert_source_to_output_epoch(output.epoch(), out_tlb, &rt_tlb)?;
        let rt_read = TimeShift::shift_epoch(&rt_read, right.time_shift())?;
        right.create_reader(&rt_read)
    }?;

    let dtb = if left.time_frame() > right.time_frame() {
        lt_tlb
    } else {
        rt_tlb
    }.direct().ok_or_else(|| {
        anyhow!(
            ASTError::WideMathExprNodeError(
                "Direct base lifetime evaluation requires that FTB of larger time frame is of type DirectLifetimeBuffer"
            )
        )
    })?;

    let direct_base_lifetime_translation_buffer = DirectBaseLifetimeTranslationBuffer::new(
        ctx.cursor().time_frame(),
        left.time_frame(),
        right.time_frame(),
        dtb,
    );

    match operator {
        BinaryMathOperator::Add => eval_d2d_direct_base_lifetime_inlined_reader::<AddOperator>(
            direct_base_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
        BinaryMathOperator::Sub => {
            eval_d2d_direct_base_lifetime_inlined_reader::<SubtractOperator>(
                direct_base_lifetime_translation_buffer,
                left_reader,
                left.time_shift(),
                right_reader,
                right.time_shift(),
                output,
            )
        }
        BinaryMathOperator::Mul => {
            eval_d2d_direct_base_lifetime_inlined_reader::<MultiplyOperator>(
                direct_base_lifetime_translation_buffer,
                left_reader,
                left.time_shift(),
                right_reader,
                right.time_shift(),
                output,
            )
        }
        BinaryMathOperator::Div => eval_d2d_direct_base_lifetime_inlined_reader::<DivideOperator>(
            direct_base_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
        BinaryMathOperator::Pow => eval_d2d_direct_base_lifetime_inlined_reader::<PowOperator>(
            direct_base_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
        BinaryMathOperator::Max => eval_d2d_direct_base_lifetime_inlined_reader::<MaxOperator>(
            direct_base_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
        BinaryMathOperator::Min => eval_d2d_direct_base_lifetime_inlined_reader::<MinOperator>(
            direct_base_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
    }
}

fn eval_d2d_direct_base_lifetime_inlined_reader<OP: BinaryOperator>(
    direct_base_lifetime_translation_buffer: DirectBaseLifetimeTranslationBuffer,
    left: Reader,
    left_shift: Option<TimeShift>,
    right: Reader,
    right_shift: Option<TimeShift>,
    output: SpatialWriter,
) -> Result<(), anyhow::Error> {
    match (left, right) {
        (Reader::SpatialReader(left), Reader::SpatialReader(right)) => {
            eval_d2d_direct_base_lifetime_inlined_op::<_, _, OP>(
                direct_base_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                output,
            )
        }
        (Reader::SpatialReader(left), Reader::StaticReader(right)) => {
            eval_d2d_direct_base_lifetime_inlined_op::<_, _, OP>(
                direct_base_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                output,
            )
        }
        (Reader::StaticReader(left), Reader::SpatialReader(right)) => {
            eval_d2d_direct_base_lifetime_inlined_op::<_, _, OP>(
                direct_base_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                output,
            )
        }
        (Reader::StaticReader(left), Reader::StaticReader(right)) => {
            eval_d2d_direct_base_lifetime_inlined_op::<_, _, OP>(
                direct_base_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                output,
            )
        }
    }
}

fn eval_d2d_direct_base_lifetime_inlined_op<LDR, RDR, OP>(
    direct_base_lifetime_translation_buffer: DirectBaseLifetimeTranslationBuffer,
    left: LDR,
    left_shift: Option<TimeShift>,
    right: RDR,
    right_shift: Option<TimeShift>,
    mut output: SpatialWriter,
) -> Result<(), anyhow::Error>
where
    LDR: DataReader,
    RDR: DataReader,
    OP: BinaryOperator,
{
    let mut iter = direct_base_lifetime_translation_buffer.translate(output.epoch())?;

    if let (None, None) = (left_shift, right_shift) {
        while let Some(translation) = iter.next() {
            match translation {
                DirectBaseLifetimeTranslation::Left(lt, rt) => {
                    let right = *right.get(rt);
                    left.iter(lt)
                        .for_each(|left| output.write(OP::apply(*left, right)));
                }
                DirectBaseLifetimeTranslation::Right(lt, rt) => {
                    let left = *left.get(lt);
                    right
                        .iter(rt)
                        .for_each(|right| output.write(OP::apply(left, *right)));
                }
            }
        }
    } else {
        let left_shift = left_shift.unwrap_or_else(|| TimeShift::nop());
        let right_shift = right_shift.unwrap_or_else(|| TimeShift::nop());

        while let Some(translation) = iter.next() {
            match translation {
                DirectBaseLifetimeTranslation::Left(lt, rt) => {
                    let shifted_lt = TimeShift::shift_range(lt, left_shift);
                    let shifted_rt = right_shift.apply(rt);

                    let right = *right.get(shifted_rt);
                    left.iter(shifted_lt)
                        .for_each(|left| output.write(OP::apply(*left, right)));
                }
                DirectBaseLifetimeTranslation::Right(lt, rt) => {
                    let shifted_lt = left_shift.apply(lt);
                    let shifted_rt = TimeShift::shift_range(rt, right_shift);

                    let left = *left.get(shifted_lt);
                    right
                        .iter(shifted_rt)
                        .for_each(|right| output.write(OP::apply(left, *right)));
                }
            }
        }
    }

    output.finish()
}

fn eval_d2d_direct_lifetime(
    ctx: &SpanContext,
    cursor_expansion: CursorExpansion,
    operator: BinaryMathOperator,
    left: Lane,
    right: Lane,
    output: &mut SpatialBuffer,
) -> Result<(), anyhow::Error> {
    let session = ctx.session();

    let lt_tlb = &*session.translation_buffer(left.time_frame())?;
    let rt_tlb = &*session.translation_buffer(right.time_frame())?;
    let out_tlb = &*session.translation_buffer(output.time_frame())?;

    let output = pipeline_assembly::create_writer(ctx, cursor_expansion, output)?;

    let left_reader = {
        let lt_read = cursor::convert_source_to_output_epoch(output.epoch(), out_tlb, lt_tlb)?;
        let lt_read = TimeShift::shift_epoch(&lt_read, left.time_shift())?;
        left.create_reader(&lt_read)
    }?;

    let right_reader = {
        let rt_read = cursor::convert_source_to_output_epoch(output.epoch(), out_tlb, rt_tlb)?;
        let rt_read = TimeShift::shift_epoch(&rt_read, right.time_shift())?;
        right.create_reader(&rt_read)
    }?;

    let (lt_tlb, rt_tlb) = if let (Some(lt), Some(rt)) = (lt_tlb.direct(), rt_tlb.direct()) {
        Ok((lt, rt))
    } else {
        Err(anyhow!(ASTError::WideMathExprNodeError(
            "Direct lifetime evaluation requires DirectLifetimeBuffers",
        )))
    }?;

    let direct_lifetime_translation_buffer = DirectLifetimeTranslationBuffer::new(lt_tlb, rt_tlb);

    match operator {
        BinaryMathOperator::Add => eval_d2d_direct_lifetime_inlined_reader::<AddOperator>(
            direct_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
        BinaryMathOperator::Sub => eval_d2d_direct_lifetime_inlined_reader::<SubtractOperator>(
            direct_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
        BinaryMathOperator::Mul => eval_d2d_direct_lifetime_inlined_reader::<MultiplyOperator>(
            direct_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
        BinaryMathOperator::Div => eval_d2d_direct_lifetime_inlined_reader::<DivideOperator>(
            direct_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
        BinaryMathOperator::Pow => eval_d2d_direct_lifetime_inlined_reader::<PowOperator>(
            direct_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
        BinaryMathOperator::Max => eval_d2d_direct_lifetime_inlined_reader::<MaxOperator>(
            direct_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
        BinaryMathOperator::Min => eval_d2d_direct_lifetime_inlined_reader::<MinOperator>(
            direct_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
    }
}

fn eval_d2d_direct_lifetime_inlined_reader<OP: BinaryOperator>(
    direct_lifetime_translation_buffer: DirectLifetimeTranslationBuffer,
    left: Reader,
    left_shift: Option<TimeShift>,
    right: Reader,
    right_shift: Option<TimeShift>,
    output: SpatialWriter,
) -> Result<(), anyhow::Error> {
    match (left, right) {
        (Reader::SpatialReader(left), Reader::SpatialReader(right)) => {
            eval_d2d_direct_lifetime_inlined_op::<_, _, OP>(
                direct_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                output,
            )
        }
        (Reader::SpatialReader(left), Reader::StaticReader(right)) => {
            eval_d2d_direct_lifetime_inlined_op::<_, _, OP>(
                direct_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                output,
            )
        }
        (Reader::StaticReader(left), Reader::SpatialReader(right)) => {
            eval_d2d_direct_lifetime_inlined_op::<_, _, OP>(
                direct_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                output,
            )
        }
        (Reader::StaticReader(left), Reader::StaticReader(right)) => {
            eval_d2d_direct_lifetime_inlined_op::<_, _, OP>(
                direct_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                output,
            )
        }
    }
}

fn eval_d2d_direct_lifetime_inlined_op<LDR, RDR, OP>(
    direct_lifetime_translation_buffer: DirectLifetimeTranslationBuffer,
    left: LDR,
    left_shift: Option<TimeShift>,
    right: RDR,
    right_shift: Option<TimeShift>,
    mut output: SpatialWriter,
) -> Result<(), anyhow::Error>
where
    LDR: DataReader,
    RDR: DataReader,
    OP: BinaryOperator,
{
    if let (None, None) = (left_shift, right_shift) {
        let mut direct_lifetime_translation_iter =
            direct_lifetime_translation_buffer.translate(output.epoch())?;

        while let Some(translation) = direct_lifetime_translation_iter.next() {
            match translation {
                DirectLifetimeTranslation::Left(lt, rt) => {
                    let right = *right.get(rt);
                    left.iter(lt)
                        .for_each(|left| output.write(OP::apply(*left, right)));
                }
                DirectLifetimeTranslation::Right(lt, rt) => {
                    let left = *left.get(lt);
                    right
                        .iter(rt)
                        .for_each(|right| output.write(OP::apply(left, *right)));
                }
            }
        }
    } else {
        let left_shift = left_shift.unwrap_or_else(|| TimeShift::nop());
        let right_shift = right_shift.unwrap_or_else(|| TimeShift::nop());

        let mut direct_lifetime_translation_iter =
            direct_lifetime_translation_buffer.translate(output.epoch())?;

        while let Some(translation) = direct_lifetime_translation_iter.next() {
            match translation {
                DirectLifetimeTranslation::Left(lt, rt) => {
                    let shifted_lt = TimeShift::shift_range(lt, left_shift);
                    let shifted_rt = right_shift.apply(rt);

                    let right = *right.get(shifted_rt);
                    left.iter(shifted_lt)
                        .for_each(|left| output.write(OP::apply(*left, right)));
                }
                DirectLifetimeTranslation::Right(lt, rt) => {
                    let shifted_lt = left_shift.apply(lt);
                    let shifted_rt = TimeShift::shift_range(rt, right_shift);

                    let left = *left.get(shifted_lt);
                    right
                        .iter(shifted_rt)
                        .for_each(|right| output.write(OP::apply(left, *right)));
                }
            }
        }
    }

    output.finish()
}

fn eval_d2d_waterfall_base_lifetime(
    ctx: &SpanContext,
    cursor_expansion: CursorExpansion,
    operator: BinaryMathOperator,
    left: Lane,
    right: Lane,
    output: &mut SpatialBuffer,
) -> Result<(), anyhow::Error> {
    debug_assert_eq!(ctx.cursor().time_frame(), output.time_frame());
    let output = pipeline_assembly::create_writer(ctx, cursor_expansion, output)?;
    let out_tlb = &*ctx.session().translation_buffer(output.time_frame())?;

    let lt_tlb = &*ctx.session().translation_buffer(left.time_frame())?;
    let left_reader = {
        let lt_read = cursor::convert_source_to_output_epoch(output.epoch(), out_tlb, &lt_tlb)?;
        let lt_read = TimeShift::shift_epoch(&lt_read, left.time_shift())?;
        left.create_reader(&lt_read)
    }?;

    let rt_tlb = &*ctx.session().translation_buffer(right.time_frame())?;
    let right_reader = {
        let rt_read = cursor::convert_source_to_output_epoch(output.epoch(), out_tlb, &rt_tlb)?;
        let rt_read = TimeShift::shift_epoch(&rt_read, right.time_shift())?;
        right.create_reader(&rt_read)
    }?;

    let (lt_tlb, rt_tlb) = if let (Some(lt), Some(rt)) = (lt_tlb.direct(), rt_tlb.direct()) {
        Ok((lt, rt))
    } else {
        Err(anyhow!(ASTError::WideMathExprNodeError(
            "Waterfall base lifetime evaluation requires DirectLifetimeBuffers",
        )))
    }?;

    let waterfall_base_lifetime_translation_buffer =
        WaterfallBaseLifetimeTranslationBuffer::new(ctx.cursor().time_frame(), lt_tlb, rt_tlb);

    match operator {
        BinaryMathOperator::Add => eval_d2d_waterfall_base_lifetime_inlined_reader::<AddOperator>(
            waterfall_base_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
        BinaryMathOperator::Sub => {
            eval_d2d_waterfall_base_lifetime_inlined_reader::<SubtractOperator>(
                waterfall_base_lifetime_translation_buffer,
                left_reader,
                left.time_shift(),
                right_reader,
                right.time_shift(),
                output,
            )
        }
        BinaryMathOperator::Mul => {
            eval_d2d_waterfall_base_lifetime_inlined_reader::<MultiplyOperator>(
                waterfall_base_lifetime_translation_buffer,
                left_reader,
                left.time_shift(),
                right_reader,
                right.time_shift(),
                output,
            )
        }
        BinaryMathOperator::Div => {
            eval_d2d_waterfall_base_lifetime_inlined_reader::<DivideOperator>(
                waterfall_base_lifetime_translation_buffer,
                left_reader,
                left.time_shift(),
                right_reader,
                right.time_shift(),
                output,
            )
        }
        BinaryMathOperator::Pow => eval_d2d_waterfall_base_lifetime_inlined_reader::<PowOperator>(
            waterfall_base_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
        BinaryMathOperator::Max => eval_d2d_waterfall_base_lifetime_inlined_reader::<MaxOperator>(
            waterfall_base_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
        BinaryMathOperator::Min => eval_d2d_waterfall_base_lifetime_inlined_reader::<MinOperator>(
            waterfall_base_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
    }
}

fn eval_d2d_waterfall_base_lifetime_inlined_reader<OP: BinaryOperator>(
    waterfall_base_lifetime_translation_buffer: WaterfallBaseLifetimeTranslationBuffer,
    left: Reader,
    left_shift: Option<TimeShift>,
    right: Reader,
    right_shift: Option<TimeShift>,
    output: SpatialWriter,
) -> Result<(), anyhow::Error> {
    match (left, right) {
        (Reader::SpatialReader(left), Reader::SpatialReader(right)) => {
            eval_d2d_waterfall_base_lifetime_inlined_op::<_, _, OP>(
                waterfall_base_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                output,
            )
        }
        (Reader::SpatialReader(left), Reader::StaticReader(right)) => {
            eval_d2d_waterfall_base_lifetime_inlined_op::<_, _, OP>(
                waterfall_base_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                output,
            )
        }
        (Reader::StaticReader(left), Reader::SpatialReader(right)) => {
            eval_d2d_waterfall_base_lifetime_inlined_op::<_, _, OP>(
                waterfall_base_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                output,
            )
        }
        (Reader::StaticReader(left), Reader::StaticReader(right)) => {
            eval_d2d_waterfall_base_lifetime_inlined_op::<_, _, OP>(
                waterfall_base_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                output,
            )
        }
    }
}

fn eval_d2d_waterfall_base_lifetime_inlined_op<LDR, RDR, OP>(
    waterfall_base_lifetime_translation_buffer: WaterfallBaseLifetimeTranslationBuffer,
    left: LDR,
    left_shift: Option<TimeShift>,
    right: RDR,
    right_shift: Option<TimeShift>,
    mut output: SpatialWriter,
) -> Result<(), anyhow::Error>
where
    LDR: DataReader,
    RDR: DataReader,
    OP: BinaryOperator,
{
    let mut iter = waterfall_base_lifetime_translation_buffer.translate(output.epoch())?;

    if let (None, None) = (left_shift, right_shift) {
        while let Some(translation) = iter.next() {
            let value = OP::apply(*left.get(translation.lt), *right.get(translation.rt));

            for _ in translation.lifetime {
                output.write(value);
            }
        }
    } else {
        let left_shift = left_shift.unwrap_or_else(|| TimeShift::nop());
        let right_shift = right_shift.unwrap_or_else(|| TimeShift::nop());

        while let Some(translation) = iter.next() {
            let shifted_lt = left_shift.apply(translation.lt);
            let shifted_rt = right_shift.apply(translation.rt);

            let value = OP::apply(*left.get(shifted_lt), *right.get(shifted_rt));

            for _ in translation.lifetime {
                output.write(value);
            }
        }
    }

    output.finish()
}

fn eval_d2d_waterfall_lifetime(
    ctx: &SpanContext,
    cursor_expansion: CursorExpansion,
    operator: BinaryMathOperator,
    left: Lane,
    right: Lane,
    output: &mut SpatialBuffer,
) -> Result<(), anyhow::Error> {
    let session = ctx.session();

    let lt_tlb = &*session.translation_buffer(left.time_frame())?;
    let rt_tlb = &*session.translation_buffer(right.time_frame())?;
    let out_tlb = &*session.translation_buffer(output.time_frame())?;

    let output = pipeline_assembly::create_writer(ctx, cursor_expansion, output)?;

    let left_reader = {
        let lt_read = cursor::convert_source_to_output_epoch(output.epoch(), out_tlb, lt_tlb)?;
        let lt_read = TimeShift::shift_epoch(&lt_read, left.time_shift())?;
        left.create_reader(&lt_read)
    }?;

    let right_reader = {
        let rt_read = cursor::convert_source_to_output_epoch(output.epoch(), out_tlb, rt_tlb)?;
        let rt_read = TimeShift::shift_epoch(&rt_read, right.time_shift())?;
        right.create_reader(&rt_read)
    }?;

    let (lt_tlb, rt_tlb, out_tlb) = if let (Some(lt), Some(rt), Some(out)) =
        (lt_tlb.direct(), rt_tlb.direct(), out_tlb.direct())
    {
        Ok((lt, rt, out))
    } else {
        Err(anyhow!(ASTError::WideMathExprNodeError(
            "Waterfall lifetime evaluation requires DirectLifetimeBuffers",
        )))
    }?;

    let waterfall_lifetime_translation_buffer =
        WaterfallLifetimeTranslationBuffer::new(lt_tlb, rt_tlb, out_tlb);

    match operator {
        BinaryMathOperator::Add => eval_d2d_waterfall_lifetime_inlined_reader::<AddOperator>(
            waterfall_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
        BinaryMathOperator::Sub => eval_d2d_waterfall_lifetime_inlined_reader::<SubtractOperator>(
            waterfall_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
        BinaryMathOperator::Mul => eval_d2d_waterfall_lifetime_inlined_reader::<MultiplyOperator>(
            waterfall_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
        BinaryMathOperator::Div => eval_d2d_waterfall_lifetime_inlined_reader::<DivideOperator>(
            waterfall_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
        BinaryMathOperator::Pow => eval_d2d_waterfall_lifetime_inlined_reader::<PowOperator>(
            waterfall_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
        BinaryMathOperator::Max => eval_d2d_waterfall_lifetime_inlined_reader::<MaxOperator>(
            waterfall_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
        BinaryMathOperator::Min => eval_d2d_waterfall_lifetime_inlined_reader::<MinOperator>(
            waterfall_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            output,
        ),
    }
}

fn eval_d2d_waterfall_lifetime_inlined_reader<OP: BinaryOperator>(
    waterfall_lifetime_translation_buffer: WaterfallLifetimeTranslationBuffer,
    left: Reader,
    left_shift: Option<TimeShift>,
    right: Reader,
    right_shift: Option<TimeShift>,
    output: SpatialWriter,
) -> Result<(), anyhow::Error> {
    match (left, right) {
        (Reader::SpatialReader(left), Reader::SpatialReader(right)) => {
            eval_d2d_waterfall_lifetime_inlined_op::<_, _, OP>(
                waterfall_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                output,
            )
        }
        (Reader::SpatialReader(left), Reader::StaticReader(right)) => {
            eval_d2d_waterfall_lifetime_inlined_op::<_, _, OP>(
                waterfall_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                output,
            )
        }
        (Reader::StaticReader(left), Reader::SpatialReader(right)) => {
            eval_d2d_waterfall_lifetime_inlined_op::<_, _, OP>(
                waterfall_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                output,
            )
        }
        (Reader::StaticReader(left), Reader::StaticReader(right)) => {
            eval_d2d_waterfall_lifetime_inlined_op::<_, _, OP>(
                waterfall_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                output,
            )
        }
    }
}

fn eval_d2d_waterfall_lifetime_inlined_op<LDR, RDR, OP>(
    waterfall_lifetime_translation_buffer: WaterfallLifetimeTranslationBuffer,
    left: LDR,
    left_shift: Option<TimeShift>,
    right: RDR,
    right_shift: Option<TimeShift>,
    mut output: SpatialWriter,
) -> Result<(), anyhow::Error>
where
    LDR: DataReader,
    RDR: DataReader,
    OP: BinaryOperator,
{
    let mut iter = waterfall_lifetime_translation_buffer.translate(output.epoch())?;

    if let (None, None) = (left_shift, right_shift) {
        while let Some(translation) = iter.next() {
            let value = OP::apply(*left.get(translation.lt), *right.get(translation.rt));

            for _ in translation.lifetime {
                output.write(value);
            }
        }
    } else {
        let left_shift = left_shift.unwrap_or_else(|| TimeShift::nop());
        let right_shift = right_shift.unwrap_or_else(|| TimeShift::nop());

        while let Some(translation) = iter.next() {
            let shifted_lt = left_shift.apply(translation.lt);
            let shifted_rt = right_shift.apply(translation.rt);

            let value = OP::apply(*left.get(shifted_lt), *right.get(shifted_rt));

            for _ in translation.lifetime {
                output.write(value);
            }
        }
    }

    output.finish()
}

fn eval_d2c(
    ctx: &SpanContext,
    cursor_expansion: CursorExpansion,
    operator: BinaryMathOperator,
    left: Lane,
    right: &Constant,
    output: &mut SpatialBuffer,
) -> Result<(), anyhow::Error> {
    let writer = pipeline_assembly::create_writer(ctx, cursor_expansion, output)?;

    let lt_read = TimeShift::shift_epoch(writer.epoch(), left.time_shift())?;
    let left = left.create_reader(&lt_read)?;

    match operator {
        BinaryMathOperator::Add => {
            eval_d2c_inlined_reader::<AddOperator>(left, lt_read, right, writer)
        }
        BinaryMathOperator::Sub => {
            eval_d2c_inlined_reader::<SubtractOperator>(left, lt_read, right, writer)
        }
        BinaryMathOperator::Mul => {
            eval_d2c_inlined_reader::<MultiplyOperator>(left, lt_read, right, writer)
        }
        BinaryMathOperator::Div => {
            eval_d2c_inlined_reader::<DivideOperator>(left, lt_read, right, writer)
        }
        BinaryMathOperator::Pow => {
            eval_d2c_inlined_reader::<PowOperator>(left, lt_read, right, writer)
        }
        BinaryMathOperator::Max => {
            eval_d2c_inlined_reader::<MaxOperator>(left, lt_read, right, writer)
        }
        BinaryMathOperator::Min => {
            eval_d2c_inlined_reader::<MinOperator>(left, lt_read, right, writer)
        }
    }
}

fn eval_d2c_inlined_reader<OP: BinaryOperator>(
    left: Reader,
    lt_read: Epoch,
    right: &Constant,
    output: SpatialWriter,
) -> Result<(), anyhow::Error> {
    match left {
        Reader::SpatialReader(left) => eval_d2c_inlined_op::<_, OP>(left, lt_read, right, output),
        Reader::StaticReader(left) => eval_d2c_inlined_op::<_, OP>(left, lt_read, right, output),
    }
}

fn eval_d2c_inlined_op<LDR, OP>(
    left: LDR,
    lt_read: Epoch,
    right: &Constant,
    mut output: SpatialWriter,
) -> Result<(), anyhow::Error>
where
    LDR: DataReader,
    OP: BinaryOperator,
{
    match right.value() {
        Scalar::F64(right) => {
            left.iter(lt_read.as_range())
                .for_each(|left| output.write(OP::apply(*left, *right)));
        }
    }

    output.finish()
}

fn eval_c2d(
    ctx: &SpanContext,
    cursor_expansion: CursorExpansion,
    operator: BinaryMathOperator,
    left: &Constant,
    right: Lane,
    output: &mut SpatialBuffer,
) -> Result<(), anyhow::Error> {
    let writer = pipeline_assembly::create_writer(ctx, cursor_expansion, output)?;

    let rt_read = TimeShift::shift_epoch(writer.epoch(), right.time_shift())?;
    let right = right.create_reader(&rt_read)?;

    match operator {
        BinaryMathOperator::Add => {
            eval_c2d_inlined_reader::<AddOperator>(left, right, rt_read, writer)
        }
        BinaryMathOperator::Sub => {
            eval_c2d_inlined_reader::<SubtractOperator>(left, right, rt_read, writer)
        }
        BinaryMathOperator::Mul => {
            eval_c2d_inlined_reader::<MultiplyOperator>(left, right, rt_read, writer)
        }
        BinaryMathOperator::Div => {
            eval_c2d_inlined_reader::<DivideOperator>(left, right, rt_read, writer)
        }
        BinaryMathOperator::Pow => {
            eval_c2d_inlined_reader::<PowOperator>(left, right, rt_read, writer)
        }
        BinaryMathOperator::Max => {
            eval_c2d_inlined_reader::<MaxOperator>(left, right, rt_read, writer)
        }
        BinaryMathOperator::Min => {
            eval_c2d_inlined_reader::<MinOperator>(left, right, rt_read, writer)
        }
    }
}

fn eval_c2d_inlined_reader<OP: BinaryOperator>(
    left: &Constant,
    right: Reader,
    rt_read: Epoch,
    output: SpatialWriter,
) -> Result<(), anyhow::Error> {
    match right {
        Reader::SpatialReader(right) => eval_c2d_inlined_op::<_, OP>(left, right, rt_read, output),
        Reader::StaticReader(right) => eval_c2d_inlined_op::<_, OP>(left, right, rt_read, output),
    }
}

fn eval_c2d_inlined_op<RDR, OP>(
    left: &Constant,
    right: RDR,
    rt_read: Epoch,
    mut output: SpatialWriter,
) -> Result<(), anyhow::Error>
where
    RDR: DataReader,
    OP: BinaryOperator,
{
    match left.value() {
        Scalar::F64(left) => {
            right
                .iter(rt_read.as_range())
                .for_each(|right| output.write(OP::apply(*left, *right)));
        }
    }

    output.finish()
}

impl WideUnaryMathExpr {
    pub fn new(
        expr: Box<ProcessorHandle>,
        operator: UnaryMathOperator,
        output: SpatialBuffer,
    ) -> Self {
        Self {
            expr,
            operator,
            output,
        }
    }
}

impl WideProcessor for WideUnaryMathExpr {
    fn eval(
        &mut self,
        ctx: &SpanContext,
        cursor_expansion: CursorExpansion,
        _: &DynamicStore,
    ) -> Result<(), anyhow::Error> {
        fn apply_unary_operator(
            operator: UnaryMathOperator,
            lane: Lane,
            output: SpatialWriter,
        ) -> Result<(), anyhow::Error> {
            fn apply_inlined_op<DR, OP>(
                lane: DR,
                lane_read: Epoch,
                mut output: SpatialWriter,
            ) -> Result<(), anyhow::Error>
            where
                DR: DataReader,
                OP: UnaryOperator,
            {
                lane.iter(lane_read.as_range())
                    .for_each(|v| output.write(OP::apply(*v)));

                output.finish()
            }

            fn apply_inlined_reader<OP: UnaryOperator>(
                lane: Reader,
                lane_read: Epoch,
                output: SpatialWriter,
            ) -> Result<(), anyhow::Error> {
                match lane {
                    Reader::SpatialReader(input) => {
                        apply_inlined_op::<_, OP>(input, lane_read, output)
                    }
                    Reader::StaticReader(input) => {
                        apply_inlined_op::<_, OP>(input, lane_read, output)
                    }
                }
            }

            let lane_read = TimeShift::shift_epoch(output.epoch(), lane.time_shift)?;
            let lane = lane.create_reader(&lane_read)?;

            match operator {
                UnaryMathOperator::Sqrt => {
                    apply_inlined_reader::<SqrtOperator>(lane, lane_read, output)
                }
                UnaryMathOperator::Abs => {
                    apply_inlined_reader::<AbsOperator>(lane, lane_read, output)
                }
            }
        }

        let _: () = self.expr.eval(
            ctx,
            TimeShift::max_expansion(cursor_expansion, self.expr.time_shift),
        )?;

        let writer = pipeline_assembly::create_writer(ctx, cursor_expansion, &mut self.output)?;

        match self.expr.take()? {
            ReadHandle::LaneProcessor(lane_processor) => apply_unary_operator(
                self.operator,
                Lane::new(lane_processor.try_read()?, self.expr.time_shift),
                writer,
            ),
            ReadHandle::Constant(_) => Err(ASTError::WideMathExprNodeError(
                "Scalar unary math expr should be used for constant input",
            )
            .into()),
        }
    }

    fn output_buffer(&self, output_selector: usize) -> Result<&SpatialBuffer, anyhow::Error> {
        match SingleOutputSelector::from_ordinal(output_selector)? {
            SingleOutputSelector => Ok(&self.output),
        }
    }

    fn epoch(&self) -> &Epoch {
        self.output.epoch()
    }
}

impl MathExprNode {
    pub fn eval(
        &mut self,
        ctx: &SpanContext,
        cursor_expansion: CursorExpansion,
    ) -> Result<(), anyhow::Error> {
        match self {
            MathExprNode::Unary(unary) => unary.eval(ctx, cursor_expansion),
            MathExprNode::Binary(binary) => binary.eval(ctx, cursor_expansion),
        }
    }

    pub fn output(&self, output_selector: usize) -> Result<&Constant, anyhow::Error> {
        debug_assert_eq!(
            0, output_selector,
            "Output selector for math expr should be equal to 0"
        );

        match self {
            MathExprNode::Unary(unary) => unary.output(),
            MathExprNode::Binary(binary) => binary.output(),
        }
    }
}

// TODO - we should not cache, but rather fold constant expression,
//        we can have problems if constant is not stable like clock
impl UnaryMathExpr {
    pub fn new(expr: Box<ProcessorHandle>, operator: UnaryMathOperator) -> Self {
        Self {
            expr,
            operator,
            output: None,
        }
    }

    pub fn eval(
        &mut self,
        ctx: &SpanContext,
        cursor_expansion: CursorExpansion,
    ) -> Result<(), anyhow::Error> {
        if let Some(_) = &self.output {
            Ok(())
        } else {
            let _: () = self.expr.eval(ctx, cursor_expansion)?;

            match self.expr.take()? {
                ReadHandle::Constant(constant) => {
                    self.output = Some(UnaryMathExpr::apply_operator(
                        self.operator,
                        constant.try_read()?,
                    ));
                    Ok(())
                }
                _ => Err(ASTError::MathExprNodeError(
                    "Scalar unary math expr cannot process any other input than constant",
                )
                .into()),
            }
        }
    }

    pub fn output(&self) -> Result<&Constant, anyhow::Error> {
        self.output.as_ref().ok_or_else(|| {
            ExecutionError::NotInitializedError("Scalar unary math expr was not evaluated").into()
        })
    }

    pub fn apply_operator(operator: UnaryMathOperator, constant: &Constant) -> Constant {
        let scalar = match constant.value() {
            Scalar::F64(constant) => match operator {
                UnaryMathOperator::Sqrt => Scalar::F64(SqrtOperator::apply(*constant)),
                UnaryMathOperator::Abs => Scalar::F64(AbsOperator::apply(*constant)),
            },
        };

        Constant::new(scalar)
    }
}

// TODO - we should not cache, but rather fold constant expression,
//        we can have problems if constant is not stable like clock
impl BinaryMathExpr {
    pub fn new(
        left: Box<ProcessorHandle>,
        operator: BinaryMathOperator,
        right: Box<ProcessorHandle>,
    ) -> Self {
        Self {
            left,
            operator,
            right,
            output: None,
        }
    }
    pub fn eval(
        &mut self,
        ctx: &SpanContext,
        cursor_expansion: CursorExpansion,
    ) -> Result<(), anyhow::Error> {
        if let Some(_) = &self.output {
            Ok(())
        } else {
            let _: () = self.left.eval(ctx, cursor_expansion)?;
            let _: () = self.right.eval(ctx, cursor_expansion)?;

            match (self.left.take()?, self.right.take()?) {
                (ReadHandle::Constant(left), ReadHandle::Constant(right)) => {
                    let left = left.try_read()?;
                    let right = right.try_read()?;

                    self.output = Some(BinaryMathExpr::apply_operator(self.operator, left, right));
                    Ok(())
                }
                _ => Err(ASTError::MathExprNodeError(
                    "Scalar binary math expr cannot process any other input than constant",
                )
                .into()),
            }
        }
    }

    pub fn output(&self) -> Result<&Constant, anyhow::Error> {
        self.output.as_ref().ok_or_else(|| {
            ExecutionError::NotInitializedError("Scalar binary math expr was not evaluated").into()
        })
    }

    pub fn apply_operator(
        operator: BinaryMathOperator,
        left: &Constant,
        right: &Constant,
    ) -> Constant {
        let scalar = match (left.value(), right.value()) {
            (Scalar::F64(a), Scalar::F64(b)) => match operator {
                BinaryMathOperator::Add => Scalar::F64(AddOperator::apply(*a, *b)),
                BinaryMathOperator::Sub => Scalar::F64(SubtractOperator::apply(*a, *b)),
                BinaryMathOperator::Mul => Scalar::F64(MultiplyOperator::apply(*a, *b)),
                BinaryMathOperator::Div => Scalar::F64(DivideOperator::apply(*a, *b)),
                BinaryMathOperator::Pow => Scalar::F64(PowOperator::apply(*a, *b)),
                BinaryMathOperator::Max => Scalar::F64(MaxOperator::apply(*a, *b)),
                BinaryMathOperator::Min => Scalar::F64(MinOperator::apply(*a, *b)),
            },
        };

        Constant::new(scalar)
    }
}
