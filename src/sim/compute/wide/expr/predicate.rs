/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::cursor;
use crate::sim::bb::cursor::{CursorExpansion, Epoch, TimeShift};
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::collections::bitmap::Bitmap;
use crate::sim::collections::iter64::Iter64;
use crate::sim::compute::operators::ordering::{
    Equal, Greater, GreaterOrEqual, Lower, LowerOrEqual, NotEqual, OrderingOperator,
};
use crate::sim::compute::operators::{BinaryMathOperator, Ordering};
use crate::sim::compute::translation_strategy::{
    TranslationStrategy, TranslationStrategyDescriptor,
};
use crate::sim::compute::wide::expr::{
    Constant, Lane, PredicateNode, ProcessorHandle, ReadHandle, Scalar,
};
use crate::sim::context::SpanContext;
use crate::sim::error::ASTError;
use crate::sim::mediator::Mediator;
use crate::sim::reader::{DataReader, Reader, ReaderFactory};
use crate::sim::spatial_buffer::{SpatialBuffer, SpatialReader, SpatialWriter};
use crate::sim::tlb::{
    DirectBaseLifetimeTranslation, DirectBaseLifetimeTranslationBuffer, DirectLifetimeTranslation,
    DirectLifetimeTranslationBuffer, DirectLifetimeTranslationIter, FrameTranslationBuffer,
    InlinedTranslation, WaterfallBaseLifetimeTranslationBuffer, WaterfallLifetimeTranslationBuffer,
    WaterfallLifetimeTranslationIter,
};
use anyhow::anyhow;

impl PredicateNode {
    pub fn new(
        left: Box<ProcessorHandle>,
        ordering: Ordering,
        right: Box<ProcessorHandle>,
        strategy: TranslationStrategy,
    ) -> PredicateNode {
        Self {
            left,
            ordering,
            right,
            strategy,
        }
    }

    pub fn eval(
        &self,
        ctx: &SpanContext,
        cursor_expansion: CursorExpansion,
    ) -> Result<Mediator, anyhow::Error> {
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

                match &self.strategy {
                    TranslationStrategy::Direct(_) => {
                        eval_d2d_direct(ctx, self.ordering, left, right)
                    }
                    TranslationStrategy::DirectBaseLifetime(descriptor) => {
                        eval_d2d_direct_base_lifetime(ctx, self.ordering, left, right, descriptor)
                    }
                    TranslationStrategy::DirectLifetime(descriptor) => {
                        eval_d2d_direct_lifetime(ctx, self.ordering, left, right, descriptor)
                    }
                    TranslationStrategy::WaterfallBaseLifetime(descriptor) => {
                        eval_d2d_waterfall_base_lifetime(
                            ctx,
                            self.ordering,
                            left,
                            right,
                            descriptor,
                        )
                    }
                    TranslationStrategy::WaterfallLifetime(descriptor) => {
                        eval_d2d_waterfall_lifetime(ctx, self.ordering, left, right, descriptor)
                    }
                }
            }
            (ReadHandle::LaneProcessor(left), ReadHandle::Constant(right)) => eval_d2c(
                ctx,
                self.ordering,
                Lane::new(left.try_read()?, self.left.time_shift),
                right.try_read()?,
            ),
            (ReadHandle::Constant(left), ReadHandle::LaneProcessor(right)) => eval_c2d(
                ctx,
                self.ordering,
                left.try_read()?,
                Lane::new(right.try_read()?, self.right.time_shift),
            ),
            (ReadHandle::Constant(left), ReadHandle::Constant(right)) => {
                eval_c2c(ctx, self.ordering, left.try_read()?, right.try_read()?)
            }
        }
    }
}

fn eval_d2d_direct(
    ctx: &SpanContext,
    ordering: Ordering,
    left: Lane,
    right: Lane,
) -> Result<Mediator, anyhow::Error> {
    let ftb = ctx.session().translation_buffer(left.time_frame())?;
    let base_indices = cursor::cursor_to_base_frame_indices(ctx.cursor(), &ftb);
    let offset_indices = cursor::cursor_to_offset_frame_indices(ctx.cursor(), &ftb);

    let left_read = TimeShift::shift_epoch(&offset_indices, left.time_shift())?;
    let left = left.create_reader(&left_read)?;

    let right_read = TimeShift::shift_epoch(&offset_indices, right.time_shift())?;
    let right = right.create_reader(&right_read)?;

    let mediator = match ordering {
        Ordering::Equal => eval_d2d_direct_inlined_reader::<Equal>(
            ctx,
            left,
            left_read,
            right,
            right_read,
            base_indices,
            offset_indices,
        ),
        Ordering::NotEqual => eval_d2d_direct_inlined_reader::<NotEqual>(
            ctx,
            left,
            left_read,
            right,
            right_read,
            base_indices,
            offset_indices,
        ),
        Ordering::Greater => eval_d2d_direct_inlined_reader::<Greater>(
            ctx,
            left,
            left_read,
            right,
            right_read,
            base_indices,
            offset_indices,
        ),
        Ordering::Lower => eval_d2d_direct_inlined_reader::<Lower>(
            ctx,
            left,
            left_read,
            right,
            right_read,
            base_indices,
            offset_indices,
        ),
        Ordering::GreaterOrEqual => eval_d2d_direct_inlined_reader::<GreaterOrEqual>(
            ctx,
            left,
            left_read,
            right,
            right_read,
            base_indices,
            offset_indices,
        ),
        Ordering::LowerOrEqual => eval_d2d_direct_inlined_reader::<LowerOrEqual>(
            ctx,
            left,
            left_read,
            right,
            right_read,
            base_indices,
            offset_indices,
        ),
    };

    Ok(mediator)
}

fn eval_d2d_direct_inlined_reader<CMP>(
    ctx: &SpanContext,
    left: Reader,
    left_read: Epoch,
    right: Reader,
    right_read: Epoch,
    base_indices: Epoch,
    offset_indices: Epoch,
) -> Mediator
where
    CMP: OrderingOperator,
{
    match (left, right) {
        (Reader::SpatialReader(left), Reader::SpatialReader(right)) => {
            eval_d2d_direct_inlined_op::<_, _, CMP>(
                ctx,
                left,
                left_read,
                right,
                right_read,
                base_indices,
                offset_indices,
            )
        }
        (Reader::SpatialReader(left), Reader::StaticReader(right)) => {
            eval_d2d_direct_inlined_op::<_, _, CMP>(
                ctx,
                left,
                left_read,
                right,
                right_read,
                base_indices,
                offset_indices,
            )
        }
        (Reader::StaticReader(left), Reader::SpatialReader(right)) => {
            eval_d2d_direct_inlined_op::<_, _, CMP>(
                ctx,
                left,
                left_read,
                right,
                right_read,
                base_indices,
                offset_indices,
            )
        }
        (Reader::StaticReader(left), Reader::StaticReader(right)) => {
            eval_d2d_direct_inlined_op::<_, _, CMP>(
                ctx,
                left,
                left_read,
                right,
                right_read,
                base_indices,
                offset_indices,
            )
        }
    }
}

fn eval_d2d_direct_inlined_op<LDR, RDR, CMP>(
    ctx: &SpanContext,
    left: LDR,
    left_read: Epoch,
    right: RDR,
    right_read: Epoch,
    base_indices: Epoch,
    offset_indices: Epoch,
) -> Mediator
where
    LDR: DataReader,
    RDR: DataReader,
    CMP: OrderingOperator,
{
    let base_offset = offset_indices.start() - base_indices.start();
    let cursor = ctx.cursor();
    let mut bitmap = Bitmap::with_capacity(cursor.step());

    if base_offset == 0 {
        left.iter(left_read.as_range())
            .zip(right.iter(right_read.as_range()))
            .enumerate()
            .for_each(|(position, (l, r))| unsafe {
                bitmap.insert_binary_unsafe(position, CMP::compare(*l, *r));
            });
    } else {
        let position = base_offset;
        left.iter(left_read.as_range())
            .zip(right.iter(right_read.as_range()))
            .enumerate()
            .for_each(|(offset, (l, r))| unsafe {
                bitmap.insert_binary_unsafe(position + offset, CMP::compare(*l, *r));
            });
    }

    Mediator::new(bitmap, cursor.snapshot(), base_indices)
}

fn eval_d2d_direct_base_lifetime(
    ctx: &SpanContext,
    ordering: Ordering,
    left: Lane,
    right: Lane,
    descriptor: &TranslationStrategyDescriptor,
) -> Result<Mediator, anyhow::Error> {
    let out_tlb = &*ctx.session().translation_buffer(descriptor.output())?;

    let base_indices = cursor::cursor_to_base_frame_indices(ctx.cursor(), out_tlb);
    let offset_indices = cursor::cursor_to_offset_frame_indices(ctx.cursor(), out_tlb);

    let lt_tlb = &*ctx.session().translation_buffer(left.time_frame())?;
    let left_reader = {
        let lt_read = cursor::convert_source_to_output_epoch(&offset_indices, out_tlb, &lt_tlb)?;
        let lt_read = TimeShift::shift_epoch(&lt_read, left.time_shift())?;
        left.create_reader(&lt_read)
    }?;

    let rt_tlb = &*ctx.session().translation_buffer(right.time_frame())?;
    let right_reader = {
        let rt_read = cursor::convert_source_to_output_epoch(&offset_indices, out_tlb, &rt_tlb)?;
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

    match ordering {
        Ordering::Equal => eval_d2d_direct_base_lifetime_inlined_reader::<Equal>(
            ctx,
            direct_base_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
        Ordering::NotEqual => eval_d2d_direct_base_lifetime_inlined_reader::<NotEqual>(
            ctx,
            direct_base_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
        Ordering::Greater => eval_d2d_direct_base_lifetime_inlined_reader::<Greater>(
            ctx,
            direct_base_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
        Ordering::Lower => eval_d2d_direct_base_lifetime_inlined_reader::<Lower>(
            ctx,
            direct_base_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
        Ordering::GreaterOrEqual => eval_d2d_direct_base_lifetime_inlined_reader::<GreaterOrEqual>(
            ctx,
            direct_base_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
        Ordering::LowerOrEqual => eval_d2d_direct_base_lifetime_inlined_reader::<LowerOrEqual>(
            ctx,
            direct_base_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
    }
}

fn eval_d2d_direct_base_lifetime_inlined_reader<CMP>(
    ctx: &SpanContext,
    direct_base_lifetime_translation_buffer: DirectBaseLifetimeTranslationBuffer,
    left: Reader,
    left_shift: Option<TimeShift>,
    right: Reader,
    right_shift: Option<TimeShift>,
    base_indices: Epoch,
    offset_indices: Epoch,
) -> Result<Mediator, anyhow::Error>
where
    CMP: OrderingOperator,
{
    match (left, right) {
        (Reader::SpatialReader(left), Reader::SpatialReader(right)) => {
            eval_d2d_direct_base_lifetime_op::<_, _, CMP>(
                ctx,
                direct_base_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                base_indices,
                offset_indices,
            )
        }
        (Reader::SpatialReader(left), Reader::StaticReader(right)) => {
            eval_d2d_direct_base_lifetime_op::<_, _, CMP>(
                ctx,
                direct_base_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                base_indices,
                offset_indices,
            )
        }
        (Reader::StaticReader(left), Reader::SpatialReader(right)) => {
            eval_d2d_direct_base_lifetime_op::<_, _, CMP>(
                ctx,
                direct_base_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                base_indices,
                offset_indices,
            )
        }
        (Reader::StaticReader(left), Reader::StaticReader(right)) => {
            eval_d2d_direct_base_lifetime_op::<_, _, CMP>(
                ctx,
                direct_base_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                base_indices,
                offset_indices,
            )
        }
    }
}

fn eval_d2d_direct_base_lifetime_op<LDR, RDR, CMP>(
    ctx: &SpanContext,
    direct_base_lifetime_translation_buffer: DirectBaseLifetimeTranslationBuffer,
    left: LDR,
    left_shift: Option<TimeShift>,
    right: RDR,
    right_shift: Option<TimeShift>,
    base_indices: Epoch,
    offset_indices: Epoch,
) -> Result<Mediator, anyhow::Error>
where
    LDR: DataReader,
    RDR: DataReader,
    CMP: OrderingOperator,
{
    let base_offset = offset_indices.start() - base_indices.start();
    let cursor = ctx.cursor();
    let mut bitmap = Bitmap::with_capacity(cursor.step());

    let mut iter = direct_base_lifetime_translation_buffer.translate(&offset_indices)?;
    let mut position = base_offset;

    if let (None, None) = (left_shift, right_shift) {
        while let Some(translation) = iter.next() {
            match translation {
                DirectBaseLifetimeTranslation::Left(lt, rt) => {
                    let span = lt.end - lt.start;

                    let right = *right.get(rt);
                    left.iter(lt).enumerate().for_each(|(offset, left)| unsafe {
                        bitmap.insert_binary_unsafe(position + offset, CMP::compare(*left, right));
                    });

                    position += span;
                }
                DirectBaseLifetimeTranslation::Right(lt, rt) => {
                    let span = rt.end - rt.start;

                    let left = *left.get(lt);
                    right
                        .iter(rt)
                        .enumerate()
                        .for_each(|(offset, right)| unsafe {
                            bitmap.insert_binary_unsafe(
                                position + offset,
                                CMP::compare(left, *right),
                            );
                        });

                    position += span;
                }
            }
        }
    } else {
        let left_shift = left_shift.unwrap_or_else(|| TimeShift::nop());
        let right_shift = right_shift.unwrap_or_else(|| TimeShift::nop());

        while let Some(translation) = iter.next() {
            match translation {
                DirectBaseLifetimeTranslation::Left(lt, rt) => {
                    let span = lt.end - lt.start;

                    let shifted_lt = TimeShift::shift_range(lt, left_shift);
                    let shifted_rt = right_shift.apply(rt);

                    let right = *right.get(shifted_rt);
                    left.iter(shifted_lt)
                        .enumerate()
                        .for_each(|(offset, left)| unsafe {
                            bitmap.insert_binary_unsafe(
                                position + offset,
                                CMP::compare(*left, right),
                            );
                        });

                    position += span;
                }
                DirectBaseLifetimeTranslation::Right(lt, rt) => {
                    let span = rt.end - rt.start;

                    let shifted_lt = left_shift.apply(lt);
                    let shifted_rt = TimeShift::shift_range(rt, right_shift);

                    let left = *left.get(shifted_lt);
                    right
                        .iter(shifted_rt)
                        .enumerate()
                        .for_each(|(offset, right)| unsafe {
                            bitmap.insert_binary_unsafe(
                                position + offset,
                                CMP::compare(left, *right),
                            );
                        });

                    position += span;
                }
            }
        }
    }

    Ok(Mediator::new(bitmap, cursor.snapshot(), base_indices))
}

fn eval_d2d_direct_lifetime(
    ctx: &SpanContext,
    ordering: Ordering,
    left: Lane,
    right: Lane,
    descriptor: &TranslationStrategyDescriptor,
) -> Result<Mediator, anyhow::Error> {
    let session = ctx.session();

    let lt_tlb = &*session.translation_buffer(left.time_frame())?;
    let rt_tlb = &*session.translation_buffer(right.time_frame())?;
    let out_tlb = &*session.translation_buffer(descriptor.output())?;

    let base_indices = cursor::cursor_to_base_frame_indices(ctx.cursor(), out_tlb);
    let offset_indices = cursor::cursor_to_offset_frame_indices(ctx.cursor(), out_tlb);

    let left_reader = {
        let lt_read = cursor::convert_source_to_output_epoch(&offset_indices, out_tlb, lt_tlb)?;
        let lt_read = TimeShift::shift_epoch(&lt_read, left.time_shift())?;
        left.create_reader(&lt_read)
    }?;

    let right_reader = {
        let rt_read = cursor::convert_source_to_output_epoch(&offset_indices, out_tlb, rt_tlb)?;
        let rt_read = TimeShift::shift_epoch(&rt_read, right.time_shift())?;
        right.create_reader(&rt_read)
    }?;

    let (lt_tlb, rt_tlb) = if let (Some(lt), Some(rt)) = (lt_tlb.direct(), rt_tlb.direct()) {
        Ok((lt, rt))
    } else {
        Err(anyhow!(ASTError::PredicateExprNodeError(
            "Direct lifetime evaluation requires DirectLifetimeBuffers",
        )))
    }?;

    let direct_lifetime_translation_buffer = DirectLifetimeTranslationBuffer::new(lt_tlb, rt_tlb);

    match ordering {
        Ordering::Equal => eval_d2d_direct_lifetime_inlined_reader::<Equal>(
            ctx,
            direct_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
        Ordering::NotEqual => eval_d2d_direct_lifetime_inlined_reader::<NotEqual>(
            ctx,
            direct_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
        Ordering::Greater => eval_d2d_direct_lifetime_inlined_reader::<Greater>(
            ctx,
            direct_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
        Ordering::Lower => eval_d2d_direct_lifetime_inlined_reader::<Lower>(
            ctx,
            direct_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
        Ordering::GreaterOrEqual => eval_d2d_direct_lifetime_inlined_reader::<GreaterOrEqual>(
            ctx,
            direct_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
        Ordering::LowerOrEqual => eval_d2d_direct_lifetime_inlined_reader::<LowerOrEqual>(
            ctx,
            direct_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
    }
}

fn eval_d2d_direct_lifetime_inlined_reader<CMP>(
    ctx: &SpanContext,
    direct_lifetime_translation_buffer: DirectLifetimeTranslationBuffer,
    left: Reader,
    left_shift: Option<TimeShift>,
    right: Reader,
    right_shift: Option<TimeShift>,
    base_indices: Epoch,
    offset_indices: Epoch,
) -> Result<Mediator, anyhow::Error>
where
    CMP: OrderingOperator,
{
    match (left, right) {
        (Reader::SpatialReader(left), Reader::SpatialReader(right)) => {
            eval_d2d_direct_lifetime_op::<_, _, CMP>(
                ctx,
                direct_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                base_indices,
                offset_indices,
            )
        }
        (Reader::SpatialReader(left), Reader::StaticReader(right)) => {
            eval_d2d_direct_lifetime_op::<_, _, CMP>(
                ctx,
                direct_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                base_indices,
                offset_indices,
            )
        }
        (Reader::StaticReader(left), Reader::SpatialReader(right)) => {
            eval_d2d_direct_lifetime_op::<_, _, CMP>(
                ctx,
                direct_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                base_indices,
                offset_indices,
            )
        }
        (Reader::StaticReader(left), Reader::StaticReader(right)) => {
            eval_d2d_direct_lifetime_op::<_, _, CMP>(
                ctx,
                direct_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                base_indices,
                offset_indices,
            )
        }
    }
}

fn eval_d2d_direct_lifetime_op<LDR, RDR, CMP>(
    ctx: &SpanContext,
    direct_lifetime_translation_buffer: DirectLifetimeTranslationBuffer,
    left: LDR,
    left_shift: Option<TimeShift>,
    right: RDR,
    right_shift: Option<TimeShift>,
    base_indices: Epoch,
    offset_indices: Epoch,
) -> Result<Mediator, anyhow::Error>
where
    LDR: DataReader,
    RDR: DataReader,
    CMP: OrderingOperator,
{
    let base_offset = offset_indices.start() - base_indices.start();
    let cursor = ctx.cursor();
    let mut bitmap = Bitmap::with_capacity(cursor.step());

    let mut iter = direct_lifetime_translation_buffer.translate(&offset_indices)?;
    let mut position = base_offset;

    if let (None, None) = (left_shift, right_shift) {
        while let Some(translation) = iter.next() {
            match translation {
                DirectLifetimeTranslation::Left(lt, rt) => {
                    let span = lt.end - lt.start;

                    let right = *right.get(rt);
                    left.iter(lt).enumerate().for_each(|(offset, left)| unsafe {
                        bitmap.insert_binary_unsafe(position + offset, CMP::compare(*left, right));
                    });

                    position += span;
                }
                DirectLifetimeTranslation::Right(lt, rt) => {
                    let span = rt.end - rt.start;

                    let left = *left.get(lt);
                    right
                        .iter(rt)
                        .enumerate()
                        .for_each(|(offset, right)| unsafe {
                            bitmap.insert_binary_unsafe(
                                position + offset,
                                CMP::compare(left, *right),
                            );
                        });

                    position += span;
                }
            }
        }
    } else {
        let left_shift = left_shift.unwrap_or_else(|| TimeShift::nop());
        let right_shift = right_shift.unwrap_or_else(|| TimeShift::nop());

        while let Some(translation) = iter.next() {
            match translation {
                DirectLifetimeTranslation::Left(lt, rt) => {
                    let span = lt.end - lt.start;

                    let shifted_lt = TimeShift::shift_range(lt, left_shift);
                    let shifted_rt = right_shift.apply(rt);

                    let right = *right.get(shifted_rt);
                    left.iter(shifted_lt)
                        .enumerate()
                        .for_each(|(offset, left)| unsafe {
                            bitmap.insert_binary_unsafe(
                                position + offset,
                                CMP::compare(*left, right),
                            );
                        });

                    position += span;
                }
                DirectLifetimeTranslation::Right(lt, rt) => {
                    let span = rt.end - rt.start;

                    let shifted_lt = left_shift.apply(lt);
                    let shifted_rt = TimeShift::shift_range(rt, right_shift);

                    let left = *left.get(shifted_lt);
                    right
                        .iter(shifted_rt)
                        .enumerate()
                        .for_each(|(offset, right)| unsafe {
                            bitmap.insert_binary_unsafe(
                                position + offset,
                                CMP::compare(left, *right),
                            );
                        });

                    position += span;
                }
            }
        }
    }

    Ok(Mediator::new(bitmap, cursor.snapshot(), base_indices))
}

fn eval_d2d_waterfall_base_lifetime(
    ctx: &SpanContext,
    ordering: Ordering,
    left: Lane,
    right: Lane,
    descriptor: &TranslationStrategyDescriptor,
) -> Result<Mediator, anyhow::Error> {
    let out_tlb = &*ctx.session().translation_buffer(descriptor.output())?;

    let base_indices = cursor::cursor_to_base_frame_indices(ctx.cursor(), out_tlb);
    let offset_indices = cursor::cursor_to_offset_frame_indices(ctx.cursor(), out_tlb);

    let lt_tlb = &*ctx.session().translation_buffer(left.time_frame())?;
    let left_reader = {
        let lt_read = cursor::convert_source_to_output_epoch(&offset_indices, out_tlb, &lt_tlb)?;
        let lt_read = TimeShift::shift_epoch(&lt_read, left.time_shift())?;
        left.create_reader(&lt_read)
    }?;

    let rt_tlb = &*ctx.session().translation_buffer(right.time_frame())?;
    let right_reader = {
        let rt_read = cursor::convert_source_to_output_epoch(&offset_indices, out_tlb, &rt_tlb)?;
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

    match ordering {
        Ordering::Equal => eval_d2d_waterfall_base_lifetime_inlined_reader::<Equal>(
            ctx,
            waterfall_base_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
        Ordering::NotEqual => eval_d2d_waterfall_base_lifetime_inlined_reader::<NotEqual>(
            ctx,
            waterfall_base_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
        Ordering::Greater => eval_d2d_waterfall_base_lifetime_inlined_reader::<Greater>(
            ctx,
            waterfall_base_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
        Ordering::Lower => eval_d2d_waterfall_base_lifetime_inlined_reader::<Lower>(
            ctx,
            waterfall_base_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
        Ordering::GreaterOrEqual => {
            eval_d2d_waterfall_base_lifetime_inlined_reader::<GreaterOrEqual>(
                ctx,
                waterfall_base_lifetime_translation_buffer,
                left_reader,
                left.time_shift(),
                right_reader,
                right.time_shift(),
                base_indices,
                offset_indices,
            )
        }
        Ordering::LowerOrEqual => eval_d2d_waterfall_base_lifetime_inlined_reader::<LowerOrEqual>(
            ctx,
            waterfall_base_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
    }
}

fn eval_d2d_waterfall_base_lifetime_inlined_reader<CMP>(
    ctx: &SpanContext,
    waterfall_base_lifetime_translation_buffer: WaterfallBaseLifetimeTranslationBuffer,
    left: Reader,
    left_shift: Option<TimeShift>,
    right: Reader,
    right_shift: Option<TimeShift>,
    base_indices: Epoch,
    offset_indices: Epoch,
) -> Result<Mediator, anyhow::Error>
where
    CMP: OrderingOperator,
{
    match (left, right) {
        (Reader::SpatialReader(left), Reader::SpatialReader(right)) => {
            eval_d2d_waterfall_base_lifetime_op::<_, _, CMP>(
                ctx,
                waterfall_base_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                base_indices,
                offset_indices,
            )
        }
        (Reader::SpatialReader(left), Reader::StaticReader(right)) => {
            eval_d2d_waterfall_base_lifetime_op::<_, _, CMP>(
                ctx,
                waterfall_base_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                base_indices,
                offset_indices,
            )
        }
        (Reader::StaticReader(left), Reader::SpatialReader(right)) => {
            eval_d2d_waterfall_base_lifetime_op::<_, _, CMP>(
                ctx,
                waterfall_base_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                base_indices,
                offset_indices,
            )
        }
        (Reader::StaticReader(left), Reader::StaticReader(right)) => {
            eval_d2d_waterfall_base_lifetime_op::<_, _, CMP>(
                ctx,
                waterfall_base_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                base_indices,
                offset_indices,
            )
        }
    }
}

fn eval_d2d_waterfall_base_lifetime_op<LDR, RDR, CMP>(
    ctx: &SpanContext,
    waterfall_base_lifetime_translation_buffer: WaterfallBaseLifetimeTranslationBuffer,
    left: LDR,
    left_shift: Option<TimeShift>,
    right: RDR,
    right_shift: Option<TimeShift>,
    base_indices: Epoch,
    offset_indices: Epoch,
) -> Result<Mediator, anyhow::Error>
where
    LDR: DataReader,
    RDR: DataReader,
    CMP: OrderingOperator,
{
    let base_offset = offset_indices.start() - base_indices.start();
    let cursor = ctx.cursor();
    let mut bitmap = Bitmap::with_capacity(cursor.step());

    let mut iter = waterfall_base_lifetime_translation_buffer.translate(&offset_indices)?;
    let mut position = base_offset;

    if let (None, None) = (left_shift, right_shift) {
        while let Some(translation) = iter.next() {
            let span = translation.lifetime.end - translation.lifetime.start;
            let satisfied = CMP::compare(*left.get(translation.lt), *right.get(translation.rt));

            unsafe {
                insert_lifetime(&mut bitmap, position, span, satisfied);
            }

            position += span;
        }
    } else {
        let left_shift = left_shift.unwrap_or_else(|| TimeShift::nop());
        let right_shift = right_shift.unwrap_or_else(|| TimeShift::nop());

        while let Some(translation) = iter.next() {
            let span = translation.lifetime.end - translation.lifetime.start;

            let shifted_lt = left_shift.apply(translation.lt);
            let shifted_rt = right_shift.apply(translation.rt);
            let satisfied = CMP::compare(*left.get(shifted_lt), *right.get(shifted_rt));

            unsafe {
                insert_lifetime(&mut bitmap, position, span, satisfied);
            }

            position += span;
        }
    }

    Ok(Mediator::new(bitmap, cursor.snapshot(), base_indices))
}

fn eval_d2d_waterfall_lifetime(
    ctx: &SpanContext,
    ordering: Ordering,
    left: Lane,
    right: Lane,
    descriptor: &TranslationStrategyDescriptor,
) -> Result<Mediator, anyhow::Error> {
    let session = ctx.session();

    let lt_tlb = &*session.translation_buffer(left.time_frame())?;
    let rt_tlb = &*session.translation_buffer(right.time_frame())?;
    let out_tlb = &*session.translation_buffer(descriptor.output())?;

    let base_indices = cursor::cursor_to_base_frame_indices(ctx.cursor(), out_tlb);
    let offset_indices = cursor::cursor_to_offset_frame_indices(ctx.cursor(), out_tlb);

    let left_reader = {
        let lt_read = cursor::convert_source_to_output_epoch(&offset_indices, out_tlb, lt_tlb)?;
        let lt_read = TimeShift::shift_epoch(&lt_read, left.time_shift())?;
        left.create_reader(&lt_read)
    }?;

    let right_reader = {
        let rt_read = cursor::convert_source_to_output_epoch(&offset_indices, out_tlb, rt_tlb)?;
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

    match ordering {
        Ordering::Equal => eval_d2d_waterfall_lifetime_inlined_reader::<Equal>(
            ctx,
            waterfall_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
        Ordering::NotEqual => eval_d2d_waterfall_lifetime_inlined_reader::<NotEqual>(
            ctx,
            waterfall_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
        Ordering::Greater => eval_d2d_waterfall_lifetime_inlined_reader::<Greater>(
            ctx,
            waterfall_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
        Ordering::Lower => eval_d2d_waterfall_lifetime_inlined_reader::<Lower>(
            ctx,
            waterfall_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
        Ordering::GreaterOrEqual => eval_d2d_waterfall_lifetime_inlined_reader::<GreaterOrEqual>(
            ctx,
            waterfall_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
        Ordering::LowerOrEqual => eval_d2d_waterfall_lifetime_inlined_reader::<LowerOrEqual>(
            ctx,
            waterfall_lifetime_translation_buffer,
            left_reader,
            left.time_shift(),
            right_reader,
            right.time_shift(),
            base_indices,
            offset_indices,
        ),
    }
}

fn eval_d2d_waterfall_lifetime_inlined_reader<CMP>(
    ctx: &SpanContext,
    waterfall_lifetime_translation_buffer: WaterfallLifetimeTranslationBuffer,
    left: Reader,
    left_shift: Option<TimeShift>,
    right: Reader,
    right_shift: Option<TimeShift>,
    base_indices: Epoch,
    offset_indices: Epoch,
) -> Result<Mediator, anyhow::Error>
where
    CMP: OrderingOperator,
{
    match (left, right) {
        (Reader::SpatialReader(left), Reader::SpatialReader(right)) => {
            eval_d2d_waterfall_lifetime_op::<_, _, CMP>(
                ctx,
                waterfall_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                base_indices,
                offset_indices,
            )
        }
        (Reader::SpatialReader(left), Reader::StaticReader(right)) => {
            eval_d2d_waterfall_lifetime_op::<_, _, CMP>(
                ctx,
                waterfall_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                base_indices,
                offset_indices,
            )
        }
        (Reader::StaticReader(left), Reader::SpatialReader(right)) => {
            eval_d2d_waterfall_lifetime_op::<_, _, CMP>(
                ctx,
                waterfall_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                base_indices,
                offset_indices,
            )
        }
        (Reader::StaticReader(left), Reader::StaticReader(right)) => {
            eval_d2d_waterfall_lifetime_op::<_, _, CMP>(
                ctx,
                waterfall_lifetime_translation_buffer,
                left,
                left_shift,
                right,
                right_shift,
                base_indices,
                offset_indices,
            )
        }
    }
}

fn eval_d2d_waterfall_lifetime_op<LDR, RDR, CMP>(
    ctx: &SpanContext,
    waterfall_lifetime_translation_buffer: WaterfallLifetimeTranslationBuffer,
    left: LDR,
    left_shift: Option<TimeShift>,
    right: RDR,
    right_shift: Option<TimeShift>,
    base_indices: Epoch,
    offset_indices: Epoch,
) -> Result<Mediator, anyhow::Error>
where
    LDR: DataReader,
    RDR: DataReader,
    CMP: OrderingOperator,
{
    let base_offset = offset_indices.start() - base_indices.start();
    let cursor = ctx.cursor();
    let mut bitmap = Bitmap::with_capacity(cursor.step());

    let mut iter = waterfall_lifetime_translation_buffer.translate(&offset_indices)?;
    let mut position = base_offset;

    if let (None, None) = (left_shift, right_shift) {
        while let Some(translation) = iter.next() {
            let span = translation.lifetime.end - translation.lifetime.start;
            let satisfied = CMP::compare(*left.get(translation.lt), *right.get(translation.rt));

            unsafe {
                insert_lifetime(&mut bitmap, position, span, satisfied);
            }

            position += span;
        }
    } else {
        let left_shift = left_shift.unwrap_or_else(|| TimeShift::nop());
        let right_shift = right_shift.unwrap_or_else(|| TimeShift::nop());

        while let Some(translation) = iter.next() {
            let span = translation.lifetime.end - translation.lifetime.start;

            let shifted_lt = left_shift.apply(translation.lt);
            let shifted_rt = right_shift.apply(translation.rt);
            let satisfied = CMP::compare(*left.get(shifted_lt), *right.get(shifted_rt));

            unsafe {
                insert_lifetime(&mut bitmap, position, span, satisfied);
            }

            position += span;
        }
    }

    Ok(Mediator::new(bitmap, cursor.snapshot(), base_indices))
}

///
/// The goal is to separate two branches always-taken from rarely-taken, so we could achieve higher
/// branch prediction.
///
#[inline(always)]
unsafe fn insert_lifetime(bitmap: &mut Bitmap, position: usize, span: usize, satisfied: bool) {
    let start = position;
    let end = position + span;

    // always taken branch that has been taken out of the loop
    let first_slice = Iter64::slice64_from(start, end);
    bitmap.insert_binary_unsafe_64(first_slice.start, first_slice.len(), satisfied);

    // rarely taken branch
    let mut iter64 = Iter64::new(first_slice.end, end);

    while let Some(next_slice) = iter64.next() {
        bitmap.insert_binary_unsafe_64(next_slice.start, next_slice.len(), satisfied);
    }
}

fn eval_d2c(
    ctx: &SpanContext,
    ordering: Ordering,
    left: Lane,
    right: &Constant,
) -> Result<Mediator, anyhow::Error> {
    let ftb = ctx.session().translation_buffer(left.time_frame())?;
    let base_indices = cursor::cursor_to_base_frame_indices(ctx.cursor(), &ftb);
    let offset_indices = cursor::cursor_to_offset_frame_indices(ctx.cursor(), &ftb);

    let lt_read = TimeShift::shift_epoch(&offset_indices, left.time_shift())?;
    let left = left.create_reader(&lt_read)?;

    let right = {
        match right.value() {
            Scalar::F64(f64) => *f64,
        }
    };

    let mediator = match ordering {
        Ordering::Equal => eval_d2c_inlined_reader::<Equal>(
            ctx,
            left,
            lt_read,
            right,
            base_indices,
            offset_indices,
        ),
        Ordering::NotEqual => eval_d2c_inlined_reader::<NotEqual>(
            ctx,
            left,
            lt_read,
            right,
            base_indices,
            offset_indices,
        ),
        Ordering::Greater => eval_d2c_inlined_reader::<Greater>(
            ctx,
            left,
            lt_read,
            right,
            base_indices,
            offset_indices,
        ),
        Ordering::Lower => eval_d2c_inlined_reader::<Lower>(
            ctx,
            left,
            lt_read,
            right,
            base_indices,
            offset_indices,
        ),
        Ordering::GreaterOrEqual => eval_d2c_inlined_reader::<GreaterOrEqual>(
            ctx,
            left,
            lt_read,
            right,
            base_indices,
            offset_indices,
        ),
        Ordering::LowerOrEqual => eval_d2c_inlined_reader::<LowerOrEqual>(
            ctx,
            left,
            lt_read,
            right,
            base_indices,
            offset_indices,
        ),
    };

    Ok(mediator)
}

fn eval_d2c_inlined_reader<CMP>(
    ctx: &SpanContext,
    left: Reader,
    lt_read: Epoch,
    right: f64,
    base_indices: Epoch,
    offset_indices: Epoch,
) -> Mediator
where
    CMP: OrderingOperator,
{
    match left {
        Reader::SpatialReader(left) => {
            eval_d2c_inlined_op::<_, CMP>(ctx, left, lt_read, right, base_indices, offset_indices)
        }
        Reader::StaticReader(left) => {
            eval_d2c_inlined_op::<_, CMP>(ctx, left, lt_read, right, base_indices, offset_indices)
        }
    }
}

fn eval_d2c_inlined_op<LDR, CMP>(
    ctx: &SpanContext,
    left: LDR,
    lt_read: Epoch,
    right: f64,
    base_indices: Epoch,
    offset_indices: Epoch,
) -> Mediator
where
    LDR: DataReader,
    CMP: OrderingOperator,
{
    let base_offset = offset_indices.start() - base_indices.start();
    let cursor = ctx.cursor();
    let mut bitmap = Bitmap::with_capacity(cursor.step());

    if base_offset == 0 {
        left.iter(lt_read.as_range())
            .enumerate()
            .for_each(|(i, left)| unsafe {
                bitmap.insert_binary_unsafe(i, CMP::compare(*left, right));
            });
    } else {
        left.iter(lt_read.as_range())
            .enumerate()
            .for_each(|(i, left)| unsafe {
                bitmap.insert_binary_unsafe(i + base_offset, CMP::compare(*left, right));
            });
    }

    Mediator::new(bitmap, cursor.snapshot(), base_indices)
}

fn eval_c2d(
    ctx: &SpanContext,
    ordering: Ordering,
    left: &Constant,
    right: Lane,
) -> Result<Mediator, anyhow::Error> {
    let ftb = ctx.session().translation_buffer(right.time_frame())?;
    let base_indices = cursor::cursor_to_base_frame_indices(ctx.cursor(), &ftb);
    let offset_indices = cursor::cursor_to_offset_frame_indices(ctx.cursor(), &ftb);

    let left = {
        match left.value() {
            Scalar::F64(f64) => *f64,
        }
    };

    let rt_read = TimeShift::shift_epoch(&offset_indices, right.time_shift())?;
    let right = right.create_reader(&rt_read)?;

    let mediator = match ordering {
        Ordering::Equal => eval_c2d_inlined_reader::<Equal>(
            ctx,
            left,
            right,
            rt_read,
            base_indices,
            offset_indices,
        ),
        Ordering::NotEqual => eval_c2d_inlined_reader::<NotEqual>(
            ctx,
            left,
            right,
            rt_read,
            base_indices,
            offset_indices,
        ),
        Ordering::Greater => eval_c2d_inlined_reader::<Greater>(
            ctx,
            left,
            right,
            rt_read,
            base_indices,
            offset_indices,
        ),
        Ordering::Lower => eval_c2d_inlined_reader::<Lower>(
            ctx,
            left,
            right,
            rt_read,
            base_indices,
            offset_indices,
        ),
        Ordering::GreaterOrEqual => eval_c2d_inlined_reader::<GreaterOrEqual>(
            ctx,
            left,
            right,
            rt_read,
            base_indices,
            offset_indices,
        ),
        Ordering::LowerOrEqual => eval_c2d_inlined_reader::<LowerOrEqual>(
            ctx,
            left,
            right,
            rt_read,
            base_indices,
            offset_indices,
        ),
    };

    Ok(mediator)
}

fn eval_c2d_inlined_reader<CMP>(
    ctx: &SpanContext,
    left: f64,
    right: Reader,
    rt_read: Epoch,
    base_indices: Epoch,
    offset_indices: Epoch,
) -> Mediator
where
    CMP: OrderingOperator,
{
    match right {
        Reader::SpatialReader(right) => {
            eval_c2d_inlined_op::<_, CMP>(ctx, left, right, rt_read, base_indices, offset_indices)
        }
        Reader::StaticReader(right) => {
            eval_c2d_inlined_op::<_, CMP>(ctx, left, right, rt_read, base_indices, offset_indices)
        }
    }
}

fn eval_c2d_inlined_op<RDR, CMP>(
    ctx: &SpanContext,
    left: f64,
    right: RDR,
    rt_read: Epoch,
    base_indices: Epoch,
    offset_indices: Epoch,
) -> Mediator
where
    RDR: DataReader,
    CMP: OrderingOperator,
{
    let base_offset = offset_indices.start() - base_indices.start();
    let cursor = ctx.cursor();
    let mut bitmap = Bitmap::with_capacity(cursor.step());

    if base_offset == 0 {
        right
            .iter(rt_read.as_range())
            .enumerate()
            .for_each(|(i, right)| unsafe {
                bitmap.insert_binary_unsafe(i, CMP::compare(left, *right));
            });
    } else {
        right
            .iter(rt_read.as_range())
            .enumerate()
            .for_each(|(i, right)| unsafe {
                bitmap.insert_binary_unsafe(i + base_offset, CMP::compare(left, *right));
            });
    }

    Mediator::new(bitmap, cursor.snapshot(), base_indices)
}

fn eval_c2c(
    ctx: &SpanContext,
    ordering: Ordering,
    left: &Constant,
    right: &Constant,
) -> Result<Mediator, anyhow::Error> {
    match (left.value(), right.value()) {
        (Scalar::F64(left), Scalar::F64(right)) => {
            let satisfied = match ordering {
                Ordering::Equal => Equal::compare(*left, *right),
                Ordering::NotEqual => NotEqual::compare(*left, *right),
                Ordering::Greater => Greater::compare(*left, *right),
                Ordering::Lower => Lower::compare(*left, *right),
                Ordering::GreaterOrEqual => GreaterOrEqual::compare(*left, *right),
                Ordering::LowerOrEqual => LowerOrEqual::compare(*left, *right),
            };

            let cursor = ctx.cursor();
            let mut bitmap = Bitmap::with_capacity(cursor.step());

            if satisfied {
                bitmap.insert_range(cursor.offset_range());
            }

            Ok(Mediator::new(
                bitmap,
                cursor.snapshot(),
                Epoch::new(cursor.base_range(), cursor.time_frame()),
            ))
        }
    }
}
