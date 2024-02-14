/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::cursor;
use crate::sim::bb::cursor::{Cursor, CursorSnapshot, Epoch};
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::collections::bitmap::Bitmap;
use crate::sim::context::SpanContext;
use crate::sim::error::ContractError;
use crate::sim::tlb::{FrameTranslationBuffer, InlinedReverseTranslation};
use anyhow::anyhow;
use smallvec::{smallvec, SmallVec};
use std::ops::Range;

///
/// Mediator describes true bits for which condition was fulfilled.
///
/// Rules:
/// * If mediator epochs matches and timeframe is larger than base quotes time frame,
///   then intersection and union should be performed directly in upper time frames
///   preserving superset offset range.
/// * If mediator epoch is larger than base quotes timeframe then negate should be performed
///   within 0..epoch.span() range
/// * if either mediators epochs doesn't match or one of them is base mediator, then we have to
///   expand upper time frame mediator to base mediator and perform operations
///   within cursor offset range. Still we have to preserve superset offset range.
///
pub struct Mediator {
    bitmap: Bitmap,
    cursor_snapshot: CursorSnapshot,
    epoch: Epoch,
}

impl Mediator {
    pub fn new(bitmap: Bitmap, cursor_snapshot: CursorSnapshot, epoch: Epoch) -> Self {
        Self {
            bitmap,
            cursor_snapshot,
            epoch,
        }
    }
    pub fn time_frame(&self) -> TimeFrame {
        self.epoch.time_frame()
    }
}

impl Mediator {
    pub fn intersect(
        ctx: &SpanContext,
        this: Mediator,
        other: Mediator,
    ) -> Result<Mediator, anyhow::Error> {
        let cursor = ctx.cursor();
        debug_assert_mediator(ctx, smallvec![&this, &other]);

        if this.time_frame() != cursor.time_frame() && this.epoch == other.epoch {
            let (mut superset, subset) = sort_desc_by_superset(this, other);

            debug_assert_words(&superset, &subset);
            let operation_range = 0..superset.epoch.span();
            superset.bitmap.iand_range(&subset.bitmap, operation_range);

            Ok(superset)
        } else {
            let (mut superset, subset) =
                sort_desc_by_superset(expand_mediator(ctx, this)?, expand_mediator(ctx, other)?);

            debug_assert_mediator(ctx, smallvec![&superset, &subset]);

            superset
                .bitmap
                .iand_range(&subset.bitmap, subset.cursor_snapshot.offset_range());

            Ok(superset)
        }
    }

    pub fn union(
        ctx: &SpanContext,
        this: Mediator,
        other: Mediator,
    ) -> Result<Mediator, anyhow::Error> {
        let cursor = ctx.cursor();
        debug_assert_mediator(ctx, smallvec![&this, &other]);

        if this.time_frame() != cursor.time_frame() && this.epoch == other.epoch {
            let (mut superset, subset) = sort_desc_by_superset(this, other);

            debug_assert_words(&superset, &subset);
            let operation_range = 0..superset.epoch.span();
            superset.bitmap.ior_range(&subset.bitmap, operation_range);

            Ok(superset)
        } else {
            let (mut superset, subset) =
                sort_desc_by_superset(expand_mediator(ctx, this)?, expand_mediator(ctx, other)?);

            debug_assert_mediator(ctx, smallvec![&superset, &subset]);

            superset
                .bitmap
                .ior_range(&subset.bitmap, subset.cursor_snapshot.offset_range());

            Ok(superset)
        }
    }

    pub fn negate(ctx: &SpanContext, mut this: Mediator) -> Result<Mediator, anyhow::Error> {
        let cursor = ctx.cursor();
        debug_assert_mediator(ctx, smallvec![&this]);

        if this.time_frame() != cursor.time_frame() {
            let operation_range = 0..this.epoch.span();
            this.bitmap.inot_range(operation_range);
            Ok(this)
        } else {
            let mut base_mediator = expand_mediator(ctx, this)?;
            debug_assert_mediator(ctx, smallvec![&base_mediator]);

            base_mediator
                .bitmap
                .inot_range(base_mediator.cursor_snapshot.offset_range());

            Ok(base_mediator)
        }
    }
}

fn expand_mediator(ctx: &SpanContext, mediator: Mediator) -> Result<Mediator, anyhow::Error> {
    debug_assert_mediator(ctx, smallvec![&mediator]);

    debug_assert!(
        ctx.cursor().time_frame() <= mediator.time_frame(),
        "Mediator time_frame({:?}) must be greater or equal than base time_frame({:?})",
        mediator.time_frame(),
        ctx.cursor().time_frame()
    );

    if mediator.time_frame() == ctx.cursor().time_frame() {
        Ok(mediator)
    } else {
        match &*ctx.session().translation_buffer(mediator.time_frame())? {
            FrameTranslationBuffer::DirectTranslationBuffer(dtb) => {
                let epoch_start = mediator.epoch.start();

                let cursor_snapshot = &mediator.cursor_snapshot;
                let base_ptr = cursor_snapshot.base_ptr();
                let step = cursor_snapshot.step();
                let cursor_start = cursor_snapshot.start();
                let cursor_end = cursor_snapshot.end();

                let mut accumulator = Bitmap::with_capacity(step);
                let mut run_iter = mediator.bitmap.iter_run();

                while let Some(run) = run_iter.next() {
                    let frame_start = epoch_start + run.position as usize;
                    let offset_start =
                        dtb.reverse_validity_lifetime(frame_start).max(cursor_start) - base_ptr;

                    let frame_end = frame_start + run.length as usize;
                    let offset_end =
                        dtb.reverse_validity_lifetime(frame_end).min(cursor_end) - base_ptr;

                    accumulator.insert_range(offset_start..offset_end);
                }

                Ok(Mediator {
                    bitmap: accumulator,
                    epoch: Epoch::new(base_ptr..(base_ptr + step), cursor_snapshot.time_frame()),
                    cursor_snapshot: mediator.cursor_snapshot,
                })
            }
            FrameTranslationBuffer::IdentityTranslationBuffer(_) => {
                Err(anyhow!(ContractError::MediatorError(
                    format!("Identity TLB cannot be expanded to base quotes as it should represent them already,\
                             mediator time_frame({:?}), base time_frame({:?})",
                    mediator.time_frame(),
                    ctx.cursor().time_frame()
                ).into())))
            }
        }
    }
}

fn sort_desc_by_superset(
    left_mediator: Mediator,
    right_mediator: Mediator,
) -> (Mediator, Mediator) {
    let left = left_mediator.cursor_snapshot.offset_range();
    let right = right_mediator.cursor_snapshot.offset_range();

    let start = left.start.max(right.start);
    let end = left.end.min(right.end);
    let intersection = start..end;

    match intersection {
        r if r == left => (right_mediator, left_mediator),
        r if r == right => (left_mediator, right_mediator),
        _ => {
            panic!(
                "Neither of mediators is a superset of the other, offset ranges: {{left: {:?}, right: {:?}}}",
                left, right
            );
        }
    }
}

fn debug_assert_mediator(ctx: &SpanContext, mediators: SmallVec<[&Mediator; 2]>) {
    #[cfg(debug_assertions)]
    {
        for mediator in mediators {
            debug_assert_cursor_id(ctx.cursor(), mediator);
            debug_assert_mediator_epoch(ctx, mediator);
        }
    }
}

fn debug_assert_cursor_id(cursor: &Cursor, mediator: &Mediator) {
    debug_assert_eq!(
        cursor.id(),
        mediator.cursor_snapshot.id(),
        "Mediator cursor doesn't match SpanContext cursor, cannot expand mediator"
    );
}

///
/// Mediators should be aligned within their time frames and contain results from offset range.
///
/// In short:
/// * Base mediator: [base_ptr .. base_ptr + step]
/// * Upper mediator: [ftb(base_ptr) .. ftb(base_ptr + step - 1) + 1]
///
fn debug_assert_mediator_epoch(ctx: &SpanContext, mediator: &Mediator) {
    #[cfg(debug_assertions)]
    {
        let cursor = ctx.cursor();
        if cursor.time_frame() == mediator.time_frame() {
            let start = cursor.base_ptr();
            let end = cursor.base_ptr() + cursor.step();
            debug_assert_eq!(
                start..end,
                mediator.epoch.as_range(),
                "Base Mediator must be aligned to [base_ptr .. base_ptr + step]"
            );
        } else {
            let ftb = ctx
                .session()
                .translation_buffer(mediator.time_frame())
                .expect("Mediator: expects that ftb exists");

            let start = ftb.translate(cursor.base_ptr());
            let end = ftb.translate(cursor.base_ptr() + cursor.step() - 1) + 1;
            debug_assert_eq!(
                start..end,
                mediator.epoch.as_range(),
                "Mediator must be aligned [ftb(base_ptr) .. ftb(base_ptr + step - 1) + 1]"
            );
        }
    }
}

fn debug_assert_words(left: &Mediator, right: &Mediator) {
    debug_assert_eq!(
        left.bitmap.words().len(),
        right.bitmap.words().len(),
        "Mediators with equal epochs should have bitmaps with equal number of words"
    );
}

#[cfg(test)]
mod tests {
    use crate::sim::bb::cursor::{Cursor, Epoch};
    use crate::sim::bb::quotes::{BaseQuotes, FrameQuotes, TimestampVector};
    use crate::sim::bb::time_frame;
    use crate::sim::bb::time_frame::TimeFrame;
    use crate::sim::builder::frame_builder::build_frame_to_frame;
    use crate::sim::builder::translation_builder::build_translation_buffer;
    use crate::sim::collections::bitmap::Bitmap;
    use crate::sim::context::{SessionConfiguration, SessionContext, SpanContext};
    use crate::sim::mediator::{debug_assert_mediator_epoch, expand_mediator, Mediator};
    use crate::sim::tlb::{FrameTranslationBuffer, InlinedReverseTranslation};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn should_negate_mediator_by_epoch() {
        let cursor = Cursor::new(0, 22, 40, time_frame::SECOND_1);
        let ctx = prepare_span_context(&cursor);

        let mediator = Mediator::new(
            Bitmap::from_bits(vec![3, 4, 6, 9]),
            cursor.snapshot(),
            Epoch::new(4..13, time_frame::SECOND_5),
        );

        let result = Mediator::negate(&ctx, mediator).unwrap();

        assert_eq!(7, result.bitmap.cardinality());
        assert!(result.bitmap.contains(0));
        assert!(result.bitmap.contains(1));
        assert!(result.bitmap.contains(2));
        assert!(result.bitmap.contains(5));
        assert!(result.bitmap.contains(7));
        assert!(result.bitmap.contains(8));
    }

    #[test]
    fn should_negate_base_mediator_by_offset_range() {
        let cursor = Cursor::new(0, 50, 128, time_frame::SECOND_1);
        let ctx = prepare_span_context(&cursor);

        let mediator = Mediator::new(
            Bitmap::from_bits(vec![10, 13, 15, 18, 19]),
            cursor.shrink_offset_range(10..20).snapshot(),
            Epoch::new(50..178, time_frame::SECOND_1),
        );

        let result = Mediator::negate(&ctx, mediator).unwrap();

        assert_eq!(5, result.bitmap.cardinality());
        assert!(result.bitmap.contains(11));
        assert!(result.bitmap.contains(12));
        assert!(result.bitmap.contains(14));
        assert!(result.bitmap.contains(16));
        assert!(result.bitmap.contains(17));
        assert_eq!(Epoch::new(50..178, time_frame::SECOND_1), result.epoch);
    }

    #[test]
    fn should_intersect_mediator_by_epoch() {
        let cursor = Cursor::new(0, 20, 60, time_frame::SECOND_1);
        let ctx = prepare_span_context(&cursor);

        let this = Mediator::new(
            Bitmap::from_bits(vec![0, 3, 4, 6, 8]),
            cursor.snapshot(),
            Epoch::new(4..16, time_frame::SECOND_5),
        );

        let other = Mediator::new(
            Bitmap::from_bits(vec![2, 3, 4, 6, 7]),
            cursor.snapshot(),
            Epoch::new(4..16, time_frame::SECOND_5),
        );

        let result = Mediator::intersect(&ctx, this, other).unwrap();

        assert_eq!(3, result.bitmap.cardinality());
        assert!(result.bitmap.contains(3));
        assert!(result.bitmap.contains(4));
        assert!(result.bitmap.contains(6));
        assert_eq!(Epoch::new(4..16, time_frame::SECOND_5), result.epoch);
    }

    #[test]
    fn should_intersect_mediator_by_subset_of_offset_range() {
        let cursor = Cursor::new(0, 0, 128, time_frame::SECOND_1);
        let ctx = prepare_span_context(&cursor);

        let this = Mediator::new(
            Bitmap::from_bits(vec![40, 80, 100, 122, 124, 126]),
            cursor.shrink_offset_range(20..128).snapshot(),
            Epoch::new(0..128, time_frame::SECOND_1),
        );

        let other = Mediator::new(
            Bitmap::from_bits(vec![122, 124, 127]),
            cursor.shrink_offset_range(120..128).snapshot(),
            Epoch::new(0..128, time_frame::SECOND_1),
        );

        let result = Mediator::intersect(&ctx, this, other).unwrap();

        assert_eq!(5, result.bitmap.cardinality());
        assert!(result.bitmap.contains(40));
        assert!(result.bitmap.contains(80));
        assert!(result.bitmap.contains(100));
        assert!(result.bitmap.contains(122));
        assert!(result.bitmap.contains(124));
    }

    #[test]
    fn should_yield_superset_from_intersection() {
        let cursor = Cursor::new(0, 0, 1000, time_frame::SECOND_1);
        let ctx = prepare_span_context(&cursor);

        let this = Mediator::new(
            Bitmap::with_capacity(256),
            cursor.snapshot(),
            Epoch::new(0..200, time_frame::SECOND_5),
        );

        let other = Mediator::new(
            Bitmap::with_capacity(256),
            cursor.shrink_offset_range(200..800).snapshot(),
            Epoch::new(0..200, time_frame::SECOND_5),
        );

        let result = Mediator::intersect(&ctx, this, other).unwrap();

        assert_eq!(cursor.snapshot(), result.cursor_snapshot);
    }

    #[test]
    fn should_union_mediator_by_epoch() {
        let cursor = Cursor::new(0, 0, 100, time_frame::SECOND_1);
        let ctx = prepare_span_context(&cursor);

        let this = Mediator::new(
            Bitmap::from_bits(vec![0, 3, 4, 6, 8]),
            cursor.snapshot(),
            Epoch::new(0..20, time_frame::SECOND_5),
        );

        let other = Mediator::new(
            Bitmap::from_bits(vec![2, 3, 4, 6, 7]),
            cursor.snapshot(),
            Epoch::new(0..20, time_frame::SECOND_5),
        );

        let result = Mediator::union(&ctx, this, other).unwrap();

        assert_eq!(7, result.bitmap.cardinality());
        assert!(result.bitmap.contains(0));
        assert!(result.bitmap.contains(2));
        assert!(result.bitmap.contains(3));
        assert!(result.bitmap.contains(4));
        assert!(result.bitmap.contains(6));
        assert!(result.bitmap.contains(7));
        assert!(result.bitmap.contains(8));
        assert_eq!(Epoch::new(0..20, time_frame::SECOND_5), result.epoch);
    }

    #[test]
    fn should_yield_superset_from_union() {
        let cursor = Cursor::new(0, 0, 1000, time_frame::SECOND_1);
        let ctx = prepare_span_context(&cursor);

        let this = Mediator::new(
            Bitmap::with_capacity(256),
            cursor.snapshot(),
            Epoch::new(0..200, time_frame::SECOND_5),
        );

        let other = Mediator::new(
            Bitmap::with_capacity(256),
            cursor.shrink_offset_range(200..800).snapshot(),
            Epoch::new(0..200, time_frame::SECOND_5),
        );

        let result = Mediator::union(&ctx, this, other).unwrap();

        assert_eq!(cursor.snapshot(), result.cursor_snapshot);
    }

    #[test]
    fn should_union_mediator_by_subset_of_offset_range() {
        let cursor = Cursor::new(0, 0, 128, time_frame::SECOND_1);
        let ctx = prepare_span_context(&cursor);

        let this = Mediator::new(
            Bitmap::from_bits(vec![40, 80, 100, 122, 124, 126]),
            cursor.shrink_offset_range(20..128).snapshot(),
            Epoch::new(0..128, time_frame::SECOND_1),
        );

        let other = Mediator::new(
            Bitmap::from_bits(vec![122, 124, 127]),
            cursor.shrink_offset_range(120..128).snapshot(),
            Epoch::new(0..128, time_frame::SECOND_1),
        );

        let result = Mediator::union(&ctx, this, other).unwrap();

        assert_eq!(7, result.bitmap.cardinality());
        assert!(result.bitmap.contains(40));
        assert!(result.bitmap.contains(80));
        assert!(result.bitmap.contains(100));
        assert!(result.bitmap.contains(122));
        assert!(result.bitmap.contains(124));
        assert!(result.bitmap.contains(126));
        assert!(result.bitmap.contains(127));
    }

    #[test]
    fn should_expand_mediator() {
        let cursor = Cursor::new(0, 20, 200, time_frame::SECOND_1);
        let ctx = prepare_span_context(&cursor);

        let this = Mediator::new(
            Bitmap::from_bits(vec![7, 10, 11, 12, 25, 32]),
            cursor.shrink_offset_range(37..162).snapshot(),
            Epoch::new(4..44, time_frame::SECOND_5),
        );

        let result = expand_mediator(&ctx, this).unwrap();

        assert_eq!(0, result.bitmap.cardinality_in_range(35..37));
        assert_eq!(3, result.bitmap.cardinality_in_range(37..40));
        assert_eq!(0, result.bitmap.cardinality_in_range(40..50));
        assert_eq!(15, result.bitmap.cardinality_in_range(50..65));
        assert_eq!(0, result.bitmap.cardinality_in_range(65..125));
        assert_eq!(5, result.bitmap.cardinality_in_range(125..130));
        assert_eq!(0, result.bitmap.cardinality_in_range(130..160));
        assert_eq!(2, result.bitmap.cardinality_in_range(160..162));
        assert_eq!(0, result.bitmap.cardinality_in_range(162..200));
        assert_eq!(25, result.bitmap.cardinality());
    }

    #[test]
    fn should_intersect_expanded_mediators_by_cursor_offsets() {
        let cursor = Cursor::new(0, 20, 200, time_frame::SECOND_1);
        let ctx = prepare_span_context(&cursor);

        let this = Mediator::new(
            Bitmap::from_bits(vec![8, 9, 16, 22, 23, 27, 28, 39]),
            cursor.shrink_offset_range(40..200).snapshot(),
            Epoch::new(4..44, time_frame::SECOND_5),
        );

        let other = Mediator::new(
            Bitmap::from_bits(vec![5, 7, 8, 11, 13]),
            cursor.shrink_offset_range(80..200).snapshot(),
            Epoch::new(1..15, time_frame::SECOND_15),
        );

        let mediator = Mediator::intersect(&ctx, this, other).unwrap();

        assert_eq!(0, mediator.bitmap.cardinality_in_range(0..40));
        assert_eq!(10, mediator.bitmap.cardinality_in_range(40..50));
        assert_eq!(0, mediator.bitmap.cardinality_in_range(50..80));
        assert_eq!(5, mediator.bitmap.cardinality_in_range(80..85));
        assert_eq!(0, mediator.bitmap.cardinality_in_range(85..110));
        assert_eq!(10, mediator.bitmap.cardinality_in_range(110..120));
        assert_eq!(0, mediator.bitmap.cardinality_in_range(120..195));
        assert_eq!(5, mediator.bitmap.cardinality_in_range(195..200));
        assert_eq!(30, mediator.bitmap.cardinality());
        assert_eq!(
            cursor.shrink_offset_range(40..200).snapshot(),
            mediator.cursor_snapshot
        );
        assert_eq!(Epoch::new(20..220, time_frame::SECOND_1), mediator.epoch);
    }

    #[test]
    fn should_union_expanded_mediators_by_cursor_offsets() {
        let cursor = Cursor::new(0, 20, 200, time_frame::SECOND_1);
        let ctx = prepare_span_context(&cursor);

        let this = Mediator::new(
            Bitmap::from_bits(vec![10, 11, 19, 38]),
            cursor.shrink_offset_range(60..200).snapshot(),
            Epoch::new(4..44, time_frame::SECOND_5),
        );

        let other = Mediator::new(
            Bitmap::from_bits(vec![0, 3, 4, 7, 13]),
            cursor.snapshot(),
            Epoch::new(1..15, time_frame::SECOND_15),
        );

        let mediator = Mediator::union(&ctx, this, other).unwrap();

        assert_eq!(10, mediator.bitmap.cardinality_in_range(0..10));
        assert_eq!(0, mediator.bitmap.cardinality_in_range(10..40));
        assert_eq!(30, mediator.bitmap.cardinality_in_range(40..70));
        assert_eq!(0, mediator.bitmap.cardinality_in_range(70..95));
        assert_eq!(20, mediator.bitmap.cardinality_in_range(95..115));
        assert_eq!(0, mediator.bitmap.cardinality_in_range(115..190));
        assert_eq!(10, mediator.bitmap.cardinality_in_range(190..200));
        assert_eq!(70, mediator.bitmap.cardinality());

        assert_eq!(cursor.snapshot(), mediator.cursor_snapshot);
        assert_eq!(Epoch::new(20..220, time_frame::SECOND_1), mediator.epoch);
    }

    fn prepare_span_context(cursor: &Cursor) -> SpanContext {
        let base_quotes = Arc::new(BaseQuotes::new(Arc::new(create_frame_quotes(
            time_frame::SECOND_1,
            1000,
        ))));

        let arc_base_quotes = Arc::clone(&**base_quotes);

        let session_context = Arc::new(SessionContext::new(
            String::from("GOLD"),
            Arc::clone(&base_quotes),
            BTreeMap::from([
                (time_frame::SECOND_1, arc_base_quotes),
                (
                    time_frame::SECOND_5,
                    Arc::new(
                        build_frame_to_frame(&base_quotes, time_frame::SECOND_5)
                            .expect("frame quotes created"),
                    ),
                ),
                (
                    time_frame::SECOND_15,
                    Arc::new(
                        build_frame_to_frame(&base_quotes, time_frame::SECOND_15)
                            .expect("frame quotes created"),
                    ),
                ),
            ]),
            BTreeMap::from([
                (
                    base_quotes.time_frame(),
                    Arc::new(
                        build_translation_buffer(&base_quotes, base_quotes.time_frame()).unwrap(),
                    ),
                ),
                (
                    time_frame::SECOND_5,
                    Arc::new(build_translation_buffer(&base_quotes, time_frame::SECOND_5).unwrap()),
                ),
                (
                    time_frame::SECOND_15,
                    Arc::new(
                        build_translation_buffer(&base_quotes, time_frame::SECOND_15).unwrap(),
                    ),
                ),
            ]),
            SessionConfiguration::new(cursor.step(), cursor.step().next_power_of_two()),
        ));

        SpanContext::new(cursor.clone(), session_context)
    }

    fn create_frame_quotes(time_frame: TimeFrame, samples: usize) -> FrameQuotes {
        FrameQuotes::new(
            vec![0f64; samples],
            vec![0f64; samples],
            vec![0f64; samples],
            vec![0f64; samples],
            TimestampVector::from_utc(
                (0..samples)
                    .map(|i| i as u64 * u64::from(*time_frame))
                    .collect(),
            ),
            time_frame,
        )
    }
}
