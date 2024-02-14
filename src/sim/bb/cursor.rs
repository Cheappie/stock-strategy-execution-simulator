/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use std::cmp::Ordering;
use std::ops::{Deref, Range};

use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::context::SpanContext;
use crate::sim::error::{BoundsError, ContractError};
use crate::sim::tlb::{
    DirectTranslationBuffer, FrameTranslationBuffer, InlinedReverseTranslation, InlinedTranslation,
    TranslationUnitDescriptor,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Cursor {
    id: usize,
    base_ptr: usize,
    step: usize,
    offset_range: Range<usize>,
    time_frame: TimeFrame,
}

impl Cursor {
    ///
    /// Optimal step parameter is power of two and greater or equal than 64
    ///
    pub fn new(id: usize, base_ptr: usize, step: usize, time_frame: TimeFrame) -> Self {
        Self {
            id,
            base_ptr,
            step,
            time_frame,
            offset_range: 0..step,
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn base_ptr(&self) -> usize {
        self.base_ptr
    }

    pub fn step(&self) -> usize {
        self.step
    }

    pub fn base_range(&self) -> Range<usize> {
        self.base_ptr..(self.base_ptr + self.step)
    }

    pub fn start(&self) -> usize {
        self.base_ptr + self.offset_range.start
    }

    pub fn end(&self) -> usize {
        self.base_ptr + self.offset_range.end
    }

    pub fn cursor_range(&self) -> Range<usize> {
        self.start()..self.end()
    }

    pub fn offset_range(&self) -> Range<usize> {
        self.offset_range.clone()
    }

    pub fn is_empty(&self) -> bool {
        self.offset_range.is_empty()
    }

    pub fn time_frame(&self) -> TimeFrame {
        self.time_frame
    }

    pub fn snapshot(&self) -> CursorSnapshot {
        CursorSnapshot(self.clone())
    }

    pub fn shrink_offset_range(&self, offset_range_subset: Range<usize>) -> Cursor {
        debug_assert!(
            self.offset_range.start <= offset_range_subset.start
                && offset_range_subset.end <= self.offset_range.end,
            "Provided offset_range_subset({:?}) is not a subset of actual offset_range({:?})",
            offset_range_subset,
            self.offset_range
        );

        Cursor {
            id: self.id,
            base_ptr: self.base_ptr,
            step: self.step,
            offset_range: offset_range_subset,
            time_frame: self.time_frame,
        }
    }
}

#[derive(Eq, PartialEq, Debug)]
pub struct CursorSnapshot(Cursor);

impl Deref for CursorSnapshot {
    type Target = Cursor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Copy, Clone, Debug)]
pub enum CursorExpansion {
    Identity,
    Universe { past: usize, future: usize },
}

impl CursorExpansion {
    pub fn expand_by(&self, time_shift: TimeShift) -> Self {
        match self {
            CursorExpansion::Identity => match time_shift {
                TimeShift::Past(past) => CursorExpansion::Universe { past, future: 0 },
                TimeShift::Future(future) => CursorExpansion::Universe { past: 0, future },
            },
            CursorExpansion::Universe { past, future } => match time_shift {
                TimeShift::Past(n) => CursorExpansion::Universe {
                    past: past + n,
                    future: *future,
                },
                TimeShift::Future(n) => CursorExpansion::Universe {
                    past: *past,
                    future: future + n,
                },
            },
        }
    }

    pub fn max_by(&self, time_shift: TimeShift) -> Self {
        match self {
            CursorExpansion::Identity => self.expand_by(time_shift),
            CursorExpansion::Universe { past, future } => match time_shift {
                TimeShift::Past(n) => CursorExpansion::Universe {
                    past: *past.max(&n),
                    future: *future,
                },
                TimeShift::Future(n) => CursorExpansion::Universe {
                    past: *past,
                    future: *future.max(&n),
                },
            },
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum TimeShift {
    Past(usize),
    Future(usize),
}

impl TimeShift {
    pub fn apply(&self, index: usize) -> usize {
        match self {
            TimeShift::Past(n) => index - n,
            TimeShift::Future(n) => index + n,
        }
    }

    pub fn shift_epoch(
        epoch: &Epoch,
        time_shift: Option<TimeShift>,
    ) -> Result<Epoch, anyhow::Error> {
        if let Some(ts) = time_shift {
            epoch.shift_by(ts)
        } else {
            Ok(epoch.clone())
        }
    }

    pub fn shift_range(range: Range<usize>, time_shift: TimeShift) -> Range<usize> {
        match time_shift {
            TimeShift::Past(past) => (range.start - past)..(range.end - past),
            TimeShift::Future(future) => (range.start + future)..(range.end + future),
        }
    }

    pub fn max_expansion(
        cursor_expansion: CursorExpansion,
        time_shift: Option<TimeShift>,
    ) -> CursorExpansion {
        if let Some(ts) = time_shift {
            cursor_expansion.max_by(ts)
        } else {
            cursor_expansion
        }
    }

    pub fn nop() -> Self {
        TimeShift::Future(0)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Epoch {
    range: Range<usize>,
    time_frame: TimeFrame,
}

impl Epoch {
    pub fn new(range: Range<usize>, time_frame: TimeFrame) -> Epoch {
        Self { range, time_frame }
    }

    pub fn empty(time_frame: TimeFrame) -> Self {
        Epoch::new(0..0, time_frame)
    }

    pub fn start(&self) -> usize {
        self.range.start
    }

    pub fn end(&self) -> usize {
        self.range.end
    }

    pub fn time_frame(&self) -> TimeFrame {
        self.time_frame
    }

    pub fn span(&self) -> usize {
        self.range.end.saturating_sub(self.range.start)
    }

    pub fn is_superset(lhs: &Epoch, rhs: &Epoch) -> Result<bool, anyhow::Error> {
        if lhs.time_frame() == rhs.time_frame() {
            Ok(lhs.start() <= rhs.start() && rhs.end() <= lhs.end())
        } else {
            Err(ContractError::TimeFrameMismatchError(
                format!(
                    "Cannot assess superset due to time frame mismatch, lhs({:?}), rhs({:?})",
                    lhs, rhs
                )
                .into(),
            )
            .into())
        }
    }

    pub fn contains(&self, index: usize) -> bool {
        self.range.contains(&index)
    }

    pub fn as_range(&self) -> Range<usize> {
        self.range.clone()
    }

    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    pub fn shift_by(&self, time_shift: TimeShift) -> Result<Epoch, anyhow::Error> {
        match time_shift {
            TimeShift::Past(past) => {
                let tail = self
                    .range
                    .start
                    .checked_sub(past)
                    .ok_or_else(|| BoundsError::LookBackOutOfBounds)?;
                let head = self
                    .range
                    .end
                    .checked_sub(past)
                    .ok_or_else(|| BoundsError::LookBackOutOfBounds)?;
                Ok(Epoch::new(tail..head, self.time_frame))
            }
            TimeShift::Future(future) => {
                let tail = self.range.start + future;
                let head = self.range.end + future;
                Ok(Epoch::new(tail..head, self.time_frame))
            }
        }
    }
}

pub fn cursor_to_frame_indices(
    cursor: &Cursor,
    cursor_expansion: CursorExpansion,
    translation_buffer: &FrameTranslationBuffer,
) -> Result<Epoch, anyhow::Error> {
    let tail = translation_buffer.translate(cursor.start());

    let inclusive = translation_buffer.translate(cursor.end() - 1);
    let head = inclusive + 1;

    apply_cursor_expansion(
        tail..head,
        cursor_expansion,
        translation_buffer.time_frame(),
    )
}

fn apply_cursor_expansion(
    range: Range<usize>,
    cursor_expansion: CursorExpansion,
    time_frame: TimeFrame,
) -> Result<Epoch, anyhow::Error> {
    match cursor_expansion {
        CursorExpansion::Identity => Ok(Epoch::new(range, time_frame)),
        CursorExpansion::Universe { past, future } => {
            let tail = range
                .start
                .checked_sub(past)
                .ok_or_else(|| BoundsError::CursorHasBeenExpandedBeyondBounds)?;

            let head = range
                .end
                .checked_add(future)
                .ok_or_else(|| BoundsError::CursorHasBeenExpandedBeyondBounds)?;

            Ok(Epoch::new(tail..head, time_frame))
        }
    }
}

pub fn epoch_look_back(epoch: &Epoch, look_back_period: usize) -> Result<Epoch, anyhow::Error> {
    let historical_start = epoch
        .start()
        .checked_sub(look_back_period)
        .ok_or_else(|| BoundsError::LookBackOutOfBounds)?;

    Ok(Epoch::new(
        historical_start..epoch.end(),
        epoch.time_frame(),
    ))
}

pub fn convert_source_to_output_epoch(
    epoch: &Epoch,
    source: &FrameTranslationBuffer,
    output: &FrameTranslationBuffer,
) -> Result<Epoch, anyhow::Error> {
    debug_assert_eq!(epoch.time_frame(), source.time_frame());

    match source.time_frame().cmp(&output.time_frame()) {
        Ordering::Greater => Err(ContractError::CursorError(
            format!(
                "Conversion of epochs is allowed only from lower or equal time frame, \
                 but source({:?}) is greater than output({:?})",
                source.time_frame(),
                output.time_frame()
            )
            .into(),
        )
        .into()),
        Ordering::Equal => Ok(epoch.clone()),
        Ordering::Less => {
            match (source, output) {
                (
                    FrameTranslationBuffer::IdentityTranslationBuffer(_),
                    FrameTranslationBuffer::DirectTranslationBuffer(output),
                ) => {
                    // by design assumes identity tlb means base time frame
                    let tail = output.translate(epoch.start());
                    let head = output.translate(epoch.end() - 1) + 1;
                    Ok(Epoch::new(tail..head, output.time_frame()))
                }
                (
                    FrameTranslationBuffer::DirectTranslationBuffer(source),
                    FrameTranslationBuffer::DirectTranslationBuffer(output),
                ) => {
                    let source_frame_start = source.reverse_validity_lifetime(epoch.start());
                    let source_frame_end = source.reverse_validity_lifetime(epoch.end());

                    let output_frame_start = output.translate(source_frame_start);
                    let output_frame_end = output.translate(source_frame_end) + 1;

                    Ok(Epoch::new(
                        output_frame_start..output_frame_end,
                        output.time_frame(),
                    ))
                }
                _ => {
                    // By design there cannot be two identity tlbs with different time frames because
                    // identity tlb represent base time frame and there cannot be direct to identity pair
                    // because direct represent upper time frame where identity base time frame.
                    Err(ContractError::CursorError(
                        format!("Cannot convert source to output epoch for either pair of (direct, identity) or (identity, identity) TLB's, \
                                 with time_frames source({:?}), output({:?})",
                                source.time_frame(),
                                output.time_frame()
                        ).into()
                    ).into())
                }
            }
        }
    }
}

pub fn cursor_to_base_frame_indices(cursor: &Cursor, ftb: &FrameTranslationBuffer) -> Epoch {
    let start = ftb.translate(cursor.base_ptr());
    let end = ftb.translate(cursor.base_ptr() + cursor.step() - 1) + 1;
    Epoch::new(start..end, ftb.time_frame())
}

pub fn cursor_to_offset_frame_indices(cursor: &Cursor, ftb: &FrameTranslationBuffer) -> Epoch {
    let start = ftb.translate(cursor.start());
    let end = ftb.translate(cursor.end() - 1) + 1;
    Epoch::new(start..end, ftb.time_frame())
}

#[cfg(test)]
mod tests {
    use crate::sim::bb::cursor::{
        convert_source_to_output_epoch, cursor_to_base_frame_indices, cursor_to_frame_indices,
        cursor_to_offset_frame_indices, epoch_look_back, Cursor, CursorExpansion, Epoch, TimeShift,
    };
    use crate::sim::bb::quotes::{BaseQuotes, FrameQuotes, TimestampVector};
    use crate::sim::bb::time_frame;
    use crate::sim::bb::time_frame::TimeFrame;
    use crate::sim::builder::frame_builder::build_frame_to_frame;
    use crate::sim::builder::translation_builder::build_translation_buffer;
    use crate::sim::tlb::TranslationUnitDescriptor;
    use std::sync::Arc;

    #[test]
    fn should_convert_source_to_output_same_tlb() {
        let base_quotes = base_quotes_with_timestamp(
            (0u64..).step_by(1000).take(100).collect(),
            time_frame::SECOND_1,
        );

        let source = build_translation_buffer(&base_quotes, time_frame::SECOND_5).unwrap();

        assert_eq!(
            Epoch::new(7..17, time_frame::SECOND_5),
            convert_source_to_output_epoch(
                &Epoch::new(7..17, source.time_frame()),
                &source,
                &source
            )
            .unwrap()
        );
    }

    #[test]
    fn should_convert_source_to_output_identity_2_direct() {
        let base_quotes = base_quotes_with_timestamp(
            (0u64..).step_by(1000).take(100).collect(),
            time_frame::SECOND_1,
        );

        let source = build_translation_buffer(&base_quotes, time_frame::SECOND_1).unwrap();
        let output = build_translation_buffer(&base_quotes, time_frame::SECOND_5).unwrap();

        assert_eq!(
            Epoch::new(1..5, time_frame::SECOND_5),
            convert_source_to_output_epoch(
                &Epoch::new(5..21, source.time_frame()),
                &source,
                &output
            )
            .unwrap()
        );
    }

    #[test]
    fn should_convert_source_to_output_epoch_seconds_based_epochs() {
        let base_quotes = base_quotes_with_timestamp(
            (0u64..).step_by(1000).take(100).collect(),
            time_frame::SECOND_1,
        );

        let source = build_translation_buffer(&base_quotes, time_frame::SECOND_5).unwrap();
        let output = build_translation_buffer(&base_quotes, time_frame::SECOND_15).unwrap();

        assert_eq!(
            Epoch::new(2..6, time_frame::SECOND_15),
            convert_source_to_output_epoch(
                &Epoch::new(7..17, source.time_frame()),
                &source,
                &output
            )
            .unwrap()
        );
    }

    #[test]
    fn should_convert_source_to_output_epoch_minutes_based_epochs() {
        let base_quotes = base_quotes_with_timestamp(
            (0u64..).step_by(60000).take(100).collect(),
            time_frame::MINUTE_1,
        );

        let source = build_translation_buffer(&base_quotes, time_frame::MINUTE_3).unwrap();
        let output = build_translation_buffer(&base_quotes, time_frame::MINUTE_5).unwrap();

        assert_eq!(
            Epoch::new(4..11, time_frame::MINUTE_5),
            convert_source_to_output_epoch(
                &Epoch::new(7..17, source.time_frame()),
                &source,
                &output
            )
            .unwrap()
        );
    }

    #[test]
    fn should_convert_cursor_to_frame_indices() {
        let base_quotes = base_quotes_with_timestamp(
            (0u64..).step_by(1000).take(100).collect(),
            time_frame::SECOND_1,
        );

        let cursor = Cursor::new(0, 5, 16, time_frame::SECOND_1);
        let tlb = build_translation_buffer(&base_quotes, time_frame::SECOND_5).unwrap();

        assert_eq!(
            Epoch::new(1..5, time_frame::SECOND_5),
            cursor_to_frame_indices(&cursor, CursorExpansion::Identity, &tlb).unwrap()
        );
    }

    #[test]
    fn should_convert_cursor_to_frame_indices_with_expansion_into_future() {
        let base_quotes = base_quotes_with_timestamp(
            (0u64..).step_by(1000).take(100).collect(),
            time_frame::SECOND_1,
        );

        let cursor = Cursor::new(0, 5, 16, time_frame::SECOND_1);
        let tlb = build_translation_buffer(&base_quotes, time_frame::SECOND_5).unwrap();

        assert_eq!(
            Epoch::new(1..6, time_frame::SECOND_5),
            cursor_to_frame_indices(
                &cursor,
                CursorExpansion::Identity.expand_by(TimeShift::Future(1)),
                &tlb
            )
            .unwrap()
        );
    }

    #[test]
    fn should_convert_cursor_to_frame_indices_with_expansion_into_past() {
        let base_quotes = base_quotes_with_timestamp(
            (0u64..).step_by(1000).take(100).collect(),
            time_frame::SECOND_1,
        );

        let cursor = Cursor::new(0, 5, 16, time_frame::SECOND_1);
        let tlb = build_translation_buffer(&base_quotes, time_frame::SECOND_5).unwrap();

        assert_eq!(
            Epoch::new(0..5, time_frame::SECOND_5),
            cursor_to_frame_indices(
                &cursor,
                CursorExpansion::Identity.expand_by(TimeShift::Past(1)),
                &tlb
            )
            .unwrap()
        );
    }

    #[test]
    fn should_convert_cursor_to_frame_indices_with_expansion_on_both_edges() {
        let base_quotes = base_quotes_with_timestamp(
            (0u64..).step_by(1000).take(100).collect(),
            time_frame::SECOND_1,
        );

        let cursor = Cursor::new(0, 5, 16, time_frame::SECOND_1);
        let tlb = build_translation_buffer(&base_quotes, time_frame::SECOND_5).unwrap();

        assert_eq!(
            Epoch::new(0..6, time_frame::SECOND_5),
            cursor_to_frame_indices(
                &cursor,
                CursorExpansion::Identity
                    .expand_by(TimeShift::Past(1))
                    .expand_by(TimeShift::Future(1)),
                &tlb
            )
            .unwrap()
        );
    }

    #[test]
    fn should_convert_cursor_to_offset_frame_indices() {
        let base_quotes = base_quotes_with_timestamp(
            (0u64..).step_by(1000).take(100).collect(),
            time_frame::SECOND_1,
        );

        let cursor = Cursor::new(0, 5, 16, time_frame::SECOND_1).shrink_offset_range(4..12);
        let tlb = build_translation_buffer(&base_quotes, time_frame::SECOND_5).unwrap();

        assert_eq!(
            Epoch::new(1..4, time_frame::SECOND_5),
            cursor_to_offset_frame_indices(&cursor, &tlb)
        );
    }

    #[test]
    fn should_convert_cursor_to_base_frame_indices() {
        let base_quotes = base_quotes_with_timestamp(
            (0u64..).step_by(1000).take(100).collect(),
            time_frame::SECOND_1,
        );

        let cursor = Cursor::new(0, 5, 16, time_frame::SECOND_1).shrink_offset_range(4..12);
        let tlb = build_translation_buffer(&base_quotes, time_frame::SECOND_5).unwrap();

        assert_eq!(
            Epoch::new(1..5, time_frame::SECOND_5),
            cursor_to_base_frame_indices(&cursor, &tlb)
        );
    }

    #[test]
    fn should_move_back_start_by_look_back() {
        assert_eq!(
            Epoch::new(5..20, time_frame::SECOND_5),
            epoch_look_back(&Epoch::new(10..20, time_frame::SECOND_5), 5).unwrap()
        );
    }

    fn base_quotes_with_timestamp(timestamp: Vec<u64>, time_frame: TimeFrame) -> BaseQuotes {
        BaseQuotes::new(Arc::new(FrameQuotes::new(
            vec![0.0; timestamp.len()],
            vec![0.0; timestamp.len()],
            vec![0.0; timestamp.len()],
            vec![0.0; timestamp.len()],
            TimestampVector::from_utc(timestamp),
            time_frame,
        )))
    }
}
