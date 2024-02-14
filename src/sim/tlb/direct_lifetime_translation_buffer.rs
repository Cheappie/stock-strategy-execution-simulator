/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::cursor::{Cursor, Epoch};
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::error::ContractError;
use crate::sim::tlb::{
    DirectTranslationBuffer, InlinedReverseTranslation, InlinedTranslation,
    TranslationUnitDescriptor,
};
use std::ops::Range;

///
/// TODO: consider switching off to timestamp based lifetime translation, to reduce cache misses.
///       Basing on translation buffers will incur cache miss on each `lower.translate(greater_idx)`.
///       Where simple timestamp iteration might yield better results, timestamps are densely packed.
///
pub struct DirectLifetimeTranslationBuffer<'a> {
    lt: &'a DirectTranslationBuffer,
    rt: &'a DirectTranslationBuffer,
}

impl<'a> DirectLifetimeTranslationBuffer<'a> {
    pub fn new(lt: &'a DirectTranslationBuffer, rt: &'a DirectTranslationBuffer) -> Self {
        debug_assert_ne!(lt.time_frame(), rt.time_frame());
        debug_assert!(
            TimeFrame::aligned(lt.time_frame(), rt.time_frame()),
            "DirectLifetimeTranslation requires aligned timeframes"
        );

        Self { lt, rt }
    }

    pub fn translate(
        &self,
        epoch: &Epoch,
    ) -> Result<DirectLifetimeTranslationIter<'a>, anyhow::Error> {
        if epoch.time_frame() != self.lt.time_frame().min(self.rt.time_frame()) {
            Err(ContractError::DirectLifetimeTranslationBufferError(
                format!("Time frame of epoch({:?}) to translate must match smaller time_frame of either lt({:?}) or rt({:?}) tlb",
                        epoch,
                        self.lt.time_frame(),
                        self.rt.time_frame()).into(),
            )
            .into())
        } else {
            let lt_greater_rt = self.lt.time_frame() > self.rt.time_frame();

            if lt_greater_rt {
                Ok(DirectLifetimeTranslationIter {
                    greater: self.lt,
                    lower: self.rt,
                    current: epoch.start(),
                    end: epoch.end(),
                    lt_greater_rt,
                })
            } else {
                Ok(DirectLifetimeTranslationIter {
                    greater: self.rt,
                    lower: self.lt,
                    current: epoch.start(),
                    end: epoch.end(),
                    lt_greater_rt,
                })
            }
        }
    }
}

pub struct DirectLifetimeTranslationIter<'a> {
    greater: &'a DirectTranslationBuffer,
    lower: &'a DirectTranslationBuffer,
    lt_greater_rt: bool,
    current: usize,
    end: usize,
}

impl Iterator for DirectLifetimeTranslationIter<'_> {
    type Item = DirectLifetimeTranslation;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let lower_frame_start = self.current;
            let lower_base_start = self.lower.reverse_validity_lifetime(lower_frame_start);

            let greater_frame_idx = self.greater.translate(lower_base_start);
            let greater_base_end = self
                .greater
                .reverse_validity_lifetime(greater_frame_idx + 1);

            let lower_frame_end_unbounded = self.lower.translate(greater_base_end - 1) + 1;
            let lower_frame_end = self.end.min(lower_frame_end_unbounded);
            let lower_frame_bounds = lower_frame_start..lower_frame_end;

            debug_assert!(
                !lower_frame_bounds.is_empty(),
                "DirectLifetimeTranslation cannot produce empty lower frame range"
            );

            self.current = lower_frame_bounds.end;

            if self.lt_greater_rt {
                Some(DirectLifetimeTranslation::Right(
                    greater_frame_idx,
                    lower_frame_bounds,
                ))
            } else {
                Some(DirectLifetimeTranslation::Left(
                    lower_frame_bounds,
                    greater_frame_idx,
                ))
            }
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub enum DirectLifetimeTranslation {
    Left(Range<usize>, usize),
    Right(usize, Range<usize>),
}

#[cfg(test)]
mod tests {
    use crate::sim::bb::cursor::{Cursor, CursorExpansion, Epoch};
    use crate::sim::bb::quotes::{BaseQuotes, FrameQuotes, TimestampVector};
    use crate::sim::bb::time_frame::TimeFrame;
    use crate::sim::bb::{cursor, time_frame};
    use crate::sim::builder::{frame_builder, translation_builder};
    use crate::sim::error::ContractError;
    use crate::sim::tlb::direct_lifetime_translation_buffer::{
        DirectLifetimeTranslation, DirectLifetimeTranslationBuffer,
    };
    use crate::sim::tlb::{FrameTranslationBuffer, TranslationUnitDescriptor};
    use std::ops::Range;
    use std::sync::Arc;

    #[test]
    fn should_translate_whole_epoch() {
        let timestamp = (0u64..).step_by(5000).take(21).collect::<Vec<_>>();
        let base_quotes = base_quotes_with_timestamp(timestamp, time_frame::SECOND_5);

        let frame_quotes_15 =
            frame_builder::build_frame_to_frame(&base_quotes, time_frame::SECOND_15).unwrap();
        let tlb_15 =
            translation_builder::build_translation_buffer(&base_quotes, time_frame::SECOND_15)
                .expect("TLB created");

        let frame_quotes_30 =
            frame_builder::build_frame_to_frame(&base_quotes, time_frame::SECOND_30).unwrap();
        let tlb_30 =
            translation_builder::build_translation_buffer(&base_quotes, time_frame::SECOND_30)
                .expect("TLB created");

        let cursor = Cursor::new(0, 0, 21, time_frame::SECOND_5);
        let epoch = cursor::cursor_to_frame_indices(&cursor, CursorExpansion::Identity, &tlb_15)
            .expect("Epoch");

        let tlb = DirectLifetimeTranslationBuffer::new(
            tlb_15.direct().unwrap(),
            tlb_30.direct().unwrap(),
        );

        let mut iter = tlb.translate(&epoch).unwrap();
        assert_left(0..2, 0, iter.next().unwrap());
        assert_left(2..4, 1, iter.next().unwrap());
        assert_left(4..6, 2, iter.next().unwrap());
        assert_left(6..7, 3, iter.next().unwrap());
        assert!(iter.next().is_none());
    }

    #[test]
    fn should_translate_by_left_tlb() {
        let timestamp = (0u64..).step_by(5000).take(100).collect::<Vec<_>>();
        let base_quotes = base_quotes_with_timestamp(timestamp, time_frame::SECOND_5);

        let frame_quotes_15 =
            frame_builder::build_frame_to_frame(&base_quotes, time_frame::SECOND_15).unwrap();
        let tlb_15 =
            translation_builder::build_translation_buffer(&base_quotes, time_frame::SECOND_15)
                .expect("TLB created");

        let frame_quotes_30 =
            frame_builder::build_frame_to_frame(&base_quotes, time_frame::SECOND_30).unwrap();
        let tlb_30 =
            translation_builder::build_translation_buffer(&base_quotes, time_frame::SECOND_30)
                .expect("TLB created");

        let cursor = Cursor::new(0, 25, 25, time_frame::SECOND_5);
        let epoch = cursor::cursor_to_frame_indices(&cursor, CursorExpansion::Identity, &tlb_15)
            .expect("Epoch");

        let tlb = DirectLifetimeTranslationBuffer::new(
            tlb_15.direct().unwrap(),
            tlb_30.direct().unwrap(),
        );
        let mut iter = tlb.translate(&epoch).unwrap();

        let mut first: Option<(Range<usize>, usize)> = None;
        let mut previous: Option<(Range<usize>, usize)> = None;

        while let Some(e) = iter.next() {
            match e {
                DirectLifetimeTranslation::Left(lower_frame_indices, greater_frame_index) => {
                    frame_indices_within_bounds(
                        lower_frame_indices.clone(),
                        greater_frame_index,
                        |i| frame_quotes_15.timestamp(i),
                        |i| frame_quotes_30.timestamp(i),
                    );

                    assert_aligned(
                        previous.take(),
                        (lower_frame_indices.clone(), greater_frame_index),
                    );

                    if let None = first {
                        first = Some((lower_frame_indices.clone(), greater_frame_index));
                    }

                    previous = Some((lower_frame_indices, greater_frame_index));
                }
                DirectLifetimeTranslation::Right(_, _) => {
                    unreachable!();
                }
            }
        }

        assert_eq!(epoch.start(), first.unwrap().0.start);
        assert_eq!(epoch.end(), previous.unwrap().0.end);

        let mut iter = tlb.translate(&epoch).unwrap();
        assert_left(8..10, 4, iter.next().unwrap());
        assert_left(10..12, 5, iter.next().unwrap());
        assert_left(12..14, 6, iter.next().unwrap());
        assert_left(14..16, 7, iter.next().unwrap());
        assert_left(16..17, 8, iter.next().unwrap());
        assert!(iter.next().is_none());
    }

    #[test]
    fn should_translate_by_right_tlb() {
        let timestamp = (0u64..).step_by(5000).take(100).collect::<Vec<_>>();
        let base_quotes = base_quotes_with_timestamp(timestamp, time_frame::SECOND_5);

        let frame_quotes_10 =
            frame_builder::build_frame_to_frame(&base_quotes, time_frame::SECOND_10).unwrap();
        let tlb_10 =
            translation_builder::build_translation_buffer(&base_quotes, time_frame::SECOND_10)
                .expect("TLB created");

        let frame_quotes_30 =
            frame_builder::build_frame_to_frame(&base_quotes, time_frame::SECOND_30).unwrap();
        let tlb_30 =
            translation_builder::build_translation_buffer(&base_quotes, time_frame::SECOND_30)
                .expect("TLB created");

        let cursor = Cursor::new(0, 24, 35, time_frame::SECOND_5);
        let epoch = cursor::cursor_to_frame_indices(&cursor, CursorExpansion::Identity, &tlb_10)
            .expect("Epoch");

        let tlb = DirectLifetimeTranslationBuffer::new(
            tlb_30.direct().unwrap(),
            tlb_10.direct().unwrap(),
        );
        let mut iter = tlb.translate(&epoch).unwrap();

        let mut first: Option<(Range<usize>, usize)> = None;
        let mut previous: Option<(Range<usize>, usize)> = None;

        while let Some(e) = iter.next() {
            println!("{:?}", e);
            match e {
                DirectLifetimeTranslation::Left(_, _) => {
                    unreachable!();
                }
                DirectLifetimeTranslation::Right(greater_frame_index, lower_frame_indices) => {
                    frame_indices_within_bounds(
                        lower_frame_indices.clone(),
                        greater_frame_index,
                        |i| frame_quotes_10.timestamp(i),
                        |i| frame_quotes_30.timestamp(i),
                    );

                    assert_aligned(
                        previous.take(),
                        (lower_frame_indices.clone(), greater_frame_index),
                    );

                    if let None = first {
                        first = Some((lower_frame_indices.clone(), greater_frame_index));
                    }

                    previous = Some((lower_frame_indices, greater_frame_index));
                }
            }
        }

        assert_eq!(epoch.start(), first.unwrap().0.start);
        assert_eq!(epoch.end(), previous.unwrap().0.end);

        let mut iter = tlb.translate(&epoch).unwrap();
        assert_right(12..15, 4, iter.next().unwrap());
        assert_right(15..18, 5, iter.next().unwrap());
        assert_right(18..21, 6, iter.next().unwrap());
        assert_right(21..24, 7, iter.next().unwrap());
        assert_right(24..27, 8, iter.next().unwrap());
        assert_right(27..30, 9, iter.next().unwrap());
        assert!(iter.next().is_none());
    }

    #[test]
    fn should_err_on_epoch_to_translate_time_frame_mismatch() {
        let timestamp = (0u64..).step_by(5000).take(100).collect::<Vec<_>>();
        let base_quotes = base_quotes_with_timestamp(timestamp, time_frame::SECOND_5);

        let frame_quotes_10 =
            frame_builder::build_frame_to_frame(&base_quotes, time_frame::SECOND_10).unwrap();
        let tlb_10 =
            translation_builder::build_translation_buffer(&base_quotes, time_frame::SECOND_10)
                .expect("TLB created");

        let frame_quotes_30 =
            frame_builder::build_frame_to_frame(&base_quotes, time_frame::SECOND_30).unwrap();
        let tlb_30 =
            translation_builder::build_translation_buffer(&base_quotes, time_frame::SECOND_30)
                .expect("TLB created");

        let cursor = Cursor::new(0, 24, 35, time_frame::SECOND_5);
        let epoch = cursor::cursor_to_frame_indices(&cursor, CursorExpansion::Identity, &tlb_30)
            .expect("Epoch");

        let tlb = DirectLifetimeTranslationBuffer::new(
            tlb_30.direct().unwrap(),
            tlb_10.direct().unwrap(),
        );
        let error = tlb.translate(&epoch).err().unwrap();

        match error
            .downcast_ref::<ContractError>()
            .expect("ContractError")
        {
            ContractError::DirectLifetimeTranslationBufferError(msg) => {
                assert_eq!("Time frame of epoch(Epoch { range: 4..10, time_frame: TimeFrame(30000) }) to translate \
                                    must match smaller time_frame of either lt(TimeFrame(30000)) or rt(TimeFrame(10000)) tlb", 
                                   msg
                        );
            }
            _ => panic!("Expected error TimeFrameMismatch, but actual was different"),
        }
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

    fn frame_indices_within_bounds(
        lower_frame_indices: Range<usize>,
        greater_frame_index: usize,
        lower_ts_vector: impl Fn(usize) -> u64,
        greater_ts_vector: impl Fn(usize) -> u64,
    ) {
        let from = greater_ts_vector(greater_frame_index);
        let to = greater_ts_vector(greater_frame_index + 1);

        let lower = greater_ts_vector(greater_frame_index - 1);
        let lower_outer_bounds = lower..from;

        let upper = greater_ts_vector(greater_frame_index + 2);
        let upper_outer_bounds = to..upper;

        let timestamp_bounds = from..to;
        for i in lower_frame_indices {
            assert!(timestamp_bounds.contains(&lower_ts_vector(i)));
            assert!(!lower_outer_bounds.contains(&lower_ts_vector(i)));
            assert!(!upper_outer_bounds.contains(&lower_ts_vector(i)));
        }
    }

    fn assert_aligned(prev: Option<(Range<usize>, usize)>, current: (Range<usize>, usize)) {
        if let Some(prev) = prev {
            assert_eq!(prev.0.end, current.0.start);
            assert_eq!(prev.1 + 1, current.1);
        }
    }

    fn assert_left(
        lower_frame_indices: Range<usize>,
        greater_frame_index: usize,
        translation: DirectLifetimeTranslation,
    ) {
        match translation {
            DirectLifetimeTranslation::Left(indices, index) => {
                assert_eq!(lower_frame_indices, indices);
                assert_eq!(greater_frame_index, index);
            }
            DirectLifetimeTranslation::Right(_, _) => {
                unreachable!()
            }
        }
    }

    fn assert_right(
        lower_frame_indices: Range<usize>,
        greater_frame_index: usize,
        translation: DirectLifetimeTranslation,
    ) {
        match translation {
            DirectLifetimeTranslation::Left(_, _) => {
                unreachable!()
            }
            DirectLifetimeTranslation::Right(index, indices) => {
                assert_eq!(lower_frame_indices, indices);
                assert_eq!(greater_frame_index, index);
            }
        }
    }
}
