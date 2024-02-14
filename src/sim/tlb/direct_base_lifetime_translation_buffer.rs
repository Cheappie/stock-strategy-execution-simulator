/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::cursor::{Cursor, Epoch};
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::context::SpanContext;
use crate::sim::tlb::{
    DirectTranslationBuffer, InlinedReverseTranslation, InlinedTranslation,
    TranslationUnitDescriptor,
};
use std::ops::Range;

pub struct DirectBaseLifetimeTranslationBuffer<'a> {
    base: TimeFrame,
    left: TimeFrame,
    right: TimeFrame,
    dtb: &'a DirectTranslationBuffer,
}

impl<'a> DirectBaseLifetimeTranslationBuffer<'a> {
    pub fn new(
        base: TimeFrame,
        left: TimeFrame,
        right: TimeFrame,
        dtb: &'a DirectTranslationBuffer,
    ) -> Self {
        debug_assert_ne!(left, right);
        debug_assert!(TimeFrame::aligned(left, right));
        debug_assert_eq!(base, left.min(right));
        debug_assert_eq!(dtb.time_frame(), left.max(right));

        Self {
            base,
            left,
            right,
            dtb,
        }
    }

    pub fn translate(
        &self,
        base_epoch: &Epoch,
    ) -> Result<DirectBaseLifetimeTranslationIter<'_>, anyhow::Error> {
        debug_assert_eq!(self.base, base_epoch.time_frame());
        debug_assert_eq!(base_epoch.time_frame(), self.left.min(self.right));

        let lt_greater_rt = self.left > self.right;
        let dtb_current = self.dtb.translate(base_epoch.start());

        Ok(DirectBaseLifetimeTranslationIter {
            lt_greater_rt,
            dtb: self.dtb,
            dtb_current,
            base_current: base_epoch.start(),
            base_end: base_epoch.end(),
        })
    }
}

pub struct DirectBaseLifetimeTranslationIter<'a> {
    lt_greater_rt: bool,
    dtb: &'a DirectTranslationBuffer,
    dtb_current: usize,
    base_current: usize,
    base_end: usize,
}

impl Iterator for DirectBaseLifetimeTranslationIter<'_> {
    type Item = DirectBaseLifetimeTranslation;

    fn next(&mut self) -> Option<Self::Item> {
        if self.base_current < self.base_end {
            debug_assert_eq!(self.dtb_current, self.dtb.translate(self.base_current));

            let dtb_current_base_limit = self
                .base_end
                .min(self.dtb.reverse_validity_lifetime(self.dtb_current + 1));

            let dtb_index = self.dtb_current;
            let base_bounds = self.base_current..dtb_current_base_limit;

            self.dtb_current += 1;
            self.base_current = dtb_current_base_limit;

            if self.lt_greater_rt {
                Some(DirectBaseLifetimeTranslation::Right(dtb_index, base_bounds))
            } else {
                Some(DirectBaseLifetimeTranslation::Left(base_bounds, dtb_index))
            }
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub enum DirectBaseLifetimeTranslation {
    Left(Range<usize>, usize),
    Right(usize, Range<usize>),
}

#[cfg(test)]
mod tests {
    use crate::sim::bb::cursor::Epoch;
    use crate::sim::bb::quotes::{BaseQuotes, FrameQuotes, TimestampVector};
    use crate::sim::bb::time_frame;
    use crate::sim::bb::time_frame::TimeFrame;
    use crate::sim::builder::{frame_builder, translation_builder};
    use crate::sim::tlb::direct_base_lifetime_translation_buffer::{
        DirectBaseLifetimeTranslation, DirectBaseLifetimeTranslationBuffer,
    };
    use crate::sim::tlb::FrameTranslationBuffer;
    use std::ops::Range;
    use std::sync::Arc;

    #[test]
    fn should_translate_whole_base_epoch() {
        let base_quotes = base_quotes_with_timestamp(
            (0u64..).step_by(5000).take(10).collect(),
            time_frame::SECOND_5,
        );

        let (frame_quotes_15, tlb_15) =
            create_frame_quotes_with_tlb(&base_quotes, time_frame::SECOND_15);

        let tlb = DirectBaseLifetimeTranslationBuffer::new(
            time_frame::SECOND_5,
            time_frame::SECOND_5,
            time_frame::SECOND_15,
            tlb_15.direct().unwrap(),
        );

        let mut iter = tlb
            .translate(&Epoch::new(0..10, time_frame::SECOND_5))
            .unwrap();

        assert_left(0..3, 0, iter.next().unwrap());
        assert_left(3..6, 1, iter.next().unwrap());
        assert_left(6..9, 2, iter.next().unwrap());
        assert_left(9..10, 3, iter.next().unwrap());
        assert!(iter.next().is_none());
    }

    #[test]
    fn should_translate_by_right_tlb() {
        let base_quotes = base_quotes_with_timestamp(
            (0u64..).step_by(5000).take(100).collect(),
            time_frame::SECOND_5,
        );

        let (frame_quotes_15, tlb_15) =
            create_frame_quotes_with_tlb(&base_quotes, time_frame::SECOND_15);

        let tlb = DirectBaseLifetimeTranslationBuffer::new(
            time_frame::SECOND_5,
            time_frame::SECOND_5,
            time_frame::SECOND_15,
            tlb_15.direct().unwrap(),
        );

        let mut iter = tlb
            .translate(&Epoch::new(10..22, time_frame::SECOND_5))
            .unwrap();

        let mut first: Option<(Range<usize>, usize)> = None;
        let mut previous: Option<(Range<usize>, usize)> = None;

        while let Some(translation) = iter.next() {
            match translation {
                DirectBaseLifetimeTranslation::Left(lower_frame_indices, greater_frame_index) => {
                    frame_indices_within_bounds(
                        lower_frame_indices.clone(),
                        greater_frame_index,
                        |i| base_quotes.timestamp(i),
                        |i| frame_quotes_15.timestamp(i),
                    );

                    assert_aligned(
                        previous.take(),
                        (lower_frame_indices.clone(), greater_frame_index),
                    );

                    if let None = first {
                        first = Some((lower_frame_indices.clone(), greater_frame_index));
                    }

                    previous = Some((lower_frame_indices.clone(), greater_frame_index));
                }
                DirectBaseLifetimeTranslation::Right(_, _) => {
                    unreachable!()
                }
            }
        }

        assert_eq!(10, first.unwrap().0.start);
        assert_eq!(22, previous.unwrap().0.end);

        let mut iter = tlb
            .translate(&Epoch::new(10..22, time_frame::SECOND_5))
            .unwrap();

        assert_left(10..12, 3, iter.next().unwrap());
        assert_left(12..15, 4, iter.next().unwrap());
        assert_left(15..18, 5, iter.next().unwrap());
        assert_left(18..21, 6, iter.next().unwrap());
        assert_left(21..22, 7, iter.next().unwrap());
        assert!(iter.next().is_none());
    }

    #[test]
    fn should_translate_by_left_tlb() {
        let base_quotes = base_quotes_with_timestamp(
            (0u64..).step_by(5000).take(100).collect(),
            time_frame::SECOND_5,
        );

        let (frame_quotes_15, tlb_15) =
            create_frame_quotes_with_tlb(&base_quotes, time_frame::SECOND_15);

        let tlb = DirectBaseLifetimeTranslationBuffer::new(
            time_frame::SECOND_5,
            time_frame::SECOND_15,
            time_frame::SECOND_5,
            tlb_15.direct().unwrap(),
        );

        let mut iter = tlb
            .translate(&Epoch::new(4..17, time_frame::SECOND_5))
            .unwrap();

        let mut first: Option<(Range<usize>, usize)> = None;
        let mut previous: Option<(Range<usize>, usize)> = None;

        while let Some(translation) = iter.next() {
            match translation {
                DirectBaseLifetimeTranslation::Left(_, _) => {
                    unreachable!()
                }
                DirectBaseLifetimeTranslation::Right(greater_frame_index, lower_frame_indices) => {
                    frame_indices_within_bounds(
                        lower_frame_indices.clone(),
                        greater_frame_index,
                        |i| base_quotes.timestamp(i),
                        |i| frame_quotes_15.timestamp(i),
                    );

                    assert_aligned(
                        previous.take(),
                        (lower_frame_indices.clone(), greater_frame_index),
                    );

                    if let None = first {
                        first = Some((lower_frame_indices.clone(), greater_frame_index));
                    }

                    previous = Some((lower_frame_indices.clone(), greater_frame_index));
                }
            }
        }

        assert_eq!(4, first.unwrap().0.start);
        assert_eq!(17, previous.unwrap().0.end);

        let mut iter = tlb
            .translate(&Epoch::new(4..17, time_frame::SECOND_5))
            .unwrap();

        assert_right(4..6, 1, iter.next().unwrap());
        assert_right(6..9, 2, iter.next().unwrap());
        assert_right(9..12, 3, iter.next().unwrap());
        assert_right(12..15, 4, iter.next().unwrap());
        assert_right(15..17, 5, iter.next().unwrap());
        assert!(iter.next().is_none());
    }

    fn assert_left(
        lower_frame_indices: Range<usize>,
        greater_frame_index: usize,
        translation: DirectBaseLifetimeTranslation,
    ) {
        match translation {
            DirectBaseLifetimeTranslation::Left(indices, index) => {
                assert_eq!(lower_frame_indices, indices);
                assert_eq!(greater_frame_index, index);
            }
            DirectBaseLifetimeTranslation::Right(_, _) => {
                unreachable!()
            }
        }
    }

    fn assert_right(
        lower_frame_indices: Range<usize>,
        greater_frame_index: usize,
        translation: DirectBaseLifetimeTranslation,
    ) {
        match translation {
            DirectBaseLifetimeTranslation::Left(_, _) => {
                unreachable!()
            }
            DirectBaseLifetimeTranslation::Right(index, indices) => {
                assert_eq!(lower_frame_indices, indices);
                assert_eq!(greater_frame_index, index);
            }
        }
    }

    fn create_frame_quotes_with_tlb(
        base_quotes: &BaseQuotes,
        time_frame: TimeFrame,
    ) -> (FrameQuotes, FrameTranslationBuffer) {
        let frame_quotes = frame_builder::build_frame_to_frame(base_quotes, time_frame)
            .expect("FrameQuotes created");
        let tlb = translation_builder::build_translation_buffer(&base_quotes, time_frame)
            .expect("TLB created");
        (frame_quotes, tlb)
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
}
