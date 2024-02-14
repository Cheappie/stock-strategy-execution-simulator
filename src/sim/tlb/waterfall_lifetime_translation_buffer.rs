/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::cursor::Epoch;
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
pub struct WaterfallLifetimeTranslationBuffer<'a> {
    lt: &'a DirectTranslationBuffer,
    rt: &'a DirectTranslationBuffer,
    out: &'a DirectTranslationBuffer,
}

impl<'a> WaterfallLifetimeTranslationBuffer<'a> {
    pub fn new(
        lt: &'a DirectTranslationBuffer,
        rt: &'a DirectTranslationBuffer,
        out: &'a DirectTranslationBuffer,
    ) -> Self {
        debug_assert!(
            !TimeFrame::aligned(lt.time_frame(), rt.time_frame()),
            "Prefer DirectLifetimeTranslationBuffer for aligned time frames"
        );
        debug_assert!(
            TimeFrame::aligned(lt.time_frame(), out.time_frame())
                && TimeFrame::aligned(rt.time_frame(), out.time_frame()),
            "WaterfallLifetimeTranslationBuffer expects that both lt and rt tlb's are aligned with output tlb"
        );
        debug_assert!(
            lt.time_frame() > out.time_frame() && rt.time_frame() > out.time_frame(),
            "WaterfallLifetimeTranslationBuffer expects that both lt and rt tlb's have larger timeframe's than output tlb timeframe"
        );

        Self { lt, rt, out }
    }

    pub fn translate(
        &self,
        epoch: &Epoch,
    ) -> Result<WaterfallLifetimeTranslationIter<'a>, anyhow::Error> {
        if epoch.time_frame() != self.out.time_frame() {
            Err(ContractError::WaterfallLifetimeTranslationBufferError(
                format!(
                    "Time frame of epoch({:?}) to translate must match output time_frame({:?})",
                    epoch,
                    self.out.time_frame(),
                )
                .into(),
            )
            .into())
        } else {
            Ok(WaterfallLifetimeTranslationIter {
                lt: self.lt,
                lt_cache: FrameWithUpperBound::empty(),
                rt: self.rt,
                rt_cache: FrameWithUpperBound::empty(),
                out: self.out,
                current: epoch.start(),
                end: epoch.end(),
            })
        }
    }
}

pub struct WaterfallLifetimeTranslationIter<'a> {
    lt: &'a DirectTranslationBuffer,
    lt_cache: FrameWithUpperBound,
    rt: &'a DirectTranslationBuffer,
    rt_cache: FrameWithUpperBound,
    out: &'a DirectTranslationBuffer,
    current: usize,
    end: usize,
}

impl Iterator for WaterfallLifetimeTranslationIter<'_> {
    type Item = WaterfallLifetimeTranslation;

    fn next(&mut self) -> Option<Self::Item> {
        fn translate<'a>(
            out_base_start: usize,
            greater: &'a DirectTranslationBuffer,
            lower: &'a DirectTranslationBuffer,
        ) -> FrameWithUpperBound {
            let greater_frame_idx = greater.translate(out_base_start);
            let greater_base_end = greater.reverse_validity_lifetime(greater_frame_idx + 1);
            let frame_lower_end_unbounded = lower.translate(greater_base_end - 1) + 1;
            FrameWithUpperBound {
                frame_idx: greater_frame_idx,
                frame_lower_end_unbounded,
            }
        }

        if self.current < self.end {
            let out_frame_start = self.current;
            let out_base_start = self.out.reverse_validity_lifetime(out_frame_start);

            let lt = if self.lt_cache.frame_lower_end_unbounded > self.current {
                &self.lt_cache
            } else {
                self.lt_cache = translate(out_base_start, self.lt, self.out);
                &self.lt_cache
            };

            let rt = if self.rt_cache.frame_lower_end_unbounded > self.current {
                &self.rt_cache
            } else {
                self.rt_cache = translate(out_base_start, self.rt, self.out);
                &self.rt_cache
            };

            let out_frame_end = self
                .end
                .min(lt.frame_lower_end_unbounded)
                .min(rt.frame_lower_end_unbounded);

            let out_frame_range = out_frame_start..out_frame_end;

            self.current = out_frame_range.end;

            Some(WaterfallLifetimeTranslation {
                lt: lt.frame_idx,
                rt: rt.frame_idx,
                lifetime: out_frame_range,
            })
        } else {
            None
        }
    }
}

struct FrameWithUpperBound {
    frame_idx: usize,
    frame_lower_end_unbounded: usize,
}

impl FrameWithUpperBound {
    pub fn empty() -> Self {
        FrameWithUpperBound {
            frame_idx: 0,
            frame_lower_end_unbounded: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WaterfallLifetimeTranslation {
    pub lt: usize,
    pub rt: usize,
    pub lifetime: Range<usize>,
}

#[cfg(test)]
mod tests {
    use crate::sim::bb::cursor::{Cursor, CursorExpansion};
    use crate::sim::bb::quotes::{BaseQuotes, FrameQuotes, TimestampVector};
    use crate::sim::bb::time_frame::TimeFrame;
    use crate::sim::bb::{cursor, time_frame};
    use crate::sim::builder::{frame_builder, translation_builder};
    use crate::sim::error::ContractError;
    use crate::sim::tlb::waterfall_lifetime_translation_buffer::{
        WaterfallLifetimeTranslation, WaterfallLifetimeTranslationBuffer,
    };
    use crate::sim::tlb::FrameTranslationBuffer;
    use std::ops::Range;
    use std::sync::Arc;

    #[test]
    fn should_translate_whole_epoch() {
        let timestamp = (0u64..).step_by(20000).take(25).collect::<Vec<_>>();
        let base_quotes = base_quotes_with_timestamp(timestamp, time_frame::SECOND_15);

        let (out_quotes, out_tlb) =
            create_frame_quotes_with_tlb(&base_quotes, time_frame::MINUTE_1);
        let (quotes_3, tlb_3) = create_frame_quotes_with_tlb(&base_quotes, time_frame::MINUTE_3);
        let (quotes_5, tlb_5) = create_frame_quotes_with_tlb(&base_quotes, time_frame::MINUTE_5);

        let cursor = Cursor::new(0, 0, 25, time_frame::SECOND_30);
        let epoch = cursor::cursor_to_frame_indices(&cursor, CursorExpansion::Identity, &out_tlb)
            .expect("Epoch");

        let tlb = WaterfallLifetimeTranslationBuffer::new(
            tlb_3.direct().unwrap(),
            tlb_5.direct().unwrap(),
            out_tlb.direct().unwrap(),
        );

        let mut iter = tlb.translate(&epoch).unwrap();

        let mut iter = tlb.translate(&epoch).unwrap();
        assert_translation(0, 0, 0..3, iter.next().unwrap());
        assert_translation(1, 0, 3..5, iter.next().unwrap());
        assert_translation(1, 1, 5..6, iter.next().unwrap());
        assert_translation(2, 1, 6..9, iter.next().unwrap());
        assert!(iter.next().is_none());
    }

    #[test]
    fn should_perform_waterfall_translation() {
        let timestamp = (0u64..).step_by(20000).take(100).collect::<Vec<_>>();
        let base_quotes = base_quotes_with_timestamp(timestamp, time_frame::SECOND_15);

        let (out_quotes, out_tlb) =
            create_frame_quotes_with_tlb(&base_quotes, time_frame::MINUTE_1);
        let (quotes_3, tlb_3) = create_frame_quotes_with_tlb(&base_quotes, time_frame::MINUTE_3);
        let (quotes_5, tlb_5) = create_frame_quotes_with_tlb(&base_quotes, time_frame::MINUTE_5);

        let cursor = Cursor::new(0, 25, 25, time_frame::SECOND_15);
        let epoch = cursor::cursor_to_frame_indices(&cursor, CursorExpansion::Identity, &out_tlb)
            .expect("Epoch");

        let tlb = WaterfallLifetimeTranslationBuffer::new(
            tlb_3.direct().unwrap(),
            tlb_5.direct().unwrap(),
            out_tlb.direct().unwrap(),
        );
        let mut iter = tlb.translate(&epoch).unwrap();

        let mut first = None;
        let mut previous = None;

        while let Some(e) = iter.next() {
            frame_indices_within_bounds(
                e.lifetime.clone(),
                e.lt,
                |i| out_quotes.timestamp(i),
                |i| quotes_3.timestamp(i),
            );

            frame_indices_within_bounds(
                e.lifetime.clone(),
                e.rt,
                |i| out_quotes.timestamp(i),
                |i| quotes_5.timestamp(i),
            );

            assert_aligned(previous.take(), e.clone());

            if let None = first {
                first = Some(e.clone());
            }

            previous = Some(e.clone());
        }

        assert_eq!(epoch.start(), first.unwrap().lifetime.start);
        assert_eq!(epoch.end(), previous.unwrap().lifetime.end);

        let mut iter = tlb.translate(&epoch).unwrap();
        assert_translation(2, 1, 8..9, iter.next().unwrap());
        assert_translation(3, 1, 9..10, iter.next().unwrap());
        assert_translation(3, 2, 10..12, iter.next().unwrap());
        assert_translation(4, 2, 12..15, iter.next().unwrap());
        assert_translation(5, 3, 15..17, iter.next().unwrap());
        assert!(iter.next().is_none());
    }

    #[test]
    fn should_err_on_epoch_to_translate_time_frame_mismatch() {
        let timestamp = (0u64..).step_by(20000).take(100).collect::<Vec<_>>();
        let base_quotes = base_quotes_with_timestamp(timestamp, time_frame::SECOND_15);

        let (out_quotes, out_tlb) =
            create_frame_quotes_with_tlb(&base_quotes, time_frame::MINUTE_1);
        let (quotes_3, tlb_3) = create_frame_quotes_with_tlb(&base_quotes, time_frame::MINUTE_3);
        let (quotes_5, tlb_5) = create_frame_quotes_with_tlb(&base_quotes, time_frame::MINUTE_5);

        let cursor = Cursor::new(0, 25, 25, time_frame::SECOND_30);
        let epoch = cursor::cursor_to_frame_indices(&cursor, CursorExpansion::Identity, &tlb_5)
            .expect("Epoch");

        let tlb = WaterfallLifetimeTranslationBuffer::new(
            tlb_3.direct().unwrap(),
            tlb_5.direct().unwrap(),
            out_tlb.direct().unwrap(),
        );

        let error = tlb.translate(&epoch).err().unwrap();

        match error
            .downcast_ref::<ContractError>()
            .expect("ContractError")
        {
            ContractError::WaterfallLifetimeTranslationBufferError(err_msg) => {
                assert_eq!(
                    "Time frame of epoch(Epoch { range: 1..4, time_frame: TimeFrame(300000) }) \
                                    to translate must match output time_frame(TimeFrame(60000))",
                    err_msg
                );
            }
            _ => panic!("Expected error TimeFrameMismatch, but actual was different"),
        }
    }

    fn assert_translation(
        lt: usize,
        rt: usize,
        lifetime: Range<usize>,
        translation: WaterfallLifetimeTranslation,
    ) {
        assert_eq!(lt, translation.lt);
        assert_eq!(rt, translation.rt);
        assert_eq!(lifetime, translation.lifetime);
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
        for i in lower_frame_indices.clone() {
            assert!(timestamp_bounds.contains(&lower_ts_vector(i)));
            assert!(!lower_outer_bounds.contains(&lower_ts_vector(i)));
            assert!(!upper_outer_bounds.contains(&lower_ts_vector(i)));
        }
    }

    fn assert_aligned(
        prev: Option<WaterfallLifetimeTranslation>,
        curr: WaterfallLifetimeTranslation,
    ) {
        if let Some(prev) = prev {
            assert_eq!(
                prev.lifetime.end, curr.lifetime.start,
                "Previous: {:?}, Current: {:?}",
                prev, curr
            );
            assert!(
                prev.lt == curr.lt || prev.lt + 1 == curr.lt,
                "Previous: {:?}, Current: {:?}",
                prev,
                curr
            );
            assert!(
                prev.rt == curr.rt || prev.rt + 1 == curr.rt,
                "Previous: {:?}, Current: {:?}",
                prev,
                curr
            );
        }
    }
}
