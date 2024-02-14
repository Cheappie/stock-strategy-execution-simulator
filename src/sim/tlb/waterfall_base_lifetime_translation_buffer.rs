/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::cursor::Epoch;
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::context::SpanContext;
use crate::sim::tlb::{
    DirectTranslationBuffer, InlinedReverseTranslation, InlinedTranslation,
    TranslationUnitDescriptor,
};
use std::ops::Range;

pub struct WaterfallBaseLifetimeTranslationBuffer<'a> {
    lt: &'a DirectTranslationBuffer,
    rt: &'a DirectTranslationBuffer,
    base: TimeFrame,
}

impl<'a> WaterfallBaseLifetimeTranslationBuffer<'a> {
    pub fn new(
        base: TimeFrame,
        lt: &'a DirectTranslationBuffer,
        rt: &'a DirectTranslationBuffer,
    ) -> Self {
        debug_assert_ne!(base, lt.time_frame());
        debug_assert_ne!(base, rt.time_frame());
        debug_assert_ne!(lt.time_frame(), rt.time_frame());
        debug_assert!(!TimeFrame::aligned(lt.time_frame(), rt.time_frame()));
        Self { base, lt, rt }
    }

    pub fn translate(
        &self,
        base_epoch: &Epoch,
    ) -> Result<WaterfallBaseLifetimeTranslationIter<'_>, anyhow::Error> {
        debug_assert_eq!(self.base, base_epoch.time_frame());

        let lt_greater_rt = self.lt.time_frame() > self.rt.time_frame();
        let lt_current = self.lt.translate(base_epoch.start());
        let rt_current = self.rt.translate(base_epoch.start());

        Ok(WaterfallBaseLifetimeTranslationIter {
            lt_greater_rt,
            lt: self.lt,
            lt_current,
            rt: self.rt,
            rt_current,
            base_current: base_epoch.start(),
            base_end: base_epoch.end(),
        })
    }
}

pub struct WaterfallBaseLifetimeTranslationIter<'a> {
    lt_greater_rt: bool,
    lt: &'a DirectTranslationBuffer,
    lt_current: usize,
    rt: &'a DirectTranslationBuffer,
    rt_current: usize,
    base_current: usize,
    base_end: usize,
}

impl Iterator for WaterfallBaseLifetimeTranslationIter<'_> {
    type Item = WaterfallBaseLifetimeTranslation;

    fn next(&mut self) -> Option<Self::Item> {
        if self.base_current < self.base_end {
            debug_assert_eq!(self.lt_current, self.lt.translate(self.base_current));
            debug_assert_eq!(self.rt_current, self.rt.translate(self.base_current));

            let lt_limit = self.lt.reverse_validity_lifetime(self.lt_current + 1);
            let rt_limit = self.rt.reverse_validity_lifetime(self.rt_current + 1);

            let base_limit = self.base_end.min(lt_limit).min(rt_limit);

            let lt = self.lt_current;
            let rt = self.rt_current;
            let lifetime = self.base_current..base_limit;

            self.lt_current += (base_limit == lt_limit) as usize;
            self.rt_current += (base_limit == rt_limit) as usize;
            self.base_current = base_limit;

            Some(WaterfallBaseLifetimeTranslation { lt, rt, lifetime })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct WaterfallBaseLifetimeTranslation {
    pub lt: usize,
    pub rt: usize,
    pub lifetime: Range<usize>,
}

#[cfg(test)]
mod tests {
    use crate::sim::bb::cursor::{Cursor, CursorExpansion, Epoch};
    use crate::sim::bb::quotes::{BaseQuotes, FrameQuotes, TimestampVector};
    use crate::sim::bb::time_frame::TimeFrame;
    use crate::sim::bb::{cursor, time_frame};
    use crate::sim::builder::{frame_builder, translation_builder};
    use crate::sim::tlb::waterfall_base_lifetime_translation_buffer::{
        WaterfallBaseLifetimeTranslation, WaterfallBaseLifetimeTranslationBuffer,
    };
    use crate::sim::tlb::FrameTranslationBuffer;
    use std::ops::Range;
    use std::sync::Arc;

    #[test]
    fn should_translate_whole_epoch() {
        let timestamp = (0u64..).step_by(60000).take(13).collect::<Vec<_>>();
        let base_quotes = base_quotes_with_timestamp(timestamp, time_frame::MINUTE_1);

        let (quotes_3, tlb_3) = create_frame_quotes_with_tlb(&base_quotes, time_frame::MINUTE_3);
        let (quotes_5, tlb_5) = create_frame_quotes_with_tlb(&base_quotes, time_frame::MINUTE_5);

        let tlb = WaterfallBaseLifetimeTranslationBuffer::new(
            time_frame::MINUTE_1,
            tlb_3.direct().unwrap(),
            tlb_5.direct().unwrap(),
        );

        let mut iter = tlb
            .translate(&Epoch::new(0..13, time_frame::MINUTE_1))
            .unwrap();

        assert_translation(0, 0, 0..3, iter.next().unwrap());
        assert_translation(1, 0, 3..5, iter.next().unwrap());
        assert_translation(1, 1, 5..6, iter.next().unwrap());
        assert_translation(2, 1, 6..9, iter.next().unwrap());
        assert_translation(3, 1, 9..10, iter.next().unwrap());
        assert_translation(3, 2, 10..12, iter.next().unwrap());
        assert_translation(4, 2, 12..13, iter.next().unwrap());
        assert!(iter.next().is_none());
    }

    #[test]
    fn should_perform_waterfall_base_translation() {
        let timestamp = (0u64..).step_by(60000).take(100).collect::<Vec<_>>();
        let base_quotes = base_quotes_with_timestamp(timestamp, time_frame::MINUTE_1);

        let (quotes_3, tlb_3) = create_frame_quotes_with_tlb(&base_quotes, time_frame::MINUTE_3);
        let (quotes_5, tlb_5) = create_frame_quotes_with_tlb(&base_quotes, time_frame::MINUTE_5);

        let tlb = WaterfallBaseLifetimeTranslationBuffer::new(
            time_frame::MINUTE_1,
            tlb_3.direct().unwrap(),
            tlb_5.direct().unwrap(),
        );

        let mut iter = tlb
            .translate(&Epoch::new(21..37, time_frame::MINUTE_1))
            .unwrap();

        let mut first = None;
        let mut previous = None;

        while let Some(translation) = iter.next() {
            println!("{:?}", translation);
            frame_indices_within_bounds(
                translation.lifetime.clone(),
                translation.lt,
                |i| base_quotes.timestamp(i),
                |i| quotes_3.timestamp(i),
            );

            frame_indices_within_bounds(
                translation.lifetime.clone(),
                translation.rt,
                |i| base_quotes.timestamp(i),
                |i| quotes_5.timestamp(i),
            );

            assert_aligned(previous.take(), translation.clone());

            if let None = first {
                first = Some(translation.clone());
            }

            previous = Some(translation.clone());
        }

        let mut iter = tlb
            .translate(&Epoch::new(21..37, time_frame::MINUTE_1))
            .unwrap();

        assert_translation(7, 4, 21..24, iter.next().unwrap());
        assert_translation(8, 4, 24..25, iter.next().unwrap());
        assert_translation(8, 5, 25..27, iter.next().unwrap());
        assert_translation(9, 5, 27..30, iter.next().unwrap());
        assert_translation(10, 6, 30..33, iter.next().unwrap());
        assert_translation(11, 6, 33..35, iter.next().unwrap());
        assert_translation(11, 7, 35..36, iter.next().unwrap());
        assert_translation(12, 7, 36..37, iter.next().unwrap());
        assert!(iter.next().is_none());
    }

    fn assert_translation(
        lt: usize,
        rt: usize,
        lifetime: Range<usize>,
        translation: WaterfallBaseLifetimeTranslation,
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
        prev: Option<WaterfallBaseLifetimeTranslation>,
        curr: WaterfallBaseLifetimeTranslation,
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
