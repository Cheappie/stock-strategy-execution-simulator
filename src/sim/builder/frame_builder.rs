/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/



use anyhow::anyhow;

use crate::sim::bb::quotes::{FrameQuotes, OfferSide, TickQuotes, TimestampVector};
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::builder::frame_builder::frame::FrameAssemblingIter;

pub fn build_tick_to_frame(
    source: &TickQuotes,
    output_time_frame: TimeFrame,
    offer_side: OfferSide,
) -> Result<FrameQuotes, anyhow::Error> {
    match source.len() {
        0 => Err(anyhow!(
            "Provided TickQuotes are empty, cannot create FrameQuotes({:?})",
            output_time_frame
        )),
        _ => {
            let mut iter = FrameAssemblingIter::new(source.timestamp_vector(), output_time_frame);

            let mut open_array = Vec::with_capacity(source.len());
            let mut high_array = Vec::with_capacity(source.len());
            let mut low_array = Vec::with_capacity(source.len());
            let mut close_array = Vec::with_capacity(source.len());
            let mut timestamp_array = Vec::with_capacity(source.len());

            while let Some(group_descriptor) = iter.next() {
                let open = offer_side.price(source, group_descriptor.first());

                let ticks = match offer_side {
                    OfferSide::Ask => &source.ask_array()[group_descriptor.group_indices.clone()],
                    OfferSide::Bid => &source.bid_array()[group_descriptor.group_indices.clone()],
                };

                let mut high = f64::MIN;
                let mut low = f64::MAX;

                for &price in ticks {
                    high = high.max(price);
                    low = low.min(price);
                }

                let close = offer_side.price(source, group_descriptor.last());

                open_array.push(open);
                high_array.push(high);
                low_array.push(low);
                close_array.push(close);
                timestamp_array.push(group_descriptor.group_timestamp);
            }

            open_array.shrink_to_fit();
            high_array.shrink_to_fit();
            low_array.shrink_to_fit();
            close_array.shrink_to_fit();
            timestamp_array.shrink_to_fit();

            Ok(FrameQuotes::new(
                open_array,
                high_array,
                low_array,
                close_array,
                TimestampVector::new(timestamp_array, source.timestamp_vector().offset()),
                output_time_frame,
            ))
        }
    }
}

pub fn build_frame_to_frame(
    source: &FrameQuotes,
    output_time_frame: TimeFrame,
) -> Result<FrameQuotes, anyhow::Error> {
    if output_time_frame <= source.time_frame()
        || !TimeFrame::aligned(source.time_frame(), output_time_frame)
    {
        Err(anyhow!(
            "Cannot create FrameQuotes from incompatible time frames, source({:?}), output({:?})",
            source.time_frame(),
            output_time_frame
        ))
    } else {
        match source.len() {
            0 => Err(anyhow!(
                "Provided FrameQuotes({:?}) are empty, cannot create FrameQuotes({:?})",
                source.time_frame(),
                output_time_frame
            )),
            _ => {
                let mut iter =
                    FrameAssemblingIter::new(source.timestamp_vector(), output_time_frame);

                let mut open_array = Vec::with_capacity(source.len());
                let mut high_array = Vec::with_capacity(source.len());
                let mut low_array = Vec::with_capacity(source.len());
                let mut close_array = Vec::with_capacity(source.len());
                let mut timestamp_array = Vec::with_capacity(source.len());

                while let Some(group_descriptor) = iter.next() {
                    let open = source.open(group_descriptor.first());

                    let high = source.high_array()[group_descriptor.group_indices.clone()]
                        .iter()
                        .fold(f64::MIN, |a, b| a.max(*b));

                    let low = source.low_array()[group_descriptor.group_indices.clone()]
                        .iter()
                        .fold(f64::MAX, |a, b| a.min(*b));

                    let close = source.close(group_descriptor.last());

                    open_array.push(open);
                    high_array.push(high);
                    low_array.push(low);
                    close_array.push(close);
                    timestamp_array.push(group_descriptor.group_timestamp);
                }

                open_array.shrink_to_fit();
                high_array.shrink_to_fit();
                low_array.shrink_to_fit();
                close_array.shrink_to_fit();
                timestamp_array.shrink_to_fit();

                Ok(FrameQuotes::new(
                    open_array,
                    high_array,
                    low_array,
                    close_array,
                    TimestampVector::new(timestamp_array, source.timestamp_vector().offset()),
                    output_time_frame,
                ))
            }
        }
    }
}

#[cfg(test)]
mod frame_builder_tests {
    use crate::sim::bb::quotes::{FrameQuotes, OfferSide, TickQuotes, TimestampVector};
    use crate::sim::bb::time_frame;
    use crate::sim::builder::frame_builder::{build_frame_to_frame, build_tick_to_frame};

    #[test]
    fn should_build_frame_quotes_from_single_tick() {
        //given
        let ask = vec![1.0];
        let bid = vec![0.8f64];
        let timestamp = vec![1200];
        let timestamp = TimestampVector::from_utc(timestamp);
        let tick_quotes = TickQuotes::new(ask, bid, timestamp);

        //when
        let frame_quotes =
            build_tick_to_frame(&tick_quotes, time_frame::SECOND_1, OfferSide::Ask).unwrap();

        //then
        assert_eq!(1, frame_quotes.len());
        assert_frame(1.0, 1.0, 1.0, 1.0, 2000, &frame_quotes, 0);
    }

    fn assert_frame(
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        timestamp: u64,
        frame_quotes: &FrameQuotes,
        index: usize,
    ) {
        assert_eq!(open, frame_quotes.open(index), "Open price mismatch");
        assert_eq!(high, frame_quotes.high(index), "High price mismatch");
        assert_eq!(low, frame_quotes.low(index), "Low price mismatch");
        assert_eq!(close, frame_quotes.close(index), "Close price mismatch");
        assert_eq!(
            timestamp,
            frame_quotes.timestamp(index),
            "Timestamp mismatch"
        );
    }

    #[test]
    fn should_build_frame_quotes_from_multiple_ticks() {
        //given
        let ask = vec![0f64; 9];
        let bid = vec![1.2, 1.4, 1.3, 1.7, 1.5, 1.9, 2.1, 2.2, 2.0];
        let timestamp = vec![1200, 1500, 16100, 21001, 21100, 21200, 21600, 21700, 22000];
        let timestamp = TimestampVector::from_utc(timestamp);
        let tick_quotes = TickQuotes::new(ask, bid, timestamp);

        //when
        let frame_quotes =
            build_tick_to_frame(&tick_quotes, time_frame::SECOND_1, OfferSide::Bid).unwrap();

        //then
        assert_eq!(3, frame_quotes.len());
        assert_frame(1.2, 1.4, 1.2, 1.4, 2000, &frame_quotes, 0);
        assert_frame(1.3, 1.3, 1.3, 1.3, 17000, &frame_quotes, 1);
        assert_frame(1.7, 2.2, 1.5, 2.0, 22000, &frame_quotes, 2);
    }

    #[test]
    fn should_build_frame_quotes_from_multiple_ticks_second_3() {
        //given
        let ask = vec![1.7, 1.2, 1.8, 1.0, 1.4, 1.1, 1.05, 1.7, 1.6];
        let bid = vec![0f64; 9];
        let timestamp = vec![300, 1000, 5000, 8000, 15001, 16100, 17200, 19212, 20000];
        let timestamp = TimestampVector::from_utc(timestamp);
        let tick_quotes = TickQuotes::new(ask, bid, timestamp);

        //when
        let frame_quotes =
            build_tick_to_frame(&tick_quotes, time_frame::SECOND_5, OfferSide::Ask).unwrap();

        //then
        assert_eq!(3, frame_quotes.len());
        assert_frame(1.7, 1.8, 1.2, 1.8, 5000, &frame_quotes, 0);
        assert_frame(1.0, 1.0, 1.0, 1.0, 10000, &frame_quotes, 1);
        assert_frame(1.4, 1.7, 1.05, 1.6, 20000, &frame_quotes, 2);
    }

    #[test]
    fn should_not_allow_building_frame_quotes_from_larger_time_frame() {
        let result = build_frame_to_frame(
            &FrameQuotes::new(
                vec![],
                vec![],
                vec![],
                vec![],
                TimestampVector::from_utc(vec![]),
                time_frame::SECOND_10,
            ),
            time_frame::SECOND_5,
        );

        assert!(result.is_err());
        let error = result.err().expect("expects error");
        assert_eq!("Cannot create FrameQuotes from incompatible time frames, source(TimeFrame(10000)), output(TimeFrame(5000))", error.to_string());
    }

    #[test]
    fn should_not_allow_building_frame_quotes_with_same_timeframe_as_source_quotes() {
        let result = build_frame_to_frame(
            &FrameQuotes::new(
                vec![],
                vec![],
                vec![],
                vec![],
                TimestampVector::from_utc(vec![]),
                time_frame::SECOND_10,
            ),
            time_frame::SECOND_10,
        );

        assert!(result.is_err());
        let error = result.err().expect("expects error");
        assert_eq!("Cannot create FrameQuotes from incompatible time frames, source(TimeFrame(10000)), output(TimeFrame(10000))", error.to_string());
    }

    #[test]
    fn should_not_build_frame_quotes_from_incompatible_source_quotes() {
        // SECOND_10 % SECOND_3 has remainder, thus we cannot build 10s frame quotes from 3s source quotes

        let result = build_frame_to_frame(
            &FrameQuotes::new(
                vec![],
                vec![],
                vec![],
                vec![],
                TimestampVector::from_utc(vec![]),
                time_frame::SECOND_10,
            ),
            time_frame::SECOND_15,
        );

        assert!(result.is_err());
        let error = result.err().expect("expects error");
        assert_eq!("Cannot create FrameQuotes from incompatible time frames, source(TimeFrame(10000)), output(TimeFrame(15000))", error.to_string());
    }

    #[test]
    fn should_build_frame_quotes_from_frame_quotes() {
        let frame_quotes = build_frame_to_frame(
            &FrameQuotes::new(
                vec![1.0, 1.1, 1.2, 1.3, 1.4],
                vec![1.4, 1.6, 1.8, 2.0, 2.2],
                vec![1.1, 1.5, 1.7, 1.9, 1.8],
                vec![1.2, 1.6, 1.7, 2.1, 2.2],
                TimestampVector::from_utc(vec![50, 100, 400, 3300, 4000]),
                time_frame::MILLISECOND_200,
            ),
            time_frame::SECOND_1,
        )
        .expect("FrameQuotes created");

        assert_eq!(2, frame_quotes.len());
        assert_frame(1.0, 1.8, 1.1, 1.7, 1000, &frame_quotes, 0);
        assert_frame(1.3, 2.2, 1.8, 2.2, 4000, &frame_quotes, 1);
    }
}

pub mod frame {
    use std::ops::Range;

    use crate::sim::bb::quotes::TimestampVector;
    use crate::sim::bb::time_frame::TimeFrame;

    ///
    /// Generates frame indices based on source timestamps and output time frame.
    ///
    /// Timestamp that passes such predicate [timestamp % output_time_frame == 0]
    /// is named `terminal quote`. Terminal quote marks an end of group.
    ///
    ///
    /// Example:
    /// ```text
    /// Assumptions:
    /// * timestamps are defined in milliseconds
    /// * timestamps: [5, 800, 1000, 1500, 1800, 4100, 6000]
    /// * output_time_frame: SECOND_1
    ///
    /// Output: [
    ///   {group_timestamp: 1000, group_indices: 0..3},
    ///   {group_timestamp: 2000, group_indices: 3..5},
    ///   {group_timestamp: 5000, group_indices: 5..6},
    ///   {group_timestamp: 6000, group_indices: 6..7},
    /// ]
    /// ```
    ///
    pub struct FrameAssemblingIter<'a> {
        source: &'a TimestampVector,
        output_time_frame: TimeFrame,
        ptr: usize,
    }

    impl<'a> FrameAssemblingIter<'a> {
        pub fn new(
            source: &'a TimestampVector,
            output_time_frame: TimeFrame,
        ) -> FrameAssemblingIter {
            FrameAssemblingIter {
                source,
                output_time_frame,
                ptr: 0,
            }
        }
    }

    impl Iterator for FrameAssemblingIter<'_> {
        type Item = GroupDescriptor;

        fn next(&mut self) -> Option<Self::Item> {
            if let Some(timestamp) = self.source.get(self.ptr) {
                let group = create_group(timestamp, u64::from(*self.output_time_frame));

                let end = {
                    let mut group_end_boundary: Option<usize> = None;

                    for i in (self.ptr + 1)..self.source.len() {
                        if !group.contains(&self.source.index(i)) {
                            group_end_boundary = Some(i);
                            break;
                        }
                    }

                    group_end_boundary.unwrap_or(self.source.len())
                };

                let group_indices = self.ptr..end;
                let group_timestamp =
                    create_group_timestamp(timestamp, u64::from(*self.output_time_frame));

                self.ptr = end;

                Some(GroupDescriptor {
                    group_indices,
                    group_timestamp,
                })
            } else {
                None
            }
        }
    }

    pub struct GroupDescriptor {
        pub group_indices: Range<usize>,
        pub group_timestamp: u64,
    }

    impl GroupDescriptor {
        pub fn first(&self) -> usize {
            debug_assert!(
                !self.group_indices.is_empty(),
                "Group indices cannot be empty"
            );
            self.group_indices.start
        }

        pub fn last(&self) -> usize {
            debug_assert!(
                !self.group_indices.is_empty(),
                "Group indices cannot be empty"
            );
            self.group_indices.end.saturating_sub(1)
        }
    }

    fn create_group(timestamp: u64, time_frame: u64) -> Range<u64> {
        if is_terminal_quote(timestamp, time_frame) {
            timestamp..(timestamp + 1)
        } else {
            let start = timestamp - (timestamp % time_frame);
            let end = start + time_frame;
            (start + 1)..(end + 1)
        }
    }

    fn is_terminal_quote(timestamp: u64, time_frame: u64) -> bool {
        timestamp % time_frame == 0
    }

    fn create_group_timestamp(timestamp: u64, time_frame: u64) -> u64 {
        if is_terminal_quote(timestamp, time_frame) {
            timestamp
        } else {
            timestamp - (timestamp % time_frame) + time_frame
        }
    }

    #[cfg(test)]
    mod tests {
        use std::ops::Range;

        use crate::sim::bb::quotes::TimestampVector;
        use crate::sim::bb::time_frame;

        use super::{create_group, FrameAssemblingIter, GroupDescriptor};

        #[test]
        fn should_create_one_second_group() {
            assert_eq!(
                3001..4001,
                create_group(3001, u64::from(*time_frame::SECOND_1))
            );
            assert_eq!(
                3001..4001,
                create_group(3500, u64::from(*time_frame::SECOND_1))
            );
            assert_eq!(
                4000..4001,
                create_group(4000, u64::from(*time_frame::SECOND_1))
            );
        }

        #[test]
        fn should_create_two_second_group() {
            assert_eq!(
                10001..15001,
                create_group(10001, u64::from(*time_frame::SECOND_5))
            );
            assert_eq!(
                10001..15001,
                create_group(12500, u64::from(*time_frame::SECOND_5))
            );
            assert_eq!(
                15000..15001,
                create_group(15000, u64::from(*time_frame::SECOND_5))
            );
        }

        #[test]
        fn should_handle_zero_timestamp() {
            assert_eq!(0..1, create_group(0, u64::from(*time_frame::SECOND_1)));
        }

        #[test]
        fn should_emit_group_indices() {
            fn assert_group(indices: Range<usize>, timestamp: u64, descriptor: GroupDescriptor) {
                assert_eq!(indices, descriptor.group_indices);
                assert_eq!(timestamp, descriptor.group_timestamp);
            }

            let source_timestamps =
                TimestampVector::from_utc(vec![5, 800, 1000, 1001, 2000, 4500, 5001]);
            let mut grouping_iter =
                FrameAssemblingIter::new(&source_timestamps, time_frame::SECOND_1);

            assert_group(0..3, 1000, grouping_iter.next().unwrap());
            assert_group(3..5, 2000, grouping_iter.next().unwrap());
            assert_group(5..6, 5000, grouping_iter.next().unwrap());
            assert_group(6..7, 6000, grouping_iter.next().unwrap());
        }
    }
}
