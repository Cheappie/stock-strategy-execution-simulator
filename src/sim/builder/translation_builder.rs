/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use std::iter::Enumerate;
use std::ops::Range;

use anyhow::anyhow;

use crate::sim::bb::quotes::{BaseQuotes, TimestampVector};
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::builder::frame_builder::frame::FrameAssemblingIter;
use crate::sim::tlb::{
    DirectTranslationBuffer, FrameTranslationBuffer, IdentityTranslationBuffer, LifetimeVector,
};

pub fn build_translation_buffer(
    base_quotes: &BaseQuotes,
    output_time_frame: TimeFrame,
) -> Result<FrameTranslationBuffer, anyhow::Error> {
    match (base_quotes.time_frame(), output_time_frame) {
        (base, output) if base > output => {
            Err(anyhow!(
                "Output time_frame({:?}) cannot be lesser than BaseQuotes time_frame({:?})",
                output,
                base
            ))
        }
        (base, output) if !TimeFrame::aligned(base, output)  => {
            Err(anyhow!(
                "Output time_frame is incompatible with BaseQuotes time_frame, because [Output({:?}) % Base({:?}) != 0]",
                output,
                base
            ))
        }
        (base, output) if base == output => {
            Ok(FrameTranslationBuffer::IdentityTranslationBuffer(
                IdentityTranslationBuffer::new(base_quotes.time_frame()),
            ))
        }
        _ => {
            let mut iter = FrameTranslationIter::new(base_quotes.timestamp_vector(), output_time_frame);

            let mut base_to_target = Vec::with_capacity(base_quotes.len());
            let mut creation_lifetime_translation = Vec::with_capacity(base_quotes.len());
            let mut validity_lifetime_translation = Vec::with_capacity(base_quotes.len());
            let mut last = None;

            while let Some(next) = iter.next() {
                for _ in next.inlined_translation.clone() {
                    base_to_target.push(next.frame_index as u32);
                }

                creation_lifetime_translation.push(next.creation_lifetime.start as u32);
                validity_lifetime_translation.push(next.validity_lifetime.start as u32);

                last = Some(next);
            }

            if let Some(last) = last {
                creation_lifetime_translation.push(last.creation_lifetime.end as u32);
                validity_lifetime_translation.push(last.validity_lifetime.end as u32)
            }

            base_to_target.shrink_to_fit();
            creation_lifetime_translation.shrink_to_fit();
            validity_lifetime_translation.shrink_to_fit();

            let creation_lifetime = LifetimeVector::new(creation_lifetime_translation);
            let validity_lifetime = LifetimeVector::new(validity_lifetime_translation);

            Ok(FrameTranslationBuffer::DirectTranslationBuffer(
                DirectTranslationBuffer::new(
                    output_time_frame,
                    base_to_target,
                    creation_lifetime,
                    validity_lifetime,
                ),
            ))
        }
    }
}

struct FrameTranslationIter<'a> {
    source: &'a TimestampVector,
    iter: FrameLifetimeDescriptorIter<'a>,
    previous: Option<FrameLifetimeDescriptor>,
}

impl<'a> FrameTranslationIter<'a> {
    pub fn new(source: &'a TimestampVector, output_time_frame: TimeFrame) -> Self {
        let mut iter = FrameLifetimeDescriptorIter::new(source, output_time_frame);
        let previous = iter.next();

        Self {
            source,
            iter,
            previous,
        }
    }
}

impl Iterator for FrameTranslationIter<'_> {
    type Item = FrameTranslationEntry;

    fn next(&mut self) -> Option<Self::Item> {
        match (self.previous.take(), self.iter.next()) {
            (Some(prev), Some(next)) => {
                let frame_index = prev.frame_index;
                let creation_lifetime = prev.creation_lifetime;
                let validity_lifetime = prev.validity_lifetime_start..next.validity_lifetime_start;

                let inlined_translation = if frame_index == 0 {
                    0..next.validity_lifetime_start
                } else {
                    validity_lifetime.clone()
                };

                self.previous = Some(next);

                Some(FrameTranslationEntry {
                    frame_index,
                    inlined_translation,
                    creation_lifetime,
                    validity_lifetime,
                })
            }
            (Some(prev), None) => {
                let frame_index = prev.frame_index;
                let creation_lifetime = prev.creation_lifetime;
                let validity_lifetime = prev.validity_lifetime_start..self.source.len();

                let inlined_translation = if frame_index == 0 {
                    0..self.source.len()
                } else {
                    validity_lifetime.clone()
                };

                Some(FrameTranslationEntry {
                    frame_index,
                    inlined_translation,
                    creation_lifetime,
                    validity_lifetime,
                })
            }
            _ => None,
        }
    }
}

#[derive(Debug)]
struct FrameTranslationEntry {
    frame_index: usize,
    inlined_translation: Range<usize>,
    creation_lifetime: Range<usize>,
    validity_lifetime: Range<usize>,
}

///
/// Generates validity lifetime starts for each frame produced by frame assembling iter.
///
/// Frame that ends with terminal quote has lifetime that starts immediately
/// at index of terminal quote. If there is no terminal quote at the end of the frame,
/// then frame is not ready for processing, It's lifetime should start at subsequent index.
///
/// Example:
/// ```text
/// * timestamps are defined in milliseconds
/// * timestamps: [5, 800, 900, 1000]
/// * output_time_frame: SECOND_1
///
/// Lifetime is immediately ready from base index: 3..
/// Even though all those timestamps belong to one frame.
/// ```
///
/// Example:
/// ```text
/// * timestamps are defined in milliseconds
/// * timestamps: [5, 800, 900, 999]
/// * output_time_frame: SECOND_1
///
/// Lifetime is ready from base index: 4..
/// Because for given timestamps frame is not ready yet, but It will be on subsequent frame.
/// ```
///
/// Example:
/// ```text
/// Assumptions:
/// * timestamps are defined in milliseconds
/// * timestamps: [5, 800, 1000, 1500, 1800, 4100, 6000]
/// * output_time_frame: SECOND_1
///
/// Output: [
///   {frame_index: 0, creation_lifetime: 0..3, validity_lifetime_start: 2},
///   {frame_index: 1, creation_lifetime: 3..5, validity_lifetime_start: 5},
///   {frame_index: 2, creation_lifetime: 5..6, validity_lifetime_start: 6},
///   {frame_index: 3, creation_lifetime: 6..7, validity_lifetime_start: 6},
/// ]
/// ```
///
struct FrameLifetimeDescriptorIter<'a> {
    source: &'a TimestampVector,
    output_time_frame: TimeFrame,
    iter: Enumerate<FrameAssemblingIter<'a>>,
}

impl<'a> FrameLifetimeDescriptorIter<'a> {
    pub fn new(source: &'a TimestampVector, output_time_frame: TimeFrame) -> Self {
        Self {
            source,
            output_time_frame,
            iter: FrameAssemblingIter::new(source, output_time_frame).enumerate(),
        }
    }
}

impl Iterator for FrameLifetimeDescriptorIter<'_> {
    type Item = FrameLifetimeDescriptor;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((frame_index, group_descriptor)) = self.iter.next() {
            #[cfg(debug_assertions)]
            {
                let terminal_quotes_count = group_descriptor
                    .group_indices
                    .clone()
                    .filter(|&i| {
                        is_terminal_quote(self.source.index(i), u64::from(*self.output_time_frame))
                    })
                    .count();

                debug_assert!(terminal_quotes_count <= 1, "Too many terminal quotes");
            }

            let last = group_descriptor.last();
            let creation_lifetime = group_descriptor.group_indices;

            let validity_lifetime_start =
                if is_terminal_quote(self.source.index(last), u64::from(*self.output_time_frame)) {
                    last
                } else {
                    last + 1
                };

            Some(FrameLifetimeDescriptor {
                frame_index,
                creation_lifetime,
                validity_lifetime_start,
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
struct FrameLifetimeDescriptor {
    frame_index: usize,
    creation_lifetime: Range<usize>,
    validity_lifetime_start: usize,
}

fn is_terminal_quote(timestamp: u64, time_frame: u64) -> bool {
    timestamp % time_frame == 0
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::Arc;

    use anyhow::Error;
    use rand::Rng;

    use crate::sim::bb::quotes::{BaseQuotes, FrameQuotes, TimestampVector};
    use crate::sim::bb::time_frame;
    use crate::sim::builder::translation_builder::build_translation_buffer;
    use crate::sim::tlb::{
        DirectTranslationBuffer, FrameTranslationBuffer, InlinedReverseTranslation,
        InlinedTranslation,
    };

    use super::{
        FrameLifetimeDescriptor, FrameLifetimeDescriptorIter, FrameTranslationEntry,
        FrameTranslationIter,
    };

    #[test]
    fn should_start_lifetime_from_terminal_quote() {
        let source_timestamps = TimestampVector::from_utc(vec![5, 100, 200, 1000]);

        let mut iter = FrameLifetimeDescriptorIter::new(&source_timestamps, time_frame::SECOND_1);

        assert_frame_lifetime(0, 0..4, 3, iter.next().unwrap());
        assert!(iter.next().is_none());
    }

    fn assert_frame_lifetime(
        frame_index: usize,
        creation_lifetime: Range<usize>,
        validity_lifetime_start: usize,
        desc: FrameLifetimeDescriptor,
    ) {
        assert_eq!(frame_index, desc.frame_index);
        assert_eq!(creation_lifetime, desc.creation_lifetime);
        assert_eq!(validity_lifetime_start, desc.validity_lifetime_start);
    }

    #[test]
    fn should_start_lifetime_immediately() {
        let source_timestamps = TimestampVector::from_utc(vec![1000]);
        let mut iter = FrameLifetimeDescriptorIter::new(&source_timestamps, time_frame::SECOND_1);

        assert_frame_lifetime(0, 0..1, 0, iter.next().unwrap());
        assert!(iter.next().is_none());
    }

    #[test]
    fn should_start_lifetime_from_subsequent_index() {
        let source_timestamps = TimestampVector::from_utc(vec![5, 100, 200]);

        let mut iter = FrameLifetimeDescriptorIter::new(&source_timestamps, time_frame::SECOND_1);

        assert_frame_lifetime(0, 0..3, 3, iter.next().unwrap());
        assert!(iter.next().is_none());
    }

    #[test]
    fn should_produce_lifetimes() {
        let source_timestamps =
            TimestampVector::from_utc(vec![5, 800, 1000, 1001, 2000, 4500, 5001, 8000]);

        let mut iter = FrameLifetimeDescriptorIter::new(&source_timestamps, time_frame::SECOND_1);

        assert_frame_lifetime(0, 0..3, 2, iter.next().unwrap());
        assert_frame_lifetime(1, 3..5, 4, iter.next().unwrap());
        assert_frame_lifetime(2, 5..6, 6, iter.next().unwrap());
        assert_frame_lifetime(3, 6..7, 7, iter.next().unwrap());
        assert_frame_lifetime(4, 7..8, 7, iter.next().unwrap());

        assert!(iter.next().is_none());
    }

    #[test]
    fn should_produce_translations_based_on_lifetimes() {
        let source_timestamps =
            TimestampVector::from_utc(vec![5, 800, 1000, 1001, 2000, 4500, 5001, 8000]);

        let mut iter = FrameTranslationIter::new(&source_timestamps, time_frame::SECOND_1);

        assert_frame_translation_entry(0, 0..4, 0..3, 2..4, iter.next().unwrap());
        assert_frame_translation_entry(1, 4..6, 3..5, 4..6, iter.next().unwrap());
        assert_frame_translation_entry(2, 6..7, 5..6, 6..7, iter.next().unwrap());
        assert_frame_translation_entry(3, 7..7, 6..7, 7..7, iter.next().unwrap());
        assert_frame_translation_entry(4, 7..8, 7..8, 7..8, iter.next().unwrap());
        assert!(iter.next().is_none());
    }

    fn assert_frame_translation_entry(
        frame_index: usize,
        inlined_translation: Range<usize>,
        creation_lifetime: Range<usize>,
        validity_lifetime: Range<usize>,
        entry: FrameTranslationEntry,
    ) {
        assert_eq!(frame_index, entry.frame_index, "{:#?}", entry);
        assert_eq!(
            inlined_translation, entry.inlined_translation,
            "{:#?}",
            entry
        );
        assert_eq!(creation_lifetime, entry.creation_lifetime, "{:#?}", entry);
        assert_eq!(validity_lifetime, entry.validity_lifetime, "{:#?}", entry);
    }

    #[test]
    fn should_expand_inlined_translation_to_start_of_timestamp_vector() {
        let source_timestamps = TimestampVector::from_utc(vec![5, 800, 1000]);
        let mut iter = FrameTranslationIter::new(&source_timestamps, time_frame::SECOND_1);

        assert_frame_translation_entry(0, 0..3, 0..3, 2..3, iter.next().unwrap());
        assert!(iter.next().is_none());
    }

    #[test]
    fn should_expand_inlined_translation_to_start_of_timestamp_vector_two_frames() {
        let source_timestamps = TimestampVector::from_utc(vec![999, 2000]);
        let mut iter = FrameTranslationIter::new(&source_timestamps, time_frame::SECOND_1);

        assert_frame_translation_entry(0, 0..1, 0..1, 1..1, iter.next().unwrap());
        assert_frame_translation_entry(1, 1..2, 1..2, 1..2, iter.next().unwrap());
        assert!(iter.next().is_none());
    }

    #[test]
    fn should_expand_inlined_translation_until_subsequent_validity_lifetime_start() {
        // [5, 800, 1000] -> 0  \
        // [1500, 1800]   -> 0  /
        // [2000]         -> 1

        let source_timestamps = TimestampVector::from_utc(vec![5, 800, 1000, 1500, 1800, 2000]);
        let mut iter = FrameTranslationIter::new(&source_timestamps, time_frame::SECOND_1);

        assert_frame_translation_entry(0, 0..5, 0..3, 2..5, iter.next().unwrap());
        assert_frame_translation_entry(1, 5..6, 3..6, 5..6, iter.next().unwrap());
        assert!(iter.next().is_none());
    }

    #[test]
    fn should_create_direct_translation_buffer_s1_to_s5() {
        let base_quotes = BaseQuotes::new(Arc::new(FrameQuotes::new(
            vec![0.0; 5],
            vec![0.0; 5],
            vec![0.0; 5],
            vec![0.0; 5],
            TimestampVector::from_utc(vec![1000, 5000, 11000, 12100, 14900]),
            time_frame::SECOND_1,
        )));

        let ftb =
            build_translation_buffer(&base_quotes, time_frame::SECOND_5).expect("TLB created");

        match ftb {
            FrameTranslationBuffer::DirectTranslationBuffer(direct) => {
                assert_base_to_target(&direct, 0..5, 0);
                assert_target_to_base(&direct, 0..2, 1..5, 0);
                assert_target_to_base(&direct, 2..5, 5..5, 1);
            }
            _ => panic!("unexpected translation buffer type"),
        }
    }

    fn assert_base_to_target(direct: &DirectTranslationBuffer, base: Range<usize>, frame: usize) {
        for i in base {
            assert_eq!(
                frame,
                direct.translate(i),
                "Translation doesn't match frame index"
            )
        }
    }

    fn assert_target_to_base(
        direct: &DirectTranslationBuffer,
        creation: Range<usize>,
        validity: Range<usize>,
        frame: usize,
    ) {
        assert_eq!(
            creation,
            direct.creation_lifetime(frame),
            "Creation lifetime mismatch"
        );
        assert_eq!(
            validity,
            direct.validity_lifetime(frame),
            "Validity lifetime mismatch"
        );
    }

    #[test]
    fn terminal_quote_should_supersede_non_terminal_quote() {
        let base_quotes = BaseQuotes::new(Arc::new(FrameQuotes::new(
            vec![0.0; 4],
            vec![0.0; 4],
            vec![0.0; 4],
            vec![0.0; 4],
            TimestampVector::from_utc(vec![1000, 5000, 9000, 15000]),
            time_frame::SECOND_1,
        )));

        let ftb =
            build_translation_buffer(&base_quotes, time_frame::SECOND_5).expect("TLB created");

        match ftb {
            FrameTranslationBuffer::DirectTranslationBuffer(direct) => {
                assert_base_to_target(&direct, 0..3, 0);
                // frame(1) is not ready immediately, It would be fine from subsequent frame
                // but It is superseded by frame(2) because frame(2) has terminal timestamp
                assert_base_to_target(&direct, 3..4, 2);

                assert_target_to_base(&direct, 0..2, 1..3, 0);
                // empty validity lifetime for frame(1)
                assert_target_to_base(&direct, 2..3, 3..3, 1);
                assert_target_to_base(&direct, 3..4, 3..4, 2);
            }
            _ => panic!("unexpected translation buffer type"),
        }
    }

    #[test]
    fn should_create_identity_translation_buffer_for_base_quotes() {
        let base_quotes = BaseQuotes::new(Arc::new(FrameQuotes::new(
            vec![],
            vec![],
            vec![],
            vec![],
            TimestampVector::from_utc(vec![]),
            time_frame::SECOND_5,
        )));

        let tlb =
            build_translation_buffer(&base_quotes, time_frame::SECOND_5).expect("TLB created");

        match tlb {
            FrameTranslationBuffer::IdentityTranslationBuffer(_) => {
                // success
            }
            _ => {
                panic!("Invalid type of FrameTranslationBuffer has been created; expected=IdentityTranslationBuffer")
            }
        }
    }

    #[test]
    fn should_fail_with_err_for_tlb_time_frame_lesser_than_base_quotes_time_frame() {
        let base_quotes = BaseQuotes::new(Arc::new(FrameQuotes::new(
            vec![],
            vec![],
            vec![],
            vec![],
            TimestampVector::from_utc(vec![]),
            time_frame::SECOND_5,
        )));

        let result = build_translation_buffer(&base_quotes, time_frame::SECOND_1);
        assert!(result.is_err());

        assert_eq!(
            "Output time_frame(TimeFrame(1000)) cannot be lesser than BaseQuotes time_frame(TimeFrame(5000))",
            result.err().unwrap().to_string()
        )
    }

    #[test]
    fn should_fail_with_err_for_tlb_incompatible_with_base_quotes() {
        let base_quotes = BaseQuotes::new(Arc::new(FrameQuotes::new(
            vec![],
            vec![],
            vec![],
            vec![],
            TimestampVector::from_utc(vec![]),
            time_frame::SECOND_10,
        )));

        let result = build_translation_buffer(&base_quotes, time_frame::SECOND_15);
        assert!(result.is_err());
        assert_eq!(
            "Output time_frame is incompatible with BaseQuotes time_frame, because [Output(TimeFrame(15000)) % Base(TimeFrame(10000)) != 0]",
            result.err().unwrap().to_string()
        );
    }
}
