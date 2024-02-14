/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::cursor::Epoch;
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::bb::ComputeId;
use crate::sim::collections::circular_buffer;
use crate::sim::collections::circular_buffer::CircularBuffer;
use crate::sim::error::{ContractError, SetupError};
use crate::sim::reader::{DataReader, Reader, ReaderFactory};
use crate::sim::selector::{LaneSelector, Selector};
use std::iter::Map;
use std::ops::{Deref, Range};

pub struct SpatialBuffer {
    epoch: Epoch,
    circular_buffer: CircularBuffer<f64>,
}

impl SpatialBuffer {
    pub fn new(time_frame: TimeFrame, capacity: usize) -> Self {
        Self {
            epoch: Epoch::empty(time_frame),
            circular_buffer: CircularBuffer::new(capacity),
        }
    }

    pub fn with_initial_position(time_frame: TimeFrame, capacity: usize, position: usize) -> Self {
        Self {
            epoch: Epoch::new(position..position, time_frame),
            circular_buffer: CircularBuffer::new(capacity),
        }
    }

    pub fn create_writer(&mut self, epoch_write: &Epoch) -> Result<WriterSpec, anyhow::Error> {
        if epoch_write.time_frame() != self.epoch.time_frame() {
            return Err(ContractError::TimeFrameMismatchError(
                format!(
                    "Epoch to write time_frame({:?}) doesn't match time_frame({:?}) of this buffer",
                    epoch_write.time_frame(),
                    self.epoch.time_frame()
                )
                .into(),
            )
            .into());
        }

        if self.epoch.start() > epoch_write.start() {
            return Err(ContractError::WriterError(
                format!(
                    "Epoch to write({:?}) cannot be older than current epoch of this SpatialBuffer({:?})",
                    epoch_write,
                    self.epoch
                )
                    .into(),
            ).into());
        }

        if self.circular_buffer.capacity() < epoch_write.span() {
            return Err(ContractError::WriterError(
                format!(
                    "Epoch to write({:?}) overflows SpatialBuffer capacity({:?})",
                    epoch_write,
                    self.circular_buffer.capacity()
                )
                .into(),
            )
            .into());
        }

        let next_epoch_start = self.epoch.end().max(epoch_write.start());
        let next_epoch_end = epoch_write.end().max(next_epoch_start);
        let epoch_for_compute =
            Epoch::new(next_epoch_start..next_epoch_end, self.epoch.time_frame());

        let difference = self.epoch.end()..epoch_for_compute.end();
        let difference = Epoch::new(difference, self.epoch.time_frame());

        if self.circular_buffer.capacity() < difference.span() {
            return Err(ContractError::WriterError(
                format!("The span of difference({:?}) between current epoch and epoch to write overflows SpatialBuffer capacity({})", 
                        difference, self.circular_buffer.capacity()
                ).into()
            ).into());
        }

        Ok(WriterSpec {
            spatial_buffer: self,
            difference,
            epoch_for_compute,
        })
    }

    pub fn create_reader(&self, epoch_read: &Epoch) -> Result<SpatialReader, anyhow::Error> {
        if Epoch::is_superset(&self.epoch, epoch_read)? {
            Ok(SpatialReader {
                buffer: &self.circular_buffer,
                epoch: self.epoch.clone(),
                tail: self.circular_buffer.tail(self.epoch.span()),
            })
        } else {
            Err(ContractError::ReaderError(
                format!(
                    "Epoch read out of SpatialBuffer bounds, SpatialBuffer({:?}), epoch_read({:?})",
                    &self.epoch, epoch_read
                )
                .into(),
            )
            .into())
        }
    }

    pub fn epoch(&self) -> &Epoch {
        &self.epoch
    }

    pub fn time_frame(&self) -> TimeFrame {
        self.epoch.time_frame()
    }

    pub fn capacity(&self) -> usize {
        self.circular_buffer.capacity()
    }

    pub fn is_empty(&self) -> bool {
        self.epoch.is_empty()
    }
}

impl ReaderFactory for SpatialBuffer {
    fn create_reader(&self, epoch_read: &Epoch) -> Result<Reader<'_>, anyhow::Error> {
        Ok(Reader::SpatialReader(SpatialBuffer::create_reader(
            self, epoch_read,
        )?))
    }

    fn epoch(&self) -> &Epoch {
        SpatialBuffer::epoch(self)
    }
}

pub struct WriterSpec<'a> {
    spatial_buffer: &'a mut SpatialBuffer,
    difference: Epoch,
    epoch_for_compute: Epoch,
}

impl<'a> WriterSpec<'a> {
    ///
    /// Unstable writer should be used for algorithms with data dependency.
    /// For example those that require previous value to compute next one, like EMA. Such algorithms
    /// cannot be computed from a random sample of superset. They require contiguous processing.
    ///
    /// Unstable writer simply fills whole difference between current and next epoch's.
    ///
    pub fn unstable_writer(self) -> SpatialWriter<'a> {
        SpatialWriter {
            spatial_buffer: self.spatial_buffer,
            add_ops: 0,
            epoch: self.difference,
            finished: false,
        }
    }

    ///
    /// Stable writer is for algorithms that can produce correct values for any random sample
    /// from dataset. For example simple moving average algorithm would produce correct values
    /// even if we would skip part of dataset.
    ///
    /// Stable writer will either compute difference if current epoch overlap with next epoch
    /// or skip the gap and compute only requested epoch.
    ///
    pub fn stable_writer(self) -> SpatialWriter<'a> {
        SpatialWriter {
            spatial_buffer: self.spatial_buffer,
            add_ops: 0,
            epoch: self.epoch_for_compute,
            finished: false,
        }
    }
}

pub struct SpatialWriter<'a> {
    spatial_buffer: &'a mut SpatialBuffer,
    add_ops: u64,
    epoch: Epoch,
    finished: bool,
}

impl SpatialWriter<'_> {
    pub fn write(&mut self, value: f64) {
        self.add_ops += 1;
        self.spatial_buffer.circular_buffer.push(value);
    }

    pub fn epoch(&self) -> &Epoch {
        &self.epoch
    }

    pub fn time_frame(&self) -> TimeFrame {
        self.epoch.time_frame()
    }

    pub fn finish(&mut self) -> Result<(), anyhow::Error> {
        self.finished = true;

        if self.add_ops != self.epoch.span() as u64 {
            Err(ContractError::WriterError(
                format!(
                    "Expected writes count({}) doesn't match actual writes count({})",
                    self.epoch.span(),
                    self.add_ops
                )
                .into(),
            )
            .into())
        } else {
            if self.spatial_buffer.epoch.end() == self.epoch.start() {
                let end = self.epoch.end();
                let start = end
                    .saturating_sub(self.spatial_buffer.capacity())
                    .max(self.spatial_buffer.epoch.start());

                self.spatial_buffer.epoch = Epoch::new(start..end, self.epoch.time_frame());
            } else {
                self.spatial_buffer.epoch = self.epoch.clone();
            }

            Ok(())
        }
    }
}

impl Drop for SpatialWriter<'_> {
    fn drop(&mut self) {
        if self.add_ops > 0 {
            assert!(self.finished);
        }
    }
}

pub struct SpatialReader<'a> {
    buffer: &'a CircularBuffer<f64>,
    epoch: Epoch,
    tail: usize,
}

impl SpatialReader<'_> {
    pub fn iter(&self, range: Range<usize>) -> SpatialReaderIter<'_> {
        debug_assert!(
            self.epoch.start() <= range.start && range.end <= self.epoch.end(),
            "Epoch read out of reader bounds, Reader({:?}), epoch_read({:?})",
            &self.epoch,
            range
        );

        let start = range.start - self.epoch.start();
        let end = start + range.end.saturating_sub(range.start);

        SpatialReaderIter {
            offset: self.tail,
            current: start,
            end,
            buffer: self.buffer,
        }
    }

    pub fn time_frame(&self) -> TimeFrame {
        self.epoch.time_frame()
    }

    pub fn get(&self, index: usize) -> &f64 {
        debug_assert!(
            self.epoch.contains(index),
            "Index({:?}) is out of epoch bounds({:?})",
            index,
            self.epoch
        );

        let norm_idx = index - self.epoch.start();
        let wrapped_idx = circular_buffer::wrap_index(self.tail + norm_idx, self.buffer.capacity());

        unsafe { self.buffer.get_direct(wrapped_idx) }
    }

    pub fn epoch(&self) -> &Epoch {
        &self.epoch
    }
}

pub struct SpatialReaderIter<'a> {
    buffer: &'a CircularBuffer<f64>,
    offset: usize,
    current: usize,
    end: usize,
}

impl<'a> Iterator for SpatialReaderIter<'a> {
    type Item = &'a f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let wrapped_idx =
                circular_buffer::wrap_index(self.current + self.offset, self.buffer.capacity());
            self.current += 1;
            Some(unsafe { self.buffer.get_direct(wrapped_idx) })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact_size = self.end.saturating_sub(self.current);
        (exact_size, Some(exact_size))
    }
}

impl<'b> DataReader for SpatialReader<'b> {
    type Iter<'a> = SpatialReaderIter<'a> where Self: 'a;

    fn iter(&self, range: Range<usize>) -> Self::Iter<'_> {
        SpatialReader::iter(self, range)
    }

    fn get(&self, index: usize) -> &f64 {
        SpatialReader::get(self, index)
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use crate::sim::bb::cursor::Epoch;
    use crate::sim::bb::time_frame::{SECOND_1, SECOND_5};
    use crate::sim::bb::{time_frame, ComputeId};
    use crate::sim::error::ContractError;
    use crate::sim::spatial_buffer::SpatialBuffer;

    #[test]
    fn should_read_by_index() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 4);
        let data = [1f64, 2f64, 3f64, 4f64];
        unstable_write(&mut spatial_buffer, 0..4, 1..5);

        let epochs = [0..2, 1..3, 2..4, 0..4];

        for epoch_read in epochs {
            let epoch_read = Epoch::new(epoch_read.clone(), SECOND_1);
            let reader = spatial_buffer.create_reader(&epoch_read).expect("reader");

            for i in epoch_read.as_range() {
                assert_eq!(data[i], *reader.get(i));
            }
        }
    }

    #[test]
    fn should_read_by_index_over_rotated_buffer() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 4);
        let data = [3f64, 4f64, 5f64, 6f64];

        unstable_write(&mut spatial_buffer, 0..4, 1..5);
        unstable_write(&mut spatial_buffer, 4..6, 5..7);

        let epochs = [2..4, 3..5, 4..6, 2..6];
        let offset = 2;

        for epoch_read in epochs {
            let epoch_read = Epoch::new(epoch_read.clone(), SECOND_1);
            let reader = spatial_buffer.create_reader(&epoch_read).expect("reader");

            for i in epoch_read.as_range() {
                assert_eq!(data[i - offset], *reader.get(i));
            }
        }
    }

    fn unstable_write(spatial_buffer: &mut SpatialBuffer, epoch: Range<usize>, data: Range<usize>) {
        let mut writer = spatial_buffer
            .create_writer(&Epoch::new(epoch.clone(), SECOND_1))
            .expect("writer")
            .unstable_writer();

        data.for_each(|i| writer.write(i as f64));
        writer.finish().unwrap();
    }

    #[test]
    fn should_iter_epoch() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 8);
        let data = [1f64, 2f64, 3f64, 4f64, 5f64, 6f64, 7f64, 8f64];
        unstable_write(&mut spatial_buffer, 0..8, 1..9);

        {
            let spatial_buffer_full_read = Epoch::new(0..8, SECOND_1);

            let reader = spatial_buffer
                .create_reader(&spatial_buffer_full_read)
                .expect("reader");
            let epochs = [0..2, 4..6, 6..8, 0..8];

            for epoch_read in epochs {
                let iter = reader.iter(epoch_read.clone());

                assert_eq!(
                    &data[epoch_read.clone()],
                    &reader.iter(epoch_read).copied().collect::<Vec<_>>()[..]
                );
            }
        }

        {
            let spatial_buffer_subset_read = Epoch::new(2..6, SECOND_1);

            let reader = spatial_buffer
                .create_reader(&spatial_buffer_subset_read)
                .expect("reader");
            let subsets_over_subset = [2..4, 3..5, 4..6, 2..6];

            for epoch_read in subsets_over_subset {
                let iter = reader.iter(epoch_read.clone());

                assert_eq!(
                    &data[epoch_read.clone()],
                    &reader.iter(epoch_read).copied().collect::<Vec<_>>()[..]
                );
            }
        }
    }

    #[test]
    fn should_iter_epoch_over_rotated_buffer() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 8);
        let data = [3f64, 4f64, 5f64, 6f64, 7f64, 8f64, 9f64, 10f64];

        unstable_write(&mut spatial_buffer, 0..8, 1..9);
        unstable_write(&mut spatial_buffer, 8..10, 9..11);

        let offset = 2;

        {
            let spatial_buffer_full_read = Epoch::new(2..10, SECOND_1);

            let reader = spatial_buffer
                .create_reader(&spatial_buffer_full_read)
                .expect("reader");
            let epochs = [2..4, 6..8, 8..10, 2..10];

            for epoch_read in epochs {
                let iter = reader.iter(epoch_read.clone());

                assert_eq!(
                    epoch_read
                        .clone()
                        .map(|i| data[i - offset])
                        .collect::<Vec<_>>()[..],
                    reader.iter(epoch_read).copied().collect::<Vec<_>>()[..]
                );
            }
        }

        {
            let spatial_buffer_subset_read = Epoch::new(4..8, SECOND_1);

            let reader = spatial_buffer
                .create_reader(&spatial_buffer_subset_read)
                .expect("reader");
            let subsets_over_subset = [4..6, 5..7, 6..8, 4..8];

            for epoch_read in subsets_over_subset {
                let iter = reader.iter(epoch_read.clone());

                assert_eq!(
                    epoch_read
                        .clone()
                        .map(|i| data[i - offset])
                        .collect::<Vec<_>>()[..],
                    reader.iter(epoch_read).copied().collect::<Vec<_>>()[..]
                );
            }
        }
    }

    #[test]
    fn should_write_epochs_stable() {
        let data = [1f64, 2f64, 3f64, 4f64];

        let start_till_middle = 0..2usize;
        let just_middle = 1..3usize;
        let middle_till_end = 2..4usize;
        let whole_range = 0..4usize;
        let epochs = [start_till_middle, just_middle, middle_till_end, whole_range];

        for epoch_write in epochs {
            let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 8);

            let epoch_write_exact = Epoch::new(epoch_write.clone(), SECOND_1);
            stable_write(&mut spatial_buffer, epoch_write_exact.as_range());

            assert_eq!(&epoch_write_exact, spatial_buffer.epoch());
        }
    }

    #[test]
    fn should_write_epochs_unstable() {
        let data = [1f64, 2f64, 3f64, 4f64];

        let start_till_middle = 0..2usize;
        let just_middle = 1..3usize;
        let middle_till_end = 2..4usize;
        let whole_range = 0..4usize;
        let epochs = [start_till_middle, just_middle, middle_till_end, whole_range];

        for epoch_write in epochs {
            let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 8);

            let epoch_write_all = Epoch::new(0..epoch_write.end, SECOND_1);
            unstable_write(
                &mut spatial_buffer,
                epoch_write_all.as_range(),
                epoch_write_all.as_range(),
            );

            assert_eq!(&epoch_write_all, spatial_buffer.epoch());
        }
    }

    #[test]
    fn should_write_epoch_stable_multiple_times() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 8);

        stable_write(&mut spatial_buffer, 0..7);
        stable_write(&mut spatial_buffer, 7..13);

        assert_eq!(&Epoch::new(5..13, SECOND_1), spatial_buffer.epoch());
    }

    #[test]
    fn should_write_epoch_unstable_multiple_times() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 8);

        unstable_write(&mut spatial_buffer, 0..7, 0..7);
        unstable_write(&mut spatial_buffer, 7..13, 7..13);

        assert_eq!(&Epoch::new(5..13, SECOND_1), spatial_buffer.epoch());
    }

    #[test]
    fn should_skip_gap_on_stable_write() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 8);

        stable_write(&mut spatial_buffer, 0..7);
        stable_write(&mut spatial_buffer, 8..13);

        assert_eq!(&Epoch::new(8..13, SECOND_1), spatial_buffer.epoch());
    }

    #[test]
    fn should_fill_gap_on_unstable_write() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 8);

        unstable_write(&mut spatial_buffer, 0..7, 0..7);

        {
            let mut writer = spatial_buffer
                .create_writer(&Epoch::new(8..13, SECOND_1))
                .expect("writer")
                .unstable_writer();

            // cover gap
            (7..13).for_each(|i| writer.write(i as f64));
            writer.finish().unwrap();
        }

        assert_eq!(&Epoch::new(5..13, SECOND_1), spatial_buffer.epoch());
    }

    fn stable_write(spatial_buffer: &mut SpatialBuffer, range: Range<usize>) {
        let mut writer = spatial_buffer
            .create_writer(&Epoch::new(range.clone(), SECOND_1))
            .expect("writer")
            .stable_writer();

        range.for_each(|i| writer.write(i as f64));
        writer.finish().unwrap();
    }

    #[test]
    fn should_not_write_already_processed_epoch() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 32);

        stable_write(&mut spatial_buffer, 0..20);

        {
            let mut writer = spatial_buffer
                .create_writer(&Epoch::new(0..20, SECOND_1))
                .expect("writer")
                .stable_writer();
            writer.finish().unwrap();
            assert_eq!(Epoch::new(20..20, SECOND_1), writer.epoch);
        }

        assert_eq!(Epoch::new(0..20, SECOND_1), spatial_buffer.epoch);
    }

    #[test]
    fn should_not_write_subset_of_already_processed_epoch() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 32);

        stable_write(&mut spatial_buffer, 0..20);

        {
            let mut writer = spatial_buffer
                .create_writer(&Epoch::new(10..18, SECOND_1))
                .expect("writer")
                .stable_writer();
            writer.finish().unwrap();
            assert_eq!(Epoch::new(20..20, SECOND_1), writer.epoch);
        }

        assert_eq!(Epoch::new(0..20, SECOND_1), spatial_buffer.epoch);
    }

    #[test]
    fn should_write_difference() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 32);

        stable_write(&mut spatial_buffer, 0..20);

        {
            let mut writer = spatial_buffer
                .create_writer(&Epoch::new(12..22, SECOND_1))
                .expect("writer")
                .stable_writer();

            writer.write(20f64);
            writer.write(21f64);
            writer.finish().unwrap();

            assert_eq!(Epoch::new(20..22, SECOND_1), writer.epoch);
        }

        assert_eq!(Epoch::new(0..22, SECOND_1), spatial_buffer.epoch);
    }

    #[test]
    fn should_write_difference_and_rotate() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 32);

        stable_write(&mut spatial_buffer, 0..20);

        {
            let mut writer = spatial_buffer
                .create_writer(&Epoch::new(16..34, SECOND_1))
                .expect("writer")
                .stable_writer();

            (20..34).for_each(|i| writer.write(i as f64));
            writer.finish().unwrap();

            assert_eq!(Epoch::new(20..34, SECOND_1), writer.epoch);
        }

        assert_eq!(Epoch::new(2..34, SECOND_1), spatial_buffer.epoch);
    }

    #[test]
    fn should_err_on_epoch_time_frame_mismatch_in_writer() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 32);

        let error = spatial_buffer
            .create_writer(&Epoch::new(0..0, SECOND_5))
            .err()
            .unwrap();

        match error
            .downcast_ref::<ContractError>()
            .expect("ContractError")
        {
            ContractError::TimeFrameMismatchError(_) => {}
            _ => panic!("Expected error TimeFrameMismatch, but actual was different"),
        }
    }

    #[test]
    fn should_err_on_stale_write_beyond_spatial_buffer_epoch_start() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 32);

        stable_write(&mut spatial_buffer, 0..32);
        stable_write(&mut spatial_buffer, 32..64);

        let error = spatial_buffer
            .create_writer(&Epoch::new(31..32, SECOND_1))
            .err()
            .unwrap();

        match error
            .downcast_ref::<ContractError>()
            .expect("ContractError")
        {
            ContractError::WriterError(err_msg) => {
                assert_eq!("Epoch to write(Epoch { range: 31..32, time_frame: TimeFrame(1000) }) \
                            cannot be older than current epoch of this SpatialBuffer(Epoch { range: 32..64, time_frame: TimeFrame(1000) })", 
                           err_msg
                );
            }
            _ => panic!("Expected error WriterError, but actual was different"),
        }
    }

    #[test]
    fn should_err_on_spatial_buffer_overflow() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 32);

        let error = spatial_buffer
            .create_writer(&Epoch::new(0..33, SECOND_1))
            .err()
            .unwrap();

        match error
            .downcast_ref::<ContractError>()
            .expect("ContractError")
        {
            ContractError::WriterError(err_msg) => {
                assert_eq!("Epoch to write(Epoch { range: 0..33, time_frame: TimeFrame(1000) }) overflows SpatialBuffer capacity(32)",
                           err_msg
                );
            }
            _ => panic!("Expected error WriterError, but actual was different"),
        }
    }

    #[test]
    fn should_err_on_difference_overflow() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 32);

        stable_write(&mut spatial_buffer, 0..32);

        let error = spatial_buffer
            .create_writer(&Epoch::new(33..65, SECOND_1))
            .err()
            .unwrap();

        match error
            .downcast_ref::<ContractError>()
            .expect("ContractError")
        {
            ContractError::WriterError(err_msg) => {
                assert_eq!("The span of difference(Epoch { range: 32..65, time_frame: TimeFrame(1000) }) \
                            between current epoch and epoch to write overflows SpatialBuffer capacity(32)",
                           err_msg
                );
            }
            _ => panic!("Expected error WriterError, but actual was different"),
        }
    }

    #[test]
    fn should_write_epochs_that_match_capacity() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 32);

        stable_write(&mut spatial_buffer, 0..32);
        stable_write(&mut spatial_buffer, 32..64);
        stable_write(&mut spatial_buffer, 64..96);

        assert_eq!(Epoch::new(64..96, SECOND_1), spatial_buffer.epoch);
    }

    #[test]
    fn should_err_on_time_frame_mismatch_in_reader() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 32);

        let error = spatial_buffer
            .create_reader(&Epoch::new(0..32, SECOND_5))
            .err()
            .unwrap();

        match error
            .downcast_ref::<ContractError>()
            .expect("ContractError")
        {
            ContractError::TimeFrameMismatchError(err_msg) => {
                assert_eq!(
                    "Cannot assess superset due to time frame mismatch, \
                            lhs(Epoch { range: 0..0, time_frame: TimeFrame(1000) }), \
                            rhs(Epoch { range: 0..32, time_frame: TimeFrame(5000) })",
                    err_msg
                );
            }
            _ => panic!("Expected error TimeFrameMismatchError, but actual was different"),
        }
    }

    #[test]
    fn should_err_on_epoch_read_out_of_spatial_bounds() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 32);

        stable_write(&mut spatial_buffer, 0..32);

        let error = spatial_buffer
            .create_reader(&Epoch::new(20..33, SECOND_1))
            .err()
            .unwrap();

        match error
            .downcast_ref::<ContractError>()
            .expect("ContractError")
        {
            ContractError::ReaderError(err_msg) => {
                assert_eq!(
                    "Epoch read out of SpatialBuffer bounds, \
                    SpatialBuffer(Epoch { range: 0..32, time_frame: TimeFrame(1000) }), \
                    epoch_read(Epoch { range: 20..33, time_frame: TimeFrame(1000) })",
                    err_msg
                );
            }
            _ => panic!("Expected error ReaderError, but actual was different"),
        }
    }

    #[test]
    fn should_read_whole_epoch_of_spatial_buffer() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 32);

        stable_write(&mut spatial_buffer, 0..32);

        {
            let mut writer = spatial_buffer
                .create_writer(&Epoch::new(20..48, SECOND_1))
                .expect("writer")
                .stable_writer();

            assert_eq!(Epoch::new(32..48, SECOND_1), writer.epoch);
            (32..48).for_each(|i| writer.write(i as f64));
            writer.finish().unwrap();
        }

        let reader = spatial_buffer
            .create_reader(&Epoch::new(16..48, SECOND_1))
            .unwrap();

        assert!((16..48).all(|i| *reader.get(i) == i as f64));
    }

    #[test]
    fn should_read_few_numbers_from_beginning() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 32);

        stable_write(&mut spatial_buffer, 0..32);
        stable_write(&mut spatial_buffer, 32..39);

        let reader = spatial_buffer
            .create_reader(&Epoch::new(7..11, SECOND_1))
            .unwrap();

        assert!((7..11).all(|i| *reader.get(i) == i as f64));
    }

    #[test]
    fn should_read_few_numbers_from_the_middle() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 32);

        stable_write(&mut spatial_buffer, 0..32);
        stable_write(&mut spatial_buffer, 32..39);

        let reader = spatial_buffer
            .create_reader(&Epoch::new(19..27, SECOND_1))
            .unwrap();

        assert!((19..27).all(|i| *reader.get(i) == i as f64));
    }

    #[test]
    fn should_read_few_numbers_from_the_end() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 32);

        stable_write(&mut spatial_buffer, 0..32);
        stable_write(&mut spatial_buffer, 32..39);

        let reader = spatial_buffer
            .create_reader(&Epoch::new(32..39, SECOND_1))
            .unwrap();

        assert!((32..39).all(|i| *reader.get(i) == i as f64));
    }

    #[test]
    fn should_err_on_writes_count_mismatch() {
        let mut spatial_buffer = SpatialBuffer::new(SECOND_1, 32);

        let mut writer = spatial_buffer
            .create_writer(&Epoch::new(0..2, SECOND_1))
            .unwrap()
            .stable_writer();

        writer.write(1f64);
        let error = writer.finish().err().unwrap();

        match error
            .downcast_ref::<ContractError>()
            .expect("ContractError")
        {
            ContractError::WriterError(err_msg) => {
                assert_eq!(
                    "Expected writes count(2) doesn't match actual writes count(1)",
                    err_msg
                );
            }
            _ => panic!("Expected error WriterError, but actual was different"),
        }
    }
}
