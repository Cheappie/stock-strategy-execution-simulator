/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::cursor::Epoch;
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::primitive_buffer::StaticReader;
use crate::sim::spatial_buffer::SpatialReader;
use std::iter::Map;
use std::ops::Range;

pub trait ReaderFactory {
    fn create_reader(&self, epoch_read: &Epoch) -> Result<Reader<'_>, anyhow::Error>;

    fn epoch(&self) -> &Epoch;

    fn time_frame(&self) -> TimeFrame {
        self.epoch().time_frame()
    }
}

pub trait DataReader {
    type Iter<'a>: Iterator<Item = &'a f64>
    where
        Self: 'a;

    fn iter(&self, range: Range<usize>) -> Self::Iter<'_>;

    fn get(&self, index: usize) -> &f64;
}

pub enum Reader<'a> {
    SpatialReader(SpatialReader<'a>),
    StaticReader(StaticReader<'a>),
}
