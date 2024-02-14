/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::cursor::Epoch;
use crate::sim::error::ContractError;
use crate::sim::reader::{DataReader, Reader, ReaderFactory};
use std::ops::Range;

pub struct PrimitiveBuffer<'a> {
    buf: &'a [f64],
    epoch: Epoch,
}

impl<'a> PrimitiveBuffer<'a> {
    pub fn new(buf: &'a [f64], epoch: Epoch) -> Self {
        assert!(epoch.start() == 0 && epoch.end() == buf.len());
        Self { buf, epoch }
    }
}

impl ReaderFactory for PrimitiveBuffer<'_> {
    fn create_reader(&self, epoch_read: &Epoch) -> Result<Reader<'_>, anyhow::Error> {
        if Epoch::is_superset(&self.epoch, epoch_read)? {
            Ok(Reader::StaticReader(StaticReader::new(
                self.buf,
                self.epoch.clone(),
            )))
        } else {
            Err(ContractError::ReaderError(
                format!(
                    "Epoch read out of PrimitiveBuffer bounds, PrimitiveBuffer({:?}), epoch_read({:?})",
                    &self.epoch, epoch_read
                ).into(),
            ).into())
        }
    }

    fn epoch(&self) -> &Epoch {
        &self.epoch
    }
}

pub struct StaticReader<'a> {
    buf: &'a [f64],
    epoch: Epoch,
}

impl<'a> StaticReader<'a> {
    pub fn new(buf: &'a [f64], epoch: Epoch) -> Self {
        assert!(epoch.start() == 0 && epoch.end() == buf.len());
        Self { buf, epoch }
    }
}

impl<'b> DataReader for StaticReader<'b> {
    type Iter<'a> = std::slice::Iter<'a, f64> where Self: 'a;

    fn iter(&self, range: Range<usize>) -> Self::Iter<'_> {
        self.buf[range].iter()
    }

    fn get(&self, index: usize) -> &f64 {
        &self.buf[index]
    }
}
