/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use std::ops::{Deref, Index, Range};
use std::slice::Iter;
use std::sync::Arc;
use std::time::Duration;

use crate::sim::bb::time_frame::TimeFrame;

#[derive(Debug, Copy, Clone)]
pub enum OfferSide {
    Ask,
    Bid,
}

impl OfferSide {
    #[inline]
    pub fn price(&self, tick_quotes: &TickQuotes, index: usize) -> f64 {
        match self {
            OfferSide::Ask => tick_quotes.ask[index],
            OfferSide::Bid => tick_quotes.bid[index],
        }
    }
}

pub struct TickQuotes {
    ask: Vec<f64>,
    bid: Vec<f64>,
    timestamp: TimestampVector,
}

impl TickQuotes {
    pub fn new(ask: Vec<f64>, bid: Vec<f64>, timestamp: TimestampVector) -> Self {
        assert_eq!(ask.len(), bid.len());
        assert_eq!(ask.len(), timestamp.len());

        Self {
            ask,
            bid,
            timestamp,
        }
    }

    pub fn ask(&self, index: usize) -> f64 {
        self.ask[index]
    }

    pub fn ask_array(&self) -> &[f64] {
        &self.ask
    }

    pub fn bid(&self, index: usize) -> f64 {
        self.bid[index]
    }

    pub fn bid_array(&self) -> &[f64] {
        &self.bid
    }

    pub fn timestamp(&self, index: usize) -> u64 {
        self.timestamp.index(index)
    }

    pub fn timestamp_vector(&self) -> &TimestampVector {
        &self.timestamp
    }

    pub fn len(&self) -> usize {
        self.ask.len()
    }
}

pub struct FrameQuotes {
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    timestamp: TimestampVector,
    time_frame: TimeFrame,
}

impl FrameQuotes {
    pub fn new(
        open: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
        timestamp: TimestampVector,
        time_frame: TimeFrame,
    ) -> Self {
        assert_eq!(open.len(), high.len());
        assert_eq!(open.len(), low.len());
        assert_eq!(open.len(), close.len());
        assert_eq!(open.len(), timestamp.len());

        Self {
            open,
            high,
            low,
            close,
            timestamp,
            time_frame,
        }
    }

    pub fn open(&self, index: usize) -> f64 {
        self.open[index]
    }

    pub fn open_array(&self) -> &[f64] {
        &self.open
    }

    pub fn high(&self, index: usize) -> f64 {
        self.high[index]
    }

    pub fn high_array(&self) -> &[f64] {
        &self.high
    }

    pub fn low(&self, index: usize) -> f64 {
        self.low[index]
    }

    pub fn low_array(&self) -> &[f64] {
        &self.low
    }

    pub fn close(&self, index: usize) -> f64 {
        self.close[index]
    }

    pub fn close_array(&self) -> &[f64] {
        &self.close
    }

    pub fn timestamp(&self, index: usize) -> u64 {
        self.timestamp.index(index)
    }

    pub fn timestamp_vector(&self) -> &TimestampVector {
        &self.timestamp
    }

    pub fn time_frame(&self) -> TimeFrame {
        self.time_frame
    }

    pub fn len(&self) -> usize {
        self.open.len()
    }
}

pub struct TimestampVector {
    timestamp: Vec<u64>,
    offset: u64,
}

impl TimestampVector {
    pub fn new(timestamp: Vec<u64>, offset: Duration) -> Self {
        let offset = u64::try_from(offset.as_millis()).expect("TZ offset should fit in u64");
        Self { timestamp, offset }
    }

    pub fn from_utc(timestamp: Vec<u64>) -> TimestampVector {
        TimestampVector {
            timestamp,
            offset: 0,
        }
    }

    pub fn get(&self, index: usize) -> Option<u64> {
        self.timestamp.get(index).map(|ts| ts + self.offset)
    }

    pub fn index(&self, index: usize) -> u64 {
        self.timestamp[index] + self.offset
    }

    pub fn utc(&self, index: usize) -> u64 {
        self.timestamp[index]
    }

    pub fn offset(&self) -> Duration {
        Duration::from_millis(self.offset)
    }

    pub fn len(&self) -> usize {
        self.timestamp.len()
    }
}

pub struct BaseQuotes(Arc<FrameQuotes>);

impl BaseQuotes {
    pub fn new(frame_quotes: Arc<FrameQuotes>) -> Self {
        Self(frame_quotes)
    }
}

impl Deref for BaseQuotes {
    type Target = Arc<FrameQuotes>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct Timestamp {
    timestamp: u64,
    offset: u64,
}

impl Timestamp {
    pub fn new(timestamp: u64, offset: u64) -> Self {
        Self { timestamp, offset }
    }

    pub fn time(&self) -> u64 {
        self.timestamp + self.offset
    }

    pub fn utc(&self) -> u64 {
        self.timestamp
    }
}
