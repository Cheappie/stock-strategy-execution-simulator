/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use std::ops::{Index, Range};

use crate::sim::bb::time_frame::TimeFrame;
pub use crate::sim::tlb::direct_translation_buffer::DirectTranslationBuffer;
pub use crate::sim::tlb::identity_translation_buffer::IdentityTranslationBuffer;

mod direct_base_lifetime_translation_buffer;
mod direct_lifetime_translation_buffer;
mod direct_translation_buffer;
mod identity_translation_buffer;
mod waterfall_base_lifetime_translation_buffer;
mod waterfall_lifetime_translation_buffer;

pub use direct_base_lifetime_translation_buffer::{
    DirectBaseLifetimeTranslation, DirectBaseLifetimeTranslationBuffer,
    DirectBaseLifetimeTranslationIter,
};
pub use direct_lifetime_translation_buffer::{
    DirectLifetimeTranslation, DirectLifetimeTranslationBuffer, DirectLifetimeTranslationIter,
};
pub use waterfall_base_lifetime_translation_buffer::{
    WaterfallBaseLifetimeTranslation, WaterfallBaseLifetimeTranslationBuffer,
    WaterfallBaseLifetimeTranslationIter,
};
pub use waterfall_lifetime_translation_buffer::{
    WaterfallLifetimeTranslation, WaterfallLifetimeTranslationBuffer,
    WaterfallLifetimeTranslationIter,
};

///
/// Provides time frame of sample contained by this buffer
///
pub trait TranslationUnitDescriptor {
    fn time_frame(&self) -> TimeFrame;
}

/// InlinedTranslation (branchless)
///
/// Translates from base index to index of *formed* frame.
///
/// Example, translate from 1 minute base to 5 minute:
/// * 1 minute data points between 00:50:00(inclusive) till 00:55:00(exclusive)
///   will point to M5 data point closed at 00:50:00
/// * 1 minute data points between 00:55:00(inclusive) till 01:00:00(exclusive)
///   will point to M5 data point closed at 00:55:00
///
pub trait InlinedTranslation: TranslationUnitDescriptor {
    fn translate(&self, index: usize) -> usize;
}

/// InlinedReverseTranslation (branchless)
///
/// Translates from frame index to the first base index of current frame quote.
///
/// Example, translate from 5 minute to 1 minute base:
/// * M5 data point formed between 00:45:00(exclusive) till 00:50:00(inclusive)
///   will point to M1 data point closed at 00:46:00
/// * M5 data point formed between 00:50:00(exclusive) till 00:55:00(inclusive)
///   will point to M1 data point closed at 00:51:00
///
/// As n+1 sample we should add len of base quotes to simplify finding end of frame in base indices.
///
pub trait InlinedReverseTranslation: TranslationUnitDescriptor {
    /// Returns creation lifetime start for given index
    ///
    /// Creation lifetime describes span where given index was under construction in base quotes
    ///
    fn reverse_creation_lifetime(&self, index: usize) -> usize;

    /// Returns creation lifetime defined in base quotes
    fn creation_lifetime(&self, index: usize) -> Range<usize> {
        let creation_start_stamp = self.reverse_creation_lifetime(index);
        let creation_end_stamp = self.reverse_creation_lifetime(index + 1);
        creation_start_stamp..creation_end_stamp
    }

    /// Returns validity lifetime start for given index
    ///
    /// Validity lifetime describes span where given index is valid for in base quotes
    ///
    fn reverse_validity_lifetime(&self, index: usize) -> usize;

    /// Returns validity lifetime defined in base quotes.
    ///
    /// Validity lifetime might be empty if subsequent bar superseded previous, it happens
    /// when subsequent bar is terminal quote and previous is non terminal quote.
    ///
    fn validity_lifetime(&self, index: usize) -> Range<usize> {
        let validity_start_stamp = self.reverse_validity_lifetime(index);
        let validity_end_stamp = self.reverse_validity_lifetime(index + 1);
        validity_start_stamp..validity_end_stamp
    }
}

pub enum FrameTranslationBuffer {
    IdentityTranslationBuffer(IdentityTranslationBuffer),
    DirectTranslationBuffer(DirectTranslationBuffer),
}

impl FrameTranslationBuffer {
    pub fn time_frame(&self) -> TimeFrame {
        match self {
            FrameTranslationBuffer::IdentityTranslationBuffer(tlb) => tlb.time_frame(),
            FrameTranslationBuffer::DirectTranslationBuffer(tlb) => tlb.time_frame(),
        }
    }

    pub fn translate(&self, index: usize) -> usize {
        match self {
            FrameTranslationBuffer::IdentityTranslationBuffer(tlb) => tlb.translate(index),
            FrameTranslationBuffer::DirectTranslationBuffer(tlb) => tlb.translate(index),
        }
    }

    pub fn identity(&self) -> Option<&IdentityTranslationBuffer> {
        match self {
            FrameTranslationBuffer::IdentityTranslationBuffer(identity) => Some(identity),
            FrameTranslationBuffer::DirectTranslationBuffer(_) => None,
        }
    }

    pub fn direct(&self) -> Option<&DirectTranslationBuffer> {
        match self {
            FrameTranslationBuffer::IdentityTranslationBuffer(_) => None,
            FrameTranslationBuffer::DirectTranslationBuffer(direct) => Some(direct),
        }
    }
}

pub struct LifetimeVector {
    lifetime: Vec<u32>,
}

impl LifetimeVector {
    pub fn new(lifetime: Vec<u32>) -> Self {
        Self { lifetime }
    }

    pub fn get_checked(&self, index: usize) -> usize {
        self.lifetime[index] as usize
    }

    pub fn lifetime(&self, index: usize) -> Range<usize> {
        self.get_checked(index)..self.get_checked(index + 1)
    }
}
