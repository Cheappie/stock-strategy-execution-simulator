/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::tlb::{InlinedReverseTranslation, InlinedTranslation, TranslationUnitDescriptor};

pub struct IdentityTranslationBuffer(TimeFrame);

impl IdentityTranslationBuffer {
    pub fn new(time_frame: TimeFrame) -> Self {
        Self(time_frame)
    }
}

impl InlinedTranslation for IdentityTranslationBuffer {
    fn translate(&self, index: usize) -> usize {
        index
    }
}

impl TranslationUnitDescriptor for IdentityTranslationBuffer {
    fn time_frame(&self) -> TimeFrame {
        self.0
    }
}
