/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::tlb::{
    InlinedReverseTranslation, InlinedTranslation, LifetimeVector, TranslationUnitDescriptor,
};

pub struct DirectTranslationBuffer {
    time_frame: TimeFrame,
    base_to_target: Vec<u32>,
    creation_lifetime: LifetimeVector,
    validity_lifetime: LifetimeVector,
}

impl DirectTranslationBuffer {
    pub fn new(
        time_frame: TimeFrame,
        base_to_target: Vec<u32>,
        creation_lifetime: LifetimeVector,
        validity_lifetime: LifetimeVector,
    ) -> Self {
        DirectTranslationBuffer {
            time_frame,
            base_to_target,
            creation_lifetime,
            validity_lifetime,
        }
    }
}

impl InlinedTranslation for DirectTranslationBuffer {
    fn translate(&self, index: usize) -> usize {
        self.base_to_target[index] as usize
    }
}

impl InlinedReverseTranslation for DirectTranslationBuffer {
    fn reverse_creation_lifetime(&self, index: usize) -> usize {
        self.creation_lifetime.get_checked(index)
    }

    fn reverse_validity_lifetime(&self, index: usize) -> usize {
        self.validity_lifetime.get_checked(index)
    }
}

impl TranslationUnitDescriptor for DirectTranslationBuffer {
    fn time_frame(&self) -> TimeFrame {
        self.time_frame
    }
}
