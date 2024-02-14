/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use std::collections::BTreeMap;
use std::ops::Deref;
use std::sync::{Arc, LockResult, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockResult};

use anyhow::anyhow;

use crate::sim::error::SetupError;

use super::bb::cursor::Cursor;
use super::bb::quotes::{BaseQuotes, FrameQuotes};
use super::bb::time_frame::TimeFrame;
use super::tlb::FrameTranslationBuffer;

pub type SessionContextRef = Arc<SessionContext>;

///
/// SpanContext aka evaluation of partition within dataset for strategy
///
pub struct SpanContext {
    cursor: Cursor,
    session_context: SessionContextRef,
}

impl SpanContext {
    pub fn new(cursor: Cursor, session_context: SessionContextRef) -> Self {
        Self {
            cursor,
            session_context,
        }
    }

    pub fn cursor(&self) -> &Cursor {
        &self.cursor
    }

    pub fn session(&self) -> &SessionContext {
        &self.session_context
    }
}

///
/// Strategy evaluation context
///
pub struct SessionContext {
    instrument: String,
    base_quotes: Arc<BaseQuotes>,
    frame_quotes: BTreeMap<TimeFrame, Arc<FrameQuotes>>,
    translation_buffers: BTreeMap<TimeFrame, Arc<FrameTranslationBuffer>>,
    configuration: SessionConfiguration,
}

impl SessionContext {
    pub fn new(
        instrument: String,
        base_quotes: Arc<BaseQuotes>,
        frame_quotes: BTreeMap<TimeFrame, Arc<FrameQuotes>>,
        translation_buffers: BTreeMap<TimeFrame, Arc<FrameTranslationBuffer>>,
        configuration: SessionConfiguration,
    ) -> Self {
        Self {
            instrument,
            base_quotes,
            frame_quotes,
            translation_buffers,
            configuration,
        }
    }
}

impl SessionContext {
    pub fn frame_quotes(&self, time_frame: TimeFrame) -> Result<Arc<FrameQuotes>, anyhow::Error> {
        self.frame_quotes
            .get(&time_frame)
            .map(|fq| Arc::clone(fq))
            .ok_or_else(|| SetupError::FrameQuotesNotCreated(time_frame).into())
    }

    pub fn translation_buffer(
        &self,
        time_frame: TimeFrame,
    ) -> Result<Arc<FrameTranslationBuffer>, anyhow::Error> {
        self.translation_buffers
            .get(&time_frame)
            .map(|ftb| Arc::clone(ftb))
            .ok_or_else(|| SetupError::FrameTranslationBufferNotCreated(time_frame).into())
    }

    pub fn configuration(&self) -> &SessionConfiguration {
        &self.configuration
    }
}

pub struct SessionConfiguration {
    step: usize,
    buffer_capacity: usize,
}

impl SessionConfiguration {
    pub fn new(step: usize, buffer_capacity: usize) -> Self {
        debug_assert!(buffer_capacity >= step);
        debug_assert!(buffer_capacity.is_power_of_two());

        Self {
            step,
            buffer_capacity,
        }
    }
}

impl SessionConfiguration {
    pub fn step(&self) -> usize {
        self.step
    }

    pub fn buffer_capacity(&self) -> usize {
        self.buffer_capacity
    }
}
