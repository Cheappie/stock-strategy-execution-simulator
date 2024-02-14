/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use std::borrow::Cow;

use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::bb::ComputeId;

/// TODO - make errors more precise, create more groups of errors

#[derive(thiserror::Error, Debug)]
pub enum SetupError {
    #[error("Frame quotes weren't created for timeframe: {0:?}")]
    FrameQuotesNotCreated(TimeFrame),
    #[error("Frame translation buffer wasn't created for timeframe: {0:?}")]
    FrameTranslationBufferNotCreated(TimeFrame),
    #[error("Indicator requested by lane selector doesn't exist")]
    IndicatorNotCreated,
    #[error("Lane for selector({selector}) doesn't exist for compute_id({producer_compute_id:?}), check data flow assembly pipeline")]
    LaneNotCreated {
        producer_compute_id: ComputeId,
        selector: usize,
    },
    #[error("There is no lane read handle in dynamic store for selector({selector}) and receiver({receiver_compute_id:?})")]
    LaneReadHandleNotCreated {
        receiver_compute_id: ComputeId,
        selector: usize,
    },
    #[error("Time frame mismatch({0})")]
    TimeFrameMismatch(Cow<'static, str>),
    #[error("Strategy contains processor with time_frame({lower:?}) lower than base quotes time_frame({base:?})")]
    TimeFrameLowerThanBase { base: TimeFrame, lower: TimeFrame },
}

#[derive(thiserror::Error, Debug)]
pub enum ASTError {
    #[error("MathExprNodeError: {0}")]
    MathExprNodeError(&'static str),
    #[error("WideMathExprNodeError: {0}")]
    WideMathExprNodeError(&'static str),
    #[error("PredicateExprNodeError: {0}")]
    PredicateExprNodeError(&'static str),
}

#[derive(thiserror::Error, Debug)]
pub enum ExecutionError {
    #[error("NotInitializedError: {0}")]
    NotInitializedError(&'static str),
}

#[derive(thiserror::Error, Debug)]
pub enum ContractError {
    #[error("CursorError: {0}")]
    CursorError(Cow<'static, str>),
    #[error("TimeFrameMismatchError: {0}")]
    TimeFrameMismatchError(Cow<'static, str>),
    #[error("ReaderError: {0}")]
    ReaderError(Cow<'static, str>),
    #[error("WriterError: {0}")]
    WriterError(Cow<'static, str>),
    #[error("TranslationStrategyError: {0}")]
    TranslationStrategyError(Cow<'static, str>),
    #[error("DirectLifetimeTranslationBufferError: {0}")]
    DirectLifetimeTranslationBufferError(Cow<'static, str>),
    #[error("WaterfallLifetimeTranslationBufferError: {0}")]
    WaterfallLifetimeTranslationBufferError(Cow<'static, str>),
    #[error("MediatorError: {0}")]
    MediatorError(Cow<'static, str>),
}

#[derive(thiserror::Error, Debug)]
pub enum BoundsError {
    #[error("Look back attempt has fallen out of bounds")]
    LookBackOutOfBounds,
    #[error("Cursor has been expanded beyond bounds")]
    CursorHasBeenExpandedBeyondBounds,
}

#[derive(thiserror::Error, Debug)]
pub enum DefinitionError {
    #[error("Unrecognized selector('{unrecognized}') for component('{component}')")]
    UnrecognizedSelector {
        unrecognized: String,
        component: &'static str,
    },
    #[error("Unrecognized selector ordinal('{unrecognized}') for component('{component}')")]
    UnrecognizedSelectorOrdinal {
        unrecognized: usize,
        component: &'static str,
    },
}
