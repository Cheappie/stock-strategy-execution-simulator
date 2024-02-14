/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::cursor::Cursor;
use crate::sim::bb::time_frame::TimeFrame;
use crate::sim::error::{ContractError, SetupError};

///
/// Selects translation strategy for given time frames.
///
pub fn select_translation_strategy(
    cursor: &Cursor,
    left: TimeFrame,
    right: TimeFrame,
) -> Result<TranslationStrategy, anyhow::Error> {
    let min_time_frame = left.min(right);

    if min_time_frame < cursor.time_frame() {
        Err(SetupError::TimeFrameLowerThanBase {
            base: cursor.time_frame(),
            lower: min_time_frame,
        }
        .into())
    } else {
        if left == right {
            Ok(TranslationStrategyDescriptor::direct(left, right))
        } else {
            if min_time_frame == cursor.time_frame() {
                Ok(TranslationStrategyDescriptor::direct_base_lifetime(
                    cursor, left, right,
                ))
            } else if TimeFrame::aligned(left, right) {
                Ok(TranslationStrategyDescriptor::direct_lifetime(
                    cursor, left, right,
                ))
            } else {
                let output = TimeFrame::find_common_divisor(left, right)
                    .filter(|common_divisor| *common_divisor >= cursor.time_frame())
                    .ok_or_else(|| ContractError::TranslationStrategyError(
                        format!("There is no compatible output time frame for waterfall lifetime translation, left({:?}), right({:?}), base({:?})",
                                left, right, cursor.time_frame()).into())
                    )?;

                if output == cursor.time_frame() {
                    Ok(TranslationStrategyDescriptor::waterfall_base_strategy(
                        cursor, left, right, output,
                    ))
                } else {
                    Ok(TranslationStrategyDescriptor::waterfall_strategy(
                        cursor, left, right, output,
                    ))
                }
            }
        }
    }
}

pub enum TranslationStrategy {
    Direct(TranslationStrategyDescriptor),
    DirectLifetime(TranslationStrategyDescriptor),
    DirectBaseLifetime(TranslationStrategyDescriptor),
    WaterfallLifetime(TranslationStrategyDescriptor),
    WaterfallBaseLifetime(TranslationStrategyDescriptor),
}

impl TranslationStrategy {
    pub fn descriptor(&self) -> &TranslationStrategyDescriptor {
        match self {
            TranslationStrategy::Direct(descriptor) => descriptor,
            TranslationStrategy::DirectLifetime(descriptor) => descriptor,
            TranslationStrategy::DirectBaseLifetime(descriptor) => descriptor,
            TranslationStrategy::WaterfallLifetime(descriptor) => descriptor,
            TranslationStrategy::WaterfallBaseLifetime(descriptor) => descriptor,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TranslationStrategyDescriptor {
    left: TimeFrame,
    right: TimeFrame,
    output: TimeFrame,
}

impl TranslationStrategyDescriptor {
    pub fn left(&self) -> TimeFrame {
        self.left
    }

    pub fn right(&self) -> TimeFrame {
        self.right
    }

    pub fn output(&self) -> TimeFrame {
        self.output
    }

    fn direct(left: TimeFrame, right: TimeFrame) -> TranslationStrategy {
        debug_assert_eq!(left, right);
        TranslationStrategy::Direct(Self {
            left,
            right,
            output: left,
        })
    }

    fn direct_lifetime(cursor: &Cursor, left: TimeFrame, right: TimeFrame) -> TranslationStrategy {
        debug_assert_ne!(left, right);
        debug_assert_ne!(cursor.time_frame(), left.min(right));
        debug_assert!(TimeFrame::aligned(left, right));
        TranslationStrategy::DirectLifetime(Self {
            left,
            right,
            output: left.min(right),
        })
    }

    pub fn direct_base_lifetime(
        cursor: &Cursor,
        left: TimeFrame,
        right: TimeFrame,
    ) -> TranslationStrategy {
        debug_assert_eq!(cursor.time_frame(), left.min(right));
        debug_assert_ne!(left, right);
        debug_assert!(TimeFrame::aligned(left, right));
        TranslationStrategy::DirectBaseLifetime(Self {
            left,
            right,
            output: cursor.time_frame(),
        })
    }

    fn waterfall_strategy(
        cursor: &Cursor,
        left: TimeFrame,
        right: TimeFrame,
        output: TimeFrame,
    ) -> TranslationStrategy {
        debug_assert!(!TimeFrame::aligned(left, right));
        debug_assert_ne!(cursor.time_frame(), left.min(right));
        debug_assert_ne!(cursor.time_frame(), output);
        TranslationStrategy::WaterfallLifetime(Self {
            left,
            right,
            output,
        })
    }

    fn waterfall_base_strategy(
        cursor: &Cursor,
        left: TimeFrame,
        right: TimeFrame,
        output: TimeFrame,
    ) -> TranslationStrategy {
        debug_assert!(!TimeFrame::aligned(left, right));
        debug_assert_ne!(cursor.time_frame(), left.min(right));
        debug_assert_eq!(cursor.time_frame(), output);
        TranslationStrategy::WaterfallBaseLifetime(Self {
            left,
            right,
            output,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::sim::bb::cursor::Cursor;
    use crate::sim::bb::time_frame;
    use crate::sim::compute::translation_strategy::{
        select_translation_strategy, TranslationStrategy,
    };
    use crate::sim::error::SetupError;

    #[test]
    fn should_fail_selection_when_time_frame_is_lower_than_base_quotes_time_frame() {
        let cursor = Cursor::new(0, 0, 1024, time_frame::MINUTE_1);

        assert_time_frame_lower_than_base_err(
            select_translation_strategy(&cursor, time_frame::SECOND_1, time_frame::MINUTE_5)
                .err()
                .unwrap(),
        );

        assert_time_frame_lower_than_base_err(
            select_translation_strategy(&cursor, time_frame::MINUTE_5, time_frame::SECOND_1)
                .err()
                .unwrap(),
        );
    }

    fn assert_time_frame_lower_than_base_err(error: anyhow::Error) {
        match error
            .downcast_ref::<SetupError>()
            .expect("Expects SetupError")
        {
            SetupError::TimeFrameLowerThanBase { .. } => {}
            _ => panic!("invalid error"),
        }
    }

    #[test]
    fn should_select_direct_translation_for_same_time_frames() {
        let cursor = Cursor::new(0, 0, 1024, time_frame::MINUTE_1);
        let translation_strategy =
            select_translation_strategy(&cursor, time_frame::MINUTE_5, time_frame::MINUTE_5)
                .unwrap();

        match translation_strategy {
            TranslationStrategy::Direct(descriptor) => {
                assert_eq!(time_frame::MINUTE_5, descriptor.left);
                assert_eq!(time_frame::MINUTE_5, descriptor.right);
                assert_eq!(time_frame::MINUTE_5, descriptor.output);
            }
            _ => panic!("Expected Direct strategy but actual is different"),
        }
    }

    #[test]
    fn should_select_direct_lifetime_translation_strategy() {
        let cursor = Cursor::new(0, 0, 1024, time_frame::MINUTE_1);
        let translation_strategy =
            select_translation_strategy(&cursor, time_frame::MINUTE_5, time_frame::MINUTE_15)
                .unwrap();

        match translation_strategy {
            TranslationStrategy::DirectLifetime(descriptor) => {
                assert_eq!(time_frame::MINUTE_5, descriptor.left);
                assert_eq!(time_frame::MINUTE_15, descriptor.right);
                assert_eq!(time_frame::MINUTE_5, descriptor.output);
            }
            _ => panic!("Expected DirectLifetime strategy but actual is different"),
        }
    }

    #[test]
    fn should_select_direct_base_lifetime_strategy() {
        let cursor = Cursor::new(0, 0, 1024, time_frame::MINUTE_1);
        let translation_strategy =
            select_translation_strategy(&cursor, time_frame::MINUTE_1, time_frame::HOUR_1).unwrap();

        match translation_strategy {
            TranslationStrategy::DirectBaseLifetime(descriptor) => {
                assert_eq!(time_frame::MINUTE_1, descriptor.left);
                assert_eq!(time_frame::HOUR_1, descriptor.right);
                assert_eq!(time_frame::MINUTE_1, descriptor.output);
            }
            _ => panic!("Expected DirectLifetime strategy but actual is different"),
        }
    }

    #[test]
    fn should_select_waterfall_lifetime_strategy() {
        let cursor = Cursor::new(0, 0, 1024, time_frame::SECOND_1);
        let translation_strategy =
            select_translation_strategy(&cursor, time_frame::MINUTE_3, time_frame::MINUTE_5)
                .unwrap();

        match translation_strategy {
            TranslationStrategy::WaterfallLifetime(descriptor) => {
                assert_eq!(time_frame::MINUTE_3, descriptor.left);
                assert_eq!(time_frame::MINUTE_5, descriptor.right);
                assert_eq!(time_frame::MINUTE_1, descriptor.output);
            }
            _ => panic!("Expected WaterfallLifetime strategy but actual is different"),
        }
    }

    #[test]
    fn should_select_waterfall_base_lifetime_strategy() {
        let cursor = Cursor::new(0, 0, 1024, time_frame::MINUTE_1);
        let translation_strategy =
            select_translation_strategy(&cursor, time_frame::MINUTE_3, time_frame::MINUTE_5)
                .unwrap();

        match translation_strategy {
            TranslationStrategy::WaterfallBaseLifetime(descriptor) => {
                assert_eq!(time_frame::MINUTE_3, descriptor.left);
                assert_eq!(time_frame::MINUTE_5, descriptor.right);
                assert_eq!(time_frame::MINUTE_1, descriptor.output);
            }
            _ => panic!("Expected WaterfallBaseLifetime strategy but actual is different"),
        }
    }
}
