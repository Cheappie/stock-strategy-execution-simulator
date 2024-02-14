/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use num_integer::Integer;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::time::Duration;

///
/// TimeFrame represents a duration for how long data points were aggregated in milliseconds.  
///
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct TimeFrame(u32);

impl Deref for TimeFrame {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl TimeFrame {
    pub fn aligned(this: TimeFrame, other: TimeFrame) -> bool {
        let (larger, smaller) = if this > other {
            (this, other)
        } else {
            (other, this)
        };

        0 == (*larger % *smaller)
    }

    ///
    /// Supported time frame's
    ///
    pub fn from_millis(millis: u32) -> Option<TimeFrame> {
        match millis {
            MILLISECOND_200_VALUE => Some(MILLISECOND_200),
            MILLISECOND_500_VALUE => Some(MILLISECOND_500),
            SECOND_1_VALUE => Some(SECOND_1),
            SECOND_5_VALUE => Some(SECOND_5),
            SECOND_10_VALUE => Some(SECOND_10),
            SECOND_15_VALUE => Some(SECOND_15),
            SECOND_30_VALUE => Some(SECOND_30),
            MINUTE_1_VALUE => Some(MINUTE_1),
            MINUTE_3_VALUE => Some(MINUTE_3),
            MINUTE_5_VALUE => Some(MINUTE_5),
            MINUTE_10_VALUE => Some(MINUTE_10),
            MINUTE_15_VALUE => Some(MINUTE_15),
            MINUTE_30_VALUE => Some(MINUTE_30),
            HOUR_1_VALUE => Some(HOUR_1),
            HOUR_2_VALUE => Some(HOUR_2),
            HOUR_3_VALUE => Some(HOUR_3),
            HOUR_4_VALUE => Some(HOUR_4),
            DAY_1_VALUE => Some(DAY_1),
            _ => None,
        }
    }

    ///
    /// Supported time frame's
    ///
    pub fn ordinal(&self) -> usize {
        match self.0 {
            MILLISECOND_200_VALUE => 0,
            MILLISECOND_500_VALUE => 1,
            SECOND_1_VALUE => 2,
            SECOND_5_VALUE => 3,
            SECOND_10_VALUE => 4,
            SECOND_15_VALUE => 5,
            SECOND_30_VALUE => 6,
            MINUTE_1_VALUE => 7,
            MINUTE_3_VALUE => 8,
            MINUTE_5_VALUE => 9,
            MINUTE_10_VALUE => 10,
            MINUTE_15_VALUE => 11,
            MINUTE_30_VALUE => 12,
            HOUR_1_VALUE => 13,
            HOUR_2_VALUE => 14,
            HOUR_3_VALUE => 15,
            HOUR_4_VALUE => 16,
            DAY_1_VALUE => 17,
            _ => panic!(
                "There is no ordinal for TimeFrame({}), this time frame is not supported yet",
                self.0
            ),
        }
    }

    pub fn prev(&self) -> Option<TimeFrame> {
        self.ordinal()
            .checked_sub(1)
            .map(|i| SUPPORTED_TIME_FRAMES[i])
    }

    pub fn next(&self) -> Option<TimeFrame> {
        self.ordinal()
            .checked_add(1)
            .map(|i| SUPPORTED_TIME_FRAMES[i])
    }

    pub fn find_common_divisor(this: TimeFrame, other: TimeFrame) -> Option<TimeFrame> {
        let mut common_divisor = Some(if this > other { other } else { this });

        while let Some(divisor) = common_divisor.take() {
            if *this % *divisor == 0 && *other % *divisor == 0 {
                return Some(divisor);
            } else {
                common_divisor = divisor.prev();
            }
        }

        None
    }
}

pub const SUPPORTED_TIME_FRAMES: [TimeFrame; 18] = [
    MILLISECOND_200,
    MILLISECOND_500,
    SECOND_1,
    SECOND_5,
    SECOND_10,
    SECOND_15,
    SECOND_30,
    MINUTE_1,
    MINUTE_3,
    MINUTE_5,
    MINUTE_10,
    MINUTE_15,
    MINUTE_30,
    HOUR_1,
    HOUR_2,
    HOUR_3,
    HOUR_4,
    DAY_1,
];

const MILLISECOND_200_VALUE: u32 = 200;
pub const MILLISECOND_200: TimeFrame = TimeFrame(MILLISECOND_200_VALUE);

const MILLISECOND_500_VALUE: u32 = 500;
pub const MILLISECOND_500: TimeFrame = TimeFrame(MILLISECOND_500_VALUE);

const SECOND_1_VALUE: u32 = from_secs(1);
pub const SECOND_1: TimeFrame = TimeFrame(SECOND_1_VALUE);

const SECOND_5_VALUE: u32 = from_secs(5);
pub const SECOND_5: TimeFrame = TimeFrame(SECOND_5_VALUE);

const SECOND_10_VALUE: u32 = from_secs(10);
pub const SECOND_10: TimeFrame = TimeFrame(SECOND_10_VALUE);

const SECOND_15_VALUE: u32 = from_secs(15);
pub const SECOND_15: TimeFrame = TimeFrame(SECOND_15_VALUE);

const SECOND_30_VALUE: u32 = from_secs(30);
pub const SECOND_30: TimeFrame = TimeFrame(SECOND_30_VALUE);

const MINUTE_1_VALUE: u32 = from_minutes(1);
pub const MINUTE_1: TimeFrame = TimeFrame(MINUTE_1_VALUE);

const MINUTE_3_VALUE: u32 = from_minutes(3);
pub const MINUTE_3: TimeFrame = TimeFrame(MINUTE_3_VALUE);

const MINUTE_5_VALUE: u32 = from_minutes(5);
pub const MINUTE_5: TimeFrame = TimeFrame(MINUTE_5_VALUE);

const MINUTE_10_VALUE: u32 = from_minutes(10);
pub const MINUTE_10: TimeFrame = TimeFrame(MINUTE_10_VALUE);

const MINUTE_15_VALUE: u32 = from_minutes(15);
pub const MINUTE_15: TimeFrame = TimeFrame(MINUTE_15_VALUE);

const MINUTE_30_VALUE: u32 = from_minutes(30);
pub const MINUTE_30: TimeFrame = TimeFrame(MINUTE_30_VALUE);

const HOUR_1_VALUE: u32 = from_hours(1);
pub const HOUR_1: TimeFrame = TimeFrame(HOUR_1_VALUE);

const HOUR_2_VALUE: u32 = from_hours(2);
pub const HOUR_2: TimeFrame = TimeFrame(HOUR_2_VALUE);

const HOUR_3_VALUE: u32 = from_hours(3);
pub const HOUR_3: TimeFrame = TimeFrame(HOUR_3_VALUE);

const HOUR_4_VALUE: u32 = from_hours(4);
pub const HOUR_4: TimeFrame = TimeFrame(HOUR_4_VALUE);

const DAY_1_VALUE: u32 = from_hours(24);
pub const DAY_1: TimeFrame = TimeFrame(DAY_1_VALUE);

const fn from_secs(secs: u32) -> u32 {
    secs * 1_000
}

const fn from_minutes(minutes: u32) -> u32 {
    minutes * from_secs(60)
}

const fn from_hours(hours: u32) -> u32 {
    hours * from_minutes(60)
}

#[cfg(test)]
mod tests {
    use crate::sim::bb::time_frame;
    use crate::sim::bb::time_frame::{TimeFrame, SUPPORTED_TIME_FRAMES};

    #[test]
    fn assert_ordinal_matches() {
        for (ordinal, time_frame) in SUPPORTED_TIME_FRAMES.iter().enumerate() {
            assert_eq!(ordinal, time_frame.ordinal());
        }
    }

    #[test]
    fn assert_supported_time_frames_are_sorted() {
        assert!(SUPPORTED_TIME_FRAMES.windows(2).all(|s| s[1] > s[0]));
    }

    #[test]
    fn should_deserialize_from_millis_to_time_frame() {
        assert!(SUPPORTED_TIME_FRAMES
            .iter()
            .all(|tf| *tf == TimeFrame::from_millis(**tf).unwrap()));
    }

    #[test]
    fn should_produce_next_values() {
        assert!(SUPPORTED_TIME_FRAMES
            .windows(2)
            .all(|s| s[0].next().unwrap() == s[1]));
    }

    #[test]
    fn should_produce_prev_values() {
        assert!(SUPPORTED_TIME_FRAMES
            .windows(2)
            .all(|s| s[0] == s[1].prev().unwrap()));
    }

    #[test]
    fn should_find_common_divisor() {
        assert_eq!(
            time_frame::MINUTE_1,
            TimeFrame::find_common_divisor(time_frame::MINUTE_3, time_frame::MINUTE_5)
                .expect("Found")
        );
    }
}
