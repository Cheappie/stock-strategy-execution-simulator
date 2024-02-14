/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::cursor::Epoch;
use crate::sim::bb::time_frame::TimeFrame;

macro_rules! epoch_matches {
    ($e1:expr, $e2:expr) => {{
        let ep1 = $e1.epoch();
        let ep2 = $e2.epoch();
        debug_assert_eq!(ep1, ep2);
        ep1
    }};
    ($e1:expr, $e2:expr, $e3:expr) => {{
        let ep1 = $e1.epoch();
        let ep2 = $e2.epoch();
        let ep3 = $e3.epoch();
        debug_assert_eq!(ep1, ep2);
        debug_assert_eq!(ep1, ep3);
        ep1
    }};
    ($e1:expr, $e2:expr, $e3:expr, $e4:expr) => {{
        let ep1 = $e1.epoch();
        let ep2 = $e2.epoch();
        let ep3 = $e3.epoch();
        let ep4 = $e4.epoch();
        debug_assert_eq!(ep1, ep2);
        debug_assert_eq!(ep1, ep3);
        debug_assert_eq!(ep1, ep4);
        ep1
    }};
    ($e1:expr, $e2:expr, $e3:expr, $e4:expr, $e5:expr) => {{
        let ep1 = $e1.epoch();
        let ep2 = $e2.epoch();
        let ep3 = $e3.epoch();
        let ep4 = $e4.epoch();
        let ep5 = $e5.epoch();
        debug_assert_eq!(ep1, ep2);
        debug_assert_eq!(ep1, ep3);
        debug_assert_eq!(ep1, ep4);
        debug_assert_eq!(ep1, ep5);
        ep1
    }};
}

pub(crate) use epoch_matches;
