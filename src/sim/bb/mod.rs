/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::cursor::Epoch;

pub mod cursor;
pub mod quotes;
pub mod time_frame;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ComputeId {
    Simple(usize),
}
