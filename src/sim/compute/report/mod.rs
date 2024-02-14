/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/
use crate::sim::bb::quotes::TimestampVector;
use crate::sim::bb::ComputeId;
use crate::sim::tlb::LifetimeVector;

///
/// This module implements support for economic market data.
///
mod expr;

struct Report {
    id: usize,
    description: String,
    values: Option<Vec<f64>>,
    timestamp: TimestampVector,
    lifetime: LifetimeVector,
}
