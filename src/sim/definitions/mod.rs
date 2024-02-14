/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use std::str::FromStr;

use anyhow::anyhow;
use uuid::Uuid;

///
/// do not compute definitions that are not part of relations
///
pub struct StrategyDefinition {
    relations: Vec<Node>,
    definitions: Vec<(Component, Vec<u8>)>,
}

pub struct Node {
    id: usize,
    parent: Uuid,
    left: Uuid,
    right: Uuid,
    operator: LogicalOperator,
}

pub enum LogicalOperator {
    And,
    Or,
    Not,
}

impl FromStr for LogicalOperator {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "and" => Ok(LogicalOperator::And),
            "or" => Ok(LogicalOperator::Or),
            "not" => Ok(LogicalOperator::Not),
            _ => Err(anyhow!("Unknown logical operator: {s}")),
        }
    }
}

pub enum Component {
    SimpleMovingAverage,
    ExponentialMovingAverage,
}

impl FromStr for Component {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "simple_moving_average" => Ok(Component::SimpleMovingAverage),
            "exponential_moving_average" => Ok(Component::ExponentialMovingAverage),
            _ => Err(anyhow!("Unknown component: {s}")),
        }
    }
}
