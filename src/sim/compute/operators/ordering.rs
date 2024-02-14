/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

pub trait OrderingOperator {
    fn compare(left: f64, right: f64) -> bool;
}

pub struct Equal;

impl OrderingOperator for Equal {
    fn compare(left: f64, right: f64) -> bool {
        left == right
    }
}

pub struct NotEqual;

impl OrderingOperator for NotEqual {
    fn compare(left: f64, right: f64) -> bool {
        left != right
    }
}

pub struct Greater;

impl OrderingOperator for Greater {
    fn compare(left: f64, right: f64) -> bool {
        left > right
    }
}

pub struct Lower;

impl OrderingOperator for Lower {
    fn compare(left: f64, right: f64) -> bool {
        left < right
    }
}

pub struct GreaterOrEqual;

impl OrderingOperator for GreaterOrEqual {
    fn compare(left: f64, right: f64) -> bool {
        left >= right
    }
}

pub struct LowerOrEqual;

impl OrderingOperator for LowerOrEqual {
    fn compare(left: f64, right: f64) -> bool {
        left <= right
    }
}
