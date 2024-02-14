/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

pub trait UnaryOperator {
    fn apply(arg: f64) -> f64;
}

pub trait BinaryOperator {
    fn apply(a: f64, b: f64) -> f64;
}

pub struct AddOperator;

impl BinaryOperator for AddOperator {
    fn apply(a: f64, b: f64) -> f64 {
        a + b
    }
}

pub struct SubtractOperator;

impl BinaryOperator for SubtractOperator {
    fn apply(a: f64, b: f64) -> f64 {
        a - b
    }
}

pub struct MultiplyOperator;

impl BinaryOperator for MultiplyOperator {
    fn apply(a: f64, b: f64) -> f64 {
        a * b
    }
}

pub struct DivideOperator;

impl BinaryOperator for DivideOperator {
    fn apply(a: f64, b: f64) -> f64 {
        a / b
    }
}

pub struct PowOperator;

impl BinaryOperator for PowOperator {
    fn apply(a: f64, b: f64) -> f64 {
        a.powf(b)
    }
}

pub struct MaxOperator;

impl BinaryOperator for MaxOperator {
    fn apply(a: f64, b: f64) -> f64 {
        a.max(b)
    }
}

pub struct MinOperator;

impl BinaryOperator for MinOperator {
    fn apply(a: f64, b: f64) -> f64 {
        a.min(b)
    }
}

pub struct SqrtOperator;

impl UnaryOperator for SqrtOperator {
    fn apply(arg: f64) -> f64 {
        arg.sqrt()
    }
}

pub struct AbsOperator;

impl UnaryOperator for AbsOperator {
    fn apply(arg: f64) -> f64 {
        arg.abs()
    }
}
