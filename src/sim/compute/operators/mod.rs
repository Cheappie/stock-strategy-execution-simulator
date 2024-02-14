/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

pub mod math;
pub mod ordering;

#[derive(Debug, Copy, Clone)]
pub enum BinaryMathOperator {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Min,
}

#[derive(Debug, Copy, Clone)]
pub enum UnaryMathOperator {
    Sqrt,
    Abs,
}

#[derive(Debug, Copy, Clone)]
pub enum Ordering {
    Equal,
    NotEqual,
    Greater,
    Lower,
    GreaterOrEqual,
    LowerOrEqual,
}
