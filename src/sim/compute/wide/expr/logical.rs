/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::cursor::CursorExpansion;
use crate::sim::compute::wide::expr::LogicalNode;
use crate::sim::context::SpanContext;
use crate::sim::mediator::Mediator;

impl LogicalNode {
    pub fn eval(
        &self,
        ctx: &SpanContext,
        cursor_expansion: CursorExpansion,
    ) -> Result<Mediator, anyhow::Error> {
        match self {
            LogicalNode::And(left, right) => {
                let left = left.eval(ctx, cursor_expansion)?;
                let right = right.eval(ctx, cursor_expansion)?;
                Mediator::intersect(ctx, left, right)
            }
            LogicalNode::Or(left, right) => {
                let left = left.eval(ctx, cursor_expansion)?;
                let right = right.eval(ctx, cursor_expansion)?;
                Mediator::union(ctx, left, right)
            }
            LogicalNode::Not(expr) => Mediator::negate(ctx, expr.eval(ctx, cursor_expansion)?),
            LogicalNode::Predicate(predicate) => predicate.eval(ctx, cursor_expansion),
        }
    }
}
