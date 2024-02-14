/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::bb::cursor::{Cursor, TimeShift};
use crate::sim::bb::quotes::{BaseQuotes, FrameQuotes, TimestampVector};
use crate::sim::bb::{time_frame, ComputeId};
use crate::sim::compute::wide::expr::{
    Constant, IdentifiedProcessorNode, PrimitiveNode, ProcessorHandle, ProcessorNode, Scalar,
};
use crate::sim::context::{SessionConfiguration, SessionContext, SpanContext};
use crate::sim::selector::Selector;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;

// TODO : add time shift tests

#[cfg(test)]
mod d2d_waterfall_base_lifetime {
    use crate::sim::bb::cursor::{Cursor, CursorExpansion, Epoch, TimeShift};
    use crate::sim::bb::quotes::{BaseQuotes, FrameQuotes, TimestampVector};
    use crate::sim::bb::{time_frame, ComputeId};
    use crate::sim::builder::frame_builder::build_frame_to_frame;
    use crate::sim::builder::translation_builder::build_translation_buffer;
    use crate::sim::compute::operators::BinaryMathOperator;
    use crate::sim::compute::translation_strategy::{
        select_translation_strategy, TranslationStrategy,
    };
    use crate::sim::compute::wide::expr::math_tests::primitive_processor_handle;
    use crate::sim::compute::wide::expr::primitive::PrimitiveOutputSelector;
    use crate::sim::compute::wide::expr::{WideBinaryExprOutput, WideBinaryMathExpr};
    use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
    use crate::sim::compute::wide::indicator::{SingleOutputSelector, WideProcessor};
    use crate::sim::context::{SessionConfiguration, SessionContext, SpanContext};
    use crate::sim::selector::Selector;
    use crate::sim::spatial_buffer::SpatialBuffer;
    use crate::sim::tlb::{FrameTranslationBuffer, IdentityTranslationBuffer};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn should_apply_add_op() {
        let evaluated_expression = eval_d2d_waterfall_base(
            BinaryMathOperator::Add,
            Cursor::new(0, 2, 7, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..9, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(2.8, *reader.get(0));
        assert_eq!(2.8, *reader.get(1));
        assert_eq!(2.8, *reader.get(2));
        assert_eq!(3.0, *reader.get(3));
        assert_eq!(3.0, *reader.get(4));
        assert_eq!(4.4, *reader.get(5));
        assert_eq!(4.4, *reader.get(6));
        assert_eq!(4.300000000000001, *reader.get(7));
        assert_eq!(4.1, *reader.get(8));
    }

    #[test]
    fn should_apply_sub_op() {
        let evaluated_expression = eval_d2d_waterfall_base(
            BinaryMathOperator::Sub,
            Cursor::new(0, 2, 7, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..9, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(-0.40000000000000013, *reader.get(0));
        assert_eq!(-0.40000000000000013, *reader.get(1));
        assert_eq!(-0.40000000000000013, *reader.get(2));
        assert_eq!(-0.20000000000000018, *reader.get(3));
        assert_eq!(-0.20000000000000018, *reader.get(4));
        assert_eq!(0.0, *reader.get(5));
        assert_eq!(0.0, *reader.get(6));
        assert_eq!(-0.10000000000000009, *reader.get(7));
        assert_eq!(0.10000000000000009, *reader.get(8));
    }

    #[test]
    fn should_apply_mul_op() {
        let evaluated_expression = eval_d2d_waterfall_base(
            BinaryMathOperator::Mul,
            Cursor::new(0, 2, 7, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..9, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.92, *reader.get(0));
        assert_eq!(1.92, *reader.get(1));
        assert_eq!(1.92, *reader.get(2));
        assert_eq!(2.2399999999999998, *reader.get(3));
        assert_eq!(2.2399999999999998, *reader.get(4));
        assert_eq!(4.840000000000001, *reader.get(5));
        assert_eq!(4.840000000000001, *reader.get(6));
        assert_eq!(4.620000000000001, *reader.get(7));
        assert_eq!(4.2, *reader.get(8));
    }

    #[test]
    fn should_apply_div_op() {
        let evaluated_expression = eval_d2d_waterfall_base(
            BinaryMathOperator::Div,
            Cursor::new(0, 2, 7, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..9, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(0.7499999999999999, *reader.get(0));
        assert_eq!(0.7499999999999999, *reader.get(1));
        assert_eq!(0.7499999999999999, *reader.get(2));
        assert_eq!(0.8749999999999999, *reader.get(3));
        assert_eq!(0.8749999999999999, *reader.get(4));
        assert_eq!(1.0, *reader.get(5));
        assert_eq!(1.0, *reader.get(6));
        assert_eq!(0.9545454545454545, *reader.get(7));
        assert_eq!(1.05, *reader.get(8));
    }

    #[test]
    fn should_apply_pow_op() {
        let evaluated_expression = eval_d2d_waterfall_base(
            BinaryMathOperator::Pow,
            Cursor::new(0, 2, 7, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..9, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();
        assert_eq!(1.338720746075793, *reader.get(0));
        assert_eq!(1.338720746075793, *reader.get(1));
        assert_eq!(1.338720746075793, *reader.get(2));
        assert_eq!(1.7131873426422426, *reader.get(3));
        assert_eq!(1.7131873426422426, *reader.get(4));
        assert_eq!(5.666695778750081, *reader.get(5));
        assert_eq!(5.666695778750081, *reader.get(6));
        assert_eq!(5.1154335606641474, *reader.get(7));
        assert_eq!(4.41, *reader.get(8));
    }

    #[test]
    fn should_apply_max_op() {
        let evaluated_expression = eval_d2d_waterfall_base(
            BinaryMathOperator::Max,
            Cursor::new(0, 2, 7, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..9, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.6, *reader.get(0));
        assert_eq!(1.6, *reader.get(1));
        assert_eq!(1.6, *reader.get(2));
        assert_eq!(1.6, *reader.get(3));
        assert_eq!(1.6, *reader.get(4));
        assert_eq!(2.2, *reader.get(5));
        assert_eq!(2.2, *reader.get(6));
        assert_eq!(2.2, *reader.get(7));
        assert_eq!(2.1, *reader.get(8));
    }

    #[test]
    fn should_apply_min_op() {
        let evaluated_expression = eval_d2d_waterfall_base(
            BinaryMathOperator::Min,
            Cursor::new(0, 2, 7, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..9, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.2, *reader.get(0));
        assert_eq!(1.2, *reader.get(1));
        assert_eq!(1.2, *reader.get(2));
        assert_eq!(1.4, *reader.get(3));
        assert_eq!(1.4, *reader.get(4));
        assert_eq!(2.2, *reader.get(5));
        assert_eq!(2.2, *reader.get(6));
        assert_eq!(2.1, *reader.get(7));
        assert_eq!(2.0, *reader.get(8));
    }

    fn eval_d2d_waterfall_base(
        operator: BinaryMathOperator,
        cursor: Cursor,
        cursor_expansion: CursorExpansion,
    ) -> WideBinaryMathExpr {
        let base_quotes = Arc::new(BaseQuotes::new(Arc::new(FrameQuotes::new(
            vec![0f64; 16],
            vec![0f64; 16],
            vec![0f64; 16],
            vec![
                1.1, 1.2, 1.6, 1.4, 1.8, 2.2, 1.9, 2.1, 2.0, 1.4, 1.6, 1.9, 2.4, 2.6, 2.3, 2.1,
            ],
            TimestampVector::from_utc(
                (0..16)
                    .map(|i| (i + 1) as u64 * u64::from(*time_frame::SECOND_5))
                    .collect(),
            ),
            time_frame::SECOND_5,
        ))));

        let all_frame_quotes = BTreeMap::from([
            (time_frame::SECOND_5, Arc::clone(&**base_quotes)),
            (
                time_frame::SECOND_10,
                Arc::new(build_frame_to_frame(&base_quotes, time_frame::SECOND_10).unwrap()),
            ),
            (
                time_frame::SECOND_15,
                Arc::new(build_frame_to_frame(&base_quotes, time_frame::SECOND_15).unwrap()),
            ),
        ]);

        let tlbs = BTreeMap::from([
            (
                time_frame::SECOND_5,
                Arc::new(build_translation_buffer(&base_quotes, time_frame::SECOND_5).unwrap()),
            ),
            (
                time_frame::SECOND_10,
                Arc::new(build_translation_buffer(&base_quotes, time_frame::SECOND_10).unwrap()),
            ),
            (
                time_frame::SECOND_15,
                Arc::new(build_translation_buffer(&base_quotes, time_frame::SECOND_15).unwrap()),
            ),
        ]);

        let session = Arc::new(SessionContext::new(
            String::from("GOLD"),
            base_quotes,
            all_frame_quotes,
            tlbs,
            SessionConfiguration::new(8, 64),
        ));

        let mut id_gen = 0usize;
        let left = primitive_processor_handle(
            &mut id_gen,
            session.frame_quotes(time_frame::SECOND_10).unwrap(),
            PrimitiveOutputSelector::Close,
        );
        let right = primitive_processor_handle(
            &mut id_gen,
            session.frame_quotes(time_frame::SECOND_15).unwrap(),
            PrimitiveOutputSelector::Close,
        );

        let strategy =
            select_translation_strategy(&cursor, time_frame::SECOND_10, time_frame::SECOND_15)
                .unwrap();

        match &strategy {
            TranslationStrategy::WaterfallBaseLifetime(descriptor) => {
                assert_eq!(time_frame::SECOND_10, descriptor.left());
                assert_eq!(time_frame::SECOND_15, descriptor.right());
                assert_eq!(time_frame::SECOND_5, descriptor.output());
            }
            _ => panic!("invalid strategy has been chosen"),
        }

        let output =
            WideBinaryExprOutput::new(strategy, SpatialBuffer::new(time_frame::SECOND_5, 64));

        let ctx = SpanContext::new(cursor, session);
        let mut expr = WideBinaryMathExpr::new(left, operator, right, output);
        expr.eval(
            &ctx,
            cursor_expansion,
            &DynamicStore::new(ComputeId::Simple(3)),
        )
        .expect("success");

        expr
    }
}

#[cfg(test)]
mod d2d_direct_base_lifetime_base_2_upper {
    use crate::sim::bb::cursor::{Cursor, CursorExpansion, Epoch, TimeShift};
    use crate::sim::bb::quotes::{BaseQuotes, FrameQuotes, TimestampVector};
    use crate::sim::bb::{time_frame, ComputeId};
    use crate::sim::builder::frame_builder::build_frame_to_frame;
    use crate::sim::builder::translation_builder::build_translation_buffer;
    use crate::sim::compute::operators::BinaryMathOperator;
    use crate::sim::compute::translation_strategy::{
        select_translation_strategy, TranslationStrategy,
    };
    use crate::sim::compute::wide::expr::math_tests::primitive_processor_handle;
    use crate::sim::compute::wide::expr::primitive::PrimitiveOutputSelector;
    use crate::sim::compute::wide::expr::{WideBinaryExprOutput, WideBinaryMathExpr};
    use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
    use crate::sim::compute::wide::indicator::{SingleOutputSelector, WideProcessor};
    use crate::sim::context::{SessionConfiguration, SessionContext, SpanContext};
    use crate::sim::selector::Selector;
    use crate::sim::spatial_buffer::SpatialBuffer;
    use crate::sim::tlb::{FrameTranslationBuffer, IdentityTranslationBuffer};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn should_expand_epoch() {
        let evaluated_expression = eval_d2d_direct_base(
            BinaryMathOperator::Add,
            Cursor::new(0, 2, 6, time_frame::SECOND_5),
            CursorExpansion::Identity.expand_by(TimeShift::Future(2)),
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..10, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(2.3, *reader.get(0));
        assert_eq!(2.4, *reader.get(1));
        assert_eq!(2.8, *reader.get(2));
        assert_eq!(2.8, *reader.get(3));
        assert_eq!(3.2, *reader.get(4));
        assert_eq!(4.4, *reader.get(5));
        assert_eq!(4.1, *reader.get(6));
        assert_eq!(4.2, *reader.get(7));
        assert_eq!(4.1, *reader.get(8));
        assert_eq!(2.8, *reader.get(9));
    }

    #[test]
    fn should_apply_add_op() {
        let evaluated_expression = eval_d2d_direct_base(
            BinaryMathOperator::Add,
            Cursor::new(0, 2, 6, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(2.3, *reader.get(0));
        assert_eq!(2.4, *reader.get(1));
        assert_eq!(2.8, *reader.get(2));
        assert_eq!(2.8, *reader.get(3));
        assert_eq!(3.2, *reader.get(4));
        assert_eq!(4.4, *reader.get(5));
        assert_eq!(4.1, *reader.get(6));
        assert_eq!(4.2, *reader.get(7));
    }

    #[test]
    fn should_apply_sub_op() {
        let evaluated_expression = eval_d2d_direct_base(
            BinaryMathOperator::Sub,
            Cursor::new(0, 2, 6, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(-0.09999999999999987, *reader.get(0));
        assert_eq!(0.0, *reader.get(1));
        assert_eq!(0.40000000000000013, *reader.get(2));
        assert_eq!(0.0, *reader.get(3));
        assert_eq!(0.40000000000000013, *reader.get(4));
        assert_eq!(0.0, *reader.get(5));
        assert_eq!(-0.30000000000000027, *reader.get(6));
        assert_eq!(0.0, *reader.get(7));
    }

    #[test]
    fn should_apply_mul_op() {
        let evaluated_expression = eval_d2d_direct_base(
            BinaryMathOperator::Mul,
            Cursor::new(0, 2, 6, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.32, *reader.get(0));
        assert_eq!(1.44, *reader.get(1));
        assert_eq!(1.92, *reader.get(2));
        assert_eq!(1.9599999999999997, *reader.get(3));
        assert_eq!(2.52, *reader.get(4));
        assert_eq!(4.840000000000001, *reader.get(5));
        assert_eq!(4.18, *reader.get(6));
        assert_eq!(4.41, *reader.get(7));
    }

    #[test]
    fn should_apply_div_op() {
        let evaluated_expression = eval_d2d_direct_base(
            BinaryMathOperator::Div,
            Cursor::new(0, 2, 6, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(0.9166666666666667, *reader.get(0));
        assert_eq!(1.0, *reader.get(1));
        assert_eq!(1.3333333333333335, *reader.get(2));
        assert_eq!(1.0, *reader.get(3));
        assert_eq!(1.2857142857142858, *reader.get(4));
        assert_eq!(1.0, *reader.get(5));
        assert_eq!(0.8636363636363635, *reader.get(6));
        assert_eq!(1.0, *reader.get(7));
    }

    #[test]
    fn should_apply_pow_op() {
        let evaluated_expression = eval_d2d_direct_base(
            BinaryMathOperator::Pow,
            Cursor::new(0, 2, 6, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.1211693641406024, *reader.get(0));
        assert_eq!(1.2445647472039776, *reader.get(1));
        assert_eq!(1.7576968692897885, *reader.get(2));
        assert_eq!(1.601692898202212, *reader.get(3));
        assert_eq!(2.2770968742508497, *reader.get(4));
        assert_eq!(5.666695778750081, *reader.get(5));
        assert_eq!(4.1044779046045985, *reader.get(6));
        assert_eq!(4.749638091742242, *reader.get(7));
    }

    #[test]
    fn should_apply_max_op() {
        let evaluated_expression = eval_d2d_direct_base(
            BinaryMathOperator::Max,
            Cursor::new(0, 2, 6, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.2, *reader.get(0));
        assert_eq!(1.2, *reader.get(1));
        assert_eq!(1.6, *reader.get(2));
        assert_eq!(1.4, *reader.get(3));
        assert_eq!(1.8, *reader.get(4));
        assert_eq!(2.2, *reader.get(5));
        assert_eq!(2.2, *reader.get(6));
        assert_eq!(2.1, *reader.get(7));
    }

    #[test]
    fn should_apply_min_op() {
        let evaluated_expression = eval_d2d_direct_base(
            BinaryMathOperator::Min,
            Cursor::new(0, 2, 6, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.1, *reader.get(0));
        assert_eq!(1.2, *reader.get(1));
        assert_eq!(1.2, *reader.get(2));
        assert_eq!(1.4, *reader.get(3));
        assert_eq!(1.4, *reader.get(4));
        assert_eq!(2.2, *reader.get(5));
        assert_eq!(1.9, *reader.get(6));
        assert_eq!(2.1, *reader.get(7));
    }

    fn eval_d2d_direct_base(
        operator: BinaryMathOperator,
        cursor: Cursor,
        cursor_expansion: CursorExpansion,
    ) -> WideBinaryMathExpr {
        let frame_quotes = Arc::new(FrameQuotes::new(
            vec![0f64; 16],
            vec![0f64; 16],
            vec![0f64; 16],
            vec![
                1.1, 1.2, 1.6, 1.4, 1.8, 2.2, 1.9, 2.1, 2.0, 1.4, 1.6, 1.9, 2.4, 2.6, 2.3, 2.1,
            ],
            TimestampVector::from_utc(
                (0..16)
                    .map(|i| (i + 1) as u64 * u64::from(*time_frame::SECOND_5))
                    .collect(),
            ),
            time_frame::SECOND_5,
        ));

        let base_quotes = Arc::new(BaseQuotes::new(Arc::clone(&frame_quotes)));

        let all_frame_quotes = BTreeMap::from([
            (time_frame::SECOND_5, Arc::clone(&frame_quotes)),
            (
                time_frame::SECOND_10,
                Arc::new(build_frame_to_frame(&frame_quotes, time_frame::SECOND_10).unwrap()),
            ),
        ]);

        let tlbs = BTreeMap::from([
            (
                time_frame::SECOND_5,
                Arc::new(FrameTranslationBuffer::IdentityTranslationBuffer(
                    IdentityTranslationBuffer::new(time_frame::SECOND_5),
                )),
            ),
            (
                time_frame::SECOND_10,
                Arc::new(build_translation_buffer(&base_quotes, time_frame::SECOND_10).unwrap()),
            ),
        ]);

        let session = Arc::new(SessionContext::new(
            String::from("GOLD"),
            base_quotes,
            all_frame_quotes,
            tlbs,
            SessionConfiguration::new(8, 64),
        ));

        let mut id_gen = 0usize;
        let left = primitive_processor_handle(
            &mut id_gen,
            session.frame_quotes(time_frame::SECOND_5).unwrap(),
            PrimitiveOutputSelector::Close,
        );
        let right = primitive_processor_handle(
            &mut id_gen,
            session.frame_quotes(time_frame::SECOND_10).unwrap(),
            PrimitiveOutputSelector::Close,
        );

        let strategy =
            select_translation_strategy(&cursor, time_frame::SECOND_5, time_frame::SECOND_10)
                .unwrap();

        match &strategy {
            TranslationStrategy::DirectBaseLifetime(descriptor) => {
                assert_eq!(time_frame::SECOND_5, descriptor.left());
                assert_eq!(time_frame::SECOND_10, descriptor.right());
                assert_eq!(time_frame::SECOND_5, descriptor.output());
            }
            _ => panic!("invalid strategy has been chosen"),
        }

        let output =
            WideBinaryExprOutput::new(strategy, SpatialBuffer::new(time_frame::SECOND_5, 64));

        let ctx = SpanContext::new(cursor, session);
        let mut expr = WideBinaryMathExpr::new(left, operator, right, output);
        expr.eval(
            &ctx,
            cursor_expansion,
            &DynamicStore::new(ComputeId::Simple(3)),
        )
        .expect("success");

        expr
    }
}

#[cfg(test)]
mod d2d_direct_base_lifetime_upper_2_base {
    use crate::sim::bb::cursor::{Cursor, CursorExpansion, Epoch, TimeShift};
    use crate::sim::bb::quotes::{BaseQuotes, FrameQuotes, TimestampVector};
    use crate::sim::bb::{time_frame, ComputeId};
    use crate::sim::builder::frame_builder::build_frame_to_frame;
    use crate::sim::builder::translation_builder::build_translation_buffer;
    use crate::sim::compute::operators::BinaryMathOperator;
    use crate::sim::compute::translation_strategy::{
        select_translation_strategy, TranslationStrategy,
    };
    use crate::sim::compute::wide::expr::math_tests::primitive_processor_handle;
    use crate::sim::compute::wide::expr::primitive::PrimitiveOutputSelector;
    use crate::sim::compute::wide::expr::{WideBinaryExprOutput, WideBinaryMathExpr};
    use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
    use crate::sim::compute::wide::indicator::{SingleOutputSelector, WideProcessor};
    use crate::sim::context::{SessionConfiguration, SessionContext, SpanContext};
    use crate::sim::selector::Selector;
    use crate::sim::spatial_buffer::SpatialBuffer;
    use crate::sim::tlb::{FrameTranslationBuffer, IdentityTranslationBuffer};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn should_expand_epoch() {
        let evaluated_expression = eval_d2d_direct_base(
            BinaryMathOperator::Add,
            Cursor::new(0, 2, 6, time_frame::SECOND_5),
            CursorExpansion::Identity.expand_by(TimeShift::Future(2)),
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..10, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(2.3, *reader.get(0));
        assert_eq!(2.4, *reader.get(1));
        assert_eq!(2.8, *reader.get(2));
        assert_eq!(2.8, *reader.get(3));
        assert_eq!(3.2, *reader.get(4));
        assert_eq!(4.4, *reader.get(5));
        assert_eq!(4.1, *reader.get(6));
        assert_eq!(4.2, *reader.get(7));
        assert_eq!(4.1, *reader.get(8));
        assert_eq!(2.8, *reader.get(9));
    }

    #[test]
    fn should_apply_add_op() {
        let evaluated_expression = eval_d2d_direct_base(
            BinaryMathOperator::Add,
            Cursor::new(0, 2, 6, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(2.3, *reader.get(0));
        assert_eq!(2.4, *reader.get(1));
        assert_eq!(2.8, *reader.get(2));
        assert_eq!(2.8, *reader.get(3));
        assert_eq!(3.2, *reader.get(4));
        assert_eq!(4.4, *reader.get(5));
        assert_eq!(4.1, *reader.get(6));
        assert_eq!(4.2, *reader.get(7));
    }

    #[test]
    fn should_apply_sub_op() {
        let evaluated_expression = eval_d2d_direct_base(
            BinaryMathOperator::Sub,
            Cursor::new(0, 2, 6, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(0.09999999999999987, *reader.get(0));
        assert_eq!(0.0, *reader.get(1));
        assert_eq!(-0.40000000000000013, *reader.get(2));
        assert_eq!(0.0, *reader.get(3));
        assert_eq!(-0.40000000000000013, *reader.get(4));
        assert_eq!(0.0, *reader.get(5));
        assert_eq!(0.30000000000000027, *reader.get(6));
        assert_eq!(0.0, *reader.get(7));
    }

    #[test]
    fn should_apply_mul_op() {
        let evaluated_expression = eval_d2d_direct_base(
            BinaryMathOperator::Mul,
            Cursor::new(0, 2, 6, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.32, *reader.get(0));
        assert_eq!(1.44, *reader.get(1));
        assert_eq!(1.92, *reader.get(2));
        assert_eq!(1.9599999999999997, *reader.get(3));
        assert_eq!(2.52, *reader.get(4));
        assert_eq!(4.840000000000001, *reader.get(5));
        assert_eq!(4.18, *reader.get(6));
        assert_eq!(4.41, *reader.get(7));
    }

    #[test]
    fn should_apply_div_op() {
        let evaluated_expression = eval_d2d_direct_base(
            BinaryMathOperator::Div,
            Cursor::new(0, 2, 6, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.0909090909090908, *reader.get(0));
        assert_eq!(1.0, *reader.get(1));
        assert_eq!(0.7499999999999999, *reader.get(2));
        assert_eq!(1.0, *reader.get(3));
        assert_eq!(0.7777777777777777, *reader.get(4));
        assert_eq!(1.0, *reader.get(5));
        assert_eq!(1.1578947368421053, *reader.get(6));
        assert_eq!(1.0, *reader.get(7));
    }

    #[test]
    fn should_apply_pow_op() {
        let evaluated_expression = eval_d2d_direct_base(
            BinaryMathOperator::Pow,
            Cursor::new(0, 2, 6, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.222079251376429, *reader.get(0));
        assert_eq!(1.2445647472039776, *reader.get(1));
        assert_eq!(1.338720746075793, *reader.get(2));
        assert_eq!(1.601692898202212, *reader.get(3));
        assert_eq!(1.8324429572510013, *reader.get(4));
        assert_eq!(5.666695778750081, *reader.get(5));
        assert_eq!(4.4730432104685045, *reader.get(6));
        assert_eq!(4.749638091742242, *reader.get(7));
    }

    #[test]
    fn should_apply_max_op() {
        let evaluated_expression = eval_d2d_direct_base(
            BinaryMathOperator::Max,
            Cursor::new(0, 2, 6, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.2, *reader.get(0));
        assert_eq!(1.2, *reader.get(1));
        assert_eq!(1.6, *reader.get(2));
        assert_eq!(1.4, *reader.get(3));
        assert_eq!(1.8, *reader.get(4));
        assert_eq!(2.2, *reader.get(5));
        assert_eq!(2.2, *reader.get(6));
        assert_eq!(2.1, *reader.get(7));
    }

    #[test]
    fn should_apply_min_op() {
        let evaluated_expression = eval_d2d_direct_base(
            BinaryMathOperator::Min,
            Cursor::new(0, 2, 6, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_5);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.1, *reader.get(0));
        assert_eq!(1.2, *reader.get(1));
        assert_eq!(1.2, *reader.get(2));
        assert_eq!(1.4, *reader.get(3));
        assert_eq!(1.4, *reader.get(4));
        assert_eq!(2.2, *reader.get(5));
        assert_eq!(1.9, *reader.get(6));
        assert_eq!(2.1, *reader.get(7));
    }

    fn eval_d2d_direct_base(
        operator: BinaryMathOperator,
        cursor: Cursor,
        cursor_expansion: CursorExpansion,
    ) -> WideBinaryMathExpr {
        let frame_quotes = Arc::new(FrameQuotes::new(
            vec![0f64; 16],
            vec![0f64; 16],
            vec![0f64; 16],
            vec![
                1.1, 1.2, 1.6, 1.4, 1.8, 2.2, 1.9, 2.1, 2.0, 1.4, 1.6, 1.9, 2.4, 2.6, 2.3, 2.1,
            ],
            TimestampVector::from_utc(
                (0..16)
                    .map(|i| (i + 1) as u64 * u64::from(*time_frame::SECOND_5))
                    .collect(),
            ),
            time_frame::SECOND_5,
        ));

        let base_quotes = Arc::new(BaseQuotes::new(Arc::clone(&frame_quotes)));

        let all_frame_quotes = BTreeMap::from([
            (time_frame::SECOND_5, Arc::clone(&frame_quotes)),
            (
                time_frame::SECOND_10,
                Arc::new(build_frame_to_frame(&frame_quotes, time_frame::SECOND_10).unwrap()),
            ),
        ]);

        let tlbs = BTreeMap::from([
            (
                time_frame::SECOND_5,
                Arc::new(FrameTranslationBuffer::IdentityTranslationBuffer(
                    IdentityTranslationBuffer::new(time_frame::SECOND_5),
                )),
            ),
            (
                time_frame::SECOND_10,
                Arc::new(build_translation_buffer(&base_quotes, time_frame::SECOND_10).unwrap()),
            ),
        ]);

        let session = Arc::new(SessionContext::new(
            String::from("GOLD"),
            base_quotes,
            all_frame_quotes,
            tlbs,
            SessionConfiguration::new(8, 64),
        ));

        let mut id_gen = 0usize;
        let left = primitive_processor_handle(
            &mut id_gen,
            session.frame_quotes(time_frame::SECOND_10).unwrap(),
            PrimitiveOutputSelector::Close,
        );
        let right = primitive_processor_handle(
            &mut id_gen,
            session.frame_quotes(time_frame::SECOND_5).unwrap(),
            PrimitiveOutputSelector::Close,
        );

        let strategy =
            select_translation_strategy(&cursor, time_frame::SECOND_10, time_frame::SECOND_5)
                .unwrap();

        match &strategy {
            TranslationStrategy::DirectBaseLifetime(descriptor) => {
                assert_eq!(time_frame::SECOND_10, descriptor.left());
                assert_eq!(time_frame::SECOND_5, descriptor.right());
                assert_eq!(time_frame::SECOND_5, descriptor.output());
            }
            _ => panic!("invalid strategy has been chosen"),
        }

        let output =
            WideBinaryExprOutput::new(strategy, SpatialBuffer::new(time_frame::SECOND_5, 64));

        let ctx = SpanContext::new(cursor, session);
        let mut expr = WideBinaryMathExpr::new(left, operator, right, output);
        expr.eval(
            &ctx,
            cursor_expansion,
            &DynamicStore::new(ComputeId::Simple(3)),
        )
        .expect("success");

        expr
    }
}

#[cfg(test)]
mod d2d_direct {
    use crate::sim::bb::cursor::{Cursor, CursorExpansion, Epoch};
    use crate::sim::bb::quotes::{BaseQuotes, FrameQuotes, TimestampVector};
    use crate::sim::bb::{time_frame, ComputeId};
    use crate::sim::builder::frame_builder::build_frame_to_frame;
    use crate::sim::builder::translation_builder::build_translation_buffer;
    use crate::sim::compute::operators::BinaryMathOperator;
    use crate::sim::compute::translation_strategy::{
        select_translation_strategy, TranslationStrategy,
    };
    use crate::sim::compute::wide::expr::math_tests::primitive_processor_handle;
    use crate::sim::compute::wide::expr::primitive::PrimitiveOutputSelector;
    use crate::sim::compute::wide::expr::{WideBinaryExprOutput, WideBinaryMathExpr};
    use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
    use crate::sim::compute::wide::indicator::{SingleOutputSelector, WideProcessor};
    use crate::sim::context::{SessionConfiguration, SessionContext, SpanContext};
    use crate::sim::selector::Selector;
    use crate::sim::spatial_buffer::SpatialBuffer;
    use crate::sim::tlb::{FrameTranslationBuffer, IdentityTranslationBuffer};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn should_add_direct() {
        let evaluated = eval_d2d_direct(
            BinaryMathOperator::Add,
            Cursor::new(0, 0, 8, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..4, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(2.2, *reader.get(0));
        assert_eq!(2.8, *reader.get(1));
        assert_eq!(3.6, *reader.get(2));
        assert_eq!(3.8, *reader.get(3));
    }

    #[test]
    fn should_sub_direct() {
        let evaluated = eval_d2d_direct(
            BinaryMathOperator::Sub,
            Cursor::new(0, 0, 8, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..4, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(0.0, *reader.get(0));
        assert_eq!(0.0, *reader.get(1));
        assert_eq!(0.0, *reader.get(2));
        assert_eq!(0.0, *reader.get(3));
    }

    #[test]
    fn should_mul_direct() {
        let evaluated = eval_d2d_direct(
            BinaryMathOperator::Mul,
            Cursor::new(0, 0, 8, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..4, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.2100000000000002, *reader.get(0));
        assert_eq!(1.9599999999999997, *reader.get(1));
        assert_eq!(3.24, *reader.get(2));
        assert_eq!(3.61, *reader.get(3));
    }

    #[test]
    fn should_div_direct() {
        let evaluated = eval_d2d_direct(
            BinaryMathOperator::Div,
            Cursor::new(0, 0, 8, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..4, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.0, *reader.get(0));
        assert_eq!(1.0, *reader.get(1));
        assert_eq!(1.0, *reader.get(2));
        assert_eq!(1.0, *reader.get(3));
    }

    #[test]
    fn should_pow_direct() {
        let evaluated = eval_d2d_direct(
            BinaryMathOperator::Pow,
            Cursor::new(0, 0, 8, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..4, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.1105342410545758, *reader.get(0));
        assert_eq!(1.601692898202212, *reader.get(1));
        assert_eq!(2.880650097068328, *reader.get(2));
        assert_eq!(3.3855703439184803, *reader.get(3));
    }

    #[test]
    fn should_max_direct() {
        let evaluated = eval_d2d_direct(
            BinaryMathOperator::Max,
            Cursor::new(0, 0, 8, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..4, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.1, *reader.get(0));
        assert_eq!(1.4, *reader.get(1));
        assert_eq!(1.8, *reader.get(2));
        assert_eq!(1.9, *reader.get(3));
    }

    #[test]
    fn should_min_direct() {
        let evaluated = eval_d2d_direct(
            BinaryMathOperator::Min,
            Cursor::new(0, 0, 8, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..4, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.1, *reader.get(0));
        assert_eq!(1.4, *reader.get(1));
        assert_eq!(1.8, *reader.get(2));
        assert_eq!(1.9, *reader.get(3));
    }

    fn eval_d2d_direct(
        operator: BinaryMathOperator,
        cursor: Cursor,
        cursor_expansion: CursorExpansion,
    ) -> WideBinaryMathExpr {
        let frame_quotes = Arc::new(FrameQuotes::new(
            vec![0f64; 16],
            vec![0f64; 16],
            vec![
                1.1, 1.2, 1.6, 1.4, 1.8, 2.2, 1.9, 2.1, 2.0, 1.4, 1.6, 1.9, 2.4, 2.6, 2.3, 2.1,
            ],
            vec![0f64; 16],
            TimestampVector::from_utc(
                (0..16)
                    .map(|i| (i + 1) as u64 * u64::from(*time_frame::SECOND_5))
                    .collect(),
            ),
            time_frame::SECOND_5,
        ));

        let base_quotes = Arc::new(BaseQuotes::new(Arc::clone(&frame_quotes)));

        let all_frame_quotes = BTreeMap::from([
            (time_frame::SECOND_5, Arc::clone(&frame_quotes)),
            (
                time_frame::SECOND_10,
                Arc::new(build_frame_to_frame(&frame_quotes, time_frame::SECOND_10).unwrap()),
            ),
        ]);

        let tlbs = BTreeMap::from([
            (
                time_frame::SECOND_5,
                Arc::new(FrameTranslationBuffer::IdentityTranslationBuffer(
                    IdentityTranslationBuffer::new(time_frame::SECOND_5),
                )),
            ),
            (
                time_frame::SECOND_10,
                Arc::new(build_translation_buffer(&base_quotes, time_frame::SECOND_10).unwrap()),
            ),
        ]);

        let session = Arc::new(SessionContext::new(
            String::from("GOLD"),
            base_quotes,
            all_frame_quotes,
            tlbs,
            SessionConfiguration::new(8, 64),
        ));

        let mut id_gen = 0usize;
        let left = primitive_processor_handle(
            &mut id_gen,
            session.frame_quotes(time_frame::SECOND_10).unwrap(),
            PrimitiveOutputSelector::Low,
        );
        let right = primitive_processor_handle(
            &mut id_gen,
            session.frame_quotes(time_frame::SECOND_10).unwrap(),
            PrimitiveOutputSelector::Low,
        );

        let strategy =
            select_translation_strategy(&cursor, time_frame::SECOND_10, time_frame::SECOND_10)
                .unwrap();

        match &strategy {
            TranslationStrategy::Direct(descriptor) => {
                assert_eq!(time_frame::SECOND_10, descriptor.left());
                assert_eq!(time_frame::SECOND_10, descriptor.right());
                assert_eq!(time_frame::SECOND_10, descriptor.output());
            }
            _ => panic!("invalid strategy has been chosen"),
        }

        let output =
            WideBinaryExprOutput::new(strategy, SpatialBuffer::new(time_frame::SECOND_10, 64));

        let ctx = SpanContext::new(cursor, session);
        let mut expr = WideBinaryMathExpr::new(left, operator, right, output);
        expr.eval(
            &ctx,
            cursor_expansion,
            &DynamicStore::new(ComputeId::Simple(3)),
        )
        .expect("success");

        expr
    }
}

#[cfg(test)]
mod d2d_direct_lifetime {
    use crate::sim::bb::cursor::{Cursor, CursorExpansion, Epoch};
    use crate::sim::bb::quotes::{BaseQuotes, FrameQuotes, TimestampVector};
    use crate::sim::bb::{time_frame, ComputeId};
    use crate::sim::builder::frame_builder::build_frame_to_frame;
    use crate::sim::builder::translation_builder::build_translation_buffer;
    use crate::sim::compute::operators::BinaryMathOperator;
    use crate::sim::compute::translation_strategy::{
        select_translation_strategy, TranslationStrategy,
    };
    use crate::sim::compute::wide::expr::math_tests::primitive_processor_handle;
    use crate::sim::compute::wide::expr::primitive::PrimitiveOutputSelector;
    use crate::sim::compute::wide::expr::{WideBinaryExprOutput, WideBinaryMathExpr};
    use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
    use crate::sim::compute::wide::indicator::{SingleOutputSelector, WideProcessor};
    use crate::sim::context::{SessionConfiguration, SessionContext, SpanContext};
    use crate::sim::selector::Selector;
    use crate::sim::spatial_buffer::SpatialBuffer;
    use crate::sim::tlb::{FrameTranslationBuffer, IdentityTranslationBuffer};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn should_add_by_direct_lifetime() {
        let evaluated = eval_d2d_direct_lifetime(
            BinaryMathOperator::Add,
            Cursor::new(0, 0, 16, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..5, time_frame::SECOND_15);
        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(3.8000000000000003, *reader.get(0));
        assert_eq!(4.4, *reader.get(1));
        assert_eq!(4.300000000000001, *reader.get(2));
        assert_eq!(4.0, *reader.get(3));
        assert_eq!(4.7, *reader.get(4));
    }

    #[test]
    fn should_sub_by_direct_lifetime() {
        let evaluated = eval_d2d_direct_lifetime(
            BinaryMathOperator::Sub,
            Cursor::new(0, 0, 16, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..5, time_frame::SECOND_15);
        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(-0.6000000000000001, *reader.get(0));
        assert_eq!(0.0, *reader.get(1));
        assert_eq!(-0.10000000000000009, *reader.get(2));
        assert_eq!(-0.20000000000000018, *reader.get(3));
        assert_eq!(0.5, *reader.get(4));
    }

    #[test]
    fn should_mul_by_direct_lifetime() {
        let evaluated = eval_d2d_direct_lifetime(
            BinaryMathOperator::Mul,
            Cursor::new(0, 0, 16, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..5, time_frame::SECOND_15);
        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(3.5200000000000005, *reader.get(0));
        assert_eq!(4.840000000000001, *reader.get(1));
        assert_eq!(4.620000000000001, *reader.get(2));
        assert_eq!(3.9899999999999998, *reader.get(3));
        assert_eq!(5.460000000000001, *reader.get(4));
    }

    #[test]
    fn should_div_by_direct_lifetime() {
        let evaluated = eval_d2d_direct_lifetime(
            BinaryMathOperator::Div,
            Cursor::new(0, 0, 16, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..5, time_frame::SECOND_15);
        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(0.7272727272727273, *reader.get(0));
        assert_eq!(1.0, *reader.get(1));
        assert_eq!(0.9545454545454545, *reader.get(2));
        assert_eq!(0.9047619047619047, *reader.get(3));
        assert_eq!(1.2380952380952381, *reader.get(4));
    }

    #[test]
    fn should_pow_by_direct_lifetime() {
        let evaluated = eval_d2d_direct_lifetime(
            BinaryMathOperator::Pow,
            Cursor::new(0, 0, 16, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..5, time_frame::SECOND_15);
        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(2.812314990863662, *reader.get(0));
        assert_eq!(5.666695778750081, *reader.get(1));
        assert_eq!(5.1154335606641474, *reader.get(2));
        assert_eq!(3.8493071111074784, *reader.get(3));
        assert_eq!(7.437792029396684, *reader.get(4));
    }

    #[test]
    fn should_max_by_direct_lifetime() {
        let evaluated = eval_d2d_direct_lifetime(
            BinaryMathOperator::Max,
            Cursor::new(0, 0, 16, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..5, time_frame::SECOND_15);
        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(2.2, *reader.get(0));
        assert_eq!(2.2, *reader.get(1));
        assert_eq!(2.2, *reader.get(2));
        assert_eq!(2.1, *reader.get(3));
        assert_eq!(2.6, *reader.get(4));
    }

    #[test]
    fn should_min_by_direct_lifetime() {
        let evaluated = eval_d2d_direct_lifetime(
            BinaryMathOperator::Min,
            Cursor::new(0, 0, 16, time_frame::SECOND_5),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..5, time_frame::SECOND_15);
        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.6, *reader.get(0));
        assert_eq!(2.2, *reader.get(1));
        assert_eq!(2.1, *reader.get(2));
        assert_eq!(1.9, *reader.get(3));
        assert_eq!(2.1, *reader.get(4));
    }

    fn eval_d2d_direct_lifetime(
        operator: BinaryMathOperator,
        cursor: Cursor,
        cursor_expansion: CursorExpansion,
    ) -> WideBinaryMathExpr {
        let frame_quotes = Arc::new(FrameQuotes::new(
            vec![0f64; 64],
            vec![
                1.1, 1.2, 1.6, 1.4, 1.8, 2.2, 1.9, 2.1, 2.0, 1.4, 1.6, 1.9, 2.4, 2.6, 2.3, 2.1,
                1.1, 1.2, 1.6, 1.4, 1.8, 2.2, 1.9, 2.1, 2.0, 1.4, 1.6, 1.9, 2.4, 2.6, 2.3, 2.1,
                1.1, 1.2, 1.6, 1.4, 1.8, 2.2, 1.9, 2.1, 2.0, 1.4, 1.6, 1.9, 2.4, 2.6, 2.3, 2.1,
                1.1, 1.2, 1.6, 1.4, 1.8, 2.2, 1.9, 2.1, 2.0, 1.4, 1.6, 1.9, 2.4, 2.6, 2.3, 2.1,
            ],
            vec![0f64; 64],
            vec![0f64; 64],
            TimestampVector::from_utc(
                (0..64)
                    .map(|i| (i + 1) as u64 * u64::from(*time_frame::SECOND_5))
                    .collect(),
            ),
            time_frame::SECOND_5,
        ));

        let base_quotes = Arc::new(BaseQuotes::new(Arc::clone(&frame_quotes)));

        let all_frame_quotes = BTreeMap::from([
            (time_frame::SECOND_5, Arc::clone(&frame_quotes)),
            (
                time_frame::SECOND_15,
                Arc::new(build_frame_to_frame(&frame_quotes, time_frame::SECOND_15).unwrap()),
            ),
            (
                time_frame::SECOND_30,
                Arc::new(build_frame_to_frame(&frame_quotes, time_frame::SECOND_30).unwrap()),
            ),
        ]);

        let tlbs = BTreeMap::from([
            (
                time_frame::SECOND_5,
                Arc::new(FrameTranslationBuffer::IdentityTranslationBuffer(
                    IdentityTranslationBuffer::new(time_frame::SECOND_5),
                )),
            ),
            (
                time_frame::SECOND_15,
                Arc::new(build_translation_buffer(&base_quotes, time_frame::SECOND_15).unwrap()),
            ),
            (
                time_frame::SECOND_30,
                Arc::new(build_translation_buffer(&base_quotes, time_frame::SECOND_30).unwrap()),
            ),
        ]);

        let session = Arc::new(SessionContext::new(
            String::from("GOLD"),
            base_quotes,
            all_frame_quotes,
            tlbs,
            SessionConfiguration::new(8, 64),
        ));

        let mut id_gen = 0usize;
        let left = primitive_processor_handle(
            &mut id_gen,
            session.frame_quotes(time_frame::SECOND_15).unwrap(),
            PrimitiveOutputSelector::High,
        );
        let right = primitive_processor_handle(
            &mut id_gen,
            session.frame_quotes(time_frame::SECOND_30).unwrap(),
            PrimitiveOutputSelector::High,
        );

        let strategy =
            select_translation_strategy(&cursor, time_frame::SECOND_15, time_frame::SECOND_30)
                .unwrap();

        match &strategy {
            TranslationStrategy::DirectLifetime(descriptor) => {
                assert_eq!(time_frame::SECOND_15, descriptor.left());
                assert_eq!(time_frame::SECOND_30, descriptor.right());
                assert_eq!(time_frame::SECOND_15, descriptor.output());
            }
            _ => panic!("invalid strategy has been chosen"),
        }

        let output =
            WideBinaryExprOutput::new(strategy, SpatialBuffer::new(time_frame::SECOND_15, 64));

        let ctx = SpanContext::new(cursor, session);
        let mut expr = WideBinaryMathExpr::new(left, operator, right, output);
        expr.eval(
            &ctx,
            cursor_expansion,
            &DynamicStore::new(ComputeId::Simple(3)),
        )
        .expect("success");

        expr
    }
}

#[cfg(test)]
mod d2d_waterfall_lifetime {
    use crate::sim::bb::cursor::{Cursor, CursorExpansion, Epoch};
    use crate::sim::bb::quotes::{BaseQuotes, FrameQuotes, TimestampVector};
    use crate::sim::bb::{time_frame, ComputeId};
    use crate::sim::builder::frame_builder::build_frame_to_frame;
    use crate::sim::builder::translation_builder::build_translation_buffer;
    use crate::sim::compute::operators::BinaryMathOperator;
    use crate::sim::compute::translation_strategy::{
        select_translation_strategy, TranslationStrategy,
    };
    use crate::sim::compute::wide::expr::math_tests::primitive_processor_handle;
    use crate::sim::compute::wide::expr::primitive::PrimitiveOutputSelector;
    use crate::sim::compute::wide::expr::{WideBinaryExprOutput, WideBinaryMathExpr};
    use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
    use crate::sim::compute::wide::indicator::{SingleOutputSelector, WideProcessor};
    use crate::sim::context::{SessionConfiguration, SessionContext, SpanContext};
    use crate::sim::selector::Selector;
    use crate::sim::spatial_buffer::SpatialBuffer;
    use crate::sim::tlb::{FrameTranslationBuffer, IdentityTranslationBuffer};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn should_add_by_waterfall_lifetime() {
        let evaluated = eval_d2d_waterfall_lifetime(
            BinaryMathOperator::Add,
            Cursor::new(0, 0, 16, time_frame::SECOND_30),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::MINUTE_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(2.2, *reader.get(0));
        assert_eq!(2.2, *reader.get(1));
        assert_eq!(2.2, *reader.get(2));
        assert_eq!(2.2, *reader.get(3));
        assert_eq!(2.2, *reader.get(4));
        assert_eq!(2.5, *reader.get(5));
        assert_eq!(2.5, *reader.get(6));
        assert_eq!(2.5, *reader.get(7));
    }

    #[test]
    fn should_sub_by_waterfall_lifetime() {
        let evaluated = eval_d2d_waterfall_lifetime(
            BinaryMathOperator::Sub,
            Cursor::new(0, 0, 16, time_frame::SECOND_30),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::MINUTE_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(0.0, *reader.get(0));
        assert_eq!(0.0, *reader.get(1));
        assert_eq!(0.0, *reader.get(2));
        assert_eq!(0.0, *reader.get(3));
        assert_eq!(0.0, *reader.get(4));
        assert_eq!(0.2999999999999998, *reader.get(5));
        assert_eq!(0.2999999999999998, *reader.get(6));
        assert_eq!(0.2999999999999998, *reader.get(7));
    }

    #[test]
    fn should_mul_by_waterfall_lifetime() {
        let evaluated = eval_d2d_waterfall_lifetime(
            BinaryMathOperator::Mul,
            Cursor::new(0, 0, 16, time_frame::SECOND_30),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::MINUTE_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.2100000000000002, *reader.get(0));
        assert_eq!(1.2100000000000002, *reader.get(1));
        assert_eq!(1.2100000000000002, *reader.get(2));
        assert_eq!(1.2100000000000002, *reader.get(3));
        assert_eq!(1.2100000000000002, *reader.get(4));
        assert_eq!(1.54, *reader.get(5));
        assert_eq!(1.54, *reader.get(6));
        assert_eq!(1.54, *reader.get(7));
    }

    #[test]
    fn should_div_by_waterfall_lifetime() {
        let evaluated = eval_d2d_waterfall_lifetime(
            BinaryMathOperator::Div,
            Cursor::new(0, 0, 16, time_frame::SECOND_30),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::MINUTE_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.0, *reader.get(0));
        assert_eq!(1.0, *reader.get(1));
        assert_eq!(1.0, *reader.get(2));
        assert_eq!(1.0, *reader.get(3));
        assert_eq!(1.0, *reader.get(4));
        assert_eq!(1.2727272727272725, *reader.get(5));
        assert_eq!(1.2727272727272725, *reader.get(6));
        assert_eq!(1.2727272727272725, *reader.get(7));
    }

    #[test]
    fn should_pow_by_waterfall_lifetime() {
        let evaluated = eval_d2d_waterfall_lifetime(
            BinaryMathOperator::Pow,
            Cursor::new(0, 0, 16, time_frame::SECOND_30),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::MINUTE_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.1105342410545758, *reader.get(0));
        assert_eq!(1.1105342410545758, *reader.get(1));
        assert_eq!(1.1105342410545758, *reader.get(2));
        assert_eq!(1.1105342410545758, *reader.get(3));
        assert_eq!(1.1105342410545758, *reader.get(4));
        assert_eq!(1.4479075717811323, *reader.get(5));
        assert_eq!(1.4479075717811323, *reader.get(6));
        assert_eq!(1.4479075717811323, *reader.get(7));
    }

    #[test]
    fn should_max_by_waterfall_lifetime() {
        let evaluated = eval_d2d_waterfall_lifetime(
            BinaryMathOperator::Max,
            Cursor::new(0, 0, 16, time_frame::SECOND_30),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::MINUTE_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.1, *reader.get(0));
        assert_eq!(1.1, *reader.get(1));
        assert_eq!(1.1, *reader.get(2));
        assert_eq!(1.1, *reader.get(3));
        assert_eq!(1.1, *reader.get(4));
        assert_eq!(1.4, *reader.get(5));
        assert_eq!(1.4, *reader.get(6));
        assert_eq!(1.4, *reader.get(7));
    }

    #[test]
    fn should_min_by_waterfall_lifetime() {
        let evaluated = eval_d2d_waterfall_lifetime(
            BinaryMathOperator::Min,
            Cursor::new(0, 0, 16, time_frame::SECOND_30),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated
            .output_buffer(SingleOutputSelector.ordinal())
            .unwrap();

        let epoch_read = Epoch::new(0..8, time_frame::MINUTE_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.1, *reader.get(0));
        assert_eq!(1.1, *reader.get(1));
        assert_eq!(1.1, *reader.get(2));
        assert_eq!(1.1, *reader.get(3));
        assert_eq!(1.1, *reader.get(4));
        assert_eq!(1.1, *reader.get(5));
        assert_eq!(1.1, *reader.get(6));
        assert_eq!(1.1, *reader.get(7));
    }

    fn eval_d2d_waterfall_lifetime(
        operator: BinaryMathOperator,
        cursor: Cursor,
        cursor_expansion: CursorExpansion,
    ) -> WideBinaryMathExpr {
        let frame_quotes = Arc::new(FrameQuotes::new(
            vec![0f64; 64],
            vec![0f64; 64],
            vec![
                1.1, 1.2, 1.6, 1.4, 1.8, 2.2, 1.9, 2.1, 2.0, 1.4, 1.6, 1.9, 2.4, 2.6, 2.3, 2.1,
                1.1, 1.2, 1.6, 1.4, 1.8, 2.2, 1.9, 2.1, 2.0, 1.4, 1.6, 1.9, 2.4, 2.6, 2.3, 2.1,
                1.1, 1.2, 1.6, 1.4, 1.8, 2.2, 1.9, 2.1, 2.0, 1.4, 1.6, 1.9, 2.4, 2.6, 2.3, 2.1,
                1.1, 1.2, 1.6, 1.4, 1.8, 2.2, 1.9, 2.1, 2.0, 1.4, 1.6, 1.9, 2.4, 2.6, 2.3, 2.1,
            ],
            vec![0f64; 64],
            TimestampVector::from_utc(
                (0..64)
                    .map(|i| (i + 1) as u64 * u64::from(*time_frame::SECOND_30))
                    .collect(),
            ),
            time_frame::SECOND_30,
        ));

        let base_quotes = Arc::new(BaseQuotes::new(Arc::clone(&frame_quotes)));

        let all_frame_quotes = BTreeMap::from([
            (time_frame::SECOND_30, Arc::clone(&frame_quotes)),
            (
                time_frame::MINUTE_1,
                Arc::new(build_frame_to_frame(&frame_quotes, time_frame::MINUTE_1).unwrap()),
            ),
            (
                time_frame::MINUTE_3,
                Arc::new(build_frame_to_frame(&frame_quotes, time_frame::MINUTE_3).unwrap()),
            ),
            (
                time_frame::MINUTE_5,
                Arc::new(build_frame_to_frame(&frame_quotes, time_frame::MINUTE_5).unwrap()),
            ),
        ]);

        let tlbs = BTreeMap::from([
            (
                time_frame::SECOND_30,
                Arc::new(FrameTranslationBuffer::IdentityTranslationBuffer(
                    IdentityTranslationBuffer::new(time_frame::SECOND_30),
                )),
            ),
            (
                time_frame::MINUTE_1,
                Arc::new(build_translation_buffer(&base_quotes, time_frame::MINUTE_1).unwrap()),
            ),
            (
                time_frame::MINUTE_3,
                Arc::new(build_translation_buffer(&base_quotes, time_frame::MINUTE_3).unwrap()),
            ),
            (
                time_frame::MINUTE_5,
                Arc::new(build_translation_buffer(&base_quotes, time_frame::MINUTE_5).unwrap()),
            ),
        ]);

        let session = Arc::new(SessionContext::new(
            String::from("GOLD"),
            base_quotes,
            all_frame_quotes,
            tlbs,
            SessionConfiguration::new(8, 64),
        ));

        let mut id_gen = 0usize;
        let left = primitive_processor_handle(
            &mut id_gen,
            session.frame_quotes(time_frame::MINUTE_3).unwrap(),
            PrimitiveOutputSelector::Low,
        );
        let right = primitive_processor_handle(
            &mut id_gen,
            session.frame_quotes(time_frame::MINUTE_5).unwrap(),
            PrimitiveOutputSelector::Low,
        );

        let strategy =
            select_translation_strategy(&cursor, time_frame::MINUTE_3, time_frame::MINUTE_5)
                .unwrap();

        match &strategy {
            TranslationStrategy::WaterfallLifetime(descriptor) => {
                assert_eq!(time_frame::MINUTE_3, descriptor.left());
                assert_eq!(time_frame::MINUTE_5, descriptor.right());
                assert_eq!(time_frame::MINUTE_1, descriptor.output());
            }
            _ => panic!("invalid strategy has been chosen"),
        }

        let output =
            WideBinaryExprOutput::new(strategy, SpatialBuffer::new(time_frame::MINUTE_1, 64));

        let ctx = SpanContext::new(cursor, session);
        let mut expr = WideBinaryMathExpr::new(left, operator, right, output);
        expr.eval(
            &ctx,
            cursor_expansion,
            &DynamicStore::new(ComputeId::Simple(3)),
        )
        .expect("success");

        expr
    }
}

#[cfg(test)]
mod d2c {
    use crate::sim::bb::cursor::{Cursor, CursorExpansion, Epoch, TimeShift};
    use crate::sim::bb::quotes::{BaseQuotes, FrameQuotes, TimestampVector};
    use crate::sim::bb::{time_frame, ComputeId};
    use crate::sim::compute::operators::BinaryMathOperator;
    use crate::sim::compute::translation_strategy::select_translation_strategy;
    use crate::sim::compute::wide::expr::math_tests::{
        const_processor_handle, primitive_processor_handle,
    };
    use crate::sim::compute::wide::expr::primitive::PrimitiveOutputSelector;
    use crate::sim::compute::wide::expr::{WideBinaryExprOutput, WideBinaryMathExpr};
    use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
    use crate::sim::compute::wide::indicator::{SingleOutputSelector, WideProcessor};
    use crate::sim::context::{SessionConfiguration, SessionContext, SpanContext};
    use crate::sim::selector::Selector;
    use crate::sim::spatial_buffer::SpatialBuffer;
    use crate::sim::tlb::{FrameTranslationBuffer, IdentityTranslationBuffer};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn should_expand_epoch_to_the_past() {
        let evaluated_expression = eval_d2c(
            BinaryMathOperator::Add,
            Cursor::new(0, 3, 2, time_frame::SECOND_1),
            CursorExpansion::Identity.expand_by(TimeShift::Past(2)),
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..5, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(5.4, *reader.get(0));
        assert_eq!(4.2, *reader.get(1));
        assert_eq!(4.7, *reader.get(2));
        assert_eq!(5.8, *reader.get(3));
        assert_eq!(6.2, *reader.get(4));
    }

    #[test]
    fn should_expand_epoch_to_the_future() {
        let evaluated_expression = eval_d2c(
            BinaryMathOperator::Add,
            Cursor::new(0, 1, 2, time_frame::SECOND_1),
            CursorExpansion::Identity.expand_by(TimeShift::Future(2)),
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..5, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(5.4, *reader.get(0));
        assert_eq!(4.2, *reader.get(1));
        assert_eq!(4.7, *reader.get(2));
        assert_eq!(5.8, *reader.get(3));
        assert_eq!(6.2, *reader.get(4));
    }

    #[test]
    fn should_add_d2c() {
        let evaluated_expression = eval_d2c(
            BinaryMathOperator::Add,
            Cursor::new(0, 0, 8, time_frame::SECOND_1),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(5.4, *reader.get(0));
        assert_eq!(4.2, *reader.get(1));
        assert_eq!(4.7, *reader.get(2));
        assert_eq!(5.8, *reader.get(3));
        assert_eq!(6.2, *reader.get(4));
        assert_eq!(4.6, *reader.get(5));
        assert_eq!(4.4, *reader.get(6));
        assert_eq!(5.2, *reader.get(7));
    }

    #[test]
    fn should_subtract_d2c() {
        let evaluated_expression = eval_d2c(
            BinaryMathOperator::Sub,
            Cursor::new(0, 0, 8, time_frame::SECOND_1),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(-0.6000000000000001, *reader.get(0));
        assert_eq!(-1.8, *reader.get(1));
        assert_eq!(-1.3, *reader.get(2));
        assert_eq!(-0.20000000000000018, *reader.get(3));
        assert_eq!(0.20000000000000018, *reader.get(4));
        assert_eq!(-1.4, *reader.get(5));
        assert_eq!(-1.6, *reader.get(6));
        assert_eq!(-0.7999999999999998, *reader.get(7));
    }

    #[test]
    fn should_multiply_d2c() {
        let evaluated_expression = eval_d2c(
            BinaryMathOperator::Mul,
            Cursor::new(0, 0, 8, time_frame::SECOND_1),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(7.199999999999999, *reader.get(0));
        assert_eq!(3.5999999999999996, *reader.get(1));
        assert_eq!(5.1, *reader.get(2));
        assert_eq!(8.399999999999999, *reader.get(3));
        assert_eq!(9.600000000000001, *reader.get(4));
        assert_eq!(4.800000000000001, *reader.get(5));
        assert_eq!(4.199999999999999, *reader.get(6));
        assert_eq!(6.6000000000000005, *reader.get(7));
    }

    #[test]
    fn should_divide_d2c() {
        let evaluated_expression = eval_d2c(
            BinaryMathOperator::Div,
            Cursor::new(0, 0, 8, time_frame::SECOND_1),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(0.7999999999999999, *reader.get(0));
        assert_eq!(0.39999999999999997, *reader.get(1));
        assert_eq!(0.5666666666666667, *reader.get(2));
        assert_eq!(0.9333333333333332, *reader.get(3));
        assert_eq!(1.0666666666666667, *reader.get(4));
        assert_eq!(0.5333333333333333, *reader.get(5));
        assert_eq!(0.4666666666666666, *reader.get(6));
        assert_eq!(0.7333333333333334, *reader.get(7));
    }

    #[test]
    fn should_pow_d2c() {
        let evaluated_expression = eval_d2c(
            BinaryMathOperator::Pow,
            Cursor::new(0, 0, 8, time_frame::SECOND_1),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(13.823999999999998, *reader.get(0));
        assert_eq!(1.7279999999999998, *reader.get(1));
        assert_eq!(4.912999999999999, *reader.get(2));
        assert_eq!(21.951999999999995, *reader.get(3));
        assert_eq!(32.76800000000001, *reader.get(4));
        assert_eq!(4.096000000000001, *reader.get(5));
        assert_eq!(2.7439999999999993, *reader.get(6));
        assert_eq!(10.648000000000003, *reader.get(7));
    }

    #[test]
    fn should_max_d2c() {
        let evaluated_expression = eval_d2c(
            BinaryMathOperator::Max,
            Cursor::new(0, 0, 8, time_frame::SECOND_1),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(3.0, *reader.get(0));
        assert_eq!(3.0, *reader.get(1));
        assert_eq!(3.0, *reader.get(2));
        assert_eq!(3.0, *reader.get(3));
        assert_eq!(3.2, *reader.get(4));
        assert_eq!(3.0, *reader.get(5));
        assert_eq!(3.0, *reader.get(6));
        assert_eq!(3.0, *reader.get(7));
    }

    #[test]
    fn should_min_d2c() {
        let evaluated_expression = eval_d2c(
            BinaryMathOperator::Min,
            Cursor::new(0, 0, 8, time_frame::SECOND_1),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(2.4, *reader.get(0));
        assert_eq!(1.2, *reader.get(1));
        assert_eq!(1.7, *reader.get(2));
        assert_eq!(2.8, *reader.get(3));
        assert_eq!(3.0, *reader.get(4));
        assert_eq!(1.6, *reader.get(5));
        assert_eq!(1.4, *reader.get(6));
        assert_eq!(2.2, *reader.get(7));
    }

    fn eval_d2c(
        operator: BinaryMathOperator,
        cursor: Cursor,
        cursor_expansion: CursorExpansion,
    ) -> WideBinaryMathExpr {
        let frame_quotes = Arc::new(FrameQuotes::new(
            vec![0f64; 8],
            vec![2.4f64, 1.2, 1.7, 2.8, 3.2, 1.6, 1.4, 2.2],
            vec![0f64; 8],
            vec![0f64; 8],
            TimestampVector::from_utc(
                (0..8)
                    .map(|i| i as u64 * u64::from(*time_frame::SECOND_1))
                    .collect(),
            ),
            time_frame::SECOND_1,
        ));

        let mut id_gen = 0usize;
        let left = primitive_processor_handle(
            &mut id_gen,
            Arc::clone(&frame_quotes),
            PrimitiveOutputSelector::High,
        );
        let right = const_processor_handle(&mut id_gen, 3f64);

        let strategy =
            select_translation_strategy(&cursor, time_frame::SECOND_1, time_frame::SECOND_1)
                .unwrap();

        let mut expr = WideBinaryMathExpr::new(
            left,
            operator,
            right,
            WideBinaryExprOutput::new(strategy, SpatialBuffer::new(time_frame::SECOND_1, 32)),
        );

        let base_quotes = Arc::new(BaseQuotes::new(Arc::clone(&frame_quotes)));
        let session = SessionContext::new(
            String::from("GOLD"),
            base_quotes,
            BTreeMap::from([(time_frame::SECOND_1, Arc::clone(&frame_quotes))]),
            BTreeMap::from([(
                time_frame::SECOND_1,
                Arc::new(FrameTranslationBuffer::IdentityTranslationBuffer(
                    IdentityTranslationBuffer::new(time_frame::SECOND_1),
                )),
            )]),
            SessionConfiguration::new(cursor.step(), 32),
        );

        let ctx = SpanContext::new(cursor, Arc::new(session));

        expr.eval(
            &ctx,
            cursor_expansion,
            &DynamicStore::new(ComputeId::Simple(0)),
        )
        .expect("success");

        expr
    }
}

#[cfg(test)]
mod c2d_const_to_upper {
    use crate::sim::bb::cursor::{Cursor, CursorExpansion, Epoch};
    use crate::sim::bb::quotes::{BaseQuotes, FrameQuotes, TimestampVector};
    use crate::sim::bb::{time_frame, ComputeId};
    use crate::sim::builder::frame_builder::build_frame_to_frame;
    use crate::sim::builder::translation_builder::build_translation_buffer;
    use crate::sim::compute::operators::BinaryMathOperator;
    use crate::sim::compute::translation_strategy::select_translation_strategy;
    use crate::sim::compute::wide::expr::math_tests::{
        const_processor_handle, primitive_processor_handle,
    };
    use crate::sim::compute::wide::expr::primitive::PrimitiveOutputSelector;
    use crate::sim::compute::wide::expr::{WideBinaryExprOutput, WideBinaryMathExpr};
    use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
    use crate::sim::compute::wide::indicator::{SingleOutputSelector, WideProcessor};
    use crate::sim::context::{SessionConfiguration, SessionContext, SpanContext};
    use crate::sim::selector::Selector;
    use crate::sim::spatial_buffer::SpatialBuffer;
    use crate::sim::tlb::{FrameTranslationBuffer, IdentityTranslationBuffer};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn should_apply_add_op() {
        let cursor = Cursor::new(0, 0, 11, time_frame::SECOND_5);
        let evaluated_expression =
            eval_c2d(BinaryMathOperator::Add, cursor, CursorExpansion::Identity);

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..6, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(3.0, *reader.get(0));
        assert_eq!(4.0, *reader.get(1));
        assert_eq!(4.6, *reader.get(2));
        assert_eq!(4.5, *reader.get(3));
        assert_eq!(3.1, *reader.get(4));
        assert_eq!(3.2, *reader.get(5));
    }

    #[test]
    fn should_apply_sub_op() {
        let cursor = Cursor::new(0, 0, 11, time_frame::SECOND_5);
        let evaluated_expression =
            eval_c2d(BinaryMathOperator::Sub, cursor, CursorExpansion::Identity);

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..6, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        (0..6).for_each(|i| {
            let value = *reader.get(i);
            println!("assert_eq!({}, *reader.get({}));", value, i);
        });

        assert_eq!(1.0, *reader.get(0));
        assert_eq!(0.0, *reader.get(1));
        assert_eq!(-0.6000000000000001, *reader.get(2));
        assert_eq!(-0.5, *reader.get(3));
        assert_eq!(0.8999999999999999, *reader.get(4));
        assert_eq!(0.8, *reader.get(5));
    }

    #[test]
    fn should_apply_mul_op() {
        let cursor = Cursor::new(0, 0, 11, time_frame::SECOND_5);
        let evaluated_expression =
            eval_c2d(BinaryMathOperator::Mul, cursor, CursorExpansion::Identity);

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..6, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(2.0, *reader.get(0));
        assert_eq!(4.0, *reader.get(1));
        assert_eq!(5.2, *reader.get(2));
        assert_eq!(5.0, *reader.get(3));
        assert_eq!(2.2, *reader.get(4));
        assert_eq!(2.4, *reader.get(5));
    }

    #[test]
    fn should_apply_div_op() {
        let cursor = Cursor::new(0, 0, 11, time_frame::SECOND_5);
        let evaluated_expression =
            eval_c2d(BinaryMathOperator::Div, cursor, CursorExpansion::Identity);

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..6, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(2.0, *reader.get(0));
        assert_eq!(1.0, *reader.get(1));
        assert_eq!(0.7692307692307692, *reader.get(2));
        assert_eq!(0.8, *reader.get(3));
        assert_eq!(1.8181818181818181, *reader.get(4));
        assert_eq!(1.6666666666666667, *reader.get(5));
    }

    #[test]
    fn should_apply_pow_op() {
        let cursor = Cursor::new(0, 0, 11, time_frame::SECOND_5);
        let evaluated_expression =
            eval_c2d(BinaryMathOperator::Pow, cursor, CursorExpansion::Identity);

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..6, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(2.0, *reader.get(0));
        assert_eq!(4.0, *reader.get(1));
        assert_eq!(6.062866266041593, *reader.get(2));
        assert_eq!(5.656854249492381, *reader.get(3));
        assert_eq!(2.1435469250725863, *reader.get(4));
        assert_eq!(2.2973967099940698, *reader.get(5));
    }

    #[test]
    fn should_apply_max_op() {
        let cursor = Cursor::new(0, 0, 11, time_frame::SECOND_5);
        let evaluated_expression =
            eval_c2d(BinaryMathOperator::Max, cursor, CursorExpansion::Identity);

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..6, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(2.0, *reader.get(0));
        assert_eq!(2.0, *reader.get(1));
        assert_eq!(2.6, *reader.get(2));
        assert_eq!(2.5, *reader.get(3));
        assert_eq!(2.0, *reader.get(4));
        assert_eq!(2.0, *reader.get(5));
    }

    #[test]
    fn should_apply_min_op() {
        let cursor = Cursor::new(0, 0, 11, time_frame::SECOND_5);
        let evaluated_expression =
            eval_c2d(BinaryMathOperator::Min, cursor, CursorExpansion::Identity);

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..6, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.0, *reader.get(0));
        assert_eq!(2.0, *reader.get(1));
        assert_eq!(2.0, *reader.get(2));
        assert_eq!(2.0, *reader.get(3));
        assert_eq!(1.1, *reader.get(4));
        assert_eq!(1.2, *reader.get(5));
    }

    fn eval_c2d(
        operator: BinaryMathOperator,
        cursor: Cursor,
        cursor_expansion: CursorExpansion,
    ) -> WideBinaryMathExpr {
        let base_quotes_5 = Arc::new(BaseQuotes::new(Arc::new(FrameQuotes::new(
            vec![0f64; 16],
            vec![0f64; 16],
            vec![
                1f64, 2.0, 3.0, 3.2, 2.6, 2.5, 3.6, 3.7, 1.1, 1.2, 1.4, 1.6, 2.0, 0.5, 1.7, 3.4,
            ],
            vec![0f64; 16],
            TimestampVector::from_utc(
                (0..16)
                    .map(|i| i as u64 * u64::from(*time_frame::SECOND_5))
                    .collect(),
            ),
            time_frame::SECOND_5,
        ))));

        let frame_quotes_10 =
            Arc::new(build_frame_to_frame(&base_quotes_5, time_frame::SECOND_10).unwrap());

        let mut id_gen = 0usize;
        let left = const_processor_handle(&mut id_gen, 2f64);
        let right = primitive_processor_handle(
            &mut id_gen,
            Arc::clone(&frame_quotes_10),
            PrimitiveOutputSelector::Low,
        );

        let strategy =
            select_translation_strategy(&cursor, time_frame::SECOND_10, time_frame::SECOND_10)
                .unwrap();

        let mut expr = WideBinaryMathExpr::new(
            left,
            operator,
            right,
            WideBinaryExprOutput::new(strategy, SpatialBuffer::new(time_frame::SECOND_10, 32)),
        );

        let session = SessionContext::new(
            String::from("GOLD"),
            Arc::clone(&base_quotes_5),
            BTreeMap::from([
                (time_frame::SECOND_5, Arc::clone(&**base_quotes_5)),
                (time_frame::SECOND_10, Arc::clone(&frame_quotes_10)),
            ]),
            BTreeMap::from([
                (
                    time_frame::SECOND_5,
                    Arc::new(
                        build_translation_buffer(&base_quotes_5, time_frame::SECOND_5).unwrap(),
                    ),
                ),
                (
                    time_frame::SECOND_10,
                    Arc::new(
                        build_translation_buffer(&base_quotes_5, time_frame::SECOND_10).unwrap(),
                    ),
                ),
            ]),
            SessionConfiguration::new(cursor.step(), 32),
        );

        let ctx = SpanContext::new(cursor, Arc::new(session));

        expr.eval(
            &ctx,
            cursor_expansion,
            &DynamicStore::new(ComputeId::Simple(0)),
        )
        .expect("success");

        expr
    }
}

#[cfg(test)]
mod c2d {
    use crate::sim::bb::cursor::{Cursor, CursorExpansion, Epoch, TimeShift};
    use crate::sim::bb::quotes::{BaseQuotes, FrameQuotes, TimestampVector};
    use crate::sim::bb::{time_frame, ComputeId};
    use crate::sim::compute::operators::BinaryMathOperator;
    use crate::sim::compute::translation_strategy::{
        select_translation_strategy, TranslationStrategy, TranslationStrategyDescriptor,
    };
    use crate::sim::compute::wide::expr::math_tests::{
        const_processor_handle, empty_span_ctx, primitive_processor_handle,
    };
    use crate::sim::compute::wide::expr::primitive::PrimitiveOutputSelector;
    use crate::sim::compute::wide::expr::{
        Constant, IdentifiedProcessorNode, ProcessorHandle, ProcessorNode, Scalar,
        WideBinaryExprOutput, WideBinaryMathExpr,
    };
    use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
    use crate::sim::compute::wide::indicator::{SingleOutputSelector, WideProcessor};
    use crate::sim::context::{SessionConfiguration, SessionContext, SpanContext};
    use crate::sim::error::{ASTError, ContractError};
    use crate::sim::selector::Selector;
    use crate::sim::spatial_buffer::SpatialBuffer;
    use crate::sim::tlb::{FrameTranslationBuffer, IdentityTranslationBuffer};
    use std::collections::BTreeMap;
    use std::rc::Rc;
    use std::sync::Arc;

    #[test]
    fn should_expand_epoch_to_the_past() {
        let evaluated_expression = eval_c2d(
            BinaryMathOperator::Add,
            Cursor::new(0, 3, 2, time_frame::SECOND_1),
            CursorExpansion::Identity.expand_by(TimeShift::Past(2)),
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..5, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(3.0, *reader.get(0));
        assert_eq!(4.0, *reader.get(1));
        assert_eq!(5.0, *reader.get(2));
        assert_eq!(5.2, *reader.get(3));
        assert_eq!(4.6, *reader.get(4));
    }

    #[test]
    fn should_expand_epoch_to_the_future() {
        let evaluated_expression = eval_c2d(
            BinaryMathOperator::Add,
            Cursor::new(0, 1, 2, time_frame::SECOND_1),
            CursorExpansion::Identity.expand_by(TimeShift::Future(2)),
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..5, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(3.0, *reader.get(0));
        assert_eq!(4.0, *reader.get(1));
        assert_eq!(5.0, *reader.get(2));
        assert_eq!(5.2, *reader.get(3));
        assert_eq!(4.6, *reader.get(4));
    }

    #[test]
    fn should_add_c2d() {
        let evaluated_expression = eval_c2d(
            BinaryMathOperator::Add,
            Cursor::new(0, 0, 8, time_frame::SECOND_1),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(3.0, *reader.get(0));
        assert_eq!(4.0, *reader.get(1));
        assert_eq!(5.0, *reader.get(2));
        assert_eq!(5.2, *reader.get(3));
        assert_eq!(4.6, *reader.get(4));
        assert_eq!(4.5, *reader.get(5));
        assert_eq!(5.6, *reader.get(6));
        assert_eq!(5.7, *reader.get(7));
    }

    #[test]
    fn should_sub_c2d() {
        let evaluated_expression = eval_c2d(
            BinaryMathOperator::Sub,
            Cursor::new(0, 0, 8, time_frame::SECOND_1),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.0, *reader.get(0));
        assert_eq!(0.0, *reader.get(1));
        assert_eq!(-1.0, *reader.get(2));
        assert_eq!(-1.2000000000000002, *reader.get(3));
        assert_eq!(-0.6000000000000001, *reader.get(4));
        assert_eq!(-0.5, *reader.get(5));
        assert_eq!(-1.6, *reader.get(6));
        assert_eq!(-1.7000000000000002, *reader.get(7));
    }

    #[test]
    fn should_multiply_c2d() {
        let evaluated_expression = eval_c2d(
            BinaryMathOperator::Mul,
            Cursor::new(0, 0, 8, time_frame::SECOND_1),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(2.0, *reader.get(0));
        assert_eq!(4.0, *reader.get(1));
        assert_eq!(6.0, *reader.get(2));
        assert_eq!(6.4, *reader.get(3));
        assert_eq!(5.2, *reader.get(4));
        assert_eq!(5.0, *reader.get(5));
        assert_eq!(7.2, *reader.get(6));
        assert_eq!(7.4, *reader.get(7));
    }

    #[test]
    fn should_divide_c2d() {
        let evaluated_expression = eval_c2d(
            BinaryMathOperator::Div,
            Cursor::new(0, 0, 8, time_frame::SECOND_1),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(2.0, *reader.get(0));
        assert_eq!(1.0, *reader.get(1));
        assert_eq!(0.6666666666666666, *reader.get(2));
        assert_eq!(0.625, *reader.get(3));
        assert_eq!(0.7692307692307692, *reader.get(4));
        assert_eq!(0.8, *reader.get(5));
        assert_eq!(0.5555555555555556, *reader.get(6));
        assert_eq!(0.5405405405405405, *reader.get(7));
    }

    #[test]
    fn should_pow_c2d() {
        let evaluated_expression = eval_c2d(
            BinaryMathOperator::Pow,
            Cursor::new(0, 0, 8, time_frame::SECOND_1),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(2.0, *reader.get(0));
        assert_eq!(4.0, *reader.get(1));
        assert_eq!(8.0, *reader.get(2));
        assert_eq!(9.18958683997628, *reader.get(3));
        assert_eq!(6.062866266041593, *reader.get(4));
        assert_eq!(5.656854249492381, *reader.get(5));
        assert_eq!(12.125732532083186, *reader.get(6));
        assert_eq!(12.99603834169977, *reader.get(7));
    }

    #[test]
    fn should_max_c2d() {
        let evaluated_expression = eval_c2d(
            BinaryMathOperator::Max,
            Cursor::new(0, 0, 8, time_frame::SECOND_1),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(2.0, *reader.get(0));
        assert_eq!(2.0, *reader.get(1));
        assert_eq!(3.0, *reader.get(2));
        assert_eq!(3.2, *reader.get(3));
        assert_eq!(2.6, *reader.get(4));
        assert_eq!(2.5, *reader.get(5));
        assert_eq!(3.6, *reader.get(6));
        assert_eq!(3.7, *reader.get(7));
    }

    #[test]
    fn should_min_c2d() {
        let evaluated_expression = eval_c2d(
            BinaryMathOperator::Min,
            Cursor::new(0, 0, 8, time_frame::SECOND_1),
            CursorExpansion::Identity,
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.0, *reader.get(0));
        assert_eq!(2.0, *reader.get(1));
        assert_eq!(2.0, *reader.get(2));
        assert_eq!(2.0, *reader.get(3));
        assert_eq!(2.0, *reader.get(4));
        assert_eq!(2.0, *reader.get(5));
        assert_eq!(2.0, *reader.get(6));
        assert_eq!(2.0, *reader.get(7));
    }

    fn eval_c2d(
        operator: BinaryMathOperator,
        cursor: Cursor,
        cursor_expansion: CursorExpansion,
    ) -> WideBinaryMathExpr {
        let frame_quotes = Arc::new(FrameQuotes::new(
            vec![0f64; 8],
            vec![0f64; 8],
            vec![1f64, 2.0, 3.0, 3.2, 2.6, 2.5, 3.6, 3.7],
            vec![0f64; 8],
            TimestampVector::from_utc(
                (0..8)
                    .map(|i| i as u64 * u64::from(*time_frame::SECOND_1))
                    .collect(),
            ),
            time_frame::SECOND_1,
        ));

        let mut id_gen = 0usize;
        let left = const_processor_handle(&mut id_gen, 2f64);
        let right = primitive_processor_handle(
            &mut id_gen,
            Arc::clone(&frame_quotes),
            PrimitiveOutputSelector::Low,
        );

        let strategy =
            select_translation_strategy(&cursor, time_frame::SECOND_1, time_frame::SECOND_1)
                .unwrap();

        let mut expr = WideBinaryMathExpr::new(
            left,
            operator,
            right,
            WideBinaryExprOutput::new(strategy, SpatialBuffer::new(time_frame::SECOND_1, 32)),
        );

        let base_quotes = Arc::new(BaseQuotes::new(Arc::clone(&frame_quotes)));
        let session = SessionContext::new(
            String::from("GOLD"),
            base_quotes,
            BTreeMap::from([(time_frame::SECOND_1, Arc::clone(&frame_quotes))]),
            BTreeMap::from([(
                time_frame::SECOND_1,
                Arc::new(FrameTranslationBuffer::IdentityTranslationBuffer(
                    IdentityTranslationBuffer::new(time_frame::SECOND_1),
                )),
            )]),
            SessionConfiguration::new(cursor.step(), 32),
        );

        let ctx = SpanContext::new(cursor, Arc::new(session));

        expr.eval(
            &ctx,
            cursor_expansion,
            &DynamicStore::new(ComputeId::Simple(0)),
        )
        .expect("success");

        expr
    }

    #[test]
    fn should_err_on_constant_left_and_right_handles() {
        let mut id_gen = 0usize;

        let cursor = Cursor::new(0, 0, 1024, time_frame::SECOND_1);
        let strategy =
            select_translation_strategy(&cursor, time_frame::SECOND_1, time_frame::SECOND_1)
                .unwrap();

        let output =
            WideBinaryExprOutput::new(strategy, SpatialBuffer::new(time_frame::SECOND_1, 2048));

        let mut expr = WideBinaryMathExpr::new(
            const_processor_handle(&mut id_gen, 2f64),
            BinaryMathOperator::Sub,
            const_processor_handle(&mut id_gen, 2f64),
            output,
        );

        let ctx = empty_span_ctx(cursor);

        let error = expr
            .eval(
                &ctx,
                CursorExpansion::Identity,
                &DynamicStore::new(ComputeId::Simple(usize::MAX)),
            )
            .err()
            .unwrap();

        match error.downcast_ref::<ASTError>().expect("Expects ASTError") {
            ASTError::WideMathExprNodeError(err_msg) => {
                assert_eq!(
                    &"Scalar binary math expr should be used for constant input's",
                    err_msg
                );
            }
            _ => {
                panic!("unexpected type of error");
            }
        }
    }
}

#[cfg(test)]
mod d2c_upper_to_const {
    use crate::sim::bb::cursor::{Cursor, CursorExpansion, Epoch};
    use crate::sim::bb::quotes::{BaseQuotes, FrameQuotes, TimestampVector};
    use crate::sim::bb::{time_frame, ComputeId};
    use crate::sim::builder::frame_builder::build_frame_to_frame;
    use crate::sim::builder::translation_builder::build_translation_buffer;
    use crate::sim::compute::operators::BinaryMathOperator;
    use crate::sim::compute::translation_strategy::select_translation_strategy;
    use crate::sim::compute::wide::expr::math_tests::{
        const_processor_handle, primitive_processor_handle,
    };
    use crate::sim::compute::wide::expr::primitive::PrimitiveOutputSelector;
    use crate::sim::compute::wide::expr::{WideBinaryExprOutput, WideBinaryMathExpr};
    use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
    use crate::sim::compute::wide::indicator::{SingleOutputSelector, WideProcessor};
    use crate::sim::context::{SessionConfiguration, SessionContext, SpanContext};
    use crate::sim::selector::Selector;
    use crate::sim::spatial_buffer::SpatialBuffer;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn should_apply_add_op() {
        let cursor = Cursor::new(0, 5, 9, time_frame::SECOND_5);
        let evaluated_expression =
            eval_d2c(BinaryMathOperator::Add, cursor, CursorExpansion::Identity);

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..7, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(2.9, *reader.get(0));
        assert_eq!(3.9, *reader.get(1));
        assert_eq!(4.5, *reader.get(2));
        assert_eq!(4.4, *reader.get(3));
        assert_eq!(3.0, *reader.get(4));
        assert_eq!(3.0999999999999996, *reader.get(5));
        assert_eq!(3.5, *reader.get(6));
    }

    #[test]
    fn should_apply_sub_op() {
        let cursor = Cursor::new(0, 5, 9, time_frame::SECOND_5);
        let evaluated_expression =
            eval_d2c(BinaryMathOperator::Sub, cursor, CursorExpansion::Identity);

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..7, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(-0.8999999999999999, *reader.get(0));
        assert_eq!(0.10000000000000009, *reader.get(1));
        assert_eq!(0.7000000000000002, *reader.get(2));
        assert_eq!(0.6000000000000001, *reader.get(3));
        assert_eq!(-0.7999999999999998, *reader.get(4));
        assert_eq!(-0.7, *reader.get(5));
        assert_eq!(-0.2999999999999998, *reader.get(6));
    }

    #[test]
    fn should_apply_mul_op() {
        let cursor = Cursor::new(0, 5, 9, time_frame::SECOND_5);
        let evaluated_expression =
            eval_d2c(BinaryMathOperator::Mul, cursor, CursorExpansion::Identity);

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..7, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.9, *reader.get(0));
        assert_eq!(3.8, *reader.get(1));
        assert_eq!(4.9399999999999995, *reader.get(2));
        assert_eq!(4.75, *reader.get(3));
        assert_eq!(2.09, *reader.get(4));
        assert_eq!(2.28, *reader.get(5));
        assert_eq!(3.04, *reader.get(6));
    }

    #[test]
    fn should_apply_div_op() {
        let cursor = Cursor::new(0, 5, 9, time_frame::SECOND_5);
        let evaluated_expression =
            eval_d2c(BinaryMathOperator::Div, cursor, CursorExpansion::Identity);

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..7, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(0.5263157894736842, *reader.get(0));
        assert_eq!(1.0526315789473684, *reader.get(1));
        assert_eq!(1.368421052631579, *reader.get(2));
        assert_eq!(1.3157894736842106, *reader.get(3));
        assert_eq!(0.5789473684210527, *reader.get(4));
        assert_eq!(0.631578947368421, *reader.get(5));
        assert_eq!(0.8421052631578948, *reader.get(6));
    }

    #[test]
    fn should_apply_pow_op() {
        let cursor = Cursor::new(0, 5, 9, time_frame::SECOND_5);
        let evaluated_expression =
            eval_d2c(BinaryMathOperator::Pow, cursor, CursorExpansion::Identity);

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..7, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.0, *reader.get(0));
        assert_eq!(3.7321319661472296, *reader.get(1));
        assert_eq!(6.143973886253817, *reader.get(2));
        assert_eq!(5.702772103471755, *reader.get(3));
        assert_eq!(1.1985222524395716, *reader.get(4));
        assert_eq!(1.4139835841691542, *reader.get(5));
        assert_eq!(2.442462851401401, *reader.get(6));
    }

    #[test]
    fn should_apply_max_op() {
        let cursor = Cursor::new(0, 5, 9, time_frame::SECOND_5);
        let evaluated_expression =
            eval_d2c(BinaryMathOperator::Max, cursor, CursorExpansion::Identity);

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..7, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.9, *reader.get(0));
        assert_eq!(2.0, *reader.get(1));
        assert_eq!(2.6, *reader.get(2));
        assert_eq!(2.5, *reader.get(3));
        assert_eq!(1.9, *reader.get(4));
        assert_eq!(1.9, *reader.get(5));
        assert_eq!(1.9, *reader.get(6));
    }

    #[test]
    fn should_apply_min_op() {
        let cursor = Cursor::new(0, 5, 9, time_frame::SECOND_5);
        let evaluated_expression =
            eval_d2c(BinaryMathOperator::Min, cursor, CursorExpansion::Identity);

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..7, time_frame::SECOND_10);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.0, *reader.get(0));
        assert_eq!(1.9, *reader.get(1));
        assert_eq!(1.9, *reader.get(2));
        assert_eq!(1.9, *reader.get(3));
        assert_eq!(1.1, *reader.get(4));
        assert_eq!(1.2, *reader.get(5));
        assert_eq!(1.6, *reader.get(6));
    }

    fn eval_d2c(
        operator: BinaryMathOperator,
        cursor: Cursor,
        cursor_expansion: CursorExpansion,
    ) -> WideBinaryMathExpr {
        let base_quotes_5 = Arc::new(BaseQuotes::new(Arc::new(FrameQuotes::new(
            vec![0f64; 16],
            vec![0f64; 16],
            vec![
                1f64, 2.0, 3.0, 3.2, 2.6, 2.5, 3.6, 3.7, 1.1, 1.2, 1.4, 1.6, 2.0, 0.5, 1.7, 3.4,
            ],
            vec![0f64; 16],
            TimestampVector::from_utc(
                (0..16)
                    .map(|i| i as u64 * u64::from(*time_frame::SECOND_5))
                    .collect(),
            ),
            time_frame::SECOND_5,
        ))));

        let frame_quotes_10 =
            Arc::new(build_frame_to_frame(&base_quotes_5, time_frame::SECOND_10).unwrap());

        let mut id_gen = 0usize;
        let left = primitive_processor_handle(
            &mut id_gen,
            Arc::clone(&frame_quotes_10),
            PrimitiveOutputSelector::Low,
        );

        let right = const_processor_handle(&mut id_gen, 1.9f64);

        let strategy =
            select_translation_strategy(&cursor, time_frame::SECOND_10, time_frame::SECOND_10)
                .unwrap();

        let mut expr = WideBinaryMathExpr::new(
            left,
            operator,
            right,
            WideBinaryExprOutput::new(strategy, SpatialBuffer::new(time_frame::SECOND_10, 32)),
        );

        let session = SessionContext::new(
            String::from("GOLD"),
            Arc::clone(&base_quotes_5),
            BTreeMap::from([
                (time_frame::SECOND_5, Arc::clone(&**base_quotes_5)),
                (time_frame::SECOND_10, Arc::clone(&frame_quotes_10)),
            ]),
            BTreeMap::from([
                (
                    time_frame::SECOND_5,
                    Arc::new(
                        build_translation_buffer(&base_quotes_5, time_frame::SECOND_5).unwrap(),
                    ),
                ),
                (
                    time_frame::SECOND_10,
                    Arc::new(
                        build_translation_buffer(&base_quotes_5, time_frame::SECOND_10).unwrap(),
                    ),
                ),
            ]),
            SessionConfiguration::new(cursor.step(), 32),
        );

        let ctx = SpanContext::new(cursor, Arc::new(session));

        expr.eval(
            &ctx,
            cursor_expansion,
            &DynamicStore::new(ComputeId::Simple(0)),
        )
        .expect("success");

        expr
    }
}

#[cfg(test)]
mod tests_wide_unary_math_expr {
    use crate::sim::bb::cursor::{Cursor, CursorExpansion, Epoch, TimeShift};
    use crate::sim::bb::quotes::{BaseQuotes, FrameQuotes, TimestampVector};
    use crate::sim::bb::{time_frame, ComputeId};
    use crate::sim::compute::operators::{BinaryMathOperator, UnaryMathOperator};
    use crate::sim::compute::translation_strategy::select_translation_strategy;
    use crate::sim::compute::wide::expr::math_tests::{
        const_processor_handle, primitive_processor_handle,
    };
    use crate::sim::compute::wide::expr::primitive::PrimitiveOutputSelector;
    use crate::sim::compute::wide::expr::{
        WideBinaryExprOutput, WideBinaryMathExpr, WideUnaryMathExpr,
    };
    use crate::sim::compute::wide::indicator::dynamic_store::DynamicStore;
    use crate::sim::compute::wide::indicator::{SingleOutputSelector, WideProcessor};
    use crate::sim::context::{SessionConfiguration, SessionContext, SpanContext};
    use crate::sim::selector::Selector;
    use crate::sim::spatial_buffer::SpatialBuffer;
    use crate::sim::tlb::{FrameTranslationBuffer, IdentityTranslationBuffer};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn should_expand_epoch_to_the_past() {
        let evaluated_expression = eval_wide_unary_expr(
            UnaryMathOperator::Abs,
            Cursor::new(0, 3, 2, time_frame::SECOND_1),
            CursorExpansion::Identity.expand_by(TimeShift::Past(2)),
            [-1f64, -2.4, 3.5, -7.6, 5.5, -5.0, 4.2, 2.0],
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..5, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.0, *reader.get(0));
        assert_eq!(2.4, *reader.get(1));
        assert_eq!(3.5, *reader.get(2));
        assert_eq!(7.6, *reader.get(3));
        assert_eq!(5.5, *reader.get(4));
    }

    #[test]
    fn should_expand_epoch_to_the_future() {
        let evaluated_expression = eval_wide_unary_expr(
            UnaryMathOperator::Abs,
            Cursor::new(0, 1, 2, time_frame::SECOND_1),
            CursorExpansion::Identity.expand_by(TimeShift::Future(2)),
            [-1f64, -2.4, 3.5, -7.6, 5.5, -5.0, 4.2, 2.0],
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..5, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.0, *reader.get(0));
        assert_eq!(2.4, *reader.get(1));
        assert_eq!(3.5, *reader.get(2));
        assert_eq!(7.6, *reader.get(3));
        assert_eq!(5.5, *reader.get(4));
    }

    #[test]
    fn should_apply_sqrt() {
        let evaluated_expression = eval_wide_unary_expr(
            UnaryMathOperator::Sqrt,
            Cursor::new(0, 0, 8, time_frame::SECOND_1),
            CursorExpansion::Identity,
            [1f64, 2.4, 3.5, 7.6, 5.5, 5.0, 4.2, 2.0],
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.0, *reader.get(0));
        assert_eq!(1.5491933384829668, *reader.get(1));
        assert_eq!(1.8708286933869707, *reader.get(2));
        assert_eq!(2.756809750418044, *reader.get(3));
        assert_eq!(2.345207879911715, *reader.get(4));
        assert_eq!(2.23606797749979, *reader.get(5));
        assert_eq!(2.04939015319192, *reader.get(6));
        assert_eq!(1.4142135623730951, *reader.get(7));
    }

    #[test]
    fn should_apply_abs() {
        let evaluated_expression = eval_wide_unary_expr(
            UnaryMathOperator::Abs,
            Cursor::new(0, 0, 8, time_frame::SECOND_1),
            CursorExpansion::Identity,
            [-1f64, -2.4, 3.5, -7.6, 5.5, -5.0, 4.2, 2.0],
        );

        let output_buf = evaluated_expression
            .output_buffer(SingleOutputSelector.ordinal())
            .expect("has output buf");

        let epoch_read = Epoch::new(0..8, time_frame::SECOND_1);
        assert_eq!(&epoch_read, output_buf.epoch());

        let reader = output_buf.create_reader(&epoch_read).unwrap();

        assert_eq!(1.0, *reader.get(0));
        assert_eq!(2.4, *reader.get(1));
        assert_eq!(3.5, *reader.get(2));
        assert_eq!(7.6, *reader.get(3));
        assert_eq!(5.5, *reader.get(4));
        assert_eq!(5.0, *reader.get(5));
        assert_eq!(4.2, *reader.get(6));
        assert_eq!(2.0, *reader.get(7));
    }

    fn eval_wide_unary_expr(
        operator: UnaryMathOperator,
        cursor: Cursor,
        cursor_expansion: CursorExpansion,
        data: [f64; 8],
    ) -> WideUnaryMathExpr {
        let frame_quotes = Arc::new(FrameQuotes::new(
            vec![0f64; 8],
            vec![0f64; 8],
            vec![0f64; 8],
            Vec::from(data),
            TimestampVector::from_utc(
                (0..8)
                    .map(|i| i as u64 * u64::from(*time_frame::SECOND_1))
                    .collect(),
            ),
            time_frame::SECOND_1,
        ));

        let mut id_gen = 0usize;
        let expr = primitive_processor_handle(
            &mut id_gen,
            Arc::clone(&frame_quotes),
            PrimitiveOutputSelector::Close,
        );

        let mut expr =
            WideUnaryMathExpr::new(expr, operator, SpatialBuffer::new(time_frame::SECOND_1, 32));

        let base_quotes = Arc::new(BaseQuotes::new(Arc::clone(&frame_quotes)));
        let session = SessionContext::new(
            String::from("GOLD"),
            base_quotes,
            BTreeMap::from([(time_frame::SECOND_1, Arc::clone(&frame_quotes))]),
            BTreeMap::from([(
                time_frame::SECOND_1,
                Arc::new(FrameTranslationBuffer::IdentityTranslationBuffer(
                    IdentityTranslationBuffer::new(time_frame::SECOND_1),
                )),
            )]),
            SessionConfiguration::new(cursor.step(), 32),
        );

        let ctx = SpanContext::new(cursor, Arc::new(session));

        expr.eval(
            &ctx,
            cursor_expansion,
            &DynamicStore::new(ComputeId::Simple(0)),
        )
        .expect("success");

        expr
    }
}

#[cfg(test)]
mod tests_binary_math_expr {
    use crate::sim::bb::cursor::{Cursor, CursorExpansion, TimeShift};
    use crate::sim::bb::{time_frame, ComputeId};
    use crate::sim::compute::operators::BinaryMathOperator;
    use crate::sim::compute::wide::expr::math_tests::empty_span_ctx;
    use crate::sim::compute::wide::expr::{
        BinaryMathExpr, Constant, IdentifiedProcessorNode, MathExprNode, ProcessorHandle,
        ProcessorNode, Scalar, UnaryMathExpr,
    };
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn ensure_all_operators_tested() {
        let operators = [
            BinaryMathOperator::Add,
            BinaryMathOperator::Sub,
            BinaryMathOperator::Mul,
            BinaryMathOperator::Div,
            BinaryMathOperator::Pow,
            BinaryMathOperator::Max,
            BinaryMathOperator::Min,
        ];

        for op in operators {
            let is_tested = match op {
                BinaryMathOperator::Add => "OK",
                BinaryMathOperator::Sub => "OK",
                BinaryMathOperator::Mul => "OK",
                BinaryMathOperator::Div => "OK",
                BinaryMathOperator::Pow => "OK",
                BinaryMathOperator::Max => "OK",
                BinaryMathOperator::Min => "OK",
            };

            assert_eq!("OK", is_tested);
        }
    }

    #[test]
    fn should_add_numbers() {
        assert_binary_operator(9.8f64, 3.5f64, 6.3f64, BinaryMathOperator::Add);
    }

    #[test]
    fn should_subtract_numbers() {
        assert_binary_operator(-2.8f64, 3.5f64, 6.3f64, BinaryMathOperator::Sub);
    }

    #[test]
    fn should_multiply_numbers() {
        assert_binary_operator(7f64, 3.5f64, 2f64, BinaryMathOperator::Mul);
    }

    #[test]
    fn should_divide_numbers() {
        assert_binary_operator(4.1f64, 8.2f64, 2f64, BinaryMathOperator::Div);
    }

    #[test]
    fn should_pow_numbers() {
        assert_binary_operator(16f64, 4f64, 2f64, BinaryMathOperator::Pow);
    }

    #[test]
    fn should_select_maximal_number() {
        assert_binary_operator(3.5f64, 3.5f64, 2f64, BinaryMathOperator::Max);
    }

    #[test]
    fn should_select_minimal_number() {
        assert_binary_operator(2f64, 3.5f64, 2f64, BinaryMathOperator::Min);
    }

    #[test]
    fn should_execute_chain_of_binary_expressions() {
        let mut id_gen = 0usize;

        let left = create_binary_math_expr(&mut id_gen, 2f64, 4.5f64, BinaryMathOperator::Mul);
        let right = create_binary_math_expr(&mut id_gen, 2f64, 4.5f64, BinaryMathOperator::Sub);

        let left = IdentifiedProcessorNode::new(
            ComputeId::Simple(id_gen),
            ProcessorNode::MathExpr(RefCell::new(MathExprNode::Binary(left))),
        );
        id_gen += 1;

        let right = IdentifiedProcessorNode::new(
            ComputeId::Simple(id_gen),
            ProcessorNode::MathExpr(RefCell::new(MathExprNode::Binary(right))),
        );

        let left_handle = ProcessorHandle::new(Rc::new(left), 0, None);
        let right_handle = ProcessorHandle::new(Rc::new(right), 0, None);

        let ctx = empty_span_ctx(Cursor::new(0, 0, 1024, time_frame::SECOND_1));

        let mut expr = BinaryMathExpr::new(
            Box::new(left_handle),
            BinaryMathOperator::Add,
            Box::new(right_handle),
        );

        expr.eval(&ctx, CursorExpansion::Identity).unwrap();

        match expr.output().unwrap().value() {
            Scalar::F64(output) => {
                assert_eq!(6.5f64, *output);
            }
        }
    }

    fn assert_binary_operator(expected: f64, left: f64, right: f64, operator: BinaryMathOperator) {
        let ctx = empty_span_ctx(Cursor::new(0, 0, 1024, time_frame::SECOND_1));

        let mut id_gen = 0usize;
        let mut expr = create_binary_math_expr(&mut id_gen, left, right, operator);

        expr.eval(&ctx, CursorExpansion::Identity).unwrap();

        match expr.output().unwrap().value() {
            Scalar::F64(output) => {
                assert_eq!(expected, *output);
            }
        }
    }

    fn create_binary_math_expr(
        id: &mut usize,
        left: f64,
        right: f64,
        operator: BinaryMathOperator,
    ) -> BinaryMathExpr {
        let left = IdentifiedProcessorNode::new(
            ComputeId::Simple(*id),
            ProcessorNode::Constant(Constant::new(Scalar::F64(left))),
        );
        *id += 1;

        let right = IdentifiedProcessorNode::new(
            ComputeId::Simple(*id),
            ProcessorNode::Constant(Constant::new(Scalar::F64(right))),
        );
        *id += 1;

        let left_handle = ProcessorHandle::new(Rc::new(left), 0, None);
        let right_handle = ProcessorHandle::new(Rc::new(right), 0, None);

        BinaryMathExpr::new(Box::new(left_handle), operator, Box::new(right_handle))
    }
}

#[cfg(test)]
mod tests_unary_math_expr {
    use crate::sim::bb::cursor::{Cursor, CursorExpansion, TimeShift};
    use crate::sim::bb::{time_frame, ComputeId};
    use crate::sim::compute::operators::UnaryMathOperator;
    use crate::sim::compute::wide::expr::math_tests::empty_span_ctx;
    use crate::sim::compute::wide::expr::{
        Constant, IdentifiedProcessorNode, MathExprNode, ProcessorHandle, ProcessorNode, Scalar,
        UnaryMathExpr,
    };
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn ensure_all_operators_tested() {
        let operators = [UnaryMathOperator::Abs, UnaryMathOperator::Sqrt];

        for op in operators {
            let is_tested = match op {
                UnaryMathOperator::Sqrt => "OK",
                UnaryMathOperator::Abs => "OK",
            };

            assert_eq!("OK", is_tested);
        }
    }

    #[test]
    fn should_apply_absolute_value_operator() {
        assert_unary_operator(1f64, -1f64, UnaryMathOperator::Abs);
    }

    #[test]
    fn should_apply_sqrt_operator() {
        assert_unary_operator(2f64, 4f64, UnaryMathOperator::Sqrt);
    }

    #[test]
    fn should_execute_chain_of_unary_expressions() {
        let node = IdentifiedProcessorNode::new(
            ComputeId::Simple(0),
            ProcessorNode::Constant(Constant::new(Scalar::F64(-16f64))),
        );

        let unary_abs = UnaryMathExpr::new(
            Box::new(ProcessorHandle::new(Rc::new(node), 0, None)),
            UnaryMathOperator::Abs,
        );

        let node = IdentifiedProcessorNode::new(
            ComputeId::Simple(1),
            ProcessorNode::MathExpr(RefCell::new(MathExprNode::Unary(unary_abs))),
        );

        let mut unary_abs_then_sqrt = UnaryMathExpr::new(
            Box::new(ProcessorHandle::new(Rc::new(node), 0, None)),
            UnaryMathOperator::Sqrt,
        );

        let ctx = empty_span_ctx(Cursor::new(0, 0, 1024, time_frame::SECOND_1));

        unary_abs_then_sqrt
            .eval(&ctx, CursorExpansion::Identity)
            .unwrap();

        match unary_abs_then_sqrt.output().unwrap().value() {
            Scalar::F64(output) => {
                assert_eq!(4f64, *output);
            }
        }
    }

    fn assert_unary_operator(expected: f64, input: f64, operator: UnaryMathOperator) {
        let node = IdentifiedProcessorNode::new(
            ComputeId::Simple(0),
            ProcessorNode::Constant(Constant::new(Scalar::F64(input))),
        );

        let handle = ProcessorHandle::new(Rc::new(node), 0, None);
        let ctx = empty_span_ctx(Cursor::new(0, 0, 1024, time_frame::SECOND_1));

        let mut expr = UnaryMathExpr::new(Box::new(handle), operator);
        expr.eval(&ctx, CursorExpansion::Identity).unwrap();

        match expr.output().unwrap().value() {
            Scalar::F64(output) => {
                assert_eq!(expected, *output);
            }
        }
    }
}

fn primitive_processor_handle(
    id_gen: &mut usize,
    frame_quotes: Arc<FrameQuotes>,
    output_selector: impl Selector,
) -> Box<ProcessorHandle> {
    let node = IdentifiedProcessorNode::new(
        ComputeId::Simple(*id_gen),
        ProcessorNode::Primitive(PrimitiveNode::new(frame_quotes)),
    );

    *id_gen += 1;

    Box::new(ProcessorHandle::new(
        Rc::new(node),
        output_selector.ordinal(),
        None,
    ))
}

fn const_processor_handle(id_gen: &mut usize, value: f64) -> Box<ProcessorHandle> {
    let node = IdentifiedProcessorNode::new(
        ComputeId::Simple(*id_gen),
        ProcessorNode::Constant(Constant::new(Scalar::F64(value))),
    );

    *id_gen += 1;

    Box::new(ProcessorHandle::new(Rc::new(node), 0, None))
}

fn empty_span_ctx(cursor: Cursor) -> SpanContext {
    let base_quotes = Arc::new(BaseQuotes::new(Arc::new(FrameQuotes::new(
        vec![],
        vec![],
        vec![],
        vec![],
        TimestampVector::from_utc(vec![]),
        time_frame::SECOND_1,
    ))));
    let session_configuration = SessionConfiguration::new(1024, 2048);
    let session = SessionContext::new(
        String::from("GOLD"),
        base_quotes,
        BTreeMap::new(),
        BTreeMap::new(),
        session_configuration,
    );
    SpanContext::new(cursor, Arc::new(session))
}
