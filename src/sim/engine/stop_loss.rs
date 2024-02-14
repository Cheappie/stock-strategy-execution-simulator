/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

pub struct StopLoss {
    stop_loss: f64,
}

///
/// If close strategy is static, then we can do smarter
///
pub enum Characteristic {
    Dynamic,
    Static,
}
