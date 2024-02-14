/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/
///
/// Simply averaged price with actual volume.
///
/// Enables us to calculate either PROFIT/LOSS for total buys or sells in O(1).
///
pub struct PriceAggregate {
    price: f64,
    volume: f64,
}

impl PriceAggregate {
    pub fn new() -> Self {
        Self {
            price: 0f64,
            volume: 0f64,
        }
    }

    pub fn averaged_price(&self) -> f64 {
        self.price
    }

    pub fn volume(&self) -> f64 {
        self.volume
    }

    pub fn accumulate(&mut self, price: f64, volume: f64) {
        self.price = (self.volume * self.price + volume * price) / (self.volume + volume);
        self.volume += volume;
    }

    pub fn drop(&mut self, price: f64, volume: f64) {
        self.price = (self.volume * self.price - volume * price) / (self.volume - volume);
        self.volume -= volume;
    }
}

#[cfg(test)]
mod tests {
    use crate::sim::engine::transaction::price_aggregate::PriceAggregate;

    #[test]
    fn should_accumulate_two_transactions() {
        let mut averaged_price = PriceAggregate::new();

        averaged_price.accumulate(1.0, 10.0);
        averaged_price.accumulate(1.5, 10.0);

        assert_eq!(1.25, averaged_price.averaged_price());
        assert_eq!(20.0, averaged_price.volume());
    }

    #[test]
    fn should_accumulate_three_transactions() {
        let mut averaged_price = PriceAggregate::new();

        averaged_price.accumulate(1.0, 10.0);
        averaged_price.accumulate(2.0, 10.0);
        averaged_price.accumulate(2.5, 20.0);

        assert_eq!(2.0, averaged_price.averaged_price());
        assert_eq!(40.0, averaged_price.volume());
    }

    #[test]
    fn should_subtract_transaction() {
        let mut averaged_price = PriceAggregate::new();

        averaged_price.accumulate(1.0, 10.0);
        averaged_price.drop(2.0, 4.0);
        averaged_price.accumulate(0.5, 16.0);

        assert_eq!(0.45454545454545453, averaged_price.averaged_price());
        assert_eq!(22.0, averaged_price.volume());
    }
}
