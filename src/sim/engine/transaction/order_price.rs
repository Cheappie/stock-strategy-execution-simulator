/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/
use std::cmp::Ordering;

///
/// Provides ascending order of iteration
///
#[derive(Debug)]
pub struct BuyOrderKey {
    pub id: u64,
    pub price: f64,
}

impl Eq for BuyOrderKey {}

impl PartialEq<Self> for BuyOrderKey {
    fn eq(&self, other: &Self) -> bool {
        self.id.eq(&other.id)
    }
}

impl PartialOrd<Self> for BuyOrderKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(Ord::cmp(self, other))
    }
}

impl Ord for BuyOrderKey {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.price < other.price {
            Ordering::Less
        } else if self.price > other.price {
            Ordering::Greater
        } else {
            self.id.cmp(&other.id)
        }
    }
}

///
/// Provides descending order of iteration
///
#[derive(Debug)]
pub struct SellOrderKey {
    pub id: u64,
    pub price: f64,
}

impl Eq for SellOrderKey {}

impl PartialEq<Self> for SellOrderKey {
    fn eq(&self, other: &Self) -> bool {
        self.id.eq(&other.id)
    }
}

impl PartialOrd<Self> for SellOrderKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(Ord::cmp(self, other))
    }
}

impl Ord for SellOrderKey {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.price > other.price {
            Ordering::Less
        } else if self.price < other.price {
            Ordering::Greater
        } else {
            self.id.cmp(&other.id)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::sim::engine::transaction::order_price::{BuyOrderKey, SellOrderKey};
    use std::collections::BTreeMap;

    #[test]
    fn should_sort_buy_orders_in_ascending_order() {
        let mut v = vec![
            BuyOrderKey { id: 0, price: 5f64 },
            BuyOrderKey { id: 1, price: 3f64 },
            BuyOrderKey { id: 3, price: 4f64 },
            BuyOrderKey { id: 2, price: 4f64 },
        ];

        v.sort();

        assert_eq!(&BuyOrderKey { id: 1, price: 3f64 }, &v[0]);
        assert_eq!(&BuyOrderKey { id: 2, price: 4f64 }, &v[1]);
        assert_eq!(&BuyOrderKey { id: 3, price: 4f64 }, &v[2]);
        assert_eq!(&BuyOrderKey { id: 0, price: 5f64 }, &v[3]);
    }

    #[test]
    fn should_sort_sell_orders_in_descending_order() {
        let mut v = vec![
            SellOrderKey { id: 0, price: 5f64 },
            SellOrderKey { id: 1, price: 3f64 },
            SellOrderKey { id: 2, price: 4f64 },
            SellOrderKey { id: 3, price: 5f64 },
        ];

        v.sort();

        assert_eq!(&SellOrderKey { id: 0, price: 5f64 }, &v[0]);
        assert_eq!(&SellOrderKey { id: 3, price: 5f64 }, &v[1]);
        assert_eq!(&SellOrderKey { id: 2, price: 4f64 }, &v[2]);
        assert_eq!(&SellOrderKey { id: 1, price: 3f64 }, &v[3]);
    }
}
