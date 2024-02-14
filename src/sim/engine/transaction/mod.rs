/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

mod currency;
mod order_price;
mod price_aggregate;

use crate::sim::bb::quotes::{OfferSide, Timestamp};
use crate::sim::engine::transaction::currency::Unicoin;
use crate::sim::engine::transaction::order_price::{BuyOrderKey, SellOrderKey};
use crate::sim::engine::transaction::price_aggregate::PriceAggregate;
use std::cmp::Ordering;
use std::collections::btree_map::Values;
use std::collections::{BTreeMap, VecDeque};
use std::iter::Chain;

pub struct TransactionEngine {
    id_generator: u64,
    total_buy: PriceAggregate,
    total_sell: PriceAggregate,
    buy_orders: BTreeMap<BuyOrderKey, ActiveTransaction>,
    sell_orders: BTreeMap<SellOrderKey, ActiveTransaction>,
    closed_transactions: Vec<ClosedTransaction>,
}

impl TransactionEngine {
    pub fn open_transaction(
        &mut self,
        open_price: f64,
        volume: f64,
        open_timestamp: Timestamp,
        offer_side: OfferSide,
    ) {
        let id = self.id_generator;
        self.id_generator += 1;

        let transaction = ActiveTransaction {
            id,
            volume,
            open_price,
            open_timestamp,
            offer_side,
        };

        match offer_side {
            OfferSide::Ask => {
                self.total_buy.accumulate(open_price, volume);
                self.buy_orders.insert(
                    BuyOrderKey {
                        id,
                        price: open_price,
                    },
                    transaction,
                );
            }
            OfferSide::Bid => {
                self.total_sell.accumulate(open_price, volume);
                self.sell_orders.insert(
                    SellOrderKey {
                        id,
                        price: open_price,
                    },
                    transaction,
                );
            }
        }
    }

    pub fn active_orders(&self) -> impl Iterator<Item = &ActiveTransaction> + '_ {
        self.buy_orders.values().chain(self.sell_orders.values())
    }

    pub fn active_orders_count(&self) -> usize {
        self.buy_orders.len() + self.sell_orders.len()
    }

    pub fn close_transaction(
        &mut self,
        id: u64,
        close_price: f64,
        close_timestamp: Timestamp,
    ) -> Result<(), anyhow::Error> {
        todo!()
        // match offer_side {
        //     OfferSide::Ask => {
        //         let order = self.buy_orders.remove(&BuyOrderKey { id, price }).ok_or_else(|| 0)?;
        //
        //     }
        //     OfferSide::Bid => {}
        // }
    }
}

pub struct ActiveTransaction {
    id: u64,
    volume: f64,
    open_price: f64,
    open_timestamp: Timestamp,
    offer_side: OfferSide,
}

pub struct ClosedTransaction {
    id: u64,
    volume: f64,
    open_price: f64,
    open_timestamp: Timestamp,
    close_price: f64,
    close_timestamp: Timestamp,
    offer_side: OfferSide,
}
