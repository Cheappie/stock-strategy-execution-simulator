/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use std::collections::VecDeque;

pub struct ThinCircularBuffer<E> {
    deque: VecDeque<E>,
}

impl<E> ThinCircularBuffer<E> {
    pub fn from_initialized(deque: VecDeque<E>) -> Self {
        Self { deque }
    }

    pub fn push(&mut self, element: E) -> Option<E> {
        let last = self.deque.pop_front();
        self.deque.push_back(element);
        last
    }

    pub fn iter(&self) -> impl Iterator<Item = &E> {
        self.deque.iter()
    }
}
