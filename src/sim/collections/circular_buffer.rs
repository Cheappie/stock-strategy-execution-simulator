/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/


pub struct CircularBuffer<T> {
    buf: Vec<T>,
    head: usize,
    capacity: usize,
}

impl<T> CircularBuffer<T>
where
    T: Copy + Default,
{
    pub fn new(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two());

        Self {
            buf: vec![T::default(); capacity],
            head: 0,
            capacity,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn push(&mut self, value: T) {
        unsafe {
            let idx = wrap_index(self.head, self.capacity);
            *self.buf.get_unchecked_mut(idx) = value;
        }

        self.head = self.head.wrapping_add(1);
    }

    pub fn get(&self, index: usize) -> &T {
        debug_assert!(index < self.capacity);
        let idx = wrap_index(self.head + index, self.capacity);
        unsafe { self.buf.get_unchecked(idx) }
    }

    pub unsafe fn get_direct(&self, index: usize) -> &T {
        debug_assert!(index < self.capacity);
        self.buf.get_unchecked(index)
    }

    ///
    /// Calculates tail by subtracting distance from head and wrapping if needed
    ///
    pub fn tail(&self, distance: usize) -> usize {
        debug_assert!(distance <= self.capacity);
        wrap_index(self.capacity + self.head - distance, self.capacity)
    }

    pub fn head(&self) -> usize {
        self.head
    }

    pub fn iter_top(&self, distance: usize) -> impl Iterator<Item = &T> {
        let tail = self.tail(distance);
        let head = tail + distance;

        (tail..head)
            .map(|i| wrap_index(i, self.capacity))
            .map(|i| unsafe { self.buf.get_unchecked(i) })
    }
}

pub fn wrap_index(index: usize, cap: usize) -> usize {
    debug_assert!(cap.is_power_of_two());
    index & (cap - 1)
}

#[cfg(test)]
mod tests {
    use crate::sim::collections::circular_buffer::{wrap_index, CircularBuffer};

    #[test]
    fn should_push_elements() {
        //given
        let mut buf = CircularBuffer::<f64>::new(4);

        //when
        (1..5).map(|i| i as f64).for_each(|n| buf.push(n));

        //then
        assert_eq!(1f64, *buf.get(0));
        assert_eq!(2f64, *buf.get(1));
        assert_eq!(3f64, *buf.get(2));
        assert_eq!(4f64, *buf.get(3));
    }

    #[test]
    fn should_push_elements_and_wrap_on_overflow() {
        //given
        let mut buf = CircularBuffer::<f64>::new(4);

        //when
        (1..8).map(|i| i as f64).for_each(|n| buf.push(n));

        //then
        assert_eq!(4f64, *buf.get(0));
        assert_eq!(5f64, *buf.get(1));
        assert_eq!(6f64, *buf.get(2));
        assert_eq!(7f64, *buf.get(3));
    }

    #[test]
    fn should_calculate_dynamic_tail() {
        //given
        let mut buf = CircularBuffer::<f64>::new(4);

        //when
        (0..9).map(|i| i as f64).for_each(|n| buf.push(n));

        //then
        let offset = buf.tail(1);
        assert_eq!(8f64, unsafe {
            *buf.get_direct(wrap_index(offset, buf.capacity))
        });

        let offset = buf.tail(2);
        assert_eq!(7f64, unsafe {
            *buf.get_direct(wrap_index(offset, buf.capacity))
        });
        assert_eq!(8f64, unsafe {
            *buf.get_direct(wrap_index(offset + 1, buf.capacity))
        });

        let offset = buf.tail(3);
        assert_eq!(6f64, unsafe {
            *buf.get_direct(wrap_index(offset, buf.capacity))
        });
        assert_eq!(7f64, unsafe {
            *buf.get_direct(wrap_index(offset + 1, buf.capacity))
        });
        assert_eq!(8f64, unsafe {
            *buf.get_direct(wrap_index(offset + 2, buf.capacity))
        });

        let offset = buf.tail(4);
        assert_eq!(5f64, unsafe {
            *buf.get_direct(wrap_index(offset, buf.capacity))
        });
        assert_eq!(6f64, unsafe {
            *buf.get_direct(wrap_index(offset + 1, buf.capacity))
        });
        assert_eq!(7f64, unsafe {
            *buf.get_direct(wrap_index(offset + 2, buf.capacity))
        });
        assert_eq!(8f64, unsafe {
            *buf.get_direct(wrap_index(offset + 3, buf.capacity))
        });
    }

    #[test]
    fn should_iter_top() {
        //given
        let mut buf = CircularBuffer::<f64>::new(4);

        //when
        (0..9).map(|i| i as f64).for_each(|n| buf.push(n));

        //then
        let mut iter = buf.iter_top(1);
        assert_eq!(8f64, *iter.next().unwrap());

        let mut iter = buf.iter_top(2);
        assert_eq!(7f64, *iter.next().unwrap());
        assert_eq!(8f64, *iter.next().unwrap());

        let mut iter = buf.iter_top(3);
        assert_eq!(6f64, *iter.next().unwrap());
        assert_eq!(7f64, *iter.next().unwrap());
        assert_eq!(8f64, *iter.next().unwrap());

        let mut iter = buf.iter_top(4);
        assert_eq!(5f64, *iter.next().unwrap());
        assert_eq!(6f64, *iter.next().unwrap());
        assert_eq!(7f64, *iter.next().unwrap());
        assert_eq!(8f64, *iter.next().unwrap());
    }
}
