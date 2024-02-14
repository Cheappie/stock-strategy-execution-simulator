/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use crate::sim::collections::bitmap::ADDRESS_BITS_PER_WORD;

pub struct Iter64 {
    start: usize,
    end: usize,
}

impl Iter64 {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn slice64_from(start: usize, end: usize) -> Slice64 {
        Slice64 {
            start,
            end: end.min(((start >> ADDRESS_BITS_PER_WORD) + 1) << ADDRESS_BITS_PER_WORD),
        }
    }
}

impl Iterator for Iter64 {
    type Item = Slice64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let slice64 = Iter64::slice64_from(self.start, self.end);
            self.start = slice64.end;
            Some(slice64)
        } else {
            None
        }
    }
}

pub struct Slice64 {
    pub start: usize,
    pub end: usize,
}

impl Slice64 {
    pub fn len(&self) -> usize {
        self.end - self.start
    }
}

#[cfg(test)]
mod tests {
    use crate::sim::collections::iter64::Iter64;

    #[test]
    fn should_iter_word_start() {
        let mut iter64 = Iter64::new(0, 1);
        let result = iter64.next().unwrap();
        assert_eq!(0, result.start);
        assert_eq!(1, result.end);
        assert_eq!(1, result.len());
    }

    #[test]
    fn should_iter_word_end() {
        let mut iter64 = Iter64::new(63, 64);
        let result = iter64.next().unwrap();
        assert_eq!(63, result.start);
        assert_eq!(64, result.end);
        assert_eq!(1, result.len());
    }

    #[test]
    fn should_iter_word_middle_range() {
        let mut iter64 = Iter64::new(30, 32);
        let result = iter64.next().unwrap();
        assert_eq!(30, result.start);
        assert_eq!(32, result.end);
        assert_eq!(2, result.len());
    }

    #[test]
    fn should_iter_multi_word_range() {
        let mut iter64 = Iter64::new(30, 148);

        {
            let result = iter64.next().unwrap();
            assert_eq!(30, result.start);
            assert_eq!(64, result.end);
            assert_eq!(34, result.len());
        }

        {
            let result = iter64.next().unwrap();
            assert_eq!(64, result.start);
            assert_eq!(128, result.end);
            assert_eq!(64, result.len());
        }

        {
            let result = iter64.next().unwrap();
            assert_eq!(128, result.start);
            assert_eq!(148, result.end);
            assert_eq!(20, result.len());
        }
    }
}
