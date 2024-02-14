/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

use std::ops::Range;

pub const U64_BITS: usize = u64::BITS as usize;
pub const ADDRESS_BITS_PER_WORD: usize = 6;

pub struct Bitmap {
    words: Vec<u64>,
}

impl Bitmap {
    pub fn empty() -> Self {
        Bitmap { words: Vec::new() }
    }

    pub fn with_capacity(bits: usize) -> Self {
        Self {
            words: vec![0u64; (bits.saturating_sub(1) >> ADDRESS_BITS_PER_WORD) + 1],
        }
    }

    pub fn capacity(&self) -> usize {
        self.words.len() * U64_BITS
    }

    pub fn contains(&self, bit: usize) -> bool {
        let w = &self.words[bit >> ADDRESS_BITS_PER_WORD];
        0 != (*w & (1u64 << (bit & 0x3F)))
    }

    pub fn insert(&mut self, bit: usize) {
        let w = &mut self.words[bit >> ADDRESS_BITS_PER_WORD];
        *w |= 1u64 << (bit & 0x3F);
    }

    pub fn insert_binary(&mut self, position: usize, flag: bool) {
        let w = &mut self.words[position >> ADDRESS_BITS_PER_WORD];
        *w |= (flag as u64) << (position & 0x3F);
    }

    pub unsafe fn insert_binary_unsafe(&mut self, position: usize, flag: bool) {
        let w = self.get_mut_word(position);
        *w |= (flag as u64) << (position & 0x3F);
    }

    unsafe fn get_mut_word(&mut self, position: usize) -> &mut u64 {
        #[cfg(debug_assertions)]
        {
            &mut self.words[position >> ADDRESS_BITS_PER_WORD]
        }
        #[cfg(not(debug_assertions))]
        {
            self.words
                .get_unchecked_mut(position >> ADDRESS_BITS_PER_WORD)
        }
    }

    pub unsafe fn insert_binary_unsafe_64(&mut self, position: usize, count: usize, flag: bool) {
        debug_assert!(count <= U64_BITS - (position & 0x3F));
        let mask = std::mem::transmute::<i64, u64>(-(1 & flag as i64));

        let start_mask = mask << (position & 0x3F);
        let end_mask = u64::MAX.wrapping_shr((U64_BITS - ((position + count) & 0x3F)) as u32);

        let w = self.get_mut_word(position);
        *w |= (start_mask & end_mask);
    }

    pub fn remove(&mut self, bit: usize) {
        let w = &mut self.words[bit >> ADDRESS_BITS_PER_WORD];
        *w &= !(1u64 << (bit & 0x3F));
    }

    pub fn cardinality(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub fn insert_range(&mut self, range: Range<usize>) {
        if range.is_empty() {
            return;
        }

        let first = range.start >> ADDRESS_BITS_PER_WORD;
        let last = (range.end - 1) >> ADDRESS_BITS_PER_WORD;

        let start_mask = u64::MAX << (range.start & 0x3F);
        let end_mask = u64::MAX.wrapping_shr((U64_BITS - (range.end & 0x3F)) as u32);

        if first == last {
            self.words[first] |= start_mask & end_mask;
        } else {
            self.words[first] |= start_mask;

            self.words[(first + 1)..last]
                .iter_mut()
                .for_each(|w| *w = u64::MAX);

            self.words[last] |= end_mask;
        }
    }

    pub fn cardinality_in_range(&self, range: Range<usize>) -> u32 {
        if range.is_empty() {
            return 0;
        }

        let mut cardinality = 0;

        let first = range.start >> ADDRESS_BITS_PER_WORD;
        let last = (range.end - 1) >> ADDRESS_BITS_PER_WORD;

        let start_mask = u64::MAX << (range.start & 0x3F);
        let end_mask = u64::MAX.wrapping_shr((U64_BITS - (range.end & 0x3F)) as u32);

        if first == last {
            let intersect = start_mask & end_mask;
            cardinality += (self.words[first] & intersect).count_ones();
        } else {
            cardinality += (self.words[first] & start_mask).count_ones();

            cardinality += self.words[(first + 1)..last]
                .iter()
                .map(|w| w.count_ones())
                .sum::<u32>();

            cardinality += (self.words[last] & end_mask).count_ones();
        }

        cardinality
    }

    pub fn iand_range(&mut self, other: &Bitmap, range: Range<usize>) {
        self.apply_binary_logical_operator::<And>(other, range);
    }

    pub fn ior_range(&mut self, other: &Bitmap, range: Range<usize>) {
        self.apply_binary_logical_operator::<Or>(other, range);
    }

    fn apply_binary_logical_operator<OP: LogicalOperator>(
        &mut self,
        other: &Bitmap,
        range: Range<usize>,
    ) {
        if range.is_empty() {
            return;
        }

        fn apply<OP: LogicalOperator>(this: u64, other: u64, mask: u64) -> u64 {
            let op_bits = OP::apply(this, other) & mask;
            let preserved_bits = this & (!mask);
            op_bits | preserved_bits
        }

        let first = range.start >> ADDRESS_BITS_PER_WORD;
        let last = (range.end - 1) >> ADDRESS_BITS_PER_WORD;

        let start_mask = u64::MAX << (range.start & 0x3F);
        let end_mask = u64::MAX.wrapping_shr((U64_BITS - (range.end & 0x3F)) as u32);

        if first == last {
            self.words[first] =
                apply::<OP>(self.words[first], other.words[first], start_mask & end_mask);
        } else {
            self.words[first] = apply::<OP>(self.words[first], other.words[first], start_mask);

            self.words[(first + 1)..last]
                .iter_mut()
                .zip(other.words[(first + 1)..last].iter())
                .for_each(|(this, other)| *this = OP::apply(*this, *other));

            self.words[last] = apply::<OP>(self.words[last], other.words[last], end_mask);
        }
    }

    pub fn inot_range(&mut self, range: Range<usize>) {
        if range.is_empty() {
            return;
        }

        fn negate(word: u64, mask: u64) -> u64 {
            let negated_bits = (!word) & mask;
            let preserved_bits = word & (!mask);
            negated_bits | preserved_bits
        }

        let first = range.start >> ADDRESS_BITS_PER_WORD;
        let last = (range.end - 1) >> ADDRESS_BITS_PER_WORD;

        let start_mask = u64::MAX << (range.start & 0x3F);
        let end_mask = u64::MAX.wrapping_shr((U64_BITS - (range.end & 0x3F)) as u32);

        if first == last {
            self.words[first] = negate(self.words[first], start_mask & end_mask);
        } else {
            self.words[first] = negate(self.words[first], start_mask);

            self.words[(first + 1)..last]
                .iter_mut()
                .for_each(|w| *w = !*w);

            self.words[last] = negate(self.words[last], end_mask);
        }
    }

    pub fn iter(&self) -> BitmapIter {
        BitmapIter::new(&self.words)
    }

    pub fn iter_run(&self) -> BitmapRunIter {
        BitmapRunIter::new(&self.words)
    }

    pub fn from_bits(bits: Vec<usize>) -> Self {
        let highest_bit = bits.iter().copied().max().unwrap_or(0usize);
        let mut bitmap = Bitmap::with_capacity(highest_bit);
        bits.iter().for_each(|&bit| bitmap.insert(bit));
        bitmap
    }
}

trait LogicalOperator {
    fn apply(lhs: u64, rhs: u64) -> u64;
}

pub struct And;
impl LogicalOperator for And {
    fn apply(lhs: u64, rhs: u64) -> u64 {
        lhs & rhs
    }
}

pub struct Or;
impl LogicalOperator for Or {
    fn apply(lhs: u64, rhs: u64) -> u64 {
        lhs | rhs
    }
}

impl From<Vec<u64>> for Bitmap {
    fn from(words: Vec<u64>) -> Self {
        Bitmap { words }
    }
}

pub struct BitmapIter<'a> {
    words: &'a Vec<u64>,
    current: Option<(usize, u64)>,
}

impl<'a> BitmapIter<'a> {
    pub fn new(words: &'a Vec<u64>) -> BitmapIter<'a> {
        let first = words.get(0).and_then(|&word| Some((0, word)));
        BitmapIter {
            words,
            current: first,
        }
    }

    pub fn next(&mut self) -> Option<u32> {
        while let Some((index, word)) = &mut self.current {
            if *word == 0 {
                *index += 1;
                if *index < self.words.len() {
                    *word = self.words[*index];
                } else {
                    self.current = None
                }
            } else {
                let next = (*index as u32 * u64::BITS) + word.trailing_zeros();
                *word &= *word - 1;
                return Some(next);
            }
        }

        None
    }

    pub fn advance_to(&mut self, offset: usize) {
        advance_to(self.words, &mut self.current, offset);
    }
}

fn advance_to(words: &[u64], current: &mut Option<(usize, u64)>, offset: usize) {
    let next_idx = offset >> ADDRESS_BITS_PER_WORD;

    if words.len() <= next_idx {
        *current = None;
    } else {
        if let Some((curr_idx, curr_word)) = current {
            if *curr_idx <= next_idx {
                if *curr_idx < next_idx {
                    *curr_idx = next_idx;
                    *curr_word = words[next_idx];
                }
                *curr_word &= u64::MAX << ((offset & 0x3F) as u64);
            }
        }
    }
}

pub struct BitmapRunIter<'a> {
    words: &'a Vec<u64>,
    current: Option<(usize, u64)>,
}

impl<'a> BitmapRunIter<'a> {
    pub fn new(words: &'a Vec<u64>) -> BitmapRunIter<'a> {
        let first = words.get(0).and_then(|&word| Some((0, word)));
        BitmapRunIter {
            words,
            current: first,
        }
    }

    pub fn next(&mut self) -> Option<Run> {
        while let Some((index, word)) = &mut self.current {
            if *word == 0 {
                *index += 1;
                if *index < self.words.len() {
                    *word = self.words[*index];
                } else {
                    self.current = None
                }
            } else {
                let head = word.trailing_zeros();
                let mask = u64::MAX << (head as u64);
                let inverse = !*word & mask;
                let tail = inverse.trailing_zeros();
                *word &= u64::MAX.checked_shl(tail).unwrap_or(0);

                let position = *index as u32 * u64::BITS + head;
                let length = tail - head;

                return Some(Run { position, length });
            }
        }

        None
    }

    pub fn advance_to(&mut self, offset: usize) {
        advance_to(self.words, &mut self.current, offset);
    }
}

#[derive(Debug)]
pub struct Run {
    pub position: u32,
    pub length: u32,
}

fn pretty_word(w: u64) -> String {
    let mut cbuf = String::with_capacity(64);
    pretty_word_to_buf(&mut cbuf, w);
    cbuf
}

fn pretty_word_to_buf(cbuf: &mut String, w: u64) {
    for i in (0..64).rev() {
        if (w & (1u64 << i)) != 0 {
            cbuf.push('1');
        } else {
            cbuf.push('0');
        }
    }
}

fn pretty_bitmap(bitmap: &Bitmap) -> String {
    let mut cbuf = String::with_capacity(bitmap.capacity());

    for w in &bitmap.words {
        pretty_word_to_buf(&mut cbuf, *w);
    }

    cbuf
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use crate::sim::collections::bitmap::{Bitmap, Run};

    #[test]
    fn should_insert_first_bit() {
        //given
        let mut bitmap = Bitmap::with_capacity(1);

        //when
        bitmap.insert(0);

        //then
        assert!(bitmap.contains(0));
    }

    #[test]
    fn should_insert_middle_bit() {
        //given
        let mut bitmap = Bitmap::with_capacity(1);

        //when
        bitmap.insert(32);

        //then
        assert!(bitmap.contains(32));
    }

    #[test]
    fn should_insert_last_bit() {
        //given
        let mut bitmap = Bitmap::with_capacity(1);

        //when
        bitmap.insert(63);

        //then
        assert!(bitmap.contains(63));
    }

    #[test]
    fn should_insert_bits_across_words() {
        //given
        let mut bitmap = Bitmap::with_capacity(128);

        //when
        bitmap.insert(0);
        bitmap.insert(64);

        //then
        assert!(bitmap.contains(0));
        assert!(bitmap.contains(64));
    }

    #[test]
    fn should_remove_bit() {
        //given
        let mut bitmap = Bitmap::with_capacity(1);
        bitmap.insert(0);

        //when
        bitmap.remove(0);

        //then
        assert!(!bitmap.contains(0));
    }

    #[test]
    fn should_calculate_cardinality() {
        //given
        let mut bitmap = Bitmap::with_capacity(128);

        bitmap.insert(0);
        bitmap.insert(32);
        bitmap.insert(64);
        bitmap.insert(96);

        //when
        let cardinality = bitmap.cardinality();

        //then
        assert_eq!(4, cardinality)
    }

    #[test]
    fn should_insert_range_first_word() {
        //given
        let mut bitmap = Bitmap::with_capacity(128);

        //when
        bitmap.insert_range(0..64);

        //then
        assert_eq!(64, cardinality_in_range(&bitmap, 0..64));
        assert_eq!(64, bitmap.cardinality());
    }

    #[test]
    fn should_insert_range_second_word() {
        //given
        let mut bitmap = Bitmap::with_capacity(128);

        //when
        bitmap.insert_range(64..128);

        //then
        assert_eq!(64, cardinality_in_range(&bitmap, 64..128));
        assert_eq!(64, bitmap.cardinality());
    }

    #[test]
    fn should_insert_range_partial_first_word() {
        //given
        let mut bitmap = Bitmap::with_capacity(128);

        //when
        bitmap.insert_range(24..40);

        //then
        assert_eq!(16, cardinality_in_range(&bitmap, 24..40));
        assert_eq!(16, bitmap.cardinality());
    }

    #[test]
    fn should_insert_range_partial_second_word() {
        //given
        let mut bitmap = Bitmap::with_capacity(128);

        //when
        bitmap.insert_range(88..104);

        //then
        assert_eq!(16, cardinality_in_range(&bitmap, 88..104));
        assert_eq!(16, bitmap.cardinality());
    }

    #[test]
    fn should_insert_range_full_multiple_words() {
        //given
        let mut bitmap = Bitmap::with_capacity(256);

        //when
        bitmap.insert_range(0..256);

        //then
        assert_eq!(256, bitmap.cardinality());
    }

    #[test]
    fn should_insert_range_partial_across_multiple_words() {
        //given
        let mut bitmap = Bitmap::with_capacity(256);

        //when
        bitmap.insert_range(32..160);

        //then
        assert_eq!(128, cardinality_in_range(&bitmap, 32..160));
        assert_eq!(128, bitmap.cardinality());
    }

    #[test]
    fn should_not_insert_empty_range() {
        //given
        let mut bitmap = Bitmap::with_capacity(256);

        //when
        bitmap.insert_range(128..128);

        //then
        assert_eq!(0, bitmap.cardinality());
    }

    fn cardinality_in_range(bitmap: &Bitmap, range: Range<usize>) -> u32 {
        range.fold(0u32, |acc, i| acc + if bitmap.contains(i) { 1 } else { 0 })
    }

    #[test]
    fn should_negate_range() {
        let mut bitmap = Bitmap::with_capacity(256);

        bitmap.insert(3);
        bitmap.inot_range(2..5);

        assert!(bitmap.contains(2));
        assert!(!bitmap.contains(3));
        assert!(bitmap.contains(4));
        assert_eq!(2, bitmap.cardinality());
    }

    #[test]
    fn negate_should_preserve_bits() {
        let mut bitmap = Bitmap::with_capacity(256);

        bitmap.insert(0);
        bitmap.insert(9);
        bitmap.insert(15);
        bitmap.insert(16);
        bitmap.insert(19);
        bitmap.insert(21);
        bitmap.insert(30);
        bitmap.insert(63);
        bitmap.insert(70);

        bitmap.inot_range(10..30);

        assert!(bitmap.contains(0));
        assert!(bitmap.contains(9));
        assert_eq!(5, cardinality_in_range(&bitmap, 10..15));
        assert!(!bitmap.contains(15));
        assert!(!bitmap.contains(16));
        assert_eq!(2, cardinality_in_range(&bitmap, 17..19));
        assert!(!bitmap.contains(19));
        assert!(bitmap.contains(20));
        assert!(!bitmap.contains(21));
        assert_eq!(8, cardinality_in_range(&bitmap, 22..30));
        assert!(bitmap.contains(30));
        assert!(bitmap.contains(63));
        assert!(bitmap.contains(70));
        assert_eq!(21, bitmap.cardinality());
    }

    #[test]
    fn should_negate_cross_word_range() {
        let mut bitmap = Bitmap::with_capacity(256);

        bitmap.insert(64);
        bitmap.inot_range(63..66);

        assert!(bitmap.contains(63));
        assert!(!bitmap.contains(64));
        assert!(bitmap.contains(65));
        assert_eq!(2, bitmap.cardinality());
    }

    #[test]
    fn should_negate_multi_word_range() {
        let mut bitmap = Bitmap::with_capacity(256);

        bitmap.insert(63);
        bitmap.insert(64);
        bitmap.insert(127);
        bitmap.inot_range(32..128);

        assert_eq!(31, cardinality_in_range(&bitmap, 32..63));
        assert!(!bitmap.contains(63));
        assert!(!bitmap.contains(64));
        assert_eq!(62, cardinality_in_range(&bitmap, 65..127));
        assert!(!bitmap.contains(127));
        assert_eq!(93, bitmap.cardinality());
    }

    #[test]
    fn should_intersect_range() {
        let mut this = {
            let mut bitmap = Bitmap::with_capacity(256);

            bitmap.insert(0);
            bitmap.insert_range(30..40);
            bitmap.insert(42);
            bitmap.insert(44);
            bitmap.insert(45);
            bitmap.insert(46);
            bitmap.insert(63);
            bitmap.insert(64);

            bitmap
        };

        let mut other = {
            let mut bitmap = Bitmap::with_capacity(256);

            bitmap.insert(0);
            bitmap.insert_range(38..40);
            bitmap.insert(44);
            bitmap.insert(46);
            bitmap.insert(47);
            bitmap.insert(63);
            bitmap.insert(64);

            bitmap
        };

        this.iand_range(&other, 35..50);

        assert!(this.contains(0));
        assert_eq!(5, cardinality_in_range(&this, 30..35));
        assert_eq!(2, cardinality_in_range(&this, 38..40));
        assert!(this.contains(44));
        assert!(this.contains(46));
        assert!(this.contains(63));
        assert!(this.contains(64));
        assert_eq!(12, this.cardinality());
    }

    #[test]
    fn should_intersect_multi_word_range() {
        let mut this = {
            let mut bitmap = Bitmap::with_capacity(256);

            bitmap.insert(50);
            bitmap.insert_range(64..128);
            bitmap.insert(150);

            bitmap.insert(0);
            bitmap.insert(30);
            bitmap.insert(55);
            bitmap.insert(140);

            bitmap
        };

        let mut other = {
            let mut bitmap = Bitmap::with_capacity(256);

            bitmap.insert(50);
            bitmap.insert_range(64..128);
            bitmap.insert(150);

            bitmap.insert(2);
            bitmap.insert(33);
            bitmap.insert(58);
            bitmap.insert(142);

            bitmap
        };

        this.iand_range(&other, 1..256);

        assert!(this.contains(0));
        assert!(this.contains(50));
        assert_eq!(64, cardinality_in_range(&this, 64..128));
        assert!(this.contains(150));

        assert_eq!(67, this.cardinality());
    }

    #[test]
    fn should_union_range() {
        let mut this = {
            let mut bitmap = Bitmap::with_capacity(256);

            bitmap.insert(0);
            bitmap.insert_range(30..35);
            bitmap.insert(42);
            bitmap.insert(44);
            bitmap.insert(45);
            bitmap.insert(46);
            bitmap.insert(63);
            bitmap.insert(64);

            bitmap
        };

        let mut other = {
            let mut bitmap = Bitmap::with_capacity(256);

            bitmap.insert(0);
            bitmap.insert_range(38..40);
            bitmap.insert(44);
            bitmap.insert(46);
            bitmap.insert(63);
            bitmap.insert(64);

            bitmap
        };

        this.ior_range(&other, 32..50);

        assert!(this.contains(0));
        assert_eq!(5, cardinality_in_range(&this, 30..35));
        assert_eq!(2, cardinality_in_range(&this, 38..40));
        assert!(this.contains(42));
        assert!(this.contains(44));
        assert!(this.contains(45));
        assert!(this.contains(46));
        assert!(this.contains(63));
        assert!(this.contains(64));
        assert_eq!(14, this.cardinality());
    }

    #[test]
    fn should_union_multi_word_range() {
        let mut this = {
            let mut bitmap = Bitmap::with_capacity(256);

            bitmap.insert_range(32..70);
            bitmap.insert(160);

            bitmap
        };

        let mut other = {
            let mut bitmap = Bitmap::with_capacity(256);

            bitmap.insert(0);
            bitmap.insert_range(60..108);

            bitmap
        };

        this.ior_range(&other, 1..256);

        assert_eq!(76, cardinality_in_range(&this, 32..108));
        assert!(this.contains(160));
        assert_eq!(77, this.cardinality());
    }

    #[test]
    fn should_iter_bits() {
        let mut bitmap = Bitmap::with_capacity(256);

        bitmap.insert(0);
        bitmap.insert(2);
        bitmap.insert(4);
        bitmap.insert(63);
        bitmap.insert(64);
        bitmap.insert(150);

        let mut iter = bitmap.iter();

        assert_eq!(0, iter.next().unwrap());
        assert_eq!(2, iter.next().unwrap());
        assert_eq!(4, iter.next().unwrap());
        assert_eq!(63, iter.next().unwrap());
        assert_eq!(64, iter.next().unwrap());
        assert_eq!(150, iter.next().unwrap());
        assert!(iter.next().is_none());
    }

    #[test]
    fn should_iter_run() {
        fn assert_run(position: u32, length: u32, run: Run) {
            assert_eq!(position, run.position);
            assert_eq!(length, run.length);
        }

        let mut bitmap = Bitmap::with_capacity(256);

        bitmap.insert(0);
        bitmap.insert(2);
        bitmap.insert_range(4..6);
        bitmap.insert(63);
        bitmap.insert(64);
        bitmap.insert(65);
        bitmap.insert(250);

        let mut iter = bitmap.iter_run();

        assert_run(0, 1, iter.next().unwrap());
        assert_run(2, 1, iter.next().unwrap());
        assert_run(4, 2, iter.next().unwrap());
        assert_run(63, 1, iter.next().unwrap());
        assert_run(64, 2, iter.next().unwrap());
        assert_run(250, 1, iter.next().unwrap());
        assert!(iter.next().is_none());
    }

    #[test]
    fn should_insert_binary_flag() {
        let mut bitmap = Bitmap::with_capacity(256);
        bitmap.insert_binary(100, true);
        assert!(bitmap.contains(100));
        assert_eq!(1, bitmap.cardinality());
    }

    #[test]
    fn should_not_insert_binary_flag() {
        let mut bitmap = Bitmap::with_capacity(256);
        bitmap.insert_binary(100, false);
        assert!(!bitmap.contains(100));
        assert_eq!(0, bitmap.cardinality());
    }

    #[test]
    fn should_estimate_cardinality_in_range() {
        let bitmap = Bitmap::from_bits(vec![1, 2, 4, 5, 6, 100, 500, 511, 512, 513]);

        assert_eq!(0, bitmap.cardinality_in_range(2..2));
        assert_eq!(3, bitmap.cardinality_in_range(2..6));
        assert_eq!(1, bitmap.cardinality_in_range(100..101));
        assert_eq!(3, bitmap.cardinality_in_range(6..501));
        assert_eq!(10, bitmap.cardinality_in_range(0..514));
    }

    #[test]
    fn should_insert_binary_flag_unsafe() {
        let mut bitmap = Bitmap::with_capacity(256);

        unsafe {
            bitmap.insert_binary_unsafe(100, true);
        }

        assert!(bitmap.contains(100));
        assert_eq!(1, bitmap.cardinality());
    }

    #[test]
    fn should_not_insert_binary_flag_unsafe() {
        let mut bitmap = Bitmap::with_capacity(256);

        unsafe {
            bitmap.insert_binary_unsafe(100, false);
        }

        assert!(!bitmap.contains(100));
        assert_eq!(0, bitmap.cardinality());
    }

    #[test]
    fn should_correctly_assess_bitmap_capacity() {
        let mut bitmap = Bitmap::with_capacity(64);
        assert_eq!(64, bitmap.capacity());

        let mut bitmap = Bitmap::with_capacity(256);
        assert_eq!(256, bitmap.capacity());
    }

    #[test]
    fn should_insert_binary_unsafe_64() {
        unsafe {
            // single bit
            let mut bitmap = Bitmap::with_capacity(64);
            bitmap.insert_binary_unsafe_64(0, 1, true);
            assert!(bitmap.contains(0));
            assert_eq!(bitmap.cardinality(), 1);
        }

        unsafe {
            // two consecutive bits
            let mut bitmap = Bitmap::with_capacity(64);
            bitmap.insert_binary_unsafe_64(0, 2, true);
            assert!(bitmap.contains(0));
            assert!(bitmap.contains(1));
            assert_eq!(bitmap.cardinality(), 2);
        }

        unsafe {
            // single somewhere in the middle
            let mut bitmap = Bitmap::with_capacity(64);
            bitmap.insert_binary_unsafe_64(30, 1, true);
            assert!(bitmap.contains(30));
            assert_eq!(bitmap.cardinality(), 1);
        }

        unsafe {
            // two consecutive bits somewhere in the middle
            let mut bitmap = Bitmap::with_capacity(64);
            bitmap.insert_binary_unsafe_64(30, 7, true);
            assert_eq!(7, bitmap.cardinality_in_range(30..37));
            assert_eq!(bitmap.cardinality(), 7);
        }

        unsafe {
            // single bit at the end
            let mut bitmap = Bitmap::with_capacity(64);
            bitmap.insert_binary_unsafe_64(63, 1, true);
            assert!(bitmap.contains(63));
            assert_eq!(bitmap.cardinality(), 1);
        }

        unsafe {
            // three consecutive bits at the end
            let mut bitmap = Bitmap::with_capacity(64);
            bitmap.insert_binary_unsafe_64(61, 3, true);
            assert_eq!(3, bitmap.cardinality_in_range(61..64));
            assert_eq!(bitmap.cardinality(), 3);
        }
    }
}
