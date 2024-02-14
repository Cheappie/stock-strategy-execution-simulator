/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

pub struct CycleIter<'a, T>(&'a T);

impl<'a, T> Iterator for CycleIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        Some(&self.0)
    }
}
