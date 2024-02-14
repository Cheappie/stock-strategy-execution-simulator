/*
 * Copyright (c) 2022 Kamil Konior. All rights reserved.
*/

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() {
    println!("sim-mvp");
}
