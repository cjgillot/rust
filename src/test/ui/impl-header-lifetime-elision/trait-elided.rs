#![allow(warnings)]

trait MyTrait<'a> {}

impl MyTrait for u32 {
    //~^ ERROR implicit elided lifetime not allowed here
    //~| ERROR missing lifetime specifier [E0106]
}

fn main() {}
