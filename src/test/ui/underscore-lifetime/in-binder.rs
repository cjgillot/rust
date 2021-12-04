// Check that we error when `'_` appears as the name of a lifetime parameter.
//
// Regression test for #52098.

struct IceCube<'a> {
    v: Vec<&'a char>,
}

impl<'_> IceCube<'_> {}
//~^ ERROR `'_` cannot be used here

struct Struct<'_> {
    //~^ ERROR `'_` cannot be used here
    //~| ERROR parameter `'_` is never used
    v: Vec<&'static char>,
}

enum Enum<'_> {
    //~^ ERROR `'_` cannot be used here
    //~| ERROR parameter `'_` is never used
    Variant,
}

union Union<'_> {
    //~^ ERROR `'_` cannot be used here
    //~| ERROR parameter `'_` is never used
    a: u32,
}

trait Trait<'_> {
    //~^ ERROR `'_` cannot be used here
}

fn foo<'_>() {
    //~^ ERROR `'_` cannot be used here
}

fn main() {}
