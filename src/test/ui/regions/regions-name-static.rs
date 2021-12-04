struct Foo<'static> {
    //~^ ERROR invalid lifetime parameter name: `'static`
    //~| ERROR parameter `'static` is never used [E0392]
    x: &'static isize,
}

fn main() {}
