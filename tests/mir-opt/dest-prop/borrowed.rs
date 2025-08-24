// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: DestinationPropagation

struct Foo([u8; 100]);

#[inline(never)]
fn foo() -> Foo {
    Foo([0; 100])
}

#[inline(never)]
fn observe<T>(b: *mut T) {}

// EMIT_MIR borrowed.example.DestinationPropagation.diff
pub fn example() {
    // CHECK-LABEL: fn example(
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug b => [[a]];
    let mut a = foo();
    observe(&raw mut a);
    let mut b = a;
    observe(&raw mut b);
}

// EMIT_MIR borrowed.escaping_borrow.DestinationPropagation.diff
pub fn escaping_borrow() {
    // CHECK-LABEL fn escaping_borrow(
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug b => [[b:_.*]];
    let mut a = 5_usize;
    observe(&raw mut a);
    let mut b = a;
    observe(&raw mut b);
}

fn main() {}
