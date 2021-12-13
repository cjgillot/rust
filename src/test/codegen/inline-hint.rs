// Checks that closures, constructors, and shims except
// for a drop glue receive inline hint by default.
//
// compile-flags: -Cno-prepopulate-passes -Csymbol-mangling-version=v0
#![crate_type = "lib"]

pub static CLONE: fn(&(i32, i32, i32, i32)) -> (i32, i32, i32, i32) =
    <(i32, i32, i32, i32) as Clone>::clone;
pub static CLOSURE: fn() -> () = || {};

pub fn f() {
    let a = A;
    let b = (0i32, 1i32, 2i32, 3i32);

    a(String::new(), String::new());
    CLONE(&b);
}

struct A(String, String);

// CHECK:      ; core::ptr::drop_in_place::<inline_hint::A>
// CHECK-NEXT: ; Function Attrs:
// CHECK-NOT:  inlinehint
// CHECK-SAME: {{$}}

// CHECK:      ; <(i32, i32, i32, i32) as core::clone::Clone>::clone
// CHECK-NEXT: ; Function Attrs: inlinehint

// CHECK:      ; inline_hint::CLOSURE::{closure#0}
// CHECK-NEXT: ; Function Attrs: inlinehint

// CHECK:      ; inline_hint::A
// CHECK-NEXT: ; Function Attrs: inlinehint
