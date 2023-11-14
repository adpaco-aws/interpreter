#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, bolero_generator::TypeGenerator)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub enum Var {
    X,
    Y,
    Z,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, bolero_generator::TypeGenerator)]
#[cfg_attr(kani, derive(kani::Arbitrary))]

pub enum BinaryOp {
    Eq,
    LessEq,
    Add,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, bolero_generator::TypeGenerator)]
// #[cfg_attr(kani, derive(kani::Arbitrary))]
pub enum Expr {
    Int(i32),
    Bool(bool),
    Var(Var),
    If {
        test_expr: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
    BinaryApp {
        op: BinaryOp,
        arg1: Box<Expr>,
        arg2: Box<Expr>,
    },
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone, bolero_generator::TypeGenerator)]
pub enum Value {
    Int(i32),
    Bool(bool),
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone, bolero_generator::TypeGenerator)]
pub enum Type {
    IntType,
    BoolType,
}

pub struct Typechecker {
    x: Type,
    y: Type,
    z: Type,
}

impl Typechecker {
    pub fn new(x: Type, y: Type, z: Type) -> Self {
        Self { x: x, y: y, z: z }
    }

    // Type signature has changed: We pass `e` by reference (`&Expr`).
    // That way, we avoid cloning the expression in the Kani harnesses.
    pub fn typecheck(&self, e: &Expr) -> Result<Type, ()> {
        match e {
            Expr::Int(_) => Ok(Type::IntType),
            Expr::Bool(_) => Ok(Type::BoolType),
            Expr::Var(Var::X) => Ok(self.x),
            Expr::Var(Var::Y) => Ok(self.y),
            Expr::Var(Var::Z) => Ok(self.z),
            Expr::If {
                test_expr,
                then_expr,
                else_expr,
            } => match self.typecheck(test_expr) {
                Ok(t) => match t {
                    Type::IntType => Err(()),
                    Type::BoolType => {
                        let tt = self.typecheck(then_expr);
                        let te = self.typecheck(else_expr);
                        if tt.is_err() || te.is_err() || tt != te {
                            Err(())
                        } else {
                            tt
                        }
                    }
                },
                Err(e) => Err(e),
            },
            Expr::BinaryApp { op, arg1, arg2 } => {
                let t1 = self.typecheck(arg1);
                let t2 = self.typecheck(arg2);
                if let (Ok(t1), Ok(t2)) = (t1, t2) {
                    match op {
                        BinaryOp::Eq => Ok(Type::BoolType),
                        BinaryOp::LessEq => match (t1, t2) {
                            (Type::IntType, Type::IntType) => Ok(Type::BoolType),
                            (_, _) => Err(()),
                        },
                        BinaryOp::Add => match (t1, t2) {
                            (Type::IntType, Type::IntType) => Ok(Type::IntType),
                            (_, _) => Err(()),
                        },
                    }
                } else {
                    Err(())
                }
            }
        }
    }
}
pub struct Evaluator {
    x: Option<Value>,
    y: Option<Value>,
    z: Option<Value>,
}

impl Evaluator {
    pub fn new(x: Option<Value>, y: Option<Value>, z: Option<Value>) -> Self {
        Self { x: x, y: y, z: z }
    }

    pub fn interpret(&self, e: Expr) -> Result<Value, ()> {
        match e {
            Expr::Int(i) => Ok(Value::Int(i)),
            Expr::Bool(b) => Ok(Value::Bool(b)),
            Expr::Var(Var::X) => self.x.ok_or(()),
            Expr::Var(Var::Y) => self.y.ok_or(()),
            Expr::Var(Var::Z) => self.z.ok_or(()),
            Expr::If {
                test_expr,
                then_expr,
                else_expr,
            } => match self.interpret(*test_expr) {
                Ok(v) => match v {
                    Value::Int(_) => Err(()),
                    Value::Bool(b) => {
                        if b {
                            self.interpret(*then_expr)
                        } else {
                            self.interpret(*else_expr)
                        }
                    }
                },
                Err(e) => Err(e),
            },
            Expr::BinaryApp { op, arg1, arg2 } => {
                let v1 = self.interpret(*arg1);
                let v2 = self.interpret(*arg2);
                if let (Ok(v1), Ok(v2)) = (v1, v2) {
                    match op {
                        BinaryOp::Eq => Ok(Value::Bool(v1 == v2)),
                        BinaryOp::LessEq => match (v1, v2) {
                            (Value::Int(i1), Value::Int(i2)) => Ok(Value::Bool(i1 <= i2)),
                            (_, _) => Err(()),
                        },
                        BinaryOp::Add => match (v1, v2) {
                            (Value::Int(i1), Value::Int(i2)) => Ok(Value::Int(i1.wrapping_add(i2))),
                            (_, _) => Err(()),
                        },
                    }
                } else {
                    Err(())
                }
            }
        }
    }
}

#[test]
pub fn test_eval() {
    let eval = Evaluator::new(Some(Value::Int(1)), None, None);
    assert_eq!(
        eval.interpret(Expr::BinaryApp {
            op: BinaryOp::Add,
            arg1: Box::new(Expr::Int(3)),
            arg2: Box::new(Expr::Int(4))
        }),
        Ok(Value::Int(7))
    );
}

/// The original Bolero harness written by Mike.
///
/// Using `bolero` (with `libfuzzer`), it can be run without issues, resulting
/// in a stack overflow error caught by the address sanitizer. The main problem
/// with the error output in that case is that it's confusing.
///
/// Using `kani`, it gets stuck unwinding (with or without the unwinding
/// annotation). Because of that, we wrote the `check_expr_kani` harness which
/// essentially does the same but allows us to use a bounded `any_expr(N)`
/// generator instead, where `N` represents the maximum depth of the expression
/// to be generated. Note: We could achieve the same result defining
/// `kani::any()` as `any_expr(N)` for some hard-coded `N` value.
#[test]
#[cfg_attr(kani, kani::proof)]
#[cfg_attr(kani, kani::unwind(5))]
pub fn check_expr() {
    let eval = Evaluator::new(
        Some(Value::Int(1)),
        Some(Value::Int(1)),
        Some(Value::Int(1)),
    );
    let tc = Typechecker::new(Type::IntType, Type::IntType, Type::IntType);
    bolero::check!()
        .with_type::<Expr>()
        .cloned()
        .for_each(|expr| {
            assert!(true);
            if tc.typecheck(&expr).is_ok() {
                if eval.interpret(expr).is_err() {
                    assert!(false);
                }
            }
        });
}

#[cfg(kani)]
mod verification {
    use super::*;

    // Note: When writing the unbounded version of this generator, it will be
    // useful to have a map that keeps the expected types for each expression.
    // For example, `test_expr` in `Expr::If` must always get a boolean type.
    fn any_expr(depth: u32) -> Expr {
        if depth == 0 {
            match kani::any() {
                0 => Expr::Int(kani::any()),
                1 => Expr::Bool(kani::any()),
                _ => Expr::Var(kani::any()),
            }
        } else {
            match kani::any() {
                0 => Expr::Int(kani::any()),
                1 => Expr::Bool(kani::any()),
                2 => Expr::Var(kani::any()),
                3 => Expr::If {
                    test_expr: Box::new(any_expr(depth - 1)),
                    then_expr: Box::new(any_expr(depth - 1)),
                    else_expr: Box::new(any_expr(depth - 1)),
                },
                _ => Expr::BinaryApp {
                    op: kani::any(),
                    arg1: Box::new(any_expr(depth - 1)),
                    arg2: Box::new(any_expr(depth - 1)),
                },
            }
        }
    }

    // impl kani::Arbitrary for Expr {
    //     fn any() -> Self {
    //         const N: u32 = 5;
    //         any_expr(N)
    //     }
    // }

    /// The Kani harness equivalent to `check_expr`, uses `any_expr` with `depth = 0`.
    /// Running with
    ///  * default args: causes an out-of-memory error in CBMC.
    ///  * `--no-memory-safety-checks`: returns `SUCCESSFUL` after ~77s.
    #[kani::proof]
    #[kani::unwind(2)]
    #[kani::solver(cadical)]
    fn check_expr_depth_0() {
        let eval = Evaluator::new(
            Some(Value::Int(1)),
            Some(Value::Int(1)),
            Some(Value::Int(1)),
        );
        let tc = Typechecker::new(Type::IntType, Type::IntType, Type::IntType);
        let expr: Expr = any_expr(0);
        if tc.typecheck(&expr).is_ok() {
            if eval.interpret(expr).is_err() {
                assert!(false);
            }
        }
    }

    /// The Kani harness equivalent to `check_expr`, uses `any_expr` with `depth = 1`.
    /// Running with
    ///  * default args: causes an out-of-memory error in CBMC.
    ///  * `--no-memory-safety-checks`: returns `SUCCESSFUL` after ~82s.
    #[kani::proof]
    #[kani::unwind(2)]
    #[kani::solver(cadical)]
    fn check_expr_depth_1() {
        let eval = Evaluator::new(
            Some(Value::Int(1)),
            Some(Value::Int(1)),
            Some(Value::Int(1)),
        );
        let tc = Typechecker::new(Type::IntType, Type::IntType, Type::IntType);
        let expr: Expr = any_expr(1);
        if tc.typecheck(&expr).is_ok() {
            if eval.interpret(expr).is_err() {
                assert!(false);
            }
        }
    }

    /// The Kani harness equivalent to `check_expr`, uses `any_expr` with `depth = 2`.
    /// Running with
    ///  * default args: causes an out-of-memory error in CBMC.
    ///  * `--no-memory-safety-checks`: returns `SUCCESSFUL` after ~330s.
    #[kani::proof]
    #[kani::unwind(2)]
    #[kani::solver(cadical)]
    fn check_expr_depth_2() {
        let eval = Evaluator::new(
            Some(Value::Int(1)),
            Some(Value::Int(1)),
            Some(Value::Int(1)),
        );
        let tc = Typechecker::new(Type::IntType, Type::IntType, Type::IntType);
        let expr: Expr = any_expr(2);
        if tc.typecheck(&expr).is_ok() {
            if eval.interpret(expr).is_err() {
                assert!(false);
            }
        }
    }

    /// The Kani harness that mixes the harness `check_expr` and the test `test_eval`.
    /// Note: This harness only uses concrete values. Running with
    ///  * default args: causes an out-of-memory error in CBMC.
    ///  * `--no-memory-safety-checks`: returns `SUCCESSFUL` after ~63s.
    #[kani::proof]
    #[kani::unwind(2)]
    fn check_expr_concrete() {
        let eval = Evaluator::new(
            Some(Value::Int(1)),
            Some(Value::Int(1)),
            Some(Value::Int(1)),
        );
        let tc = Typechecker::new(Type::IntType, Type::IntType, Type::IntType);
        let expr = Expr::BinaryApp {
            op: BinaryOp::Add,
            arg1: Box::new(Expr::Int(3)),
            arg2: Box::new(Expr::Int(4)),
        };
        if tc.typecheck(&expr).is_ok() {
            if eval.interpret(expr).is_err() {
                assert!(false);
            }
        }
    }

    /// The Kani harness that is the same as `test_eval`.
    /// Running with
    ///  * default args: causes an out-of-memory error in CBMC.
    ///  * `--no-memory-safety-checks`: doesn't terminate within 30min.
    #[kani::proof]
    #[kani::unwind(2)]
    fn test_eval_kani() {
        let eval = Evaluator::new(Some(Value::Int(1)), None, None);
        assert_eq!(
            eval.interpret(Expr::BinaryApp {
                op: BinaryOp::Add,
                arg1: Box::new(Expr::Int(3)),
                arg2: Box::new(Expr::Int(4))
            }),
            Ok(Value::Int(7))
        );
    }
}
