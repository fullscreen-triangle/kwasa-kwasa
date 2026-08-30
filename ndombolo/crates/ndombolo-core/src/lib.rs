//! `ndombolo-core` — the deterministic Turbulance runtime.
//!
//! The Python under `prototype/` is a reference implementation, not a
//! performance claim (paper, §8), and it is frozen: it stays off the execution
//! path and serves only as the oracle the differential harness compares
//! against. This crate is the system.
//!
//! Port order is lexer → parser → evaluator → graph, each stage checked
//! against the frozen corpus before the next begins.

pub mod ast;
pub mod lexer;
pub mod eval;
pub mod parser;
pub mod session;
pub mod value;

pub use ast::{Entry, Node};
pub use lexer::{tokenize, Kind, LexError, Token};
pub use eval::{Compiler, RuntimeErr};
pub use parser::{parse, ParseError};
pub use session::{split_cells, CellResult, Session, StoreChange};
pub use value::{render, text, Value};
