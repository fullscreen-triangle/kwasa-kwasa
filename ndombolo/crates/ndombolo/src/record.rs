//! The record: a never-resetting counter and the deposits behind it.
//!
//! The document shows the last run. The record remembers every run. Keeping
//! them separate is what lets the visible output block be *replaced* on a
//! re-run -- one block, not a growing transcript -- without that replacement
//! being a rewind (paper, B2: the record never resets, and "undo" increments
//! it).
//!
//! # Every run deposits
//!
//! Including runs whose output is thrown away (paper, B4). A cell run and then
//! cleared, a cell run whose output the user deleted, a cell that failed: each
//! deposited. This is why the record is not derivable from the document, and
//! why it is stored beside it rather than inside it.
//!
//! # Not a cache
//!
//! The record is append-only history, never consulted to answer a question
//! (paper, B3: determination by search, no cache of prior determinations).
//! Nothing in this module has a lookup that would let a repeated question skip
//! a run -- and a `get`-shaped method here would be the violation, so there
//! isn't one.
//!
//! # On-disk form
//!
//! JSON Lines beside the document: `foo.ndo` keeps `foo.ndo.record`. One
//! object per line, appended, never rewritten. A line-oriented format is the
//! point: appending cannot corrupt what is already there, and a truncated
//! final line costs one deposit rather than the file.

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde_json::{json, Value as Json};

/// One committing act.
#[derive(Debug, Clone)]
pub struct Deposit {
    /// The value of the counter *after* this deposit. Strictly increasing.
    pub record: u64,
    /// Which cell, by position among cells.
    pub cell: usize,
    /// How many trace events the run appended.
    pub events: usize,
    /// The names the run added or updated.
    pub items_touched: Vec<String>,
    /// Whether the run completed without an error. Not a verdict on the
    /// document (paper, B6 forbids success vocabulary in the runtime) -- it
    /// records whether the compiler reached the end, which the trace shows
    /// anyway; it is stored so the record is readable without replaying.
    pub completed: bool,
}

/// The append-only record for one document.
pub struct Record {
    path: PathBuf,
    count: u64,
}

impl Record {
    /// Open the record beside `doc`, counting what is already there.
    ///
    /// A missing file starts at zero; an existing one resumes at its length.
    /// There is deliberately no way to open it at any other number.
    pub fn open(doc: &Path) -> std::io::Result<Record> {
        let mut path = doc.as_os_str().to_owned();
        path.push(".record");
        let path = PathBuf::from(path);

        let count = match std::fs::File::open(&path) {
            Ok(f) => BufReader::new(f)
                .lines()
                .map_while(Result::ok)
                .filter(|l| !l.trim().is_empty())
                .count() as u64,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
            Err(e) => return Err(e),
        };

        Ok(Record { path, count })
    }

    /// The current record. Read-only: there is no setter.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Commit one act, advancing the record by one.
    ///
    /// The only mutator in this module, and it only goes up.
    pub fn deposit(
        &mut self,
        cell: usize,
        events: usize,
        items_touched: Vec<String>,
        completed: bool,
    ) -> std::io::Result<Deposit> {
        let before = self.count;
        self.count += 1;

        let d = Deposit {
            record: self.count,
            cell,
            events,
            items_touched,
            completed,
        };

        let line = json!({
            "record": d.record,
            "record_before": before,
            "cell": d.cell,
            "events": d.events,
            "items_touched": d.items_touched,
            "completed": d.completed,
            "at": now(),
        });

        // Append, never rewrite: an open-truncate here would be the rewind the
        // whole design forbids.
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(f, "{line}")?;

        Ok(d)
    }

    /// Every deposit, in order, for `ndombolo record`.
    pub fn entries(&self) -> std::io::Result<Vec<Json>> {
        match std::fs::File::open(&self.path) {
            Ok(f) => Ok(BufReader::new(f)
                .lines()
                .map_while(Result::ok)
                .filter(|l| !l.trim().is_empty())
                .filter_map(|l| serde_json::from_str(&l).ok())
                .collect()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(e),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Seconds since the epoch, as a number.
///
/// A timestamp is not what individuates a run -- the record is (paper,
/// Thm. "the pair individuates where the compiler cannot"). It is here so a
/// human reading the file can orient, and nothing reads it back.
fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("ndombolo-record-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_file(&p);
        let mut r = p.as_os_str().to_owned();
        r.push(".record");
        let _ = std::fs::remove_file(PathBuf::from(r));
        p
    }

    #[test]
    fn the_record_only_increases() {
        let doc = tmp("increase");
        let mut r = Record::open(&doc).unwrap();
        assert_eq!(r.count(), 0);

        for expect in 1..=3 {
            let d = r.deposit(0, 1, vec!["a".into()], true).unwrap();
            assert_eq!(d.record, expect);
            assert_eq!(r.count(), expect);
        }

        // Reopening resumes; it does not reset. This is B2.
        let r2 = Record::open(&doc).unwrap();
        assert_eq!(r2.count(), 3);

        let _ = std::fs::remove_file(r.path());
    }

    #[test]
    fn a_failed_run_deposits_too() {
        // B4: every propagation deposits, including those whose output is
        // discarded. A cell that errored still moved the record.
        let doc = tmp("failed");
        let mut r = Record::open(&doc).unwrap();
        r.deposit(0, 2, vec![], false).unwrap();
        assert_eq!(r.count(), 1);
        let entries = r.entries().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["completed"], json!(false));
        let _ = std::fs::remove_file(r.path());
    }
}
