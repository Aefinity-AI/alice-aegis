//! SAFE-7c fuzz target support: the pure, non-I/O parts of the `gateway
//! serve` request line protocol (see the big comment block above
//! `parse_serve_request` in `main.rs` for the wire format), split out so
//! a `cargo fuzz` target can drive them directly with arbitrary bytes
//! without needing a live `UnixStream`. `main.rs::parse_serve_request`
//! delegates to these exact functions/types (not a reimplementation), so
//! fuzzing this module fuzzes the real parser.

use std::path::PathBuf;

pub struct ServeRequest {
    pub receipt: PathBuf,
    pub action_hex: String,
    pub session: String,
    pub counter: u64,
    /// SAFE-11: the tool tag this request's eventual capability token (on
    /// ALLOW) will be bound to.
    pub tool: String,
}

/// A parsed request is either a normal decision request or a
/// `POLL ticket=<id>` request asking for that ticket's current/final
/// status.
pub enum ServeRequestKind {
    Decide(ServeRequest),
    Poll { ticket: String },
}

/// Parse the first line of a request as a `POLL ticket=<id>` line.
/// Returns `None` if the line isn't a well-formed, non-empty POLL line.
pub fn parse_poll_ticket(first: &str) -> Option<String> {
    let rest = first.strip_prefix("POLL ")?;
    let ticket = rest.trim().strip_prefix("ticket=")?.trim().to_string();
    if ticket.is_empty() {
        return None;
    }
    Some(ticket)
}

/// Blank-line/EOF terminator check shared by both request kinds.
pub fn is_blank_terminator(line: &str) -> bool {
    line.trim().is_empty()
}

/// Accumulates `RECEIPT <path>` / `ACTION <hex>` / `SESSION <s>` /
/// `COUNTER <n>` / `TOOL <tag>` lines for a DECIDE request, exactly as
/// `parse_serve_request`'s local `apply` closure used to do inline.
#[derive(Default)]
pub struct DecideAccumulator {
    receipt: Option<PathBuf>,
    action_hex: String,
    session: String,
    counter: u64,
    // SAFE-11: no default that any real shim tag would match -- an
    // omitted TOOL line must fail closed.
    tool: String,
}

impl DecideAccumulator {
    pub fn new() -> Self {
        DecideAccumulator {
            receipt: None,
            action_hex: String::new(),
            session: "default".to_string(),
            counter: 0,
            tool: String::new(),
        }
    }

    /// Apply one raw request line. Lines that don't match a known
    /// `PREFIX <value>` form are silently ignored (matching the original
    /// `apply` closure's behavior), so this never fails on malformed
    /// input -- it just leaves fields at their defaults.
    pub fn apply(&mut self, line: &str) {
        if let Some(rest) = line.strip_prefix("RECEIPT ") {
            self.receipt = Some(PathBuf::from(rest.trim()));
        } else if let Some(rest) = line.strip_prefix("ACTION ") {
            self.action_hex = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("SESSION ") {
            self.session = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("COUNTER ") {
            self.counter = rest.trim().parse().unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("TOOL ") {
            self.tool = rest.trim().to_string();
        }
    }

    /// Finish accumulating: `None` if no `RECEIPT` line was ever seen
    /// (a request without a receipt path has nothing to verify against).
    pub fn finish(self) -> Option<ServeRequest> {
        Some(ServeRequest {
            receipt: self.receipt?,
            action_hex: self.action_hex,
            session: self.session,
            counter: self.counter,
            tool: self.tool,
        })
    }
}

/// Parse a complete, already-collected request (as an ordered list of
/// lines with the terminating blank line/EOF already stripped off) into
/// a `ServeRequestKind`. This is the pure core of `parse_serve_request`,
/// used directly by the `fuzz_request_parser` target; `main.rs` builds
/// the same `Vec<String>` lazily from a live socket and calls this.
pub fn parse_request_lines(lines: &[String]) -> Option<ServeRequestKind> {
    let first = lines.first()?;
    if let Some(ticket) = parse_poll_ticket(first) {
        return Some(ServeRequestKind::Poll { ticket });
    }
    let mut acc = DecideAccumulator::new();
    for line in lines {
        if !line.trim().is_empty() {
            acc.apply(line);
        }
    }
    Some(ServeRequestKind::Decide(acc.finish()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poll_line_parses() {
        assert_eq!(
            parse_poll_ticket("POLL ticket=abc123"),
            Some("abc123".to_string())
        );
    }

    #[test]
    fn poll_line_empty_ticket_rejected() {
        assert_eq!(parse_poll_ticket("POLL ticket="), None);
    }

    #[test]
    fn poll_line_missing_prefix_rejected() {
        assert_eq!(parse_poll_ticket("ticket=abc123"), None);
    }

    #[test]
    fn decide_without_receipt_is_none() {
        let lines = vec!["ACTION deadbeef".to_string(), "TOOL shell".to_string()];
        assert!(parse_request_lines(&lines).is_none());
    }

    #[test]
    fn decide_with_receipt_parses() {
        let lines = vec![
            "RECEIPT /tmp/r.json".to_string(),
            "ACTION deadbeef".to_string(),
            "SESSION s1".to_string(),
            "COUNTER 3".to_string(),
            "TOOL shell".to_string(),
        ];
        match parse_request_lines(&lines) {
            Some(ServeRequestKind::Decide(req)) => {
                assert_eq!(req.receipt, PathBuf::from("/tmp/r.json"));
                assert_eq!(req.action_hex, "deadbeef");
                assert_eq!(req.session, "s1");
                assert_eq!(req.counter, 3);
                assert_eq!(req.tool, "shell");
            }
            other => panic!("expected Decide, got {other:?}"),
        }
    }

    #[test]
    fn malformed_counter_defaults_to_zero() {
        let lines = vec![
            "RECEIPT /tmp/r.json".to_string(),
            "COUNTER not-a-number".to_string(),
        ];
        match parse_request_lines(&lines) {
            Some(ServeRequestKind::Decide(req)) => assert_eq!(req.counter, 0),
            other => panic!("expected Decide, got {other:?}"),
        }
    }
}

impl std::fmt::Debug for ServeRequestKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServeRequestKind::Decide(_) => write!(f, "Decide(..)"),
            ServeRequestKind::Poll { ticket } => write!(f, "Poll{{ticket={ticket:?}}}"),
        }
    }
}
