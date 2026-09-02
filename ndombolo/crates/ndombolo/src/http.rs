//! A minimal HTTP/1.1 server and client, enough for one local document.
//!
//! Hand-written rather than pulled from a framework because the whole surface
//! is a handful of routes on loopback for a single user. A framework would add
//! an async runtime and a dependency tree to a program whose editor is one page
//! and whose concurrency is one browser tab.
//!
//! # Loopback only, and not negotiable
//!
//! [`serve`] binds `127.0.0.1` explicitly, never `0.0.0.0`. This server reads
//! and writes a file on disk and forwards prompts to a local model; there is no
//! authentication and none is planned, so reachability from another host would
//! be a file-read primitive for anyone on the network. The bind address is not
//! configurable for that reason.
//!
//! # What it deliberately does not do
//!
//! No keep-alive, no chunked request bodies, no TLS, no compression. Each
//! connection serves one request and closes. `Content-Length` is required on
//! any request with a body -- the browser's `fetch` always sends it.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

/// A parsed request: enough of one to route on.
///
/// No query string. Every route here either takes no argument or takes a JSON
/// body, so a request is `(method, path, body)` and a parameter parser would be
/// a second way to say the same thing.
pub struct Request {
    pub method: String,
    /// Path with any query string stripped.
    pub path: String,
    pub body: Vec<u8>,
}

/// A response to write back.
pub struct Response {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

impl Response {
    pub fn json(value: &serde_json::Value) -> Response {
        Response {
            status: 200,
            content_type: "application/json; charset=utf-8",
            body: serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec()),
        }
    }

    pub fn html(text: &str) -> Response {
        Response {
            status: 200,
            content_type: "text/html; charset=utf-8",
            body: text.as_bytes().to_vec(),
        }
    }

    /// An error as JSON, so the page can display it rather than guess from a
    /// status code.
    pub fn error(status: u16, message: &str) -> Response {
        Response {
            status,
            content_type: "application/json; charset=utf-8",
            body: serde_json::to_vec(&serde_json::json!({ "error": message }))
                .unwrap_or_else(|_| b"{}".to_vec()),
        }
    }
}

/// Serve on loopback until the process is killed.
///
/// Single-threaded on purpose: the handler mutates one document on disk, and
/// serialising requests is how two overlapping runs are prevented without a
/// lock. The cost is that a long model call blocks the page, which is why the
/// editor sends those to a separate endpoint it can wait on.
pub fn serve<F>(port: u16, mut handler: F) -> std::io::Result<()>
where
    F: FnMut(&Request) -> Response,
{
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    println!("ndombolo: http://127.0.0.1:{port}/");
    println!("(ctrl-c to stop)");

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            // One dropped connection is not a reason to stop serving.
            Err(_) => continue,
        };
        match read_request(&mut stream) {
            Ok(Some(req)) => {
                let res = handler(&req);
                let _ = write_response(&mut stream, &res);
            }
            Ok(None) => {}
            Err(_) => {
                let _ = write_response(&mut stream, &Response::error(400, "malformed request"));
            }
        }
    }
    Ok(())
}

fn read_request(stream: &mut TcpStream) -> std::io::Result<Option<Request>> {
    let mut reader = BufReader::new(stream.try_clone()?);

    let mut start = String::new();
    if reader.read_line(&mut start)? == 0 {
        return Ok(None);
    }
    let mut parts = start.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("/").to_string();

    let mut length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                length = v.trim().parse().unwrap_or(0);
            }
        }
    }

    let mut body = vec![0u8; length];
    if length > 0 {
        reader.read_exact(&mut body)?;
    }

    let path = match target.split_once('?') {
        Some((p, _)) => p.to_string(),
        None => target,
    };

    Ok(Some(Request {
        method,
        path: percent_decode(&path),
        body,
    }))
}

fn write_response(stream: &mut TcpStream, res: &Response) -> std::io::Result<()> {
    let reason = match res.status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "OK",
    };
    write!(
        stream,
        "HTTP/1.1 {} {reason}\r\n\
         Content-Type: {}\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-store\r\n\
         Connection: close\r\n\r\n",
        res.status,
        res.content_type,
        res.body.len()
    )?;
    stream.write_all(&res.body)?;
    stream.flush()
}

/// Decode `%XX` and `+`.
///
/// A malformed escape is left as written rather than dropped: this decodes
/// paths, and silently mangling one would turn a typo into a request for a
/// different file.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(b) => {
                        out.push(b);
                        i += 3;
                    }
                    Err(_) => {
                        out.push(bytes[i]);
                        i += 1;
                    }
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

// -- a client, for talking to ollama ---------------------------------------

/// POST JSON to a local URL and read the JSON back.
///
/// Same reasoning as the server: one endpoint on loopback does not justify an
/// HTTP client dependency. `url` is expected to be `http://host:port/path`.
pub fn post_json(
    url: &str,
    body: &serde_json::Value,
    timeout: std::time::Duration,
) -> Result<serde_json::Value, String> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| format!("only http:// is supported, got {url}"))?;
    let (authority, path) = match rest.split_once('/') {
        Some((a, p)) => (a, format!("/{p}")),
        None => (rest, "/".to_string()),
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (h, p.parse::<u16>().map_err(|_| "bad port".to_string())?),
        None => (authority, 80),
    };

    let payload = serde_json::to_vec(body).map_err(|e| e.to_string())?;
    let mut stream = TcpStream::connect((host, port))
        .map_err(|e| format!("cannot reach {host}:{port}: {e}"))?;
    stream.set_read_timeout(Some(timeout)).ok();
    stream.set_write_timeout(Some(timeout)).ok();

    write!(
        stream,
        "POST {path} HTTP/1.1\r\n\
         Host: {host}:{port}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n",
        payload.len()
    )
    .map_err(|e| e.to_string())?;
    stream.write_all(&payload).map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).map_err(|e| e.to_string())?;

    // Split headers from body on the blank line. Ollama answers with a plain
    // body and `Content-Length`, so there is no chunked case to handle here;
    // if that changes, the parse below fails loudly rather than truncating.
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| "no header terminator in response".to_string())?;
    let body = &raw[split + 4..];

    serde_json::from_slice(body).map_err(|e| {
        let head = String::from_utf8_lossy(&body[..body.len().min(200)]);
        format!("response was not json: {e} ({head})")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percent_decoding_handles_paths_and_bad_escapes() {
        assert_eq!(percent_decode("/a%20b.ndo"), "/a b.ndo");
        assert_eq!(percent_decode("a+b"), "a b");
        // A malformed escape survives rather than vanishing.
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%zz"), "%zz");
    }

    #[test]
    fn a_query_string_does_not_reach_the_path() {
        // Routing is on the path alone; `?t=1` from a cache-busting reload must
        // not turn `/api/doc` into a 404.
        let server = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = server.local_addr().unwrap().port();
        let t = std::thread::spawn(move || {
            let (mut s, _) = server.accept().unwrap();
            read_request(&mut s).unwrap().unwrap()
        });
        let mut c = TcpStream::connect(("127.0.0.1", port)).unwrap();
        write!(c, "GET /api/doc?t=1 HTTP/1.1{}{}", "\r\n", "\r\n").unwrap();
        c.flush().unwrap();
        let req = t.join().unwrap();
        assert_eq!(req.path, "/api/doc");
        assert_eq!(req.method, "GET");
    }
}
