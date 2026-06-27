use std::path::{Path, PathBuf};

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

mod site_generator;
use site_generator::{build, OUT_DIR};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("serve") => {
            let port = args.get(2).and_then(|p| p.parse().ok()).unwrap_or(8000);
            build();
            serve(port).await;
        }
        Some("build") | None => build(),
        Some(other) => eprintln!("unknown command: {other}\nusage: website [build | serve [port]]"),
    }
}

// --- web server ---------------------------------------------------------

async fn serve(port: u16) {
    let addr = format!("0.0.0.0:{port}");
    let listener = TcpListener::bind(&addr).await.expect("failed to bind");
    println!("serving {OUT_DIR}/ on http://{addr}");
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                tokio::spawn(handle(stream));
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
}

async fn handle(mut stream: TcpStream) {
    let (read_half, mut write_half) = stream.split();
    let mut reader = BufReader::new(read_half);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).await.is_err() {
        return;
    }
    let path = request_line.split_whitespace().nth(1).unwrap_or("/");
    let (status, ctype, body) = resolve(path).await;
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = write_half.write_all(header.as_bytes()).await;
    let _ = write_half.write_all(&body).await;
}

async fn resolve(path: &str) -> (&'static str, &'static str, Vec<u8>) {
    let clean = path
        .split('?')
        .next()
        .unwrap_or("/")
        .trim_start_matches('/');
    if clean.contains("..") {
        return ("403 Forbidden", "text/plain", b"forbidden".to_vec());
    }
    let mut file = PathBuf::from(OUT_DIR);
    file.push(if clean.is_empty() {
        "index.html"
    } else {
        clean
    });
    if tokio::fs::metadata(&file)
        .await
        .map(|m| m.is_dir())
        .unwrap_or(false)
    {
        file.push("index.html");
    }
    match tokio::fs::read(&file).await {
        Ok(bytes) => ("200 OK", content_type(&file), bytes),
        Err(_) => (
            "404 Not Found",
            "text/html; charset=utf-8",
            b"<h1>404</h1><p><a href=\"/\">home</a></p>".to_vec(),
        ),
    }
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("woff2") => "font/woff2",
        Some("woff") => "font/woff",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        _ => "application/octet-stream",
    }
}
