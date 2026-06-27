use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::{sleep, timeout};

// Stress tester for the website's bare TcpStream HTTP server.
//
// Point it straight at the VPS origin (its raw IP:port), NOT at the public
// domain — that bypasses Cloudflare's cache/proxy so you measure your own
// server rather than the edge. The server forces `Connection: close`, so every
// request is its own TCP connection; we match that here.

struct Config {
    addr: String,
    host: String,
    paths: Vec<String>,
    connections: usize,
    duration: Duration,
    timeout: Duration,
}

#[derive(Default)]
struct Stats {
    latencies: Vec<u64>, // microseconds, successful responses only
    bytes: u64,
    errors: u64,
    statuses: HashMap<u16, u64>,
}

impl Stats {
    fn merge(&mut self, other: Stats) {
        self.latencies.extend(other.latencies);
        self.bytes += other.bytes;
        self.errors += other.errors;
        for (code, n) in other.statuses {
            *self.statuses.entry(code).or_insert(0) += n;
        }
    }
}

#[tokio::main]
async fn main() {
    let cfg = match parse_args() {
        Ok(cfg) => Arc::new(cfg),
        Err(msg) => {
            eprintln!("{msg}\n");
            usage();
            std::process::exit(1);
        }
    };

    println!("target      {}  (Host: {})", cfg.addr, cfg.host);
    println!("connections {}", cfg.connections);
    println!("duration    {}s", cfg.duration.as_secs());
    println!("paths       {}", cfg.paths.join(", "));
    println!("\nwarming up the load...\n");

    let stop = Arc::new(AtomicBool::new(false));
    let counter = Arc::new(AtomicU64::new(0));

    let reporter = tokio::spawn(report(stop.clone(), counter.clone()));

    let started = Instant::now();
    let workers: Vec<_> = (0..cfg.connections)
        .map(|_| tokio::spawn(worker(cfg.clone(), stop.clone(), counter.clone())))
        .collect();

    sleep(cfg.duration).await;
    stop.store(true, Ordering::Relaxed);

    let mut total = Stats::default();
    for w in workers {
        if let Ok(s) = w.await {
            total.merge(s);
        }
    }
    let _ = reporter.await;

    summarize(&mut total, started.elapsed());
}

async fn worker(cfg: Arc<Config>, stop: Arc<AtomicBool>, counter: Arc<AtomicU64>) -> Stats {
    let mut stats = Stats::default();
    let mut i = 0usize;
    while !stop.load(Ordering::Relaxed) {
        let path = &cfg.paths[i % cfg.paths.len()];
        i += 1;
        let start = Instant::now();
        match timeout(cfg.timeout, request(&cfg.addr, &cfg.host, path)).await {
            Ok(Ok((status, n))) => {
                stats.latencies.push(start.elapsed().as_micros() as u64);
                stats.bytes += n as u64;
                *stats.statuses.entry(status).or_insert(0) += 1;
            }
            _ => stats.errors += 1, // connect/read error or timed out
        }
        counter.fetch_add(1, Ordering::Relaxed);
    }
    stats
}

// One request over a fresh connection, returning (status_code, body+header bytes).
async fn request(addr: &str, host: &str, path: &str) -> std::io::Result<(u16, usize)> {
    let mut stream = TcpStream::connect(addr).await?;
    stream.set_nodelay(true).ok();
    let req = format!("GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n");
    stream.write_all(req.as_bytes()).await?;
    let mut buf = Vec::with_capacity(16 * 1024);
    stream.read_to_end(&mut buf).await?; // server closes after the body
    Ok((parse_status(&buf), buf.len()))
}

fn parse_status(buf: &[u8]) -> u16 {
    let line = buf.split(|&b| b == b'\n').next().unwrap_or(&[]);
    std::str::from_utf8(line)
        .unwrap_or("")
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or(0)
}

// Live throughput, printed once a second.
async fn report(stop: Arc<AtomicBool>, counter: Arc<AtomicU64>) {
    let mut last = 0u64;
    while !stop.load(Ordering::Relaxed) {
        sleep(Duration::from_secs(1)).await;
        let now = counter.load(Ordering::Relaxed);
        eprintln!("  {:>9} req/s   ({} total)", now - last, now);
        last = now;
    }
}

fn summarize(total: &mut Stats, elapsed: Duration) {
    total.latencies.sort_unstable();
    let ok = total.latencies.len() as u64;
    let secs = elapsed.as_secs_f64();
    let done = ok + total.errors;

    println!("\n────────────────────────────────────────");
    println!("requests      {done}  ({ok} ok, {} failed)", total.errors);
    println!("throughput    {:.0} req/s", done as f64 / secs);
    println!(
        "transferred   {:.1} MB  ({:.1} MB/s)",
        total.bytes as f64 / 1e6,
        total.bytes as f64 / 1e6 / secs
    );

    if !total.statuses.is_empty() {
        let mut codes: Vec<_> = total.statuses.iter().collect();
        codes.sort();
        let line: Vec<String> = codes.iter().map(|(c, n)| format!("{c}: {n}")).collect();
        println!("status        {}", line.join("   "));
    }

    if ok > 0 {
        let sum: u64 = total.latencies.iter().sum();
        println!("\nlatency (ms)");
        println!("  avg  {:.2}", sum as f64 / ok as f64 / 1000.0);
        println!("  p50  {:.2}", pct(&total.latencies, 50.0));
        println!("  p90  {:.2}", pct(&total.latencies, 90.0));
        println!("  p99  {:.2}", pct(&total.latencies, 99.0));
        println!("  max  {:.2}", pct(&total.latencies, 100.0));
    }
    println!("────────────────────────────────────────");
}

fn pct(sorted_micros: &[u64], p: f64) -> f64 {
    if sorted_micros.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted_micros.len() - 1) as f64).round() as usize;
    sorted_micros[idx] as f64 / 1000.0
}

fn parse_args() -> Result<Config, String> {
    let mut args = std::env::args().skip(1);
    let target = args.next().ok_or("missing <target>")?;
    if target == "-h" || target == "--help" {
        usage();
        std::process::exit(0);
    }

    let (addr, default_host) = normalize_target(&target)?;
    let mut cfg = Config {
        host: default_host,
        addr,
        paths: Vec::new(),
        connections: 50,
        duration: Duration::from_secs(10),
        timeout: Duration::from_secs(5),
    };

    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "-c" | "--connections" => cfg.connections = value()?.parse().map_err(|_| "bad -c")?,
            "-d" | "--duration" => {
                cfg.duration = Duration::from_secs(value()?.parse().map_err(|_| "bad -d")?)
            }
            "-t" | "--timeout" => {
                cfg.timeout = Duration::from_secs(value()?.parse().map_err(|_| "bad -t")?)
            }
            "-p" | "--path" => cfg.paths.push(value()?),
            "-H" | "--host" => cfg.host = value()?,
            other => return Err(format!("unknown flag: {other}")),
        }
    }

    if cfg.paths.is_empty() {
        cfg.paths.push("/".to_string());
    }
    Ok(cfg)
}

// Accepts `host`, `host:port`, or `http://host[:port]`. Returns the
// connect address (host:port) and the default Host header value (the bare host).
fn normalize_target(target: &str) -> Result<(String, String), String> {
    let stripped = target
        .strip_prefix("http://")
        .or_else(|| target.strip_prefix("https://"))
        .unwrap_or(target)
        .trim_end_matches('/');
    if stripped.is_empty() {
        return Err("empty target".to_string());
    }
    let host = stripped.split(':').next().unwrap_or(stripped).to_string();
    let addr = if stripped.contains(':') {
        stripped.to_string()
    } else {
        format!("{stripped}:80")
    };
    Ok((addr, host))
}

fn usage() {
    eprintln!(
        "stress — load test the website server (hit the VPS origin directly, not Cloudflare)

usage:
  stress <target> [options]

  <target>   host | host:port | http://host[:port]   (defaults to port 80)

options:
  -c, --connections N   concurrent connections     (default 50)
  -d, --duration N      seconds to run             (default 10)
  -t, --timeout N       per-request timeout, secs  (default 5)
  -p, --path P          request path (repeatable)  (default /)
  -H, --host HDR        Host header override

examples:
  stress 203.0.113.7:8000 -c 200 -d 30
  stress 203.0.113.7:8000 -p / -p /posts/spherical-flow.html -p /BerkeleyMono-Regular.woff2
  stress 203.0.113.7:80 -H finn.example.com -c 500 -d 60"
    );
}
