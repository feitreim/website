# loadtest

A tiny stress tester for the website server. Raw HTTP/1.1 over `TcpStream`, no
dependencies beyond `tokio` — same spirit as the server it hammers.

```sh
cargo build --release
./target/release/stress <target> [options]
```

## Hit the VPS origin, not Cloudflare

Point it at the **origin IP and port directly** (`203.0.113.7:8000`, not
`https://yourdomain`). That bypasses Cloudflare's cache, proxy, and rate
limiting so you measure *your* server instead of the edge. Plain HTTP only —
no TLS, because the origin behind Cloudflare speaks plain HTTP anyway.

If the origin vhosts by Host header, keep the real domain with `-H`:

```sh
./target/release/stress 203.0.113.7:80 -H yourdomain.com -c 500 -d 60
```

## Options

| flag                 | default | meaning                          |
|----------------------|---------|----------------------------------|
| `-c, --connections`  | 50      | concurrent connections           |
| `-d, --duration`     | 10      | seconds to run                   |
| `-t, --timeout`      | 5       | per-request timeout (seconds)    |
| `-p, --path`         | `/`     | request path (repeat to mix)     |
| `-H, --host`         | target  | Host header override             |

```sh
./target/release/stress 203.0.113.7:8000 -c 200 -d 30 \
  -p / -p /posts/spherical-flow-models.html -p /BerkeleyMono-Regular.woff2
```

## Note on the connection model

The server always replies `Connection: close`, so keep-alive is impossible —
every request is its own TCP connection, and the tester matches that. This
makes `-c` "connections in flight per moment," and the run mostly exercises the
accept loop + TCP handshake, which is the realistic bottleneck for this server.
