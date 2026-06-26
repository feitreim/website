# bearblog

A bare-bones static site generator + web server in one Rust binary.

- `posts/*.md` → `dist/posts/*.html`
- `main.md` → `dist/index.html` (the `{{posts}}` marker is replaced by the auto-generated post list)
- LaTeX math (`$...$`, `$$...$$`) is rendered in the browser by KaTeX (loaded from a CDN)
- Empty `.md` files are skipped

## Usage

```sh
cargo build --release

./target/release/blog build        # generate dist/
./target/release/blog serve 8000   # build, then serve dist/ on 0.0.0.0:8000
```

Writing a new post = drop a `.md` file in `posts/` and re-run `build`. The title
comes from the filename; the URL slug is the lowercased, hyphenated filename.

## Deploying to a VPS

```sh
# on the VPS, with the repo checked out and Rust installed:
cargo build --release
./target/release/blog serve 80      # needs root/cap for port 80
```

To keep it running, drop a systemd unit at `/etc/systemd/system/bearblog.service`:

```ini
[Unit]
Description=bearblog
After=network.target

[Service]
WorkingDirectory=/path/to/blog
ExecStart=/path/to/blog/target/release/blog serve 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Then `systemctl enable --now bearblog`. Put nginx/caddy in front for TLS, or
point it at port 8000 directly.
