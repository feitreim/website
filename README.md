# personal website

A bare-bones static site generator + web server in one.

- `posts/*.md` → `dist/posts/*.html`
- `main.md` → `dist/index.html` (the `{{posts}}` marker is replaced by the auto-generated post list)
- LaTeX math (`$...$`, `$$...$$`) is rendered in the browser by KaTeX (loaded from a CDN)
- Empty `.md` files are skipped

## Usage

```sh
cargo build --release

./target/release/website build        # generate dist/
./target/release/website serve 8000   # build, then serve dist/ on 0.0.0.0:8000
```

Writing a new post = drop a `.md` file in `posts/` and re-run `build`. The title
comes from the post's first `# ` heading; the URL slug is that title, lowercased
and hyphenated. (If a post has no `# ` heading, the filename is used instead.)

## Deploying to a VPS

```sh
# on the VPS, with the repo checked out and Rust installed:
cargo build --release
./target/release/website serve 80      # needs root/cap for port 80
```

To keep it running, drop a systemd unit at `/etc/systemd/system/website.service`:

```ini
[Unit]
Description=website
After=network.target

[Service]
WorkingDirectory=/path/to/blog
ExecStart=/path/to/blog/target/release/website serve 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Then `systemctl enable --now website`. Put nginx/caddy in front for TLS, or
point it at port 8000 directly.

### Redeploying

The systemd unit's `ExecStart` rebuilds from the latest checkout on every start,
so after pushing changes just restart the service on the VPS:

```sh
ssh <your-vps> 'systemctl restart website'
```

its aliased as `update-website`
