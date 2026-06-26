use std::fs;
use std::path::{Path, PathBuf};

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

const OUT_DIR: &str = "dist";
const POSTS_DIR: &str = "posts";
const MAIN_PAGE: &str = "main.md";

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
        Some(other) => eprintln!("unknown command: {other}\nusage: blog [build | serve [port]]"),
    }
}

// --- generation ---------------------------------------------------------

struct Post {
    title: String,
    slug: String,
    body: String,
}

fn build() {
    let _ = fs::remove_dir_all(OUT_DIR);
    fs::create_dir_all(format!("{OUT_DIR}/{POSTS_DIR}")).unwrap();

    let mut posts: Vec<Post> = fs::read_dir(POSTS_DIR)
        .expect("posts/ directory not found")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .filter_map(read_post)
        .collect();
    posts.sort_by(|a, b| a.title.to_lowercase().cmp(&b.title.to_lowercase()));

    for post in &posts {
        let page = render_page(&post.title, &md_to_html(&post.body));
        fs::write(format!("{OUT_DIR}/{POSTS_DIR}/{}.html", post.slug), page).unwrap();
    }

    fs::write(format!("{OUT_DIR}/index.html"), render_index(&posts)).unwrap();
    fs::write(format!("{OUT_DIR}/style.css"), STYLE).unwrap();

    println!("built {} posts -> {OUT_DIR}/", posts.len());
}

fn read_post(path: PathBuf) -> Option<Post> {
    let body = fs::read_to_string(&path).ok()?;
    if body.trim().is_empty() {
        return None; // skip empty drafts so we don't ship dead links
    }
    let stem = path.file_stem()?.to_string_lossy().into_owned();
    let title = title_from_markdown(&body).unwrap_or_else(|| humanize(&stem));
    let slug = slugify(&title);
    Some(Post { title, slug, body })
}

/// The post title is its first level-1 heading (`# ...`); the filename is only a fallback.
fn title_from_markdown(src: &str) -> Option<String> {
    src.lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix("# ").map(|t| t.trim().to_string()))
}

fn render_index(posts: &[Post]) -> String {
    let mut list = String::from("<h2>Posts</h2>\n<ul class=\"posts\">\n");
    for p in posts {
        list.push_str(&format!(
            "  <li><a href=\"{POSTS_DIR}/{}.html\">{}</a></li>\n",
            p.slug,
            escape(&p.title)
        ));
    }
    list.push_str("</ul>\n");

    let main_md = fs::read_to_string(MAIN_PAGE).unwrap_or_default();
    // Markdown wraps a lone {{posts}} line in its own <p>; swap the whole paragraph
    // so the list isn't illegally nested inside one.
    let main_html = md_to_html(&main_md).replace("<p>{{posts}}</p>", "{{posts}}");
    let body = if main_html.contains("{{posts}}") {
        main_html.replace("{{posts}}", &list)
    } else {
        format!("{main_html}\n{list}")
    };
    render_page("Finn's Blog", &body)
}

// --- markdown + math ----------------------------------------------------

/// Convert markdown to HTML, keeping LaTeX math intact for client-side KaTeX.
fn md_to_html(src: &str) -> String {
    let (protected, math) = protect_math(src);

    use pulldown_cmark::{html, Options, Parser};
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_FOOTNOTES);
    let parser = Parser::new_ext(&protected, opts);
    let mut html_out = String::new();
    html::push_html(&mut html_out, parser);

    restore_math(html_out, &math)
}

/// Replace every `$...$` / `$$...$$` span with an inert placeholder token so the
/// markdown parser can't mangle the LaTeX (e.g. `_`, `\\`, `*` inside formulas).
fn protect_math(src: &str) -> (String, Vec<String>) {
    let chars: Vec<char> = src.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(src.len());
    let mut spans = Vec::new();
    let mut i = 0;

    while i < n {
        if chars[i] == '$' {
            let display = i + 1 < n && chars[i + 1] == '$';
            let delim = if display { 2 } else { 1 };
            let mut j = i + delim;
            let end = loop {
                if j >= n {
                    break n; // unterminated: swallow to end of input
                }
                if chars[j] == '$' && (!display || (j + 1 < n && chars[j + 1] == '$')) {
                    break j + delim;
                }
                j += 1;
            };
            spans.push(chars[i..end.min(n)].iter().collect());
            out.push_str(&format!("MATHSPAN{}END", spans.len() - 1));
            i = end;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    (out, spans)
}

fn restore_math(mut html: String, spans: &[String]) -> String {
    for (idx, raw) in spans.iter().enumerate() {
        html = html.replace(&format!("MATHSPAN{idx}END"), raw);
    }
    html
}

// --- string helpers -----------------------------------------------------

/// "basic nn math" / "forward_ad" -> "Basic Nn Math" / "Forward Ad"
fn humanize(stem: &str) -> String {
    stem.split(|c| c == '_' || c == '-' || c == ' ')
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// "Basic NN Math" -> "basic-nn-math"
fn slugify(stem: &str) -> String {
    let mut slug = String::new();
    let mut prev_dash = false;
    for c in stem.chars() {
        if c.is_alphanumeric() {
            slug.extend(c.to_lowercase());
            prev_dash = false;
        } else if !prev_dash {
            slug.push('-');
            prev_dash = true;
        }
    }
    slug.trim_matches('-').to_string()
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

// --- html template ------------------------------------------------------

fn render_page(title: &str, body: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="/style.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {{
          delimiters: [
            {{left: '$$', right: '$$', display: true}},
            {{left: '$', right: '$', display: false}}
          ]
        }});"></script>
</head>
<body>
<main>
<p class="nav"><a href="/">&larr; home</a></p>
{body}
</main>
</body>
</html>
"#,
        title = escape(title),
        body = body
    )
}

const STYLE: &str = r#"
:root { --fg: #1a1a1a; --muted: #666; --link: #2563eb; --bg: #fdfdfc; }
* { box-sizing: border-box; }
body {
  font-family: Georgia, 'Times New Roman', serif;
  line-height: 1.65; color: var(--fg); background: var(--bg);
  margin: 0; padding: 0;
}
main { max-width: 42rem; margin: 0 auto; padding: 3rem 1.25rem 6rem; }
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
h1, h2, h3 { line-height: 1.25; font-weight: 700; }
h1 { margin-top: 0; }
.nav a { color: var(--muted); font-size: 0.9rem; }
ul.posts { list-style: none; padding: 0; }
ul.posts li { margin: 0.4rem 0; font-size: 1.05rem; }
code {
  font-family: 'SF Mono', Menlo, monospace; font-size: 0.9em;
  background: #f0f0ee; padding: 0.1em 0.3em; border-radius: 3px;
}
pre {
  background: #f0f0ee; padding: 1rem; border-radius: 6px; overflow-x: auto;
}
pre code { background: none; padding: 0; }
blockquote {
  border-left: 3px solid #ddd; margin-left: 0; padding-left: 1rem; color: var(--muted);
}
.katex-display { overflow-x: auto; overflow-y: hidden; padding: 0.25rem 0; }
"#;

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
    let clean = path.split('?').next().unwrap_or("/").trim_start_matches('/');
    if clean.contains("..") {
        return ("403 Forbidden", "text/plain", b"forbidden".to_vec());
    }
    let mut file = PathBuf::from(OUT_DIR);
    file.push(if clean.is_empty() { "index.html" } else { clean });
    if tokio::fs::metadata(&file).await.map(|m| m.is_dir()).unwrap_or(false) {
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
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        _ => "application/octet-stream",
    }
}
