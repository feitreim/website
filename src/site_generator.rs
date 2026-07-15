use std::cmp::Ordering;
use std::fs;
use std::path::PathBuf;

use ratex_layout::LayoutOptions;
use ratex_svg::SvgOptions;
use ratex_types::{color::Color, math_style::MathStyle};

pub const OUT_DIR: &str = "dist";
const POSTS_DIR: &str = "posts";
const MAIN_PAGE: &str = "main.md";
const STATIC_DIR: &str = "static";

/// Math glyph color, kept in sync with `--fg` in STYLE.
const MATH_COLOR: &str = "#2b3034";

type DateKey = (u16, u8, u8);

struct Post {
    title: String,
    slug: String,
    body: String,
    date: Option<String>,
    date_key: Option<DateKey>,
}

pub fn build() {
    let _ = fs::remove_dir_all(OUT_DIR);
    fs::create_dir_all(format!("{OUT_DIR}/{POSTS_DIR}")).unwrap();

    let mut posts: Vec<Post> = fs::read_dir(POSTS_DIR)
        .expect("posts/ directory not found")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .filter_map(read_post)
        .collect();
    posts.sort_by(compare_posts);

    for post in &posts {
        let page = render_page(&post.title, &md_to_html(&post.body), true);
        fs::write(format!("{OUT_DIR}/{POSTS_DIR}/{}.html", post.slug), page).unwrap();
    }

    fs::write(format!("{OUT_DIR}/index.html"), render_index(&posts)).unwrap();
    fs::write(format!("{OUT_DIR}/style.css"), STYLE).unwrap();
    fs::write(format!("{OUT_DIR}/favicon.svg"), FAVICON).unwrap();
    copy_static();

    println!("built {} posts -> {OUT_DIR}/", posts.len());
}

/// Copy verbatim assets (fonts, etc.) from static/ into the output dir.
fn copy_static() {
    let Ok(entries) = fs::read_dir(STATIC_DIR) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_file() {
            let dest = format!("{OUT_DIR}/{}", entry.file_name().to_string_lossy());
            fs::copy(&path, dest).unwrap();
        }
    }
}

fn read_post(path: PathBuf) -> Option<Post> {
    let body = fs::read_to_string(&path).ok()?;
    if body.trim().is_empty() {
        return None; // skip empty drafts so we don't ship dead links
    }
    let stem = path.file_stem()?.to_string_lossy().into_owned();
    let title = title_from_markdown(&body).unwrap_or_else(|| humanize(&stem));
    let slug = slugify(&title);
    let date = date_from_markdown(&body);
    let date_key = date.as_deref().and_then(parse_sortable_date);
    Some(Post {
        title,
        slug,
        body,
        date,
        date_key,
    })
}

fn compare_posts(a: &Post, b: &Post) -> Ordering {
    match (a.date_key, b.date_key) {
        (Some(a_date), Some(b_date)) => b_date.cmp(&a_date).then_with(|| compare_titles(a, b)),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => compare_titles(a, b),
    }
}

fn compare_titles(a: &Post, b: &Post) -> Ordering {
    a.title.to_lowercase().cmp(&b.title.to_lowercase())
}

/// The post title is its first level-1 heading (`# ...`); the filename is only a fallback.
fn title_from_markdown(src: &str) -> Option<String> {
    src.lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix("# ").map(|t| t.trim().to_string()))
}

/// Optional post dates use a markdown line like `### date: 29 Nov, 2025`.
fn date_from_markdown(src: &str) -> Option<String> {
    src.lines().map(str::trim).find_map(|line| {
        let date = line.strip_prefix("### date:")?.trim();
        (!date.is_empty()).then(|| date.to_string())
    })
}

fn parse_sortable_date(date: &str) -> Option<DateKey> {
    let mut parts = date.split_whitespace();
    let day = parts.next()?.parse::<u8>().ok()?;
    let month = month_number(parts.next()?.trim_end_matches(','))?;
    let year = parts.next()?.parse::<u16>().ok()?;

    if day == 0 || day > 31 || parts.next().is_some() {
        return None;
    }

    Some((year, month, day))
}

fn month_number(month: &str) -> Option<u8> {
    match month {
        "Jan" => Some(1),
        "Feb" => Some(2),
        "Mar" => Some(3),
        "Apr" => Some(4),
        "May" => Some(5),
        "Jun" => Some(6),
        "Jul" => Some(7),
        "Aug" => Some(8),
        "Sep" => Some(9),
        "Oct" => Some(10),
        "Nov" => Some(11),
        "Dec" => Some(12),
        _ => None,
    }
}

fn render_index(posts: &[Post]) -> String {
    let mut list = String::from("<h2>Posts</h2>\n<ul class=\"posts\">\n");
    for p in posts {
        let date = p
            .date
            .as_ref()
            .map(|date| format!(" <span class=\"post-date\">{}</span>", escape(date)))
            .unwrap_or_default();
        list.push_str(&format!(
            "  <li>{} <a href=\"{POSTS_DIR}/{}.html\">{}</a></li>\n",
            date,
            p.slug,
            escape(&p.title),
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
    render_page("feitreim.com", &body, false)
}

// --- markdown + math ----------------------------------------------------

/// Convert markdown to HTML, rendering LaTeX math to self-contained SVG at build time.
/// The parser recognizes `$...$` / `$$...$$` itself (ENABLE_MATH), so a `$` inside a
/// code fence or inline code stays literal instead of opening a math span.
fn md_to_html(src: &str) -> String {
    use pulldown_cmark::{html, Event, Options, Parser};
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_FOOTNOTES);
    opts.insert(Options::ENABLE_MATH);
    let parser = Parser::new_ext(src, opts).map(|event| match event {
        Event::InlineMath(latex) => Event::Html(render_math(&latex, false).into()),
        Event::DisplayMath(latex) => Event::Html(render_math(&latex, true).into()),
        other => other,
    });
    let mut html_out = String::new();
    html::push_html(&mut html_out, parser);
    html_out
}

/// Render one math span to a self-contained inline SVG so pages ship no KaTeX CSS/JS
/// and pull nothing from a CDN. Unparseable LaTeX falls back to the raw source shown
/// as code.
fn render_math(latex: &str, display: bool) -> String {
    math_to_svg(latex, display).unwrap_or_else(|| {
        let delim = if display { "$$" } else { "$" };
        format!("<code>{}</code>", escape(&format!("{delim}{latex}{delim}")))
    })
}

fn math_to_svg(latex: &str, display: bool) -> Option<String> {
    let nodes = ratex_parser::parse(latex).ok()?;
    let layout_opts = LayoutOptions {
        style: if display {
            MathStyle::Display
        } else {
            MathStyle::Text
        },
        color: Color::from_hex(MATH_COLOR)?,
        ..Default::default()
    };
    let layout = ratex_layout::layout(&nodes, &layout_opts);
    let list = ratex_layout::to_display_list(&layout);

    // font_size (== em_px) of 1 and zero padding make the SVG viewBox carry raw em units,
    // so the figure can be sized and baseline-aligned with plain CSS em values below.
    let svg = ratex_svg::render_to_svg(
        &list,
        &SvgOptions {
            font_size: 1.0,
            padding: 0.0,
            embed_glyphs: true,
            ..Default::default()
        },
    );
    Some(frame_math(
        svg,
        list.height + list.depth,
        list.depth,
        display,
    ))
}

/// Size the SVG in `em` (so it scales with surrounding text) and, for inline math, drop it by
/// its depth so the math baseline sits on the text baseline. Display math is centered in a
/// horizontally scrollable block.
fn frame_math(svg: String, total_em: f64, depth_em: f64, display: bool) -> String {
    if display {
        let svg = style_svg(svg, &format!("height:{total_em:.4}em;width:auto"));
        format!("<div class=\"math-display\">{svg}</div>")
    } else {
        let style = format!("height:{total_em:.4}em;width:auto;vertical-align:-{depth_em:.4}em");
        style_svg(svg, &style)
    }
}

fn style_svg(svg: String, style: &str) -> String {
    svg.replacen(
        "<svg ",
        &format!("<svg class=\"math\" style=\"{style}\" "),
        1,
    )
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
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

// --- html template ------------------------------------------------------

fn render_page(title: &str, body: &str, nav: bool) -> String {
    let nav = if nav {
        "<p class=\"nav\"><a href=\"/\"><- home</a></p>\n"
    } else {
        ""
    };
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="/style.css">
</head>
<body>
<main>
{nav}{body}
</main>
<p class="accred"><a href="https://github.com/comfysage/evergarden">evergarden</a> summer</p>
</body>
</html>
"#,
        title = escape(title),
        nav = nav,
        body = body
    )
}

// A lowercase serif "f" on a transparent ground; flips color to stay visible
// on light or dark tab bars.
const FAVICON: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <style>
    text { fill: #1a1a1a; }
    @media (prefers-color-scheme: dark) { text { fill: #fdfdfc; } }
  </style>
  <text x="50" y="54" font-family="ui-monospace, 'SF Mono', Menlo, monospace" font-size="84" font-weight="700" text-anchor="middle" dominant-baseline="central">f</text>
</svg>
"##;

const STYLE: &str = r#"
@font-face {
  font-family: 'Berkeley Mono';
  src: url('/BerkeleyMono-Regular.woff2') format('woff2');
  font-weight: 400; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'Berkeley Mono';
  src: url('/BerkeleyMono-Oblique.woff2') format('woff2');
  font-weight: 400; font-style: italic; font-display: swap;
}
@font-face {
  font-family: 'Berkeley Mono';
  src: url('/BerkeleyMono-Bold.woff2') format('woff2');
  font-weight: 700; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'Berkeley Mono';
  src: url('/BerkeleyMono-Bold-Oblique.woff2') format('woff2');
  font-weight: 700; font-style: italic; font-display: swap;
}
@font-face {
  font-family: 'Berkeley Mono';
  src: url('/BerkeleyMono-Black.woff2') format('woff2');
  font-weight: 900; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'Berkeley Mono';
  src: url('/BerkeleyMono-Black-Oblique.woff2') format('woff2');
  font-weight: 900; font-style: italic; font-display: swap;
}
/* Evergarden Summer */
:root {
  --fg: #2b3034; --muted: #829084; --link: #f57f82; --bg: #f5efe6;
  --surface: #e6e1d3; --border: #ceccbd;
}
* { box-sizing: border-box; }
body {
  font-family: 'Berkeley Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  line-height: 1.65; color: var(--fg); background: var(--bg);
  margin: 0; padding: 0;
}
main { max-width: 42rem; margin: 0 auto; padding: 3rem 1.25rem 6rem; }
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
h1, h2, h3 { line-height: 1.25; font-weight: 900; }
h1 { margin-top: 0; }
.nav a { color: var(--muted); font-size: 0.9rem; }
ul.posts { list-style: none; padding: 0; }
ul.posts li { margin: 0.4rem 0; font-size: 1.05rem; }
.post-date { color: var(--muted); font-size: 0.9em; margin-left: 0.4rem; }
code {
  font-family: 'Berkeley Mono', ui-monospace, Menlo, monospace; font-size: 0.9em;
  background: var(--surface); padding: 0.1em 0.3em; border-radius: 3px;
}
pre {
  background: var(--surface); padding: 1rem; border-radius: 6px;
  white-space: pre-wrap; overflow-wrap: anywhere;
}
pre code { background: none; padding: 0; }
blockquote {
  border-left: 3px solid var(--border); margin-left: 0; padding-left: 1rem; color: var(--muted);
}
img { max-width: 100%; height: auto; display: block; }
figure { width: fit-content; max-width: 100%; margin: 1.5rem auto; }
figure img { margin: 0 auto; border-radius: 3px; }
figcaption { color: var(--muted); font-size: 0.85rem; text-align: center; margin-top: 0.5rem; }
svg.math { vertical-align: middle; }
.math-display {
  overflow-x: auto; overflow-y: hidden; text-align: center; padding: 0.5rem 0;
}
.accred {
  position: fixed; bottom: 0.6rem; right: 0.75rem;
  font-size: 0.72rem; color: var(--muted);
}
.accred a { color: var(--muted); }
"#;
