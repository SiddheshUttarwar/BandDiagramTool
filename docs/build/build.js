// Build the static documentation site (docs/*.html) from docs/math/*.md.
// The Markdown sources in docs/math/ are kept out of the repository (see
// .gitignore): GitHub renders their equations incorrectly. Keep a backup.
//
//   cd docs/build && npm install && node build.js
//
// Every equation is typeset to SVG here, at build time, with MathJax. The
// published pages contain no math script and do not depend on how a viewer
// (GitHub, a browser extension, ...) treats TeX.
const fs = require('fs');
const path = require('path');
const MarkdownIt = require('markdown-it');
const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const {AllPackages} = require('mathjax-full/js/input/tex/AllPackages.js');

const SRC = path.join(__dirname, '..', 'math');
const OUT = path.join(__dirname, '..');
const SITE_TITLE = 'EpiBand model reference';
const REPO = 'https://github.com/SiddheshUttarwar/BandDiagramTool';

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);

const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const slug = s => s.toLowerCase().replace(/<[^>]+>/g, '').replace(/[^\w\s.-]/g, '').trim().replace(/[\s.]+/g, '-');
const outName = f => (f === 'README.md' ? 'index.html' : f.replace(/\.md$/, '.html'));

// ---------------------------------------------------------------- sources
const files = fs.readdirSync(SRC).filter(f => f.endsWith('.md')).sort((a, b) =>
  (a === 'README.md' ? -1 : b === 'README.md' ? 1 : a.localeCompare(b)));
const pages = files.map(f => {
  const text = fs.readFileSync(path.join(SRC, f), 'utf8').replace(/\r\n/g, '\n');
  const h1 = (text.match(/^# (.+)$/m) || [, f])[1];
  return {file: f, out: outName(f), text, title: h1.replace(/^(\d+|[A-Z])\.\s*/, ''), num: (h1.match(/^(\d+|[A-Z])\./) || [])[1]};
});

// equation number -> page that defines it
const eqHome = {};
for (const p of pages) for (const m of p.text.matchAll(/\\tag\{(\d+\.\d+)\}/g)) eqHome[m[1]] = p.out;

// ---------------------------------------------------------------- math
let mjxCss = '';
function makeTypesetter() {
  const tex = new TeX({packages: AllPackages, tags: 'ams'});
  const svg = new SVG({fontCache: 'none'});
  const doc = mathjax.document('', {InputJax: tex, OutputJax: svg});
  return (src, display) => {
    const node = doc.convert(src, {display, em: 16, ex: 8, containerWidth: 760});
    const html = adaptor.outerHTML(node);
    const err = html.match(/data-mjx-error="([^"]*)"/);
    if (err) throw new Error(`TeX error: ${err[1]}\n  ${src}`);
    if (!mjxCss) mjxCss = adaptor.textContent(svg.styleSheet(doc));
    return html;
  };
}

// ---------------------------------------------------------------- page body
function renderPage(p) {
  const typeset = makeTypesetter();           // fresh tag numbering per page
  const store = [];
  const keep = html => `\u0001${store.push(html) - 1}\u0002`;
  let t = p.text;

  // display math
  t = t.replace(/^```math[ \t]*\n([\s\S]*?)\n```[ \t]*$/gm, (_, src) => {
    const tag = (src.match(/\\tag\{(\d+\.\d+)\}/) || [])[1];
    const id = tag ? ` id="eq-${tag.replace('.', '-')}"` : '';
    return '\n\n' + keep(`<div class="eq"${id} role="img" aria-label="${esc(src.replace(/\s+/g, ' '))}">${typeset(src, true)}</div>`) + '\n\n';
  });
  // other fenced blocks stay as they are
  t = t.replace(/^```[\s\S]*?^```[ \t]*$/gm, m => keep(md.render(m)));
  // inline math
  t = t.replace(/\$`([^`\n]+?)`\$/g, (_, src) =>
    keep(`<span class="m" role="img" aria-label="${esc(src)}">${typeset(src, false)}</span>`));
  // inline code is protected from the reference linker below
  t = t.replace(/`[^`\n]+`/g, m => keep(md.renderInline(m)));
  // equation references "(5.14)" -> links (not layer thicknesses such as "AlN(4.5)")
  t = t.replace(/(?<![A-Za-z0-9])\((\d{1,2}\.\d{1,2})\)/g, (m, n) => {
    if (!eqHome[n]) return m;
    const href = (eqHome[n] === p.out ? '' : eqHome[n]) + `#eq-${n.replace('.', '-')}`;
    return keep(`<a class="eqref" href="${href}">(${n})</a>`);
  });
  // links between chapters
  t = t.replace(/\]\(([^)\s#]+)\.md(#[^)]*)?\)/g, (_, f, h) => `](${outName(f + '.md')}${h || ''})`);

  let html = md.render(t);

  // callouts:  > [!NOTE]
  html = html.replace(/<blockquote>\s*<p>\[!(NOTE|IMPORTANT|WARNING|TIP|CAUTION)\]\s*/g, (_, k) =>
    `<blockquote class="callout ${k.toLowerCase()}"><p><span class="callout-title">${k[0] + k.slice(1).toLowerCase()}</span>`);
  // tables scroll on their own instead of widening the page
  html = html.replace(/<table>/g, '<div class="table-wrap"><table>').replace(/<\/table>/g, '</table></div>');
  // restore protected fragments (nested: links may sit inside nothing else)
  for (let i = 0; i < 3; i++) html = html.replace(/\u0001(\d+)\u0002/g, (_, k) => store[+k]);
  html = html.replace(/<p>\s*(<div class="eq"[\s\S]*?<\/div>)\s*<\/p>/g, '$1');

  // headings: ids + section list
  const sections = [];
  html = html.replace(/<h([23])>([\s\S]*?)<\/h\1>/g, (_, lvl, inner) => {
    const id = slug(inner);
    if (lvl === '2') sections.push({id, inner: inner.replace(/<[^>]+>/g, '')});
    return `<h${lvl} id="${id}"><a class="anchor" href="#${id}" aria-label="Link to this section">#</a>${inner}</h${lvl}>`;
  });
  return {html, sections};
}

const md = new MarkdownIt({html: true, linkify: false, typographer: false});

// ---------------------------------------------------------------- shell
function shell(p, body, sections, i) {
  const nav = pages.map(q => {
    const cur = q === p;
    const label = (q.num ? `<span class="n">${q.num}</span>` : '<span class="n"></span>') + esc(q === pages[0] ? 'Overview' : q.title);
    const sub = cur && sections.length
      ? '<ul class="sub">' + sections.map(s => `<li><a href="#${s.id}">${esc(s.inner)}</a></li>`).join('') + '</ul>' : '';
    return `<li${cur ? ' class="current"' : ''}><a href="${q.out}"${cur ? ' aria-current="page"' : ''}>${label}</a>${sub}</li>`;
  }).join('\n');
  const prev = pages[i - 1], next = pages[i + 1];
  const pager = `<nav class="pager" aria-label="Chapter">
${prev ? `<a class="prev" href="${prev.out}"><span>Previous</span>${esc(prev === pages[0] ? 'Overview' : prev.title)}</a>` : '<span></span>'}
${next ? `<a class="next" href="${next.out}"><span>Next</span>${esc(next.title)}</a>` : '<span></span>'}
</nav>`;
  const title = p === pages[0] ? SITE_TITLE : `${p.title} · ${SITE_TITLE}`;
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title)}</title>
<meta name="description" content="Equations, parameters and algorithms of EpiBand, a 1D Schrödinger-Poisson-current solver for III-nitride heterostructures.">
<link rel="stylesheet" href="assets/site.css">
</head>
<body>
<a class="skip" href="#content">Skip to content</a>
<header class="top">
  <a class="brand" href="index.html">EpiBand <span>model reference</span></a>
  <a class="repo" href="${REPO}">GitHub</a>
</header>
<div class="layout">
<aside class="side">
  <details class="navbox" open>
    <summary>Chapters</summary>
    <ol class="nav">
${nav}
    </ol>
  </details>
</aside>
<main id="content">
<article>
${body}
</article>
${pager}
<footer>Equations are typeset at build time with MathJax.</footer>
</main>
</div>
<script>
// collapse the chapter list by default on narrow screens
if (window.matchMedia('(max-width: 900px)').matches) document.querySelector('.navbox').removeAttribute('open');
</script>
</body>
</html>
`;
}

// ---------------------------------------------------------------- build
let nEq = 0;
pages.forEach((p, i) => {
  const {html, sections} = renderPage(p);
  nEq += (html.match(/<mjx-container/g) || []).length;
  if (/\u0001|\$`|```math/.test(html)) throw new Error('unrendered math left in ' + p.file);
  fs.writeFileSync(path.join(OUT, p.out), shell(p, html, sections, i));
  console.log(p.out.padEnd(34), (html.match(/<mjx-container/g) || []).length, 'equations');
});
fs.mkdirSync(path.join(OUT, 'assets'), {recursive: true});
const css = fs.readFileSync(path.join(__dirname, 'site.css'), 'utf8');
fs.writeFileSync(path.join(OUT, 'assets', 'site.css'), css + '\n/* MathJax SVG output */\n' + mjxCss + '\n');
fs.writeFileSync(path.join(OUT, '.nojekyll'), '');
console.log(`${pages.length} pages, ${nEq} equations`);
