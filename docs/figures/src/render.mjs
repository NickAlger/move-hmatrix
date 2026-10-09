// Renders every figure source in this directory (*.jsx) to a static SVG one level up
// (docs/figures/<name>.svg), the form GitHub displays. See README.md here.
//
// A source is a React component that draws one SVG. Chart sources wrap their drawing in
// <claude.Visualize sources={{rows: {kind: 'data', data: [...]}}}>{({rows, datum}) => ...}
// (the form the documents were first written in): the data rows are the numbers plotted, and
// datum() only tagged elements for an editor, so here it returns nothing.
//
// Colours are CSS variables (--cds-*) set in an embedded style sheet, with a dark-mode
// variant chosen by the viewer's prefers-color-scheme.
import * as esbuild from 'esbuild';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import fs from 'node:fs';
import path from 'node:path';

const here = path.dirname(new URL(import.meta.url).pathname);
const outDir = path.resolve(here, '..');
const names = process.argv.length > 2
  ? process.argv.slice(2)
  : fs.readdirSync(here).filter((f) => f.endsWith('.jsx')).map((f) => f.slice(0, -4)).sort();

const claude = {
  Visualize: ({ sources, children }) => children({ rows: sources.rows.data, datum: () => ({}) }),
};

const light = {
  'bg': '#ffffff', 'text-primary': '#1f2328', 'text-secondary': '#59636e',
  'chart-axis': '#818b98', 'chart-grid': '#d8dee4', 'chart-muted': '#8c959f',
  'chart-categorical-1': '#0969da', 'chart-reference': '#bc4c00', 'chart-reference-tint': '#f6f8fa',
};
const dark = {
  'bg': '#0d1117', 'text-primary': '#e6edf3', 'text-secondary': '#9198a1',
  'chart-axis': '#7d8590', 'chart-grid': '#30363d', 'chart-muted': '#7d8590',
  'chart-categorical-1': '#4493f8', 'chart-reference': '#f0883e', 'chart-reference-tint': '#161b22',
};
const vars = (t) => Object.entries(t).map(([k, v]) => `--cds-${k}:${v}`).join(';');
const style =
  `svg{${vars(light)};font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans",Helvetica,Arial,"Liberation Sans",sans-serif}` +
  `@media (prefers-color-scheme:dark){svg{${vars(dark)}}}`;

// var() is reliable in style properties but not in every renderer's presentation attributes.
function varsToStyle(svg) {
  return svg.replace(/<([a-zA-Z]+)((?:\s+[\w:-]+="[^"]*")*)\s*(\/?)>/g, (m, tag, attrs, close) => {
    const styles = [];
    const kept = attrs.replace(/\s+(fill|stroke)="(var\([^"]*\))"/g, (_, a, v) => { styles.push(`${a}:${v}`); return ''; });
    if (!styles.length) return m;
    return `<${tag}${kept} style="${styles.join(';')}"${close ? ' /' : ''}>`;
  });
}

for (const name of names) {
  const src = fs.readFileSync(path.join(here, `${name}.jsx`), 'utf8');
  const { code } = await esbuild.transform(src, { loader: 'jsx', format: 'cjs', jsx: 'transform' });
  const module = { exports: {} };
  new Function('module', 'exports', 'React', 'claude', code)(module, module.exports, React, claude);
  let svg = renderToStaticMarkup(React.createElement(module.exports.default));
  svg = svg.replace(/\s+data-claude-[\w-]+="[^"]*"/g, '');
  svg = varsToStyle(svg);
  const [, w, h] = svg.match(/viewBox="0 0 ([\d.]+) ([\d.]+)"/);
  const label = (svg.match(/aria-label="([^"]*)"/) || [])[1] || name;
  svg = svg.replace(/^<svg /, `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" `);
  svg = svg.replace(/^(<svg[^>]*>)/, `$1<title>${label}</title><style>${style}</style>` +
    `<rect x="0" y="0" width="${w}" height="${h}" rx="6" style="fill:var(--cds-bg)"/>`);
  fs.writeFileSync(path.join(outDir, `${name}.svg`), svg + '\n');
  console.log(`${name}.svg`);
}
