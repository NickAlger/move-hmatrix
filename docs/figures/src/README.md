# Figure sources

Each `*.jsx` file here draws one figure of the background docs as an SVG React component;
`render.mjs` turns it into the static SVG one directory up, which the documents embed. The
numbers a chart plots are the `data` rows at the top of its source, so changing a figure is:
edit the source, then

```sh
cd docs/figures/src
npm install          # once: esbuild, react, react-dom
npm run render       # or: node render.mjs <name>  for one figure
```

The SVGs carry their own light and dark colours (chosen by the viewer's
`prefers-color-scheme`) and use the system sans-serif font.

| Source | Figure | Used in |
|---|---|---|
| `interop-transfer-costs.jsx` | Cost of each transfer route, in source matvecs | [Moving H-matrices between libraries](../../interoperability.md#measured-costs) |
| `overview-landscape.jsx` | The twelve packages by operations and parallel reach | [Overview](../../overview.md#landscape) |
| `h2lib-h-vs-h2.jsx` | H vs H² storage and Cholesky time in H2Lib | [H2Lib](../../packages/h2lib.md#spot-check-h-vs-h-on-a-3d-bem-problem) |
| `hlibpro-memory-time.jsx` | HLIBpro memory and H-Cholesky time from 2k to 2M unknowns | [HLIBpro](../../packages/hlibpro.md#how-memory-and-time-grow-with-n) |
| `h2opus-hara-accuracy.jsx` | HARA's achieved vs requested error against tree depth | [H2Opus](../../packages/h2opus.md#hara-accuracy-and-error-compounding) |

The routes diagram in the interoperability document is Mermaid, written inline in the
Markdown.
