# Speech Dialog Systems Literature Atlas

This directory is a static GitHub Pages site. It has no build step.

Open `index.html` locally, or configure GitHub Pages to publish from the repository's `/docs` folder.

The companion `literature.md` file contains the same records sorted by component and year for easier review in GitHub.

## Extend the bibliography

Edit `docs/data/literature.js` and append records with this shape:

```js
{
  id: "stable-id",
  year: 2026,
  component: "Automatic Evaluation",
  type: "method",
  evaluation: ["automatic", "human"],
  title: "Paper title",
  authors: "Author list",
  venue: "Venue",
  url: "https://doi.org/...",
  note: "One-sentence field relevance."
}
```

The UI sorts records by component, then year. Filters, timeline graphics, and CSV export update automatically.
