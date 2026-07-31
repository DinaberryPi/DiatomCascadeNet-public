# DiatomCascadeNet Research Site

Bilingual public companion site for *Hierarchical Deep Learning for Diatom Image Classification: A Multi-Level Taxonomic Approach*.

The site presents the experiment pipeline, model stages, independently recomputed matched comparisons, and verified literature. Internal audit notes and results that have not completed the current evidence review remain in the author's local review copy. The public site intentionally provides no image upload or live inference.

## Development

Requires Node.js 22.13 or newer.

```bash
npm install
npm run dev
```

## Validation

```bash
npm test
npm run lint
```

`npm test` creates a production build and checks the public-content boundary.

## Hosting

`.openai/hosting.json` connects this directory to the existing public Sites project. The site uses no database, object storage, authentication, upload, or inference service.
