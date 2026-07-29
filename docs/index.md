# afmkit

> Modern Python toolkit for single-molecule force spectroscopy.

afmkit is a clean, extensible Python reimplementation of the workflow that
single-molecule biophysics labs have been running in Igor Pro for two
decades. It reads AFM force-extension curves, fits standard polymer models,
picks sawtooth unfolding peaks, and exports results in formats that drop
straight into Origin, Matlab, and the Python data stack.

## What you can do today (v0.1 — core only)

- Load JPK 4-column `.txt` exports from Nanowizzard / ForceRobot.
- Read legacy Igor Binary Wave (`.ibw`) data.
- Fit the Marko-Siggia WLC model with a least-squares engine.
- Store and re-load everything in a single HDF5 file.
- Export per-curve results to CSV, Matlab `.mat`, Parquet, or Markdown.

## What is coming next

- **v0.2** — extensible WLC (Wang 1997), FJC, automated sawtooth peak
  detection, and a minimal PyQt6 GUI.
- **v0.3+** — full GUI parity with the original Igor panel, plugin
  examples (Bruker loader, twist-WLC, robust fitting), official
  documentation site.

## Where to go next

- 🚀 [Quickstart](quickstart.md) — install afmkit and fit your first curve.
- 🔁 [Migration from Igor Pro](migration.md) — bring your existing data
  and workflow across.
- 🧩 [Contributing / plugin authoring](contributing.md) — extend afmkit
  with your own file format or polymer model.
- 📚 [API reference](api/index.md) — full public API.
