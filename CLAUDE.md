# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A personal blog platform that uses **Notion as a CMS**, **Python** for content synchronization, and **Hugo** (with the PaperMod theme) for static site generation. Content written in Notion is automatically converted to Hugo-compatible markdown and deployed to GitHub Pages.

- Live site: https://kevinchen1994.github.io/ (custom domain: chenmeng.blog)
- Language: Chinese (zh-cn)

## Commands

```bash
# Local development server
hugo server

# Build static site (output to public/)
hugo

# Sync content from Notion (requires NOTION_TOKEN and DATABASE_ID in env or config.py)
python notion_to_hugo.py

# Install Python dependencies
pip install notion-client requests
```

## Architecture

```
Notion Database → notion_to_hugo.py → notion_converter.py → content/posts/*.md → Hugo build → GitHub Pages
```

### Key files

- **`notion_to_hugo.py`** — Orchestrator: fetches Notion database entries, checks timestamps to skip unchanged posts, writes markdown files, cleans deleted/draft posts
- **`notion_converter.py`** — Converter: transforms Notion blocks to markdown, downloads images to `static/images/`, generates Hugo front-matter (JSON format)
- **`config.py`** — Local-only Notion API credentials template (production uses GitHub Secrets)
- **`hugo.toml`** — Hugo configuration (theme, menus, Giscus comments, search, analytics)
- **`.github/workflows/sync-and-deploy.yml`** — CI/CD: runs on push to master, daily at 00:00 UTC, or manual trigger

### Content structure

- `content/posts/` — Blog posts (auto-generated from Notion, do not edit manually)
- `content/About.md`, `content/search.md` — Custom pages (manually maintained)
- `static/images/` — Images downloaded from Notion (naming: `notion_{block_id}{uuid}{ext}`)
- `layouts/partials/` — Custom Hugo template overrides (`comments.html` for Giscus, `single.html` for post layout)
- `themes/PaperMod/` — Hugo theme (git submodule, do not edit directly)

### Front-matter format

Posts use JSON front-matter inside YAML delimiters:
```
---
{"title": "...", "date": "...", "tags": [...], "categories": [...], "summary": "...", "draft": false, "generated_time": "..."}
---
```

### CI/CD pipeline

The GitHub Actions workflow: syncs Notion → commits changes to `content/` and `static/` → builds Hugo → deploys to GitHub Pages. Python 3.12 is used in CI.
