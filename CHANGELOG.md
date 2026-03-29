# Changelog

All notable changes to **english-pos** will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [1.0.0] – 2024-01-01

### Added
- `analyze_sentence()` – single-sentence POS tagging with Penn Treebank tags.
- `analyze_batch()` – efficient multi-sentence tagging via `nlp.pipe()`.
- `find_clauses()` – clause segmentation with type/subtype/connective metadata.
- `find_connectives()` – detection of coordinating, subordinating, and relative connectives.
- `register_word_tag()` / `unregister_word_tag()` / `clear_word_tag_overrides()` / `get_word_tag_overrides()` – per-word tag override registry with callable support.
- Dual backend: spaCy `en_core_web_sm` (primary) with automatic NLTK fallback.
- 19+ context-correction rules covering auxiliary chains, passive voice, predicate adjectives, comparatives, superlatives, existential *there*, going-to future, perfect aspect, do-support, and parallel structure.
- Long-sentence / multi-clause support with clause-boundary awareness.
- `__version__` attribute and `__all__` export list.
- `pyproject.toml` build configuration for PyPI distribution.
- Non-Commercial License v1.0.
