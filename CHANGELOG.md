# Changelog

All notable changes to **GSL20051013-english** will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- `_MULTIWORD_CONJ` – recognition table for 13 multi-word connective patterns
  (e.g. "instead of", "rather than", "even though", "even if", "as long as",
  "as soon as", "as if", "as though", "so that", "in order to", "now that",
  "except that", "provided that", "in case").
- `_match_multiword()` – internal helper that matches multi-word connectives
  at a given position in a tagged sequence.
- `find_clauses()` now detects and segments on multi-word connectives in
  addition to single-word subordinating conjunctions.
- `find_connectives()` now returns multi-word connectives as single entries
  (space-joined surface form, `tag='IN'`, `position` of first token).
- New single-word connectives: "except" (subtype `'exceptive'`) and "lest"
  (subtype `'conditional'`) added to `_SUBORD_CONJ` and `_CONJ_SUBTYPES`.
- New single-word connectives: "whenever" (subtype `'temporal'`), "once"
  (subtype `'temporal'`), and "that" (subtype `'nominal'`) added to
  `_SUBORD_CONJ` to align with `_CONJ_SUBTYPES`.
- New semantic subtypes: `'contrastive'`, `'exceptive'`, `'manner'`,
  `'purpose'`, `'additive'` alongside the existing causal/concessive/temporal/conditional/nominal.
- New multi-word connectives: "as well as" (`'additive'`), "given that"
  (`'conditional'`), "in spite of" (`'concessive'`), "due to" (`'causal'`),
  "owing to" (`'causal'`), "regardless of" (`'concessive'`), "such that"
  (`'purpose'`), "for fear that" (`'conditional'`),
  "on the condition that" (`'conditional'`).
- Extended `_DEGREE_ADVERBS` with additional intensifiers: "enormously",
  "intensely", "remarkably", "surprisingly", "unusually", "exceedingly",
  "exceptionally", "extraordinarily", "particularly", "especially",
  "notably", "genuinely", "distinctly", "utterly", "immensely", "profoundly".
- Pattern 20: "most"/"least" as superlative intensifier → `RBS`; next
  adjective/adverb promoted to `JJS` (e.g. "most beautiful", "least interesting").
- Pattern 21: Sentence-initial gerund mis-tagged as `NN`/`NNS` corrected to
  `VBG` (e.g. "Running every day builds stamina.").

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
