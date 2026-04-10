# GSL20051013-english

**Context-aware English part-of-speech tagger with clause and connective detection.**

[![PyPI version](https://badge.fury.io/py/GSL20051013-english.svg)](https://pypi.org/project/GSL20051013-english/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Non-Commercial](https://img.shields.io/badge/license-Non--Commercial-red.svg)](LICENSE)

`GSL20051013-english` is a lightweight Python library for tagging English text with Penn Treebank part-of-speech tags. It combines a state-of-the-art neural backend (spaCy) with a hand-crafted context-rule layer that corrects common mis-tags in auxiliary chains, adjective/verb ambiguities, passives, and more. It also detects clause boundaries and logical connectives.

---

## Features

- **Dual backend** – uses spaCy (`en_core_web_sm`) when installed for highest accuracy; falls back to NLTK's averaged perceptron tagger automatically.
- **Context-rule layer** – 21+ heuristic patterns correct auxiliary chains, passive voice, predicate adjectives, comparative/superlative forms, gerund subjects, and more.
- **Clause detection** – `find_clauses()` segments a sentence into main, subordinate, and relative clauses with semantic subtypes (causal, concessive, temporal, conditional, nominal, contrastive, exceptive, manner, purpose, additive). Supports both single-word and multi-word connectives (e.g. "instead of", "rather than", "even though", "as long as", "as well as", "in spite of", "due to").
- **Connective detection** – `find_connectives()` identifies coordinating, subordinating, and relative connectives (including multi-word connectives) with their positions.
- **Custom overrides** – register per-word tag overrides (fixed string or callable) via `register_word_tag()`.
- **Batch processing** – `analyze_batch()` processes multiple sentences efficiently using `nlp.pipe()` when spaCy is available.
- **Zero configuration** – NLTK data is downloaded automatically on first use.

---

## Installation

### Minimal (NLTK backend only)

```bash
pip install GSL20051013-english
```

### With spaCy backend (recommended – faster and more accurate)

```bash
pip install "GSL20051013-english[spacy]"
```

> **Note:** The spaCy extra installs `en_core_web_sm` automatically via the direct wheel URL.  
> If you manage spaCy models separately, run `pip install GSL20051013-english` and then  
> `python -m spacy download en_core_web_sm`.

---

## Quick Start

```python
from Geemeth import english, POSTag

# Tag a single sentence – returns list[tuple[str, POSTag]]
tagged = english.analyze_sentence("The fast robot jumped over 2 walls!")
print(tagged)
# [('The', POSTag.DT(3)), ('fast', POSTag.JJ(7)), ('robot', POSTag.NN(12)),
#  ('jumped', POSTag.VBD(28)), ('over', POSTag.IN(6)), ('2', POSTag.CD(2)),
#  ('walls', POSTag.NNS(13)), ('!', POSTag.PERIOD(38))]

# Tags are POSTag enum members – compare, print, or use as integers
word, tag = tagged[0]
print(tag)           # DT
print(tag.full_name) # Determiner
print(int(tag))      # 3
print(tag == POSTag.DT)  # True

# Tag multiple sentences efficiently
results = english.analyze_batch([
    "She has been running every morning.",
    "The letter was written by the president.",
])
for tokens in results:
    print(tokens)

# Detect clauses
tagged = english.analyze_sentence("Although she was tired, she kept working.")
for clause in english.find_clauses(tagged):
    words = " ".join(w for w, _ in clause["tokens"])
    print(f"[{clause['type']}/{clause['subtype'] or '-'}]  {words}")
# [subordinate/concessive]  Although she was tired
# [main/-]  she kept working

# Detect connectives
for conn in english.find_connectives(tagged):
    print(conn["type"], conn["subtype"], repr(conn["word"]))
# subordinating concessive 'Although'

# Detect multi-word connectives
tagged = english.analyze_sentence("She ran instead of walking.")
for conn in english.find_connectives(tagged):
    print(conn["type"], conn["subtype"], repr(conn["word"]))
# subordinating contrastive 'instead of'
```

---

## API Reference

### `POSTag`

`IntEnum` of all Penn Treebank POS tag codes.  Every function in this library
returns `POSTag` instances rather than plain strings, enabling fast numeric
comparison and rich metadata access.

```python
from Geemeth import POSTag

POSTag.NN           # POSTag.NN(12)
int(POSTag.NN)      # 12
str(POSTag.NN)      # 'NN'
POSTag.NN.full_name # 'Noun, singular or mass'

# Tags that contain $ use a safe Python name
str(POSTag.PRPS)    # 'PRP$'
str(POSTag.WPS)     # 'WP$'

# Convert from a raw string (e.g. when using a third-party tagger)
POSTag["NN"]        # POSTag.NN(12)
```

| Property / Method | Description |
|-------------------|-------------|
| `.full_name`      | Human-readable description (e.g. `"Noun, singular or mass"`) |
| `.tag_string`     | Canonical Penn Treebank string (e.g. `'PRP$'`, `'WP$'`) |
| `int(tag)`        | Numeric value (0–45); `0` means `POSTag.UNKNOWN` |

---

### `analyze_sentence(text: str) → list[tuple[str, POSTag]]`

Tag a single English sentence.

| Parameter | Type  | Description                        |
|-----------|-------|------------------------------------|
| `text`    | `str` | Raw English text (any punctuation) |

**Returns** a list of `(word, POSTag)` tuples.  `str(tag)` gives the
Penn Treebank abbreviation (e.g. `'NN'`, `'PRP$'`).

```python
english.analyze_sentence("Dogs run faster than cats.")
# [('Dogs', POSTag.NNS(13)), ('run', POSTag.VBP(31)), ('faster', POSTag.RBR(21)),
#  ('than', POSTag.IN(6)), ('cats', POSTag.NNS(13)), ('.', POSTag.PERIOD(38))]
```

---

### `analyze_batch(texts: list[str]) → list[list[tuple[str, POSTag]]]`

Tag multiple sentences in one call.  When spaCy is available this uses
`nlp.pipe()` for significantly better throughput.

```python
results = english.analyze_batch(["The cat sat.", "Dogs run fast."])
```

---

### `find_clauses(tagged: list[tuple[str, POSTag]]) → list[dict]`

Segment a POS-tagged sentence into logical clauses.

**Each returned dict contains:**

| Key          | Type   | Values / Notes                                                   |
|--------------|--------|------------------------------------------------------------------|
| `type`       | `str`  | `'main'`, `'subordinate'`, `'relative'`                          |
| `subtype`    | `str`  | `'causal'`, `'concessive'`, `'temporal'`, `'conditional'`, `'nominal'`, `'contrastive'`, `'exceptive'`, `'manner'`, `'purpose'`, `'additive'`, or `''` |
| `connective` | `str`  | Opening conjunction / relative pronoun (may be multi-word, e.g. `'instead of'`), or `''` for root main |
| `tokens`     | `list` | `list[tuple[str, POSTag]]` – tagged tokens in this clause           |

```python
tagged = english.analyze_sentence(
    "He stayed home because it was raining."
)
for c in english.find_clauses(tagged):
    print(c["type"], c["subtype"], c["connective"])
# main      ''      ''
# subordinate causal  because
```

---

### `find_connectives(tagged: list[tuple[str, POSTag]]) → list[dict]`

Identify logical connectives in a tagged sentence.

**Each returned dict contains:**

| Key        | Type  | Values / Notes                                                    |
|------------|-------|-------------------------------------------------------------------|
| `word`     | `str` | The connective as it appears in the text (multi-word connectives are space-joined, e.g. `'instead of'`) |
| `tag`      | `POSTag` | POS tag of the connective (`POSTag.IN` for multi-word connectives)  |
| `type`     | `str` | `'subordinating'`, `'coordinating'`, `'relative'`                 |
| `subtype`  | `str` | Semantic subtype (`'causal'`, `'concessive'`, `'temporal'`, `'conditional'`, `'nominal'`, `'contrastive'`, `'exceptive'`, `'manner'`, `'purpose'`, `'additive'`) for subordinating; `''` for others |
| `position` | `int` | Index of the first token of the connective in `tagged`            |

```python
tagged = english.analyze_sentence(
    "She left early because she was tired, but he stayed."
)
for c in english.find_connectives(tagged):
    print(c["type"], c["subtype"], c["word"])
# subordinating causal   because
# coordinating  ''       but
```

---

### `register_word_tag(word: str, tag_or_fn) → None`

Register a custom POS tag (or tag-computing callable) for a specific word.
Overrides are applied **after** the tagger and all context rules.

```python
from Geemeth import POSTag
from Geemeth import english

# Fixed override (string tag still accepted; POSTag preferred)
english.register_word_tag("Python", "NNP")

# Callable override – receives and returns POSTag
def fix_data(word: str, current_tag: POSTag, context: list) -> POSTag:
    return POSTag.NNS if current_tag == POSTag.NN else current_tag

english.register_word_tag("data", fix_data)
```

The callable signature is `(word: str, current_tag: POSTag, context: list[tuple[str, POSTag]]) -> POSTag`.

---

### `unregister_word_tag(word: str) → None`

Remove the override for a specific word (case-insensitive; no-op if absent).

---

### `clear_word_tag_overrides() → None`

Remove **all** registered overrides.

---

### `get_word_tag_overrides() → dict`

Return a shallow copy of the current override registry.

---

## Penn Treebank POS Tags (Reference)

These are all members of the `POSTag` enum.  Use `POSTag.<NAME>` in code;
`str(tag)` gives the Penn Treebank abbreviation.

| Tag / Name   | `POSTag` member | Description                       |
|--------------|-----------------|-----------------------------------|
| `CC`         | `POSTag.CC`     | Coordinating conjunction          |
| `CD`         | `POSTag.CD`     | Cardinal number                   |
| `DT`         | `POSTag.DT`     | Determiner                        |
| `EX`         | `POSTag.EX`     | Existential *there*               |
| `FW`         | `POSTag.FW`     | Foreign word                      |
| `IN`         | `POSTag.IN`     | Preposition / subordinating conj  |
| `JJ`         | `POSTag.JJ`     | Adjective                         |
| `JJR`        | `POSTag.JJR`    | Adjective, comparative            |
| `JJS`        | `POSTag.JJS`    | Adjective, superlative            |
| `LS`         | `POSTag.LS`     | List item marker                  |
| `MD`         | `POSTag.MD`     | Modal                             |
| `NN`         | `POSTag.NN`     | Noun, singular                    |
| `NNS`        | `POSTag.NNS`    | Noun, plural                      |
| `NNP`        | `POSTag.NNP`    | Proper noun, singular             |
| `NNPS`       | `POSTag.NNPS`   | Proper noun, plural               |
| `PDT`        | `POSTag.PDT`    | Predeterminer                     |
| `POS`        | `POSTag.POS`    | Possessive ending                 |
| `PRP`        | `POSTag.PRP`    | Personal pronoun                  |
| `PRP$`       | `POSTag.PRPS`   | Possessive pronoun                |
| `RB`         | `POSTag.RB`     | Adverb                            |
| `RBR`        | `POSTag.RBR`    | Adverb, comparative               |
| `RBS`        | `POSTag.RBS`    | Adverb, superlative               |
| `RP`         | `POSTag.RP`     | Particle                          |
| `SYM`        | `POSTag.SYM`    | Symbol                            |
| `TO`         | `POSTag.TO`     | *to*                              |
| `UH`         | `POSTag.UH`     | Interjection                      |
| `VB`         | `POSTag.VB`     | Verb, base form                   |
| `VBD`        | `POSTag.VBD`    | Verb, past tense                  |
| `VBG`        | `POSTag.VBG`    | Verb, gerund / present part       |
| `VBN`        | `POSTag.VBN`    | Verb, past participle             |
| `VBP`        | `POSTag.VBP`    | Verb, non-3rd-person sing         |
| `VBZ`        | `POSTag.VBZ`    | Verb, 3rd-person sing             |
| `WDT`        | `POSTag.WDT`    | *Wh*-determiner                   |
| `WP`         | `POSTag.WP`     | *Wh*-pronoun                      |
| `WP$`        | `POSTag.WPS`    | Possessive *wh*-pronoun           |
| `WRB`        | `POSTag.WRB`    | *Wh*-adverb                       |
| `,`          | `POSTag.COMMA`  | Comma                             |
| `.`          | `POSTag.PERIOD` | Period / sentence-final punct     |
| `:`          | `POSTag.COLON`  | Colon or semicolon                |
| ` `` `       | `POSTag.LQUOTE` | Left quotation mark               |
| `''`         | `POSTag.RQUOTE` | Right quotation mark              |
| `-LRB-`      | `POSTag.LRB`    | Left bracket                      |
| `-RRB-`      | `POSTag.RRB`    | Right bracket                     |
| `HYPH`       | `POSTag.HYPH`   | Hyphen                            |
| `NFP`        | `POSTag.NFP`    | Superfluous punctuation           |
| *(unknown)*  | `POSTag.UNKNOWN`| Unrecognised / not-yet-mapped tag |

---

## Requirements

- Python ≥ 3.10
- `nltk` ≥ 3.8
- *(optional)* `spacy` ≥ 3.5 + `en_core_web_sm`

---

## License

**Non-Commercial Use Only.**

This software is licensed under the
[english-pos Non-Commercial License v1.0](LICENSE).

- ✅ **Free** for personal projects, academic research, and open-source work.
- 💼 **Commercial use** (production systems, SaaS, paid products/services) requires a separate paid license.

To obtain a commercial license, open an issue or contact the author at
<https://github.com/GSL20051013>.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
