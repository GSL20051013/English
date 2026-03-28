import re
import nltk
from functools import lru_cache

for _d in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
    nltk.download(_d, quiet=True)

# ---------------------------------------------------------------------------
# Optional spaCy backend – faster and more accurate than NLTK when available.
# Falls back to NLTK if spaCy or the English model is not installed.
# Only the tagger and tok2vec components are loaded (no parser / NER).
# ---------------------------------------------------------------------------
try:
    import spacy as _spacy
    _nlp = _spacy.load("en_core_web_sm", exclude=["parser", "ner", "lemmatizer"])
    _USE_SPACY = True
except (ImportError, OSError):
    _USE_SPACY = False

_TOKENIZE = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[^\s]")

# ---------------------------------------------------------------------------
# Vocabulary sets used by the context-rule engine
# ---------------------------------------------------------------------------
_MODALS = frozenset({
    "can", "will", "shall", "may", "might",
    "could", "would", "should", "must", "ought",
})

_HAVE_FORMS = frozenset({
    "have", "has", "had", "having",
})

_DO_FORMS = frozenset({
    "do", "does", "did",
})

_BE_FORMS = frozenset({
    "be", "am", "is", "are", "was", "were", "been", "being",
})

_DEGREE_ADVERBS = frozenset({
    "very", "quite", "rather", "extremely", "so", "too",
    "really", "fairly", "pretty", "somewhat", "incredibly",
    "awfully", "terribly", "deeply", "highly", "hugely",
    "perfectly", "totally", "absolutely", "completely",
})

_NEGATIONS = frozenset({"not", "n't", "never"})

_PREPOSITIONS = frozenset({
    "in", "on", "at", "by", "with", "from", "to", "of",
    "about", "above", "below", "between", "through", "during",
    "before", "after", "under", "over", "into", "onto",
    "against", "among", "around", "behind", "beside",
    "beyond", "near", "within", "without",
})

_SUBORD_CONJ = frozenset({
    "because", "although", "though", "if", "when", "while",
    "since", "unless", "until", "whereas", "whether",
})

_SUBJ_PRONOUNS = frozenset({"i", "he", "she", "it", "we", "they", "you"})

_NOUN_TAGS = frozenset({"NN", "NNS", "NNP", "NNPS"})
_VERB_TAGS = frozenset({"VB", "VBZ", "VBD", "VBG", "VBN", "VBP"})
_ADJ_TAGS  = frozenset({"JJ", "JJR", "JJS"})
_AMBIG_TAGS = frozenset({"JJ", "VB", "VBZ", "VBD"})   # tags that can be misclassified

# Pre-computed unions for use inside the hot loop
_NEG_OR_DEG   = _NEGATIONS | _DEGREE_ADVERBS
_NOUN_OR_VERB = _NOUN_TAGS | _VERB_TAGS
_AUX_FORMS    = _HAVE_FORMS | _DO_FORMS | _MODALS | _BE_FORMS

# Lookup tables used in the context-rule self-corrections (module-level for speed)
_HAVE_TAG = {"have": "VBP", "has": "VBZ", "had": "VBD", "having": "VBG"}
_DO_TAG   = {"do": "VBP", "does": "VBZ", "did": "VBD"}


@lru_cache(maxsize=4096)
def _guess_raw_morphology(word: str) -> str:
    """Fallback guesser that returns raw Penn Treebank-style tags."""
    # 1. Numbers & Symbols
    if re.match(r'^-?\d+(\.\d+)?$', word): return 'CD'   # Cardinal Digit
    if not re.match(r"^[A-Za-z]+(?:'[A-Za-z]+)*$", word): return 'SYM'  # Symbol

    w = word.lower()

    # 2. Common closed-class function words (before suffix rules)
    if w in {"the", "a", "an"}:                            return 'DT'
    if w in {"and", "or", "but", "nor", "yet", "for", "so"}: return 'CC'
    if w in {"i", "he", "she", "it", "we", "they", "you"}: return 'PRP'
    if w in {"my", "his", "her", "its", "our", "their", "your"}: return 'PRP$'
    if w in {"this", "that", "these", "those"}:            return 'DT'
    if w in {"am", "is"}:                                  return 'VBZ'
    if w in {"are"}:                                       return 'VBP'
    if w in {"was", "were"}:                               return 'VBD'
    if w in {"be"}:                                        return 'VB'
    if w in {"been"}:                                      return 'VBN'
    if w in {"being"}:                                     return 'VBG'
    if w in {"has"}:                                       return 'VBZ'
    if w in {"have"}:                                      return 'VBP'
    if w in {"had"}:                                       return 'VBD'
    if w in {"having"}:                                    return 'VBG'
    if w in {"does"}:                                      return 'VBZ'
    if w in {"do"}:                                        return 'VBP'
    if w in {"did"}:                                       return 'VBD'
    if w in _MODALS:                                       return 'MD'
    if w in _PREPOSITIONS:                                 return 'IN'
    if w in _SUBORD_CONJ:                                  return 'IN'
    if w in {"not", "never"}:                              return 'RB'
    if w in _DEGREE_ADVERBS:                               return 'RB'
    if w in {"more", "less"}:                              return 'RBR'
    if w in {"most", "least"}:                             return 'RBS'
    if w == "there":                                       return 'EX'

    # 3. Suffix rules — longer / more specific suffixes checked first
    if w.endswith(('ize', 'ise', 'ify')):   return 'VB'   # organize, realise, modify
    if w.endswith(('ism', 'ist')):           return 'NN'   # capitalism, scientist
    if w.endswith('ship'):                   return 'NN'   # friendship, hardship
    if w.endswith('hood'):                   return 'NN'   # childhood, neighborhood
    if w.endswith('dom'):                    return 'NN'   # kingdom, freedom
    if w.endswith(('tion', 'sion')):         return 'NN'   # nation, tension
    if w.endswith(('ness', 'ment', 'ity')): return 'NN'   # kindness, treatment, ability
    if w.endswith('ing'):                    return 'VBG'  # running, jumping
    if w.endswith('est'):                    return 'JJS'  # fastest, largest
    if w.endswith('ed'):                     return 'VBD'  # walked, played
    if w.endswith('ish'):                    return 'JJ'   # reddish, childish
    if w.endswith(('ful', 'less', 'ous', 'ive', 'able', 'ible', 'al', 'ic')): return 'JJ'
    if w.endswith('ly'):                     return 'RB'   # quickly, slowly

    if word[0].isupper(): return 'NNP'

    return 'NN'


def _apply_context_rules(tagged: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Refine POS tags by applying positional / contextual patterns.

    Patterns
    --------
    1.  to + WORD              → WORD is VB    (infinitive)
         e.g. "want TO run", "going TO eat"
    2.  Modal + [not] + WORD   → WORD is VB
         e.g. "can GO", "should NOT leave"
    3.  Be-verb + [neg/deg] + WORD → WORD is VBG / VBN / JJ
         e.g. "is RUNNING", "was TIRED", "were EATEN"
    4.  Degree-adverb + WORD   → WORD is JJ
         e.g. "very FAST", "quite LARGE", "incredibly SMART"
    5.  Article/Possessive + WORD + Noun → WORD is JJ
         e.g. "the FAST car", "a BEAUTIFUL song", "my OLD friend"
    6.  Parallel structure: X + CC + WORD → WORD gets tag of X
         e.g. "fast AND furious" (JJ), "cats AND dogs" (NN), "run OR jump" (VB)
    7.  Cardinal number + WORD → WORD is NNS
         e.g. "2 WALLS", "three DOGS"
    8.  Possessive 's + WORD   → WORD is NN
         e.g. "John's CAR", "the dog's COLLAR"
    9.  Preposition + WORD     → WORD is NN  (when ambiguous)
         e.g. "in the CITY", "by the RIVER"
    10. Wh-word (what/which) + WORD → WORD is NN or VB based on position
         e.g. "what IS this" → IS is VB; "which CAR" → CAR is NN
    11. Have-auxiliary + [neg] + WORD → WORD is VBN  (perfect aspect)
         e.g. "has EATEN", "have FINISHED", "had GONE"
    12. Do-support + [neg] + WORD → WORD is VB  (emphatic / interrogative)
         e.g. "do KNOW", "does SHE like" → "like" is VB, "did NOT see"
    13. "going to" future: going + to → next verb is VB
         e.g. "I am going TO LEAVE"
    14. Subordinating conjunctions → always IN
         e.g. "BECAUSE he left", "ALTHOUGH it rained", "IF you go"
    15. Existential "there" before be-form → EX
         e.g. "THERE is a problem", "THERE are many options"
    16. "more" / "less" before adj/noun/verb → RBR + JJ
         e.g. "MORE beautiful", "LESS interesting"
    17. JJR/RBR + "than" → "than" is IN  (comparative conjunction)
         e.g. "faster THAN light", "more beautiful THAN ever"
    18. Relative pronoun + ambiguous word → verb form
         e.g. "the man WHO RUNS", "a book THAT DESCRIBES"
    Post-pass 19. Passive voice: be + [advs] + -ed/-en word + "by" → VBN
         e.g. "was WRITTEN by", "is KNOWN for"
    """
    tags = list(tagged)
    n = len(tags)

    for i in range(n):
        word, tag = tags[i]
        w = word.lower()

        # ------------------------------------------------------------------
        # Self-correction: known degree adverbs should always be RB
        # NLTK sometimes mislabels them (e.g. "incredibly" → NN)
        # ------------------------------------------------------------------
        if w in _DEGREE_ADVERBS and tag not in ("RB", "RBR", "RBS"):
            tags[i] = (word, "RB")
            tag = "RB"

        # ------------------------------------------------------------------
        # Self-correction: known modal / auxiliary verbs should be verb tags.
        # Sentence-initial capitalisation can cause NLTK to produce NNP.
        # ------------------------------------------------------------------
        if w in _MODALS and tag != "MD":
            tags[i] = (word, "MD")
            tag = "MD"
        if w in _HAVE_FORMS and tag not in _VERB_TAGS:
            tags[i] = (word, _HAVE_TAG.get(w, "VBZ"))
            tag = tags[i][1]
        if w in _DO_FORMS and tag not in _VERB_TAGS:
            tags[i] = (word, _DO_TAG.get(w, "VBZ"))
            tag = tags[i][1]

        # ------------------------------------------------------------------
        # Pattern 14 — Subordinating conjunctions: force to IN
        # "BECAUSE he left", "ALTHOUGH it rained", "IF you go"
        # ------------------------------------------------------------------
        if w in _SUBORD_CONJ and tag not in ("IN", "RB"):
            tags[i] = (word, "IN")
            tag = "IN"

        # ------------------------------------------------------------------
        # Pattern 15 — Existential "there" before a be-form → EX
        # "THERE is a problem", "THERE are many options"
        # ------------------------------------------------------------------
        if w == "there" and tag != "EX" and i + 1 < n:
            if tags[i + 1][0].lower() in _BE_FORMS:
                tags[i] = (word, "EX")
                tag = "EX"

        # ------------------------------------------------------------------
        # Pattern 16 — "more" / "less" → RBR; next non-"than" word → JJ
        # "MORE beautiful", "LESS interesting", "more THAN"
        # ------------------------------------------------------------------
        if w in ("more", "less"):
            if tag not in ("RBR", "JJR"):
                tags[i] = (word, "RBR")
                tag = "RBR"
            if i + 1 < n:
                nw, nt = tags[i + 1]
                if nw.lower() != "than" and nt not in _ADJ_TAGS and nt not in ("RB", "RBR", "RBS"):
                    tags[i + 1] = (nw, "JJ")

        # ------------------------------------------------------------------
        # Pattern 17 — Comparative/superlative + "than" → "than" is IN
        # "faster THAN light", "more beautiful THAN ever"
        # ------------------------------------------------------------------
        if tag in ("JJR", "RBR") and i + 1 < n and tags[i + 1][0].lower() == "than":
            tags[i + 1] = (tags[i + 1][0], "IN")

        # ------------------------------------------------------------------
        # Pattern 18 — Relative pronoun + ambiguous word → verb form
        # "the man WHO RUNS", "a book THAT DESCRIBES", "the cat WHICH SITS"
        # Only corrects NNS (plural-noun) tags to VBZ to avoid false-positives
        # on singular nouns that legitimately end in -s (e.g. "analysis",
        # "crisis") which tend to be tagged NN rather than NNS.
        # ------------------------------------------------------------------
        if w in ("who", "which", "that") and tag in ("WP", "WDT", "IN") and i + 1 < n:
            nw, nt = tags[i + 1]
            nwl = nw.lower()
            if nwl not in _BE_FORMS and nwl not in _MODALS:
                if nwl.endswith("s") and not nwl.endswith("ss") and nt == "NNS":
                    tags[i + 1] = (nw, "VBZ")
                elif nt in ("JJ", "VB") and not nwl.endswith("s"):
                    tags[i + 1] = (nw, "VBP")

        # ------------------------------------------------------------------
        # Pattern 1 — Infinitive: "to" → next word is base verb (VB)
        # Covers: "want to run", "need to go", "going to eat", "have to try"
        # ------------------------------------------------------------------
        if w == "to" and tag == "TO" and i + 1 < n:
            nw, nt = tags[i + 1]
            if nt not in ("DT", "CD", "PRP", "PRP$", "WP", "WP$", "IN", "CC", "TO"):
                tags[i + 1] = (nw, "VB")

        # ------------------------------------------------------------------
        # Pattern 2 — Modal auxiliary → next content word is base verb (VB)
        # Skip an optional negation ("not", "n't") between modal and verb.
        # ------------------------------------------------------------------
        if w in _MODALS and tag == "MD":
            j = i + 1
            if j < n and tags[j][0].lower() in _NEGATIONS:
                j += 1
            if j < n:
                nw, nt = tags[j]
                if nt not in ("DT", "CD", "PRP", "PRP$", "IN", "CC"):
                    tags[j] = (nw, "VB")

        # ------------------------------------------------------------------
        # Pattern 3 — Be-verb as copula/auxiliary → predict complement
        # • ends in -ing  → VBG  ("is running")
        # • ends in -ed   → VBN  ("was excited")
        # • otherwise     → JJ   ("is tall", "were happy")
        # Skip optional negations and degree adverbs before the complement.
        # ------------------------------------------------------------------
        if w in _BE_FORMS and tag in _VERB_TAGS:
            j = i + 1
            while j < n and tags[j][0].lower() in _NEG_OR_DEG:
                j += 1
            if j < n:
                nw, nt = tags[j]
                nwl = nw.lower()
                if nwl.endswith("ing") and nt not in _NOUN_TAGS:
                    tags[j] = (nw, "VBG")
                elif nwl.endswith("ed") and nt not in _NOUN_TAGS:
                    tags[j] = (nw, "VBN")
                elif nt in _NOUN_OR_VERB and not nw[0].isupper():
                    # Likely a predicate adjective misclassified as noun/verb
                    tags[j] = (nw, "JJ")

        # ------------------------------------------------------------------
        # Pattern 4 — Degree / intensifier adverb → next word is adjective
        # "very fast", "quite large", "incredibly smart", "too tired"
        # VBN is included because degree adverbs precede adjectives, not
        # passive participles (e.g. "very tired" → JJ, not VBN).
        # ------------------------------------------------------------------
        if w in _DEGREE_ADVERBS and i + 1 < n:
            nw, nt = tags[i + 1]
            if nt in _NOUN_TAGS or nt in ("VB", "VBZ", "VBD", "VBP", "VBN"):
                tags[i + 1] = (nw, "JJ")

        # ------------------------------------------------------------------
        # Pattern 5 — Article / possessive sandwich: DT + ? + Noun → ? is JJ
        # "the fast car", "a beautiful song", "my old friend", "the dark forest"
        # Also handles longer chains (DT + JJ + JJ + NN) by cascading.
        # The middle word can be any tag except determiners, pronouns, and
        # prepositions — noun-tagged words are included because NLTK sometimes
        # mislabels adjectives as nouns (e.g. "dark" → NN before "forest").
        # ------------------------------------------------------------------
        if tag in ("DT", "PRP$") and i + 2 < n:
            mw, mt = tags[i + 1]
            ew, et = tags[i + 2]
            if (et in _NOUN_TAGS
                    and mt not in ("DT", "CD", "PRP", "PRP$", "IN", "CC", "TO")):
                tags[i + 1] = (mw, "JJ")

        # ------------------------------------------------------------------
        # Pattern 6 — Parallel structure via CC: X + and/or/but + Y → Y ~ X
        # "fast AND furious" (JJ), "cats AND dogs" (NN), "run OR jump" (VB)
        # ------------------------------------------------------------------
        if tag == "CC" and i > 0 and i + 1 < n:
            pw, pt = tags[i - 1]
            nw, nt = tags[i + 1]
            if pt in _ADJ_TAGS and nt not in _ADJ_TAGS:
                tags[i + 1] = (nw, "JJ")
            elif pt in _NOUN_TAGS and nt not in _NOUN_OR_VERB:
                tags[i + 1] = (nw, pt)
            elif pt in _VERB_TAGS and nt not in _NOUN_OR_VERB:
                tags[i + 1] = (nw, pt)

        # ------------------------------------------------------------------
        # Pattern 7 — Cardinal number + word → word is plural noun (NNS)
        # "2 walls", "three dogs", "100 students"
        # ------------------------------------------------------------------
        if tag == "CD" and i + 1 < n:
            nw, nt = tags[i + 1]
            if nt in ("NN", "JJ", "VB", "VBZ"):
                tags[i + 1] = (nw, "NNS")

        # ------------------------------------------------------------------
        # Pattern 8 — Possessive clitic "'s" → next word is a noun
        # "John's car", "the dog's collar", "Mary's idea"
        # ------------------------------------------------------------------
        if w == "'s" and i + 1 < n:
            nw, nt = tags[i + 1]
            if nt in _AMBIG_TAGS | {"RB"}:
                tags[i + 1] = (nw, "NN")

        # ------------------------------------------------------------------
        # Pattern 9 — Preposition + word → word is likely a noun (NN)
        # "in the city", "by the river", "with great power"
        # Only fires when the word looks ambiguous (not already a noun).
        # ------------------------------------------------------------------
        if w in _PREPOSITIONS and tag == "IN" and i + 1 < n:
            nw, nt = tags[i + 1]
            # Skip article — look one step further
            if nt == "DT" and i + 2 < n:
                nw2, nt2 = tags[i + 2]
                if nt2 in _AMBIG_TAGS and not nw2[0].isupper():
                    tags[i + 2] = (nw2, "NN")
            elif nt in ("JJ", "VB", "VBZ") and not nw[0].isupper():
                tags[i + 1] = (nw, "NN")

        # ------------------------------------------------------------------
        # Pattern 10 — Wh-determiner / pronoun context
        # "what/which + Noun-like" → NN   ("which car", "what time")
        # "what/who + be-form"     → leave as-is (interrogative)
        # ------------------------------------------------------------------
        if w in ("what", "which") and tag in ("WP", "WDT") and i + 1 < n:
            nw, nt = tags[i + 1]
            if nt in _AMBIG_TAGS and nw.lower() not in _BE_FORMS:
                tags[i + 1] = (nw, "NN")

        # ------------------------------------------------------------------
        # Pattern 11 — Have-auxiliary → perfect participle (VBN)
        # "has eaten", "have finished", "had gone", "have been told"
        # Skip an optional negation between auxiliary and participle.
        # ------------------------------------------------------------------
        if w in _HAVE_FORMS and tag in _VERB_TAGS:
            j = i + 1
            if j < n and tags[j][0].lower() in _NEGATIONS:
                j += 1
            if j < n:
                nw, nt = tags[j]
                nwl = nw.lower()
                if (nwl.endswith("ed") or nwl.endswith("en")
                        or (nt in _VERB_TAGS and nwl not in _MODALS)):
                    tags[j] = (nw, "VBN")

        # ------------------------------------------------------------------
        # Pattern 12 — Do-support → base verb (VB)
        # "do know", "does she LIKE", "did not SEE"
        # Skip an optional pronoun subject (interrogative inversion) or negation.
        # ------------------------------------------------------------------
        if w in _DO_FORMS and tag in _VERB_TAGS:
            j = i + 1
            while j < n and (
                (tags[j][1] == "PRP" and tags[j][0].lower() in _SUBJ_PRONOUNS)
                or (tags[j][0].lower() in _NEGATIONS)
            ):
                j += 1
            if j < n:
                nw, nt = tags[j]
                if nt not in ("DT", "CD", "PRP", "PRP$", "IN", "CC", "WP", "WDT"):
                    tags[j] = (nw, "VB")

        # ------------------------------------------------------------------
        # Pattern 13 — "going to" future construction → next verb is VB
        # "I am going to LEAVE", "she is going to SING"
        # ------------------------------------------------------------------
        if w == "going" and i + 1 < n and tags[i + 1][0].lower() == "to":
            if i + 2 < n:
                nw, nt = tags[i + 2]
                if nt not in ("DT", "CD", "PRP", "PRP$", "IN", "CC"):
                    tags[i + 2] = (nw, "VB")

    # ------------------------------------------------------------------
    # Post-pass 19 — Passive voice (agentive): be + [advs] + -ed/-en word + "by" → VBN
    # "was WRITTEN by the president", "is KNOWN by everyone"
    # Scope is intentionally limited to explicit "by"-passives so that
    # ambiguous cases like "was tired" (adjective) are left unchanged.
    # Runs after the main loop so it can override JJ tags set by Pattern 4.
    # ------------------------------------------------------------------
    for i in range(n - 2):
        w0 = tags[i][0].lower()
        if w0 in _BE_FORMS and tags[i][1] in _VERB_TAGS:
            j = i + 1
            while j < n and tags[j][1] in ("RB", "RBR", "RBS"):
                j += 1
            if j < n - 1:
                pw, pt = tags[j]
                pwl = pw.lower()
                if (pwl.endswith("ed") or pwl.endswith("en")) and tags[j + 1][0].lower() == "by":
                    tags[j] = (pw, "VBN")

    return tags


def analyze_sentence(text: str) -> list[tuple[str, str]]:
    """
    Takes an English string and returns a list of (word, POS_TAG) tuples,
    with contextual pattern rules applied on top of the base tagger.

    Uses spaCy (en_core_web_sm) when available for higher accuracy and speed;
    falls back to NLTK's averaged perceptron tagger otherwise.

    Example: [('The', 'DT'), ('fast', 'JJ'), ('robot', 'NN')]
    """
    if _USE_SPACY:
        doc = _nlp(text)
        tagged = [(t.text, t.tag_) for t in doc if not t.is_space]
    else:
        tokens = _TOKENIZE.findall(text)
        raw = nltk.pos_tag(tokens) if tokens else []
        tagged = [(w, t if t else _guess_raw_morphology(w)) for w, t in raw]

    return _apply_context_rules(tagged)


def analyze_batch(texts: list[str]) -> list[list[tuple[str, str]]]:
    """
    Analyze multiple sentences efficiently.

    When spaCy is available, uses nlp.pipe() for batched throughput,
    which is significantly faster than calling analyze_sentence() in a loop.
    Falls back to sequential analyze_sentence() calls when spaCy is absent.

    Example::

        results = analyze_batch(["The cat sat.", "Dogs run fast."])
        for sent, pairs in zip(["The cat sat.", "Dogs run fast."], results):
            print(sent, pairs)
    """
    if _USE_SPACY:
        return [
            _apply_context_rules([(t.text, t.tag_) for t in doc if not t.is_space])
            for doc in _nlp.pipe(texts)
        ]
    return [analyze_sentence(t) for t in texts]

if __name__ == "__main__":
    sentences = [
        # Original test cases
        "The incredibly fast robot jumped over 2 walls!",
        "I want to eat a big red apple.",
        "She can not go to the store.",
        "The dog is very tired and hungry.",
        "John's car was completely destroyed.",
        "What time should we leave?",
        "He was running through the dark forest.",
        # Perfect aspect (Pattern 11)
        "She has eaten the largest piece.",
        "They have finished the hardest exercise.",
        "He had gone to the nearest hospital.",
        # Do-support (Pattern 12)
        "Do you know the fastest route?",
        "She does not understand the simplest rule.",
        "Did they see the brightest star?",
        # Going-to future (Pattern 13)
        "We are going to visit the oldest castle.",
        "She is going to start the newest project.",
        # Subordinating conjunctions (Pattern 14)
        "He stayed home because it was raining.",
        "Although she was tired, she kept working.",
        # Existential there (Pattern 15)
        "There is a problem with the system.",
        "There are many ways to solve this.",
        # Comparative (Patterns 16 & 17)
        "She is more intelligent than her brother.",
        "This road is less dangerous than the other.",
        # Passive voice (Post-pass 19)
        "The letter was written by the president.",
        "The bridge was built by engineers.",
        # Relative clause (Pattern 18)
        "The man who runs the company is smart.",
        "A book that describes history well.",
    ]

    backend = "spaCy" if _USE_SPACY else "NLTK"
    print(f"=== Single sentence analysis  [backend: {backend}] ===\n")
    for sentence in sentences:
        print(f"INPUT: '{sentence}'")
        for word, raw_tag in analyze_sentence(sentence):
            print(f"  {word:<20} -> {raw_tag}")
        print()

    print("=== Batch analysis demo ===\n")
    batch = ["The cat sat on the mat.", "Dogs run faster than cats."]
    for sent, result in zip(batch, analyze_batch(batch)):
        print(f"INPUT: '{sent}'")
        for word, tag in result:
            print(f"  {word:<20} -> {tag}")
        print()