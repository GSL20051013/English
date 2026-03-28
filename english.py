import re
import nltk

for _d in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"): 
    nltk.download(_d, quiet=True)

_TOKENIZE = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[^\s]")

# ---------------------------------------------------------------------------
# Vocabulary sets used by the context-rule engine
# ---------------------------------------------------------------------------
_MODALS = frozenset({
    "can", "will", "shall", "may", "might",
    "could", "would", "should", "must",
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

_NOUN_TAGS = frozenset({"NN", "NNS", "NNP", "NNPS"})
_VERB_TAGS = frozenset({"VB", "VBZ", "VBD", "VBG", "VBN", "VBP"})
_ADJ_TAGS  = frozenset({"JJ", "JJR", "JJS"})
_AMBIG_TAGS = frozenset({"JJ", "VB", "VBZ", "VBD"})   # tags that can be misclassified

# Pre-computed unions for use inside the hot loop
_NEG_OR_DEG  = _NEGATIONS | _DEGREE_ADVERBS
_NOUN_OR_VERB = _NOUN_TAGS | _VERB_TAGS


def _guess_raw_morphology(word: str) -> str:
    """Fallback guesser that returns raw Penn Treebank-style tags."""
    # 1. Numbers & Symbols
    if re.match(r'^-?\d+(\.\d+)?$', word): return 'CD'   # Cardinal Digit
    if not re.match(r"^[A-Za-z]+(?:'[A-Za-z]+)*$", word): return 'SYM'  # Symbol
    
    w = word.lower()
    # 2. Suffix Rules
    if w.endswith('ly'): return 'RB'                     # Adverb
    if w.endswith('ing'): return 'VBG'                   # Verb, gerund
    if w.endswith('ed'): return 'VBD'                    # Verb, past tense
    if w.endswith(('tion', 'sion', 'ness', 'ment', 'ity')): return 'NN' # Noun, singular
    if w.endswith(('ful', 'less', 'ous', 'ive', 'able', 'al')): return 'JJ' # Adjective
    
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

    return tags


def analyze_sentence(text: str) -> list[tuple[str, str]]:
    """
    Takes an English string and returns a list of (word, RAW_NLTK_TAG) tuples,
    with contextual pattern rules applied on top of NLTK's base tagging.
    Example: [('The', 'DT'), ('fast', 'JJ'), ('robot', 'NN')]
    """
    tokens = _TOKENIZE.findall(text)
    
    tagged_tokens = nltk.pos_tag(tokens)
    
    result = []
    for word, pos_tag in tagged_tokens:
        if not pos_tag:
            pos_tag = _guess_raw_morphology(word)
        result.append((word, pos_tag))

    # Apply contextual pattern rules to refine the NLTK output
    result = _apply_context_rules(result)

    return result

if __name__ == "__main__":
    sentences = [
        "The incredibly fast robot jumped over 2 walls!",
        "I want to eat a big red apple.",
        "She can not go to the store.",
        "The dog is very tired and hungry.",
        "John's car was completely destroyed.",
        "What time should we leave?",
        "He was running through the dark forest.",
    ]

    for sentence in sentences:
        print(f"INPUT: '{sentence}'")
        for word, raw_tag in analyze_sentence(sentence):
            print(f"  {word:<15} -> {raw_tag}")
        print()