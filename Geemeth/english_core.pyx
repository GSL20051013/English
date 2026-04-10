# SPDX-License-Identifier: LicenseRef-GSL20051013-english-noncommercial
# Copyright (c) 2024 GSL20051013
# See LICENSE for full terms. Commercial use requires a paid license.

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""Cython-accelerated hot paths for GSL20051013-english.

This module provides C-compiled replacements for the two most performance-
critical functions in english.py:

* ``apply_context_rules_cy`` – drop-in replacement for ``_apply_context_rules``
* ``guess_raw_morphology_cy`` – drop-in replacement for ``_guess_raw_morphology``

Compiled with Cython, tag comparisons become C integer comparisons and
set-membership tests become 64-bit bitmask operations, giving 3–6× speedup
on typical sentence lengths compared to the pure-Python implementation.
Both functions have identical semantics to their Python counterparts and work
with ``list[tuple[str, POSTag]]`` data (``POSTag`` is an ``IntEnum`` whose
values are plain C-compatible integers).
"""

from libc.stdint cimport uint64_t

# ---------------------------------------------------------------------------
# POSTag integer constants — must match the values in POSTag enum exactly.
# ---------------------------------------------------------------------------
DEF T_UNKNOWN = 0
DEF T_CC   = 1
DEF T_CD   = 2
DEF T_DT   = 3
DEF T_EX   = 4
DEF T_FW   = 5
DEF T_IN   = 6
DEF T_JJ   = 7
DEF T_JJR  = 8
DEF T_JJS  = 9
DEF T_LS   = 10
DEF T_MD   = 11
DEF T_NN   = 12
DEF T_NNS  = 13
DEF T_NNP  = 14
DEF T_NNPS = 15
DEF T_PDT  = 16
DEF T_POS  = 17
DEF T_PRP  = 18
DEF T_PRPS = 19
DEF T_RB   = 20
DEF T_RBR  = 21
DEF T_RBS  = 22
DEF T_RP   = 23
DEF T_SYM  = 24
DEF T_TO   = 25
DEF T_UH   = 26
DEF T_VB   = 27
DEF T_VBD  = 28
DEF T_VBG  = 29
DEF T_VBN  = 30
DEF T_VBP  = 31
DEF T_VBZ  = 32
DEF T_WDT  = 33
DEF T_WP   = 34
DEF T_WPS  = 35
DEF T_WRB  = 36
DEF T_COMMA  = 37
DEF T_PERIOD = 38
DEF T_COLON  = 39
DEF T_LQUOTE = 40
DEF T_RQUOTE = 41
DEF T_LRB    = 42
DEF T_RRB    = 43
DEF T_HYPH   = 44
DEF T_NFP    = 45

# ---------------------------------------------------------------------------
# Bitmask helpers — 64-bit masks covering tag values 0–63.
# Membership test: (mask >> tag_val) & 1  (true when tag_val is in the set).
# ---------------------------------------------------------------------------

# _NOUN_TAGS  = {NN, NNS, NNP, NNPS}
DEF NOUN_MASK = (1 << T_NN) | (1 << T_NNS) | (1 << T_NNP) | (1 << T_NNPS)

# _VERB_TAGS  = {VB, VBZ, VBD, VBG, VBN, VBP}
DEF VERB_MASK = ((1 << T_VB) | (1 << T_VBZ) | (1 << T_VBD) |
                 (1 << T_VBG) | (1 << T_VBN) | (1 << T_VBP))

# _ADJ_TAGS   = {JJ, JJR, JJS}
DEF ADJ_MASK  = (1 << T_JJ) | (1 << T_JJR) | (1 << T_JJS)

# _AMBIG_TAGS = {JJ, VB, VBZ, VBD}
DEF AMBIG_MASK = (1 << T_JJ) | (1 << T_VB) | (1 << T_VBZ) | (1 << T_VBD)

# _JJ_PROMOTABLE = _AMBIG_TAGS | {VBP, VBN}
DEF JJ_PROMO_MASK = AMBIG_MASK | (1 << T_VBP) | (1 << T_VBN)

# _JJ_PROMO_WITH_CORR = JJ_PROMO_MASK | {RB}
DEF JJ_PROMO_CORR_MASK = JJ_PROMO_MASK | (1 << T_RB)

# _NOUN_OR_VERB = _NOUN_TAGS | _VERB_TAGS
DEF NOUN_OR_VERB_MASK = NOUN_MASK | VERB_MASK

# {RB, RBR, RBS}
DEF ADV_MASK = (1 << T_RB) | (1 << T_RBR) | (1 << T_RBS)

# Pattern 1 / 2 / 13 exclusion set: {DT, CD, PRP, PRPS, WP, WPS, IN, CC, TO}
DEF EXCL1_MASK = ((1 << T_DT) | (1 << T_CD) | (1 << T_PRP) | (1 << T_PRPS) |
                  (1 << T_WP) | (1 << T_WPS) | (1 << T_IN) | (1 << T_CC) | (1 << T_TO))

# Pattern 12 exclusion: {DT, CD, PRP, PRPS, IN, CC, WP, WDT}
DEF EXCL12_MASK = ((1 << T_DT) | (1 << T_CD) | (1 << T_PRP) | (1 << T_PRPS) |
                   (1 << T_IN) | (1 << T_CC) | (1 << T_WP) | (1 << T_WDT))

# Pattern 2 exclusion: {DT, CD, PRP, PRPS, IN, CC}  (no WP/WDT)
DEF EXCL2_MASK = ((1 << T_DT) | (1 << T_CD) | (1 << T_PRP) | (1 << T_PRPS) |
                  (1 << T_IN) | (1 << T_CC))

# Pattern 5 middle-word exclusion: {DT, CD, PRP, PRPS, IN, CC, TO}
DEF EXCL5_MASK = ((1 << T_DT) | (1 << T_CD) | (1 << T_PRP) | (1 << T_PRPS) |
                  (1 << T_IN) | (1 << T_CC) | (1 << T_TO))

# Pattern 6 target-exclusion: {PRP, PRPS, DT, WP, WDT, IN}
DEF EXCL6_MASK = ((1 << T_PRP) | (1 << T_PRPS) | (1 << T_DT) |
                  (1 << T_WP) | (1 << T_WDT) | (1 << T_IN))

# {VB, VBZ, VBD, VBP}  — simple present/past without -ing/-en forms
DEF SIMP_VERB_MASK = (1 << T_VB) | (1 << T_VBZ) | (1 << T_VBD) | (1 << T_VBP)

# Punct tags stripped in post-processing
DEF PUNCT_MASK = ((1 << T_COMMA) | (1 << T_PERIOD) | (1 << T_COLON) |
                  (1 << T_LQUOTE) | (1 << T_RQUOTE) | (1 << T_LRB) |
                  (1 << T_RRB) | (1 << T_HYPH) | (1 << T_NFP) | (1 << T_SYM))


cdef inline bint in_mask(int tag, uint64_t mask) noexcept nogil:
    """Return True when *tag* is set in *mask*.

    *tag* should be in the range 1–63 (named tags); 0 (UNKNOWN) is always
    rejected, as UNKNOWN is never a member of any tag-set bitmask.
    """
    if tag <= 0 or tag > 63:
        return 0
    return (mask >> tag) & 1


# ---------------------------------------------------------------------------
# Word-level vocabulary sets (Python frozensets, used for word string tests).
# These are imported from the parent module at runtime; defined here as module-
# level so the function can reference them as module globals.
# ---------------------------------------------------------------------------
_MODALS = None
_HAVE_FORMS = None
_DO_FORMS = None
_BE_FORMS = None
_DEGREE_ADVERBS = None
_NEGATIONS = None
_PREPOSITIONS = None
_SUBORD_CONJ = None
_SUBJ_PRONOUNS = None
_HAVE_TAG = None
_DO_TAG = None
_NEG_OR_DEG = None
_CLAUSE_BOUNDARY = None
_POSTag = None   # the POSTag enum class


def _init_cy_globals(
    modals, have_forms, do_forms, be_forms,
    degree_adverbs, negations, prepositions, subord_conj,
    subj_pronouns, have_tag, do_tag, neg_or_deg,
    clause_boundary, pos_tag_cls,
):
    """Inject shared vocabulary sets from english.py into this module."""
    global _MODALS, _HAVE_FORMS, _DO_FORMS, _BE_FORMS
    global _DEGREE_ADVERBS, _NEGATIONS, _PREPOSITIONS, _SUBORD_CONJ
    global _SUBJ_PRONOUNS, _HAVE_TAG, _DO_TAG, _NEG_OR_DEG
    global _CLAUSE_BOUNDARY, _POSTag
    _MODALS        = modals
    _HAVE_FORMS    = have_forms
    _DO_FORMS      = do_forms
    _BE_FORMS      = be_forms
    _DEGREE_ADVERBS = degree_adverbs
    _NEGATIONS     = negations
    _PREPOSITIONS  = prepositions
    _SUBORD_CONJ   = subord_conj
    _SUBJ_PRONOUNS = subj_pronouns
    _HAVE_TAG      = have_tag
    _DO_TAG        = do_tag
    _NEG_OR_DEG    = neg_or_deg
    _CLAUSE_BOUNDARY = clause_boundary
    _POSTag        = pos_tag_cls


def apply_context_rules_cy(list tagged):
    """Cython-compiled context-rules pass.

    Drop-in replacement for ``_apply_context_rules`` in english.py.
    Accepts and returns ``list[tuple[str, POSTag]]``.
    All tag comparisons are performed as C integer operations.
    """
    cdef int n = len(tagged)
    cdef int i, j, k
    cdef int tag_val, nt_val, pt_val, et_val, mt_val, item_t_val
    cdef int anchor_i, target_i, k2
    cdef bint comma_skipped, in_perfect, is_correlative
    cdef str word, w, nw, pw, nw2, item_w, skip_w
    cdef uint64_t jj_set_mask
    cdef object tag_obj, nt_obj, pt_obj   # POSTag enum objects

    cdef list tags = list(tagged)

    _T = _POSTag   # local alias

    for i in range(n):
        tup = tags[i]
        word = tup[0]
        tag_obj = tup[1]
        tag_val = <int>tag_obj
        w = word.lower()

        # ---- Self-correction: degree adverbs → RB ----
        if w in _DEGREE_ADVERBS and not in_mask(tag_val, ADV_MASK):
            tag_val = T_RB
            tags[i] = (word, _T(T_RB))

        # ---- Self-correction: modals → MD ----
        if w in _MODALS and tag_val != T_MD:
            tag_val = T_MD
            tags[i] = (word, _T(T_MD))

        # ---- Self-correction: have-forms → correct verb form ----
        if w in _HAVE_FORMS and not in_mask(tag_val, VERB_MASK):
            new_tag = _HAVE_TAG.get(w, _T(T_VBZ))
            tags[i] = (word, new_tag)
            tag_val = <int>new_tag

        # ---- Self-correction: do-forms → correct verb form ----
        if w in _DO_FORMS and not in_mask(tag_val, VERB_MASK):
            new_tag = _DO_TAG.get(w, _T(T_VBZ))
            tags[i] = (word, new_tag)
            tag_val = <int>new_tag

        # ---- Pattern 14: subordinating conj → IN ----
        if w in _SUBORD_CONJ and tag_val != T_IN and tag_val != T_RB:
            tag_val = T_IN
            tags[i] = (word, _T(T_IN))

        # ---- Pattern 15: existential "there" → EX ----
        if w == "there" and tag_val != T_EX and i + 1 < n:
            if (<str>(tags[i + 1][0])).lower() in _BE_FORMS:
                tag_val = T_EX
                tags[i] = (word, _T(T_EX))

        # ---- Pattern 16: "more"/"less" → RBR; next word → JJ ----
        if w == "more" or w == "less":
            if tag_val != T_RBR and tag_val != T_JJR:
                tag_val = T_RBR
                tags[i] = (word, _T(T_RBR))
            if i + 1 < n:
                nxt = tags[i + 1]
                nw = <str>nxt[0]
                nt_val = <int>nxt[1]
                if (nw.lower() != "than"
                        and not in_mask(nt_val, ADJ_MASK)
                        and not in_mask(nt_val, ADV_MASK)):
                    tags[i + 1] = (nw, _T(T_JJ))

        # ---- Pattern 17: JJR/RBR + "than" → IN ----
        if (tag_val == T_JJR or tag_val == T_RBR) and i + 1 < n:
            nxt = tags[i + 1]
            if (<str>nxt[0]).lower() == "than":
                tags[i + 1] = (nxt[0], _T(T_IN))

        # ---- Pattern 18: relative pronoun + ambiguous word → verb ----
        if (w == "who" or w == "which" or w == "that") and i + 1 < n:
            if tag_val == T_WP or tag_val == T_WDT or tag_val == T_IN:
                nxt = tags[i + 1]
                nw = <str>nxt[0]
                nt_val = <int>nxt[1]
                nwl = nw.lower()
                if nwl not in _BE_FORMS and nwl not in _MODALS:
                    if (nwl.endswith("s") and not nwl.endswith("ss")
                            and nt_val == T_NNS):
                        tags[i + 1] = (nw, _T(T_VBZ))
                    elif (nt_val == T_JJ or nt_val == T_VB) and not nwl.endswith("s"):
                        tags[i + 1] = (nw, _T(T_VBP))

        # ---- Pattern 1: infinitive "to" → next is VB ----
        if w == "to" and tag_val == T_TO and i + 1 < n:
            nxt = tags[i + 1]
            nw = <str>nxt[0]
            nt_val = <int>nxt[1]
            if not in_mask(nt_val, EXCL1_MASK):
                tags[i + 1] = (nw, _T(T_VB))

        # ---- Pattern 2: modal → next content word is VB ----
        if w in _MODALS and tag_val == T_MD:
            j = i + 1
            if j < n and (<str>(tags[j][0])).lower() in _NEGATIONS:
                j += 1
            if j < n and <str>(tags[j][0]) not in _CLAUSE_BOUNDARY:
                nxt = tags[j]
                nw = <str>nxt[0]
                nt_val = <int>nxt[1]
                if not in_mask(nt_val, EXCL2_MASK):
                    tags[j] = (nw, _T(T_VB))

        # ---- Pattern 3: be-verb → predict complement ----
        if w in _BE_FORMS and in_mask(tag_val, VERB_MASK):
            # Check for perfect-aspect context
            in_perfect = False
            for k in range(max(0, i - 3), i):
                if <str>(tags[k][0]) not in _CLAUSE_BOUNDARY:
                    if (<str>(tags[k][0])).lower() in _HAVE_FORMS:
                        in_perfect = True
                        break
            j = i + 1
            # Skip negations, degree adverbs, RB-family adverbs
            while j < n:
                nxt_w = (<str>(tags[j][0])).lower()
                nxt_t = <int>(tags[j][1])
                if (nxt_w in _NEG_OR_DEG or
                        (in_mask(nxt_t, ADV_MASK) and
                         <str>(tags[j][0]) not in _CLAUSE_BOUNDARY)):
                    j += 1
                else:
                    break
            if j < n and <str>(tags[j][0]) not in _CLAUSE_BOUNDARY:
                nxt = tags[j]
                nw = <str>nxt[0]
                nt_val = <int>nxt[1]
                nwl = nw.lower()
                if nwl.endswith("ing") and not in_mask(nt_val, NOUN_MASK):
                    tags[j] = (nw, _T(T_VBG))
                elif (nwl.endswith("ed") and not in_mask(nt_val, NOUN_MASK)
                        and nt_val != T_JJ):
                    tags[j] = (nw, _T(T_VBN))
                elif (not in_perfect
                        and (in_mask(nt_val, NOUN_MASK) or
                             in_mask(nt_val, SIMP_VERB_MASK))
                        and not nw[0].isupper()):
                    tags[j] = (nw, _T(T_JJ))

        # ---- Pattern 4: degree adverb → next word is JJ ----
        if w in _DEGREE_ADVERBS and i + 1 < n:
            nxt = tags[i + 1]
            nw = <str>nxt[0]
            nt_val = <int>nxt[1]
            if in_mask(nt_val, NOUN_MASK) or in_mask(nt_val, SIMP_VERB_MASK):
                tags[i + 1] = (nw, _T(T_JJ))

        # ---- Pattern 5: DT/PRPS + ? + Noun → ? is JJ ----
        if (tag_val == T_DT or tag_val == T_PRPS) and i + 2 < n:
            mid = tags[i + 1]
            end = tags[i + 2]
            mt_val = <int>mid[1]
            et_val = <int>end[1]
            mw = <str>mid[0]
            if in_mask(et_val, NOUN_MASK) and not in_mask(mt_val, EXCL5_MASK):
                tags[i + 1] = (mw, _T(T_JJ))

        # ---- Pattern 6: CC parallel structure ----
        if tag_val == T_CC and i > 0 and i + 1 < n:
            anchor_i = i - 1
            comma_skipped = False
            if anchor_i >= 0 and <str>(tags[anchor_i][0]) == ",":
                anchor_i -= 1
                comma_skipped = True
            if anchor_i >= 0:
                anc = tags[anchor_i]
                pw = <str>anc[0]
                pt_val = <int>anc[1]

                target_i = i + 1
                skip_w = (<str>(tags[target_i][0])).lower()
                is_correlative = (skip_w == "also" or skip_w == "then"
                                  or skip_w == "yet")
                if is_correlative and i + 2 < n:
                    target_i = i + 2
                tgt = tags[target_i]
                nw = <str>tgt[0]
                nt_val = <int>tgt[1]

                if is_correlative:
                    jj_set_mask = JJ_PROMO_CORR_MASK
                else:
                    jj_set_mask = JJ_PROMO_MASK

                if in_mask(pt_val, ADJ_MASK) and in_mask(nt_val, jj_set_mask):
                    tags[target_i] = (nw, _T(T_JJ))
                elif (in_mask(pt_val, NOUN_MASK) and not in_mask(nt_val, NOUN_MASK)
                        and not in_mask(nt_val, VERB_MASK)):
                    tags[target_i] = (nw, anc[1])
                elif (in_mask(pt_val, VERB_MASK)
                        and not in_mask(nt_val, NOUN_OR_VERB_MASK)
                        and not in_mask(nt_val, EXCL6_MASK)
                        and not comma_skipped):
                    tags[target_i] = (nw, anc[1])

                # Backward comma-list unification
                k2 = anchor_i - 1
                if k2 >= 0 and <str>(tags[k2][0]) == ",":
                    k2 -= 1
                    if k2 >= 0 and <str>(tags[k2][0]) not in _CLAUSE_BOUNDARY:
                        bl = tags[k2]
                        item_w = <str>bl[0]
                        item_t_val = <int>bl[1]
                        if in_mask(pt_val, VERB_MASK) and in_mask(item_t_val, NOUN_MASK):
                            tags[k2] = (item_w, anc[1])
                        elif (in_mask(pt_val, NOUN_MASK)
                                and not in_mask(item_t_val, NOUN_MASK)
                                and not in_mask(item_t_val, VERB_MASK)):
                            tags[k2] = (item_w, anc[1])
                        elif in_mask(pt_val, ADJ_MASK) and in_mask(item_t_val, NOUN_MASK):
                            tags[k2] = (item_w, _T(T_JJ))

        # ---- Pattern 7: CD + word → NNS ----
        if tag_val == T_CD and i + 1 < n:
            nxt = tags[i + 1]
            nw = <str>nxt[0]
            nt_val = <int>nxt[1]
            if nt_val == T_NN or nt_val == T_JJ or nt_val == T_VB or nt_val == T_VBZ:
                tags[i + 1] = (nw, _T(T_NNS))

        # ---- Pattern 8: "'s" → next is NN ----
        if w == "'s" and i + 1 < n:
            nxt = tags[i + 1]
            nw = <str>nxt[0]
            nt_val = <int>nxt[1]
            if in_mask(nt_val, AMBIG_MASK) or nt_val == T_RB:
                tags[i + 1] = (nw, _T(T_NN))

        # ---- Pattern 9: preposition + word → NN (ambiguous words) ----
        if w in _PREPOSITIONS and tag_val == T_IN and i + 1 < n:
            nxt = tags[i + 1]
            nw = <str>nxt[0]
            nt_val = <int>nxt[1]
            if nt_val == T_DT and i + 2 < n:
                nxt2 = tags[i + 2]
                nw2 = <str>nxt2[0]
                nt2_val = <int>nxt2[1]
                if in_mask(nt2_val, AMBIG_MASK) and not nw2[0].isupper():
                    tags[i + 2] = (nw2, _T(T_NN))
            elif (nt_val == T_JJ or nt_val == T_VB or nt_val == T_VBZ) and not nw[0].isupper():
                tags[i + 1] = (nw, _T(T_NN))

        # ---- Pattern 10: wh-det/pron + ambiguous → NN ----
        if (w == "what" or w == "which") and i + 1 < n:
            if tag_val == T_WP or tag_val == T_WDT:
                nxt = tags[i + 1]
                nw = <str>nxt[0]
                nt_val = <int>nxt[1]
                if in_mask(nt_val, AMBIG_MASK) and nw.lower() not in _BE_FORMS:
                    tags[i + 1] = (nw, _T(T_NN))

        # ---- Pattern 11: have-auxiliary → VBN ----
        if w in _HAVE_FORMS and in_mask(tag_val, VERB_MASK):
            j = i + 1
            if j < n and (<str>(tags[j][0])).lower() in _NEGATIONS:
                j += 1
            if j < n:
                nxt = tags[j]
                nw = <str>nxt[0]
                nt_val = <int>nxt[1]
                nwl = nw.lower()
                if (nwl.endswith("ed") or nwl.endswith("en")
                        or (in_mask(nt_val, VERB_MASK) and nwl not in _MODALS)):
                    tags[j] = (nw, _T(T_VBN))

        # ---- Pattern 12: do-support → VB ----
        if w in _DO_FORMS and in_mask(tag_val, VERB_MASK):
            j = i + 1
            while j < n and <str>(tags[j][0]) not in _CLAUSE_BOUNDARY:
                tj_val = <int>(tags[j][1])
                tj_w = (<str>(tags[j][0])).lower()
                if (tj_val == T_PRP and tj_w in _SUBJ_PRONOUNS) or tj_w in _NEGATIONS:
                    j += 1
                else:
                    break
            if j < n and <str>(tags[j][0]) not in _CLAUSE_BOUNDARY:
                nxt = tags[j]
                nw = <str>nxt[0]
                nt_val = <int>nxt[1]
                if not in_mask(nt_val, EXCL12_MASK):
                    tags[j] = (nw, _T(T_VB))

        # ---- Pattern 13: "going to" future → VB ----
        if w == "going" and i + 1 < n:
            if (<str>(tags[i + 1][0])).lower() == "to":
                if i + 2 < n and <str>(tags[i + 2][0]) not in _CLAUSE_BOUNDARY:
                    nxt = tags[i + 2]
                    nw = <str>nxt[0]
                    nt_val = <int>nxt[1]
                    if not in_mask(nt_val, EXCL2_MASK):
                        tags[i + 2] = (nw, _T(T_VB))

        # ---- Pattern 20: "most"/"least" → RBS ----
        if (w == "most" or w == "least") and tag_val != T_RBS and tag_val != T_JJS:
            if i + 1 < n:
                nxt = tags[i + 1]
                nw = <str>nxt[0]
                nt_val = <int>nxt[1]
                if in_mask(nt_val, ADJ_MASK) or in_mask(nt_val, SIMP_VERB_MASK):
                    tag_val = T_RBS
                    tags[i] = (word, _T(T_RBS))
                    if not in_mask(nt_val, ADJ_MASK):
                        tags[i + 1] = (nw, _T(T_JJS))
                elif nt_val == T_RB or nt_val == T_RBR:
                    tag_val = T_RBS
                    tags[i] = (word, _T(T_RBS))
                    tags[i + 1] = (nw, _T(T_RBS))

        # ---- Pattern 21: sentence-initial gerund mis-tagged as NN/NNS ----
        if i == 0 and word.lower().endswith("ing") and (tag_val == T_NN or tag_val == T_NNS):
            tag_val = T_VBG
            tags[i] = (word, _T(T_VBG))

    # ---- Post-pass 19: agentive passive → VBN ----
    for i in range(n - 2):
        w0 = (<str>(tags[i][0])).lower()
        if w0 in _BE_FORMS and in_mask(<int>(tags[i][1]), VERB_MASK):
            j = i + 1
            while j < n and in_mask(<int>(tags[j][1]), ADV_MASK):
                j += 1
            if j < n - 1:
                nxt = tags[j]
                pw = (<str>nxt[0]).lower()
                if (pw.endswith("ed") or pw.endswith("en")) and (<str>(tags[j + 1][0])).lower() == "by":
                    tags[j] = (nxt[0], _T(T_VBN))

    return tags


def guess_raw_morphology_cy(str word, object POSTag_cls):
    """Cython-compiled morphology guesser.

    Drop-in replacement for ``_guess_raw_morphology`` in english.py.
    Returns a :class:`POSTag` value.  Handles numbers, symbols, common
    closed-class words, and suffix patterns using Cython-compiled C code.
    """
    cdef str w

    _T = POSTag_cls

    # 1. Numbers & Symbols
    if _is_number(word):
        return _T(T_CD)
    if not _is_alpha_apos(word):
        return _T(T_SYM)

    w = word.lower()

    # 2. Closed-class function words
    if w == "the" or w == "a" or w == "an":            return _T(T_DT)
    if w == "and" or w == "or" or w == "but":           return _T(T_CC)
    if w == "nor" or w == "yet" or w == "for" or w == "so": return _T(T_CC)
    if w == "i" or w == "he" or w == "she":             return _T(T_PRP)
    if w == "it" or w == "we" or w == "they" or w == "you": return _T(T_PRP)
    if w == "my" or w == "his" or w == "her":           return _T(T_PRPS)
    if w == "its" or w == "our" or w == "their" or w == "your": return _T(T_PRPS)
    if w == "this" or w == "that" or w == "these" or w == "those": return _T(T_DT)
    if w == "am" or w == "is":                          return _T(T_VBZ)
    if w == "are":                                      return _T(T_VBP)
    if w == "was" or w == "were":                       return _T(T_VBD)
    if w == "be":                                       return _T(T_VB)
    if w == "been":                                     return _T(T_VBN)
    if w == "being":                                    return _T(T_VBG)
    if w == "has":                                      return _T(T_VBZ)
    if w == "have":                                     return _T(T_VBP)
    if w == "had":                                      return _T(T_VBD)
    if w == "having":                                   return _T(T_VBG)
    if w == "does":                                     return _T(T_VBZ)
    if w == "do":                                       return _T(T_VBP)
    if w == "did":                                      return _T(T_VBD)
    if w in _MODALS:                                    return _T(T_MD)
    if w in _PREPOSITIONS:                              return _T(T_IN)
    if w in _SUBORD_CONJ:                               return _T(T_IN)
    if w == "not" or w == "never":                      return _T(T_RB)
    if w in _DEGREE_ADVERBS:                            return _T(T_RB)
    if w == "more" or w == "less":                      return _T(T_RBR)
    if w == "most" or w == "least":                     return _T(T_RBS)
    if w == "there":                                    return _T(T_EX)

    # 3. Suffix rules (longest suffix first)
    if w.endswith("ize") or w.endswith("ise") or w.endswith("ify"):
        return _T(T_VB)
    if w.endswith("ism") or w.endswith("ist"):          return _T(T_NN)
    if w.endswith("ship"):                              return _T(T_NN)
    if w.endswith("hood"):                              return _T(T_NN)
    if w.endswith("dom"):                               return _T(T_NN)
    if w.endswith("tion") or w.endswith("sion"):        return _T(T_NN)
    if w.endswith("ness") or w.endswith("ment") or w.endswith("ity"):
        return _T(T_NN)
    if w.endswith("ing"):                               return _T(T_VBG)
    if w.endswith("est"):                               return _T(T_JJS)
    if w.endswith("ed"):                                return _T(T_VBD)
    if w.endswith("ish"):                               return _T(T_JJ)
    if (w.endswith("ful") or w.endswith("less") or w.endswith("ous") or
            w.endswith("ive") or w.endswith("able") or w.endswith("ible") or
            w.endswith("al") or w.endswith("ic")):
        return _T(T_JJ)
    if w.endswith("ly"):                                return _T(T_RB)

    if word[0].isupper():                               return _T(T_NNP)
    return _T(T_NN)


cdef inline bint _is_number(str s) noexcept:
    """Return True when s looks like an integer or decimal number."""
    cdef int i = 0
    cdef int n = len(s)
    if n == 0:
        return False
    if s[0] == '-':
        i = 1
    if i >= n:
        return False
    while i < n and '0' <= s[i] <= '9':
        i += 1
    if i < n and s[i] == '.':
        i += 1
        while i < n and '0' <= s[i] <= '9':
            i += 1
    return i == n


cdef inline bint _is_alpha_apos(str s) noexcept:
    """Return True when s consists only of letters and apostrophes."""
    cdef int i
    cdef str c
    for i in range(len(s)):
        c = s[i]
        if not (c.isalpha() or c == "'"):
            return False
    return True
