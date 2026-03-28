import re
import nltk

for _d in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"): 
    nltk.download(_d, quiet=True)

_TOKENIZE = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[^\s]")

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


def analyze_sentence(text: str) -> list[tuple[str, str]]:
    """
    Takes an English string and returns a list of (word, RAW_NLTK_TAG) tuples.
    Example: [('The', 'DT'), ('fast', 'JJ'), ('robot', 'NN')]
    """
    tokens = _TOKENIZE.findall(text)
    
    tagged_tokens = nltk.pos_tag(tokens)
    
    result = []
    
    for word, pos_tag in tagged_tokens:
        if not pos_tag:
            pos_tag = _guess_raw_morphology(word)
            
        result.append((word, pos_tag))
        
    return result

if __name__ == "__main__":
    test_sentence = "The incredibly fast robot jumped over 2 walls!"
    
    print(f"INPUT: '{test_sentence}'\n")
    
    parsed_data = analyze_sentence(test_sentence)
    
    for word, raw_tag in parsed_data:
        print(f"{word:<15} -> {raw_tag}")