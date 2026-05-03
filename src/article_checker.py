"""
Dedicated article agreement checker for ESL error detection.
Designed to run before the binary DistilBERT model as a fast pre-filter.
Returns a list of (error_type, span) pairs found in the sentence.
"""

import re
import spacy

nlp = spacy.load("en_core_web_sm")

# ── error labels ───────────────────────────────────────────────────────────────

A_AN_ERROR    = "Article error (a/an)"
UNNECESSARY   = "Unnecessary article"
OMISSION      = "Article omission"
WRONG_ARTICLE = "Wrong article (a/an vs the)"

# ── phonetic edge cases ────────────────────────────────────────────────────────

# start with vowel letter but consonant sound → use "a" not "an"
A_NOT_AN = {
    'university', 'universities', 'union', 'unions', 'unit', 'units',
    'unique', 'uniform', 'uniforms', 'use', 'user', 'users', 'useful',
    'usual', 'usually', 'usage', 'utensil', 'utensils', 'united', 'unity',
    'universe', 'utility', 'utilities', 'ukulele', 'upon',
    'european', 'europe', 'euphemism', 'eulogy', 'eunuch', 'euphoria',
    'euphoric', 'eureka', 'euthanasia',
    'one', 'once',
    'ewe', 'ewes',
}

# start with consonant letter but vowel sound → use "an" not "a"
AN_NOT_A = {
    'hour', 'hours', 'hourly', 'honest', 'honestly', 'honesty',
    'honor', 'honors', 'honour', 'honours', 'honorable', 'honourable',
    'heir', 'heiress', 'heirs', 'herb', 'herbs',
}

# letters whose names start with a vowel sound → "an FBI", "an MP", "an NGO"
VOWEL_SOUND_LETTERS = set('AEFHILMNORSX')

# ── noun lists ─────────────────────────────────────────────────────────────────

UNCOUNTABLE_NOUNS = {
    'information', 'advice', 'furniture', 'equipment', 'luggage', 'baggage',
    'homework', 'knowledge', 'evidence', 'research', 'music', 'money',
    'news', 'weather', 'traffic', 'progress', 'work', 'accommodation',
    'bread', 'butter', 'cheese', 'milk', 'rice', 'water', 'hair',
    'food', 'fruit', 'meat', 'sand', 'air', 'grass', 'wood', 'ice',
    'electricity', 'energy', 'software', 'hardware', 'vocabulary',
    'tourism', 'transportation', 'education', 'happiness', 'sadness',
    'anger', 'love', 'peace', 'violence', 'beauty', 'darkness',
    'nature', 'society', 'freedom', 'justice', 'history',
}

# singular countable nouns that almost always need an article — high confidence only
ARTICLE_REQUIRED_NOUNS = {
    'cat', 'dog', 'car', 'house', 'book', 'table', 'chair', 'phone',
    'computer', 'problem', 'question', 'answer', 'idea', 'plan', 'job',
    'meeting', 'lesson', 'teacher', 'student', 'doctor', 'friend',
    'message', 'letter', 'mistake', 'error', 'decision', 'choice',
    'chance', 'bag', 'box', 'bottle', 'cup', 'glass', 'plate',
    'shirt', 'dress', 'shoe', 'hat', 'coat', 'jacket', 'window', 'door',
    'key', 'map', 'ticket', 'receipt', 'contract', 'document', 'report',
    'test', 'exam', 'interview', 'appointment', 'reservation',
}

# nouns that appear without articles in common idiomatic phrases
ARTICLE_EXEMPT = {
    'home', 'work', 'school', 'college', 'university', 'hospital', 'church',
    'bed', 'town', 'sea', 'prison', 'court', 'class', 'lunch', 'breakfast',
    'dinner', 'time', 'life', 'love', 'nature', 'society', 'language',
    'help', 'fun', 'trouble', 'difficulty', 'permission', 'control',
    'bus', 'car', 'train', 'plane', 'foot', 'bike',  # by + transport
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _starts_with_vowel_sound(word: str) -> bool:
    """True if word starts with a vowel sound (for a/an selection)."""
    lower = word.lower()

    if lower in A_NOT_AN:
        return False
    if lower in AN_NOT_A:
        return True

    # all-caps abbreviation — check if first letter name starts with a vowel sound
    if word.isupper() and len(word) >= 2:
        return word[0] in VOWEL_SOUND_LETTERS

    return bool(re.match(r'^[aeiou]', lower))


def _has_determiner(tok) -> bool:
    """True if a noun token has any determiner, possessive, or numeral child."""
    return any(c.dep_ in ('det', 'poss', 'nummod') for c in tok.children)


def _preceding_token_is_det(tok, doc) -> bool:
    """True if the token immediately before tok is a determiner."""
    return tok.i > 0 and doc[tok.i - 1].dep_ == 'det'


# ── individual checks ──────────────────────────────────────────────────────────

def check_a_an(doc) -> list[tuple[str, str]]:
    """Returns (type, article) pairs for wrong a/an choices."""
    results = []
    tokens = list(doc)
    for i, tok in enumerate(tokens[:-1]):
        if tok.text.lower() not in ('a', 'an'):
            continue
        next_word = tokens[i + 1].text
        # skip uncountable nouns — wrong article type, not wrong a/an choice
        if next_word.lower() in UNCOUNTABLE_NOUNS:
            continue
        vowel_sound = _starts_with_vowel_sound(next_word)
        if tok.text.lower() == 'a' and vowel_sound:
            results.append((A_AN_ERROR, tok.text))
        elif tok.text.lower() == 'an' and not vowel_sound:
            results.append((A_AN_ERROR, tok.text))
    return results


def check_unnecessary_article(doc) -> list[tuple[str, str]]:
    """Returns (type, article) pairs for articles placed before uncountable or proper nouns."""
    results = []
    for tok in doc:
        if tok.dep_ != 'det':
            continue
        head = tok.head
        if tok.text.lower() in ('a', 'an') and head.lemma_.lower() in UNCOUNTABLE_NOUNS:
            results.append((UNNECESSARY, tok.text))
        elif tok.text.lower() in ('a', 'an') and head.pos_ == 'PROPN':
            results.append((UNNECESSARY, tok.text))
    return results


def check_article_omission(doc) -> list[tuple[str, str]]:
    """
    Returns (type, noun) pairs for singular countable nouns missing a determiner.
    Conservative — only fires on the ARTICLE_REQUIRED_NOUNS whitelist.
    """
    results = []
    for tok in doc:
        if (
            tok.tag_ == 'NN'
            and tok.pos_ != 'PRON'
            and tok.lemma_.lower() in ARTICLE_REQUIRED_NOUNS
            and tok.lemma_.lower() not in ARTICLE_EXEMPT
            and tok.dep_ in ('nsubj', 'nsubjpass', 'dobj', 'pobj', 'attr')
            and not _has_determiner(tok)
            and not _preceding_token_is_det(tok, doc)
        ):
            results.append((OMISSION, tok.text))
    return results


def check_wrong_article(doc) -> list[tuple[str, str]]:
    """
    Returns (type, article) pairs for a/an used where the is clearly needed —
    when the noun has a relative clause or superlative modifier.
    """
    results = []
    for tok in doc:
        if tok.dep_ != 'det' or tok.text.lower() not in ('a', 'an'):
            continue
        head = tok.head
        if head.lemma_.lower() in UNCOUNTABLE_NOUNS:
            continue  # unnecessary article check handles these
        has_relcl = any(c.dep_ == 'relcl' for c in head.children)
        is_superl  = any(c.tag_ == 'JJS'   for c in head.children)
        if has_relcl or is_superl:
            results.append((WRONG_ARTICLE, tok.text))
        elif any(c.tag_ == 'JJS' and c.dep_ == 'amod' for c in head.children):
            results.append((WRONG_ARTICLE, tok.text))
    return results


# ── main entry point ───────────────────────────────────────────────────────────

CHECKS = [
    check_a_an,
    check_unnecessary_article,
    check_article_omission,
    check_wrong_article,
]

def check_articles(sentence: str) -> list[tuple[str, str]]:
    """
    Returns a list of (error_type, span) pairs for article errors in the sentence.
    Returns [] if no article errors detected.
    """
    doc = nlp(sentence)
    seen = set()
    results = []
    for check in CHECKS:
        for pair in check(doc):
            if pair not in seen:
                seen.add(pair)
                results.append(pair)
    return results
