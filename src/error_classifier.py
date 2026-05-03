"""
Rule-based error classifier for ESL writing.
Runs after binary DistilBERT detection to label the error type.
Returns a list of (error_type, span) pairs found in a sentence.
Article errors are handled by article_checker.py instead.
"""

import re
import spacy
from spellchecker import SpellChecker

nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()

# ── error type labels ──────────────────────────────────────────────────────────

SPELLING      = "Spelling error"
SVA           = "Subject-verb agreement"
PRONOUN       = "Pronoun case error"
TENSE         = "Verb tense error"
MISSING_S     = "Missing -s ending"
PRES_CONT     = "Present simple vs continuous"
DOUBLE_NEG    = "Double negative"
IRREGULAR     = "Incorrect irregular form"
RUN_ON        = "Run-on sentence"
PREPOSITION   = "Preposition error"
WORD_ORDER    = "Word order error"
MODIFIER      = "Misplaced modifier"
PARALLELISM   = "Faulty parallelism"
VAGUE_PRONOUN = "Vague pronoun reference"
SIMILAR_WORDS = "Confused similar words"
ADJ_ADV       = "Adjective/adverb confusion"
UNKNOWN       = "Grammatical error (unclassified)"

# ── word lists ─────────────────────────────────────────────────────────────────

UNCOUNTABLE_NOUNS = {
    'information', 'advice', 'furniture', 'equipment', 'luggage', 'baggage',
    'homework', 'knowledge', 'evidence', 'research', 'music', 'money',
    'news', 'weather', 'traffic', 'progress', 'work', 'accommodation',
    'bread', 'butter', 'cheese', 'milk', 'rice', 'water', 'hair',
    'food', 'fruit', 'meat', 'sand', 'air', 'grass', 'wood', 'ice',
    'electricity', 'energy', 'software', 'hardware', 'vocabulary',
}

INCORRECT_FORMS = {
    # over-regularized past tenses
    'goed': 'went', 'runned': 'ran', 'bringed': 'brought', 'buyed': 'bought',
    'catched': 'caught', 'thinked': 'thought', 'writed': 'wrote',
    'speaked': 'spoke', 'breaked': 'broke', 'stoled': 'stole',
    'drived': 'drove', 'keeped': 'kept', 'leaved': 'left', 'maked': 'made',
    'selled': 'sold', 'sended': 'sent', 'singed': 'sang', 'sleeped': 'slept',
    'spended': 'spent', 'telled': 'told', 'throwed': 'threw', 'winned': 'won',
    'weared': 'wore', 'teached': 'taught', 'comed': 'came', 'holded': 'held',
    'fighted': 'fought', 'falled': 'fell', 'growed': 'grew', 'knowed': 'knew',
    'rided': 'rode', 'rised': 'rose', 'shooted': 'shot', 'sitted': 'sat',
    'standed': 'stood', 'swimed': 'swam', 'taked': 'took',
    'understanded': 'understood',
    # incorrect plurals
    'childs': 'children', 'mouses': 'mice', 'tooths': 'teeth',
    'foots': 'feet', 'mans': 'men', 'womans': 'women', 'gooses': 'geese',
    'leafs': 'leaves', 'knifes': 'knives', 'wolfs': 'wolves',
}

STATIVE_VERBS = {
    'know', 'believe', 'like', 'love', 'hate', 'want', 'need', 'understand',
    'remember', 'mean', 'seem', 'appear', 'own', 'belong', 'contain',
    'consist', 'prefer', 'realize', 'suppose', 'doubt', 'imagine', 'agree',
    'disagree', 'deserve', 'involve', 'include', 'concern', 'depend',
    'matter', 'weigh', 'cost', 'measure',
}

# wrong preposition collocations: (word, wrong_preposition) -> correct form
PREPOSITION_ERRORS = {
    ('married', 'with'):       'married to',
    ('interested', 'on'):      'interested in',
    ('good', 'on'):            'good at',
    ('arrive', 'to'):          'arrive at/in',
    ('depend', 'of'):          'depend on',
    ('congratulate', 'for'):   'congratulate on',
    ('consist', 'from'):       'consist of',
    ('insist', 'about'):       'insist on',
    ('participate', 'on'):     'participate in',
    ('succeed', 'to'):         'succeed in',
    ('accused', 'for'):        'accused of',
    ('afraid', 'from'):        'afraid of',
    ('capable', 'to'):         'capable of',
    ('different', 'than'):     'different from',
    ('similar', 'than'):       'similar to',
    ('bored', 'from'):         'bored of/with',
    ('responsible', 'of'):     'responsible for',
    ('famous', 'by'):          'famous for',
}

OBJECT_PRONOUNS  = {'me', 'him', 'her', 'us', 'them'}
SUBJECT_PRONOUNS = {'i', 'he', 'she', 'we', 'they'}
FREQ_ADVERBS     = {'always', 'usually', 'often', 'sometimes', 'rarely', 'seldom', 'never'}

# nouns that legitimately appear without articles in common phrases
ARTICLE_EXEMPT = {
    'home', 'work', 'school', 'college', 'university', 'hospital', 'church',
    'bed', 'town', 'sea', 'prison', 'court', 'class', 'lunch', 'breakfast',
    'dinner', 'time', 'life', 'love', 'nature', 'society', 'language',
    'help', 'fun', 'trouble', 'difficulty', 'permission', 'control',
}

# spelled-out numbers that like_num misses
NUMBER_WORDS = {
    'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
    'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty',
    'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand',
}

# ── individual checks ──────────────────────────────────────────────────────────

def check_spelling(doc) -> list[tuple[str, str]]:
    words = [
        tok.text for tok in doc
        if tok.is_alpha and not tok.is_stop
        and tok.pos_ != 'PROPN'
        and tok.text.lower() not in INCORRECT_FORMS  # caught separately
    ]
    return [(SPELLING, w) for w in spell.unknown(words)]


def check_subject_verb_agreement(doc) -> list[tuple[str, str]]:
    results = []
    for tok in doc:
        if tok.dep_ != 'nsubj' or tok.head.pos_ != 'VERB':
            continue
        subj = tok.text.lower()
        verb = tok.head
        is_3sg   = subj in ('he', 'she', 'it') or tok.tag_ == 'NN'
        is_plural = subj in ('i', 'we', 'they', 'you') or tok.tag_ == 'NNS'

        # main verb mismatch
        if is_3sg and verb.tag_ == 'VBP':
            results.append((SVA, verb.text))
            continue
        if is_plural and verb.tag_ == 'VBZ':
            results.append((SVA, verb.text))
            continue

        # was/were with wrong subject
        if subj in ('they', 'we', 'you') and verb.lemma_ == 'be' and verb.text.lower() == 'was':
            results.append((SVA, verb.text))
            continue
        if subj in ('he', 'she', 'it') and verb.lemma_ == 'be' and verb.text.lower() == 'were':
            results.append((SVA, verb.text))
            continue

        # check auxiliaries (catches "she don't", "they was sleeping")
        for child in verb.children:
            if child.dep_ in ('aux', 'auxpass') and child.lemma_ in ('do', 'have', 'be'):
                if is_3sg and child.tag_ == 'VBP':
                    results.append((SVA, child.text))
                elif is_plural and child.tag_ == 'VBZ':
                    results.append((SVA, child.text))
                elif is_plural and child.lemma_ == 'be' and child.text.lower() == 'was':
                    results.append((SVA, child.text))
                elif is_3sg and child.lemma_ == 'be' and child.text.lower() == 'were':
                    results.append((SVA, child.text))
    return results


def check_pronoun_case(doc) -> list[tuple[str, str]]:
    results = []
    for tok in doc:
        if tok.dep_ in ('nsubj', 'nsubjpass') and tok.text.lower() in OBJECT_PRONOUNS:
            results.append((PRONOUN, tok.text))
        elif tok.dep_ in ('dobj', 'pobj', 'iobj') and tok.text.lower() in SUBJECT_PRONOUNS:
            results.append((PRONOUN, tok.text))
    return results


def check_verb_tense(doc) -> list[tuple[str, str]]:
    present_markers = {'today', 'now', 'currently', 'this week', 'this year', 'nowadays'}
    has_present_marker = any(tok.text.lower() in present_markers for tok in doc)
    if not has_present_marker:
        return []
    return [
        (TENSE, tok.text)
        for tok in doc
        if tok.tag_ in ('VBD', 'VBN') and tok.dep_ not in ('aux', 'auxpass')
    ]


def check_missing_s(doc) -> list[tuple[str, str]]:
    results = []
    tokens = list(doc)
    for i, tok in enumerate(tokens[:-1]):
        is_num = tok.like_num or tok.text.lower() in NUMBER_WORDS
        is_one = tok.text in ('1', 'one')
        if is_num and not is_one and tokens[i + 1].tag_ == 'NN':
            results.append((MISSING_S, tokens[i + 1].text))
    return results


def check_present_continuous(doc) -> list[tuple[str, str]]:
    results = []
    for tok in doc:
        if tok.tag_ == 'VBG' and tok.lemma_ in STATIVE_VERBS:
            if any(child.lemma_ == 'be' for child in tok.head.children):
                results.append((PRES_CONT, tok.text))
    return results


def check_double_negative(doc) -> list[tuple[str, str]]:
    neg_indefinites = {'nothing', 'nobody', 'nowhere', 'none', 'neither'}
    has_neg = any(tok.dep_ == 'neg' for tok in doc)
    if not has_neg:
        return []
    indef_toks = [tok for tok in doc if tok.text.lower() in neg_indefinites]
    return [(DOUBLE_NEG, tok.text) for tok in indef_toks]


def check_incorrect_forms(doc) -> list[tuple[str, str]]:
    return [
        (IRREGULAR, tok.text)
        for tok in doc
        if tok.text.lower() in INCORRECT_FORMS
    ]


def check_run_on(doc) -> list[tuple[str, str]]:
    results = []
    tokens = list(doc)
    for i, tok in enumerate(tokens[1:-1], start=1):
        if tok.text != ',':
            continue
        left  = tokens[:i]
        right = tokens[i + 1:]
        right_starts_conj = right[0].pos_ == 'CCONJ' if right else False
        left_is_subordinate = any(
            t.dep_ in ('mark', 'advcl', 'relcl') or t.pos_ == 'SCONJ'
            for t in left
        )
        if (
            not left_is_subordinate and
            any(t.pos_ == 'VERB' and t.dep_ != 'aux' for t in left) and
            any(t.dep_ == 'nsubj' for t in left) and
            any(t.pos_ == 'VERB' and t.dep_ != 'aux' for t in right) and
            any(t.dep_ == 'nsubj' for t in right) and
            not right_starts_conj
        ):
            results.append((RUN_ON, ','))
    return results


def check_preposition(doc) -> list[tuple[str, str]]:
    results = []
    tokens = list(doc)
    for i, tok in enumerate(tokens[:-1]):
        pair = (tok.lemma_.lower(), tokens[i + 1].text.lower())
        if pair in PREPOSITION_ERRORS:
            results.append((PREPOSITION, tokens[i + 1].text))
    return results


def check_word_order(doc) -> list[tuple[str, str]]:
    results = []
    tokens = list(doc)
    for i, tok in enumerate(tokens):
        if tok.text.lower() not in FREQ_ADVERBS:
            continue
        if i > 0 and tokens[i - 1].pos_ == 'VERB' and tokens[i - 1].lemma_ != 'be':
            results.append((WORD_ORDER, tok.text))
        elif i < len(tokens) - 1 and tokens[i + 1].lemma_ == 'be':
            results.append((WORD_ORDER, tok.text))
    return results


def check_modifier(doc) -> list[tuple[str, str]]:
    tokens = list(doc)
    if len(tokens) > 2 and tokens[0].tag_ == 'VBG':
        if any(tok.text == ',' for tok in tokens[:5]):
            return [(MODIFIER, tokens[0].text)]
    return []


def check_parallelism(doc) -> list[tuple[str, str]]:
    results = []
    for tok in doc:
        if tok.dep_ != 'cc':
            continue
        head = tok.head
        conjuncts = [c for c in head.children if c.dep_ == 'conj']
        if conjuncts:
            tags = {head.tag_} | {c.tag_ for c in conjuncts}
            if 'VBG' in tags and ('VB' in tags or 'VBP' in tags or 'TO' in tags):
                results.append((PARALLELISM, tok.text))
    return results


def check_vague_pronoun(doc) -> list[tuple[str, str]]:
    nouns    = [tok for tok in doc if tok.pos_ == 'PROPN']
    pronouns = [tok for tok in doc if tok.pos_ == 'PRON'
                and tok.text.lower() in ('he', 'she', 'it', 'they', 'him', 'her', 'them')]
    if len(nouns) >= 2 and pronouns:
        return [(VAGUE_PRONOUN, pronouns[0].text)]
    return []


def check_similar_words(doc) -> list[tuple[str, str]]:
    pos_errors = {
        ('their',   'ADV'):   SIMILAR_WORDS,
        ('there',   'DET'):   SIMILAR_WORDS,
        ("they're", 'DET'):   SIMILAR_WORDS,
        ('your',    'ADV'):   SIMILAR_WORDS,
        ("you're",  'DET'):   SIMILAR_WORDS,
        ('its',     'VERB'):  SIMILAR_WORDS,
        ("it's",    'DET'):   SIMILAR_WORDS,
        ('then',    'SCONJ'): SIMILAR_WORDS,
        ('than',    'ADV'):   SIMILAR_WORDS,
        ('loose',   'VERB'):  SIMILAR_WORDS,
        ('lose',    'ADJ'):   SIMILAR_WORDS,
        ('advice',  'VERB'):  SIMILAR_WORDS,
        ('advise',  'NOUN'):  SIMILAR_WORDS,
        ('affect',  'NOUN'):  SIMILAR_WORDS,
        ('effect',  'VERB'):  SIMILAR_WORDS,
    }
    seen = set()
    results = []

    for tok in doc:
        if (tok.text.lower(), tok.pos_) in pos_errors and tok.text not in seen:
            seen.add(tok.text)
            results.append((SIMILAR_WORDS, tok.text))

    # "Their/Your going..." — spaCy may parse as poss or nsubj
    for tok in doc:
        if tok.text.lower() in ('their', 'your') and tok.dep_ in ('poss', 'nsubj'):
            if tok.head.tag_ == 'VBG' and tok.head.dep_ == 'ROOT':
                if tok.text not in seen:
                    seen.add(tok.text)
                    results.append((SIMILAR_WORDS, tok.text))

    return results


def check_adj_adv(doc) -> list[tuple[str, str]]:
    return [
        (ADJ_ADV, tok.text)
        for tok in doc
        if tok.pos_ == 'ADJ' and tok.dep_ == 'advmod' and tok.head.pos_ == 'VERB'
    ]


# ── main pipeline ─────────────────────────────────────────────────────────────

CHECKS = [
    check_spelling,
    check_subject_verb_agreement,
    check_pronoun_case,
    check_verb_tense,
    check_missing_s,
    check_present_continuous,
    check_double_negative,
    check_incorrect_forms,
    check_run_on,
    check_preposition,
    check_word_order,
    check_modifier,
    check_parallelism,
    check_vague_pronoun,
    check_similar_words,
    check_adj_adv,
]


def classify_errors(sentence: str) -> list[tuple[str, str]]:
    """
    Takes a sentence, returns (error_type, span) pairs for all errors found.
    Returns [(UNKNOWN, '')] if no specific rule matches.
    """
    doc = nlp(sentence)
    errors = []
    for check in CHECKS:
        errors.extend(check(doc))
    return errors if errors else [(UNKNOWN, '')]
