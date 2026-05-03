"""
Rule-based error classifier for ESL writing.
Runs after binary DistilBERT detection to label the error type.
Returns a list of all error types found in a sentence.
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
ARTICLE_AN    = "Article error (a/an)"
TENSE         = "Verb tense error"
MISSING_S     = "Missing -s ending"
PRES_CONT     = "Present simple vs continuous"
DOUBLE_NEG    = "Double negative"
ART_OMISSION  = "Article omission"
COUNTABLE     = "Countable/uncountable noun"
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
    ('addicted', 'to'):        'addicted to',  # this is correct - skip
}

# words that are often confused - checked against grammatical context
CONFUSED_WORDS = {
    'their': ('DET', 'poss'),       # possessive determiner
    "they're": ('AUX', None),       # contraction of they are
    'there': ('ADV', None),         # adverb/existential
    'your': ('DET', 'poss'),
    "you're": ('AUX', None),
    'its': ('DET', 'poss'),
    "it's": ('AUX', None),
    'then': ('ADV', None),          # time adverb
    'than': ('SCONJ', None),        # comparison
    'affect': ('VERB', None),       # usually a verb
    'effect': ('NOUN', None),       # usually a noun
    'loose': ('ADJ', None),
    'lose': ('VERB', None),
    'advice': ('NOUN', None),
    'advise': ('VERB', None),
}

OBJECT_PRONOUNS  = {'me', 'him', 'her', 'us', 'them'}
SUBJECT_PRONOUNS = {'i', 'he', 'she', 'we', 'they'}
FREQ_ADVERBS     = {'always', 'usually', 'often', 'sometimes', 'rarely', 'seldom', 'never'}

# ── individual checks ──────────────────────────────────────────────────────────

def check_spelling(doc):
    words = [
        tok.text for tok in doc
        if tok.is_alpha and not tok.is_stop
        and tok.pos_ != 'PROPN'
        and tok.text.lower() not in INCORRECT_FORMS  # caught separately
    ]
    return SPELLING if spell.unknown(words) else None


def check_subject_verb_agreement(doc):
    for tok in doc:
        if tok.dep_ == 'nsubj' and tok.head.pos_ == 'VERB':
            subj = tok.text.lower()
            verb = tok.head
            # 3rd person singular subjects need VBZ
            if subj in ('he', 'she', 'it') or tok.tag_ == 'NN':
                if verb.tag_ == 'VBP':
                    return SVA
            # plural/1st/2nd person subjects should not use VBZ
            if subj in ('i', 'we', 'they', 'you') or tok.tag_ == 'NNS':
                if verb.tag_ == 'VBZ':
                    return SVA
            # was/were with wrong subject
            if subj in ('they', 'we', 'you') and verb.lemma_ == 'be' and verb.text.lower() == 'was':
                return SVA
            if subj in ('he', 'she', 'it') and verb.lemma_ == 'be' and verb.text.lower() == 'were':
                return SVA
    return None


def check_pronoun_case(doc):
    for tok in doc:
        if tok.dep_ in ('nsubj', 'nsubjpass') and tok.text.lower() in OBJECT_PRONOUNS:
            return PRONOUN
        if tok.dep_ in ('dobj', 'pobj', 'iobj') and tok.text.lower() in SUBJECT_PRONOUNS:
            return PRONOUN
    return None


def check_article_an(doc):
    tokens = list(doc)
    for i, tok in enumerate(tokens[:-1]):
        if tok.text.lower() in ('a', 'an'):
            next_word = tokens[i + 1].text
            starts_vowel = bool(re.match(r'^[aeiouAEIOU]', next_word))
            if tok.text.lower() == 'a' and starts_vowel:
                return ARTICLE_AN
            if tok.text.lower() == 'an' and not starts_vowel:
                return ARTICLE_AN
    return None


def check_verb_tense(doc):
    # flag past tense verb in a clause that has present tense time markers
    present_markers = {'today', 'now', 'currently', 'this week', 'this year', 'nowadays'}
    has_present_marker = any(tok.text.lower() in present_markers for tok in doc)
    if has_present_marker:
        for tok in doc:
            if tok.tag_ in ('VBD', 'VBN') and tok.dep_ not in ('aux', 'auxpass'):
                return TENSE
    return None


def check_missing_s(doc):
    tokens = list(doc)
    for i, tok in enumerate(tokens[:-1]):
        if tok.like_num:
            try:
                val = float(tok.text.replace(',', ''))
                if val != 1 and tokens[i + 1].tag_ == 'NN':
                    return MISSING_S
            except ValueError:
                pass
    return None


def check_present_continuous(doc):
    for tok in doc:
        # stative verb in progressive form
        if tok.tag_ == 'VBG' and tok.lemma_ in STATIVE_VERBS:
            if any(child.lemma_ == 'be' for child in tok.head.children):
                return PRES_CONT
    return None


def check_double_negative(doc):
    neg_indefinites = {'nothing', 'nobody', 'nowhere', 'none', 'neither'}
    has_neg = any(tok.dep_ == 'neg' for tok in doc)
    has_neg_indef = any(tok.text.lower() in neg_indefinites for tok in doc)
    return DOUBLE_NEG if has_neg and has_neg_indef else None


def check_article_omission(doc):
    # singular countable noun in object/subject position with no determiner
    for tok in doc:
        if tok.tag_ == 'NN' and tok.dep_ in ('dobj', 'pobj') and tok.lemma_ not in UNCOUNTABLE_NOUNS:
            has_det = any(c.dep_ == 'det' or c.dep_ == 'poss' for c in tok.children)
            has_adj_with_det = any(
                any(c2.dep_ == 'det' for c2 in c.children)
                for c in tok.children if c.dep_ == 'amod'
            )
            if not has_det and not has_adj_with_det:
                return ART_OMISSION
    return None


def check_countable(doc):
    for tok in doc:
        # pluralized uncountable noun
        if tok.tag_ == 'NNS' and tok.lemma_ in UNCOUNTABLE_NOUNS:
            return COUNTABLE
        # uncountable noun with a/an
        if tok.dep_ == 'det' and tok.text.lower() in ('a', 'an'):
            if tok.head.lemma_ in UNCOUNTABLE_NOUNS:
                return COUNTABLE
    return None


def check_incorrect_forms(doc):
    for tok in doc:
        if tok.text.lower() in INCORRECT_FORMS:
            return IRREGULAR
    return None


def check_run_on(doc):
    tokens = list(doc)
    # comma splice: [subj + verb], [subj + verb] with only a comma between
    for i, tok in enumerate(tokens[1:-1], start=1):
        if tok.text == ',':
            left  = tokens[:i]
            right = tokens[i + 1:]
            right_starts_conj = right[0].pos_ == 'CCONJ' if right else False
            if (
                any(t.pos_ == 'VERB' and t.dep_ != 'aux' for t in left) and
                any(t.dep_ == 'nsubj' for t in left) and
                any(t.pos_ == 'VERB' and t.dep_ != 'aux' for t in right) and
                any(t.dep_ == 'nsubj' for t in right) and
                not right_starts_conj
            ):
                return RUN_ON
    return None


def check_preposition(doc):
    tokens = list(doc)
    for i, tok in enumerate(tokens[:-1]):
        pair = (tok.lemma_.lower(), tokens[i + 1].text.lower())
        if pair in PREPOSITION_ERRORS:
            return PREPOSITION
    return None


def check_word_order(doc):
    tokens = list(doc)
    for i, tok in enumerate(tokens):
        if tok.text.lower() in FREQ_ADVERBS:
            # freq adverb should not come directly after a main verb (non-be)
            if i > 0 and tokens[i - 1].pos_ == 'VERB' and tokens[i - 1].lemma_ != 'be':
                return WORD_ORDER
            # freq adverb should come after 'be', not before it
            if i < len(tokens) - 1 and tokens[i + 1].lemma_ == 'be':
                return WORD_ORDER
    return None


def check_modifier(doc):
    tokens = list(doc)
    # dangling participle: sentence opens with a VBG phrase before a comma
    if len(tokens) > 2 and tokens[0].tag_ == 'VBG':
        if any(tok.text == ',' for tok in tokens[:5]):
            return MODIFIER
    return None


def check_parallelism(doc):
    for tok in doc:
        if tok.dep_ == 'cc':
            head = tok.head
            conjuncts = [c for c in head.children if c.dep_ == 'conj']
            if conjuncts:
                tags = {head.tag_} | {c.tag_ for c in conjuncts}
                # mixing gerund and infinitive/base form
                if 'VBG' in tags and ('VB' in tags or 'VBP' in tags or 'TO' in tags):
                    return PARALLELISM
    return None


def check_vague_pronoun(doc):
    # multiple proper nouns / noun phrases + a pronoun that could refer to any of them
    nouns   = [tok for tok in doc if tok.pos_ == 'PROPN']
    pronouns = [tok for tok in doc if tok.pos_ == 'PRON' and tok.text.lower() in ('he', 'she', 'it', 'they', 'him', 'her', 'them')]
    if len(nouns) >= 2 and pronouns:
        return VAGUE_PRONOUN
    return None


def check_similar_words(doc):
    # flag common confused words used in the wrong grammatical role
    errors = {
        # (word, expected_pos_when_wrong)
        ('their',   'ADV'):  SIMILAR_WORDS,   # their used as adverb → should be there
        ('there',   'DET'):  SIMILAR_WORDS,   # there used as determiner → should be their
        ("they're", 'DET'):  SIMILAR_WORDS,
        ('your',    'ADV'):  SIMILAR_WORDS,
        ("you're",  'DET'):  SIMILAR_WORDS,
        ('its',     'VERB'): SIMILAR_WORDS,
        ("it's",    'DET'):  SIMILAR_WORDS,
        ('then',    'SCONJ'): SIMILAR_WORDS,  # then used in comparison → should be than
        ('than',    'ADV'):  SIMILAR_WORDS,   # than used as time adverb → should be then
        ('loose',   'VERB'): SIMILAR_WORDS,   # loose used as verb → should be lose
        ('lose',    'ADJ'):  SIMILAR_WORDS,
        ('advice',  'VERB'): SIMILAR_WORDS,
        ('advise',  'NOUN'): SIMILAR_WORDS,
        ('affect',  'NOUN'): SIMILAR_WORDS,
        ('effect',  'VERB'): SIMILAR_WORDS,
    }
    for tok in doc:
        key = (tok.text.lower(), tok.pos_)
        if key in errors:
            return errors[key]
    return None


def check_adj_adv(doc):
    for tok in doc:
        # adjective in advmod role (modifying verb) — should be adverb
        if tok.pos_ == 'ADJ' and tok.dep_ == 'advmod' and tok.head.pos_ == 'VERB':
            return ADJ_ADV
    return None


# ── main pipeline ─────────────────────────────────────────────────────────────

CHECKS = [
    check_spelling,
    check_subject_verb_agreement,
    check_pronoun_case,
    check_article_an,
    check_verb_tense,
    check_missing_s,
    check_present_continuous,
    check_double_negative,
    check_article_omission,
    check_countable,
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


def classify_errors(sentence: str) -> list[str]:
    """
    Takes a sentence, returns all error types found.
    Returns [UNKNOWN] if no specific rule matches.
    """
    doc = nlp(sentence)
    errors = [result for check in CHECKS if (result := check(doc))]
    return errors if errors else [UNKNOWN]
