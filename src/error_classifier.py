"""
Rule-based error classifier for ESL writing.
Runs after binary DistilBERT detection to label the error type.
Returns a list of (error_type, span) pairs found in a sentence.
Article errors are handled by article_checker.py instead.
"""

import spacy
from spellchecker import SpellChecker

nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()

# ── error type labels ──────────────────────────────────────────────────────────

SPELLING        = "Spelling error"
SVA             = "Subject-verb agreement"
PRONOUN         = "Pronoun case error"
TENSE           = "Verb tense error"
MISSING_S       = "Missing -s ending"
PRES_CONT       = "Present simple vs continuous"
DOUBLE_NEG      = "Double negative"
IRREGULAR       = "Incorrect irregular form"
RUN_ON          = "Run-on sentence"
PREPOSITION     = "Preposition error"
WORD_ORDER      = "Word order error"
MODIFIER        = "Misplaced modifier"
PARALLELISM     = "Faulty parallelism"
VAGUE_PRONOUN   = "Vague pronoun reference"
SIMILAR_WORDS   = "Confused similar words"
ADJ_ADV         = "Adjective/adverb confusion"
THERE_THEIR     = "there/their/they're confusion"
COMPARATIVE     = "Comparative/superlative error"
PARTICIPIAL_ADJ = "Participial adjective confusion"
SINCE_FOR_AGO   = "Since/for/ago error"
DOUBLED_SUBJ    = "Doubled subject"
SUBJ_ORDER      = "Subject ordering"
NEGATED_VERB    = "Incorrect verb form after auxiliary"
INF_AFTER_PREP  = "Infinitive after preposition"
GERUND_VERB     = "Gerund required after verb"
INF_FORM        = "Wrong verb form after 'to'"
MODAL_HAVE      = "Modal + 'had' error"
WRONG_REL_PRON  = "Wrong relative pronoun"
HAVE_PART       = "Missing past participle"
MODAL_TO_ERR    = "Modal + infinitive error"
MUCH_MANY       = "Much/many confusion"
FEWER_LESS      = "Fewer/less error"
UNKNOWN         = "Grammatical error (unclassified)"

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

PREPOSITION_ERRORS = {
    ('married', 'with'):     'married to',
    ('interested', 'on'):    'interested in',
    ('good', 'on'):          'good at',
    ('arrive', 'to'):        'arrive at/in',
    ('depend', 'of'):        'depend on',
    ('congratulate', 'for'): 'congratulate on',
    ('consist', 'from'):     'consist of',
    ('insist', 'about'):     'insist on',
    ('participate', 'on'):   'participate in',
    ('succeed', 'to'):       'succeed in',
    ('accused', 'for'):      'accused of',
    ('afraid', 'from'):      'afraid of',
    ('capable', 'to'):       'capable of',
    ('different', 'than'):   'different from',
    ('similar', 'than'):     'similar to',
    ('bored', 'from'):       'bored of/with',
    ('responsible', 'of'):   'responsible for',
    ('famous', 'by'):        'famous for',
    ('superior', 'than'):    'superior to',
    ('inferior', 'than'):    'inferior to',
    ('agree', 'in'):         'agree on/with',
    ('laugh', 'of'):         'laugh at',
    ('look', 'in'):          'look at',  # "look in" is sometimes valid but "look in X" for gazing is wrong
    ('listen', 'in'):        'listen to',
}

OBJECT_PRONOUNS  = {'me', 'him', 'her', 'us', 'them'}
SUBJECT_PRONOUNS = {'i', 'he', 'she', 'we', 'they'}
FREQ_ADVERBS     = {'always', 'usually', 'often', 'sometimes', 'rarely', 'seldom', 'never'}

ARTICLE_EXEMPT = {
    'home', 'work', 'school', 'college', 'university', 'hospital', 'church',
    'bed', 'town', 'sea', 'prison', 'court', 'class', 'lunch', 'breakfast',
    'dinner', 'time', 'life', 'love', 'nature', 'society', 'language',
    'help', 'fun', 'trouble', 'difficulty', 'permission', 'control',
}

NUMBER_WORDS = {
    'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
    'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty',
    'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand',
}

# -ing emotion adjectives that describe what causes a feeling
# (ESL learners often use these when they mean the -ed form)
EMOTION_ING = {
    'interesting', 'boring', 'exciting', 'surprising', 'confusing', 'tiring',
    'frustrating', 'satisfying', 'annoying', 'shocking', 'disappointing',
    'worrying', 'embarrassing', 'terrifying', 'amazing', 'fascinating',
    'overwhelming', 'depressing', 'exhausting',
}

EMOTION_ING_TO_ED = {
    'interesting': 'interested', 'boring': 'bored', 'exciting': 'excited',
    'surprising': 'surprised', 'confusing': 'confused', 'tiring': 'tired',
    'frustrating': 'frustrated', 'satisfying': 'satisfied', 'annoying': 'annoyed',
    'shocking': 'shocked', 'disappointing': 'disappointed', 'worrying': 'worried',
    'embarrassing': 'embarrassed', 'terrifying': 'terrified', 'amazing': 'amazed',
    'fascinating': 'fascinated', 'overwhelming': 'overwhelmed',
    'depressing': 'depressed', 'exhausting': 'exhausted',
}

LINKING_VERBS = {'be', 'seem', 'appear', 'become', 'feel', 'look', 'sound', 'get'}

TIME_UNITS = {
    'year', 'years', 'month', 'months', 'week', 'weeks', 'day', 'days',
    'hour', 'hours', 'minute', 'minutes', 'second', 'seconds',
}

TIME_NOUNS_PERIOD = {
    'week', 'month', 'year', 'day', 'night', 'morning', 'evening',
    'semester', 'term', 'summer', 'winter', 'spring', 'fall', 'autumn',
}

# causative verbs whose object takes a bare infinitive complement
# "have" excluded — too easily confused with possessive/auxiliary have
CAUSATIVE_VERBS = {'make', 'let', 'help'}

# person nouns that require 'who' not 'which' in relative clauses
PERSON_NOUNS = {
    'man', 'woman', 'person', 'people', 'boy', 'girl', 'teacher', 'student',
    'doctor', 'nurse', 'friend', 'colleague', 'boss', 'employee', 'worker',
    'child', 'kid', 'baby', 'parent', 'mother', 'father', 'sister', 'brother',
    'husband', 'wife', 'partner', 'customer', 'client', 'patient', 'lawyer',
    'stranger', 'neighbor', 'neighbour', 'classmate', 'roommate', 'teammate',
    'chef', 'driver', 'manager', 'officer', 'soldier', 'scientist', 'artist',
    'writer', 'speaker', 'listener', 'visitor', 'tourist', 'resident',
}

# verbs that DO NOT require a preposition but ESL learners often add one
SUPERFLUOUS_PREPS = {
    ('discuss', 'about'),
    ('mention', 'about'),
    ('explain', 'about'),
    ('emphasize', 'about'),
    ('stress', 'about'),
    ('approach', 'to'),
    ('enter', 'in'),
    ('enter', 'into'),
    ('marry', 'with'),
}

# causative/resultative verbs that use "to" + infinitive (catches "to becoming")
CAUSATIVE_INF_VERBS = {'cause', 'allow', 'force', 'enable', 'help', 'get', 'compel', 'encourage'}

# verbs that ONLY accept gerund complements — infinitive is always wrong
GERUND_ONLY_VERBS = {
    'enjoy', 'avoid', 'keep', 'miss', 'risk', 'resist', 'mind', 'deny',
    'delay', 'postpone', 'suggest', 'recommend', 'practice', 'quit', 'give',
}

# verbs that accept gerund OR infinitive (different meanings) —
# 'stop to rest' (purpose) vs 'stop smoking' (quit) — both valid, don't flag TO+VB
GERUND_OR_INF_VERBS = {
    'stop', 'finish', 'consider', 'admit', 'recall', 'remember', 'forget',
    'try', 'continue', 'begin', 'start', 'like', 'prefer', 'hate', 'love',
}

GERUND_VERBS = GERUND_ONLY_VERBS | GERUND_OR_INF_VERBS

# verbs where "verb + to + VBG" is the CORRECT construction (prep + gerund object)
# e.g. "resort to lying", "look forward to going", "be/get used to doing"
VERB_TO_GERUND = {
    'resort', 'amount', 'lead', 'object', 'commit', 'devote', 'adapt', 'adjust',
    'limit', 'contribute', 'look', 'use', 'accustom', 'belong', 'get',
    'confess', 'reduce', 'stoop',
}

# prepositions that require a gerund (-ing), not an infinitive
GERUND_PREPS = {
    'about', 'in', 'of', 'at', 'for', 'without', 'before', 'after',
    'instead', 'despite', 'by', 'with', 'on', 'from', 'into', 'through',
}

# ── individual checks ──────────────────────────────────────────────────────────

def check_spelling(doc) -> list[tuple[str, str]]:
    words = [
        tok.text for tok in doc
        if tok.is_alpha and not tok.is_stop
        and tok.pos_ != 'PROPN'
        and tok.text.lower() not in INCORRECT_FORMS
    ]
    return [(SPELLING, w) for w in spell.unknown(words)]


def _sva_check_verb(verb, is_3sg: bool, is_plural: bool) -> list[tuple[str, str]]:
    """Return SVA errors for a verb and its auxiliaries given subject agreement info."""
    results = []
    if is_3sg and verb.tag_ == 'VBP':
        results.append((SVA, verb.text))
    elif is_plural and verb.tag_ == 'VBZ':
        results.append((SVA, verb.text))
    elif is_plural and verb.lemma_ == 'be' and verb.text.lower() == 'was':
        results.append((SVA, verb.text))
    elif is_3sg and verb.lemma_ == 'be' and verb.text.lower() == 'were':
        results.append((SVA, verb.text))
    elif is_3sg and verb.tag_ == 'VB':
        # bare form in main clause with 3sg subject (e.g. "she speak") — missing -s
        # exclude: modal aux, do-support, or TO (infinitive position — not a finite verb)
        has_modal_or_do = any(
            c.dep_ == 'aux' and (c.tag_ in ('MD', 'TO') or c.lemma_ == 'do')
            for c in verb.children
        )
        if not has_modal_or_do:
            results.append((SVA, verb.text))
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


def check_subject_verb_agreement(doc) -> list[tuple[str, str]]:
    results = []

    # existential "there is/are" — the real subject is the attr noun, not "there"
    for tok in doc:
        if tok.dep_ == 'expl' and tok.text.lower() == 'there':
            verb = tok.head
            if verb.lemma_ == 'be':
                attr = next((c for c in verb.children if c.dep_ == 'attr'), None)
                if attr and attr.tag_ in ('NN', 'NNP'):
                    results.extend(_sva_check_verb(verb, is_3sg=True, is_plural=False))
                elif attr and attr.tag_ in ('NNS', 'NNPS'):
                    results.extend(_sva_check_verb(verb, is_3sg=False, is_plural=True))

    for tok in doc:
        if tok.dep_ not in ('nsubj', 'nsubjpass') or tok.head.pos_ not in ('VERB', 'AUX'):
            continue
        verb = tok.head

        # relative pronoun subject (who/which/that) — inherit number from antecedent noun
        if tok.tag_ in ('WDT', 'WP') and verb.dep_ == 'relcl':
            antecedent = verb.head
            if antecedent.tag_ in ('NN', 'NNP'):
                results.extend(_sva_check_verb(verb, is_3sg=True, is_plural=False))
            elif antecedent.tag_ in ('NNS', 'NNPS'):
                results.extend(_sva_check_verb(verb, is_3sg=False, is_plural=True))
            continue

        subj = tok.text.lower()
        # compound subject ("Me and my cousin") → treat as plural
        is_compound = any(c.dep_ == 'conj' for c in tok.children)
        is_3sg    = (subj in ('he', 'she', 'it') or tok.tag_ == 'NN') and not is_compound
        is_plural = subj in ('i', 'we', 'they', 'you') or tok.tag_ == 'NNS' or is_compound

        results.extend(_sva_check_verb(verb, is_3sg, is_plural))

        # also check conjoined verbs that inherit this subject (elided subject)
        for conj_verb in verb.children:
            if conj_verb.dep_ != 'conj' or conj_verb.pos_ not in ('VERB', 'AUX'):
                continue
            if any(c.dep_ == 'nsubj' for c in conj_verb.children):
                continue  # has its own subject
            results.extend(_sva_check_verb(conj_verb, is_3sg, is_plural))

    return results


def check_pronoun_case(doc) -> list[tuple[str, str]]:
    results = []
    for tok in doc:
        if tok.dep_ in ('nsubj', 'nsubjpass') and tok.text.lower() in OBJECT_PRONOUNS:
            results.append((PRONOUN, tok.text))
        elif tok.dep_ in ('dobj', 'pobj', 'iobj') and tok.text.lower() in SUBJECT_PRONOUNS:
            results.append((PRONOUN, tok.text))
        # "between you and I" — "I" is conj of "you" (pobj), inherits object role
        elif (tok.dep_ == 'conj'
              and tok.text.lower() in SUBJECT_PRONOUNS
              and tok.head.dep_ in ('pobj', 'dobj', 'iobj')):
            results.append((PRONOUN, tok.text))
    return results


def check_negated_verb(doc) -> list[tuple[str, str]]:
    """Catches 'didn't understood', 'doesn't runs', etc."""
    results = []
    for tok in doc:
        if tok.lemma_ != 'do' or tok.dep_ not in ('aux', 'auxpass'):
            continue
        main = tok.head
        if main.pos_ == 'VERB' and main.tag_ != 'VB':
            results.append((NEGATED_VERB, main.text))
    return results


def check_verb_tense(doc) -> list[tuple[str, str]]:
    """Extended tense checker using past, present, and future time markers."""
    results = []
    tokens = list(doc)
    lowers = [t.text.lower() for t in tokens]

    has_present = any(w in lowers for w in ('now', 'today', 'currently', 'nowadays'))
    has_past    = any(w in lowers for w in ('yesterday', 'ago'))
    has_future  = any(w in lowers for w in ('tomorrow',))

    # "last week/month/year/night/..."
    for i, w in enumerate(lowers[:-1]):
        if w == 'last' and tokens[i + 1].text.lower() in TIME_NOUNS_PERIOD:
            has_past = True
            break

    if has_present:
        for tok in tokens:
            if tok.tag_ in ('VBD', 'VBN') and tok.dep_ not in ('aux', 'auxpass'):
                results.append((TENSE, tok.text))

    if has_past:
        for tok in tokens:
            if tok.tag_ in ('VBZ', 'VBP') and tok.dep_ not in ('aux', 'auxpass'):
                results.append((TENSE, tok.text))

    if has_future:
        for tok in tokens:
            if tok.tag_ == 'VBD' and tok.dep_ not in ('aux', 'auxpass'):
                results.append((TENSE, tok.text))

    return results


ORDINALS = {
    'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh',
    'eighth', 'ninth', 'tenth', 'last', 'next', 'final',
}

def check_missing_s(doc) -> list[tuple[str, str]]:
    results = []
    tokens = list(doc)
    for i, tok in enumerate(tokens[:-1]):
        is_num = tok.like_num or tok.text.lower() in NUMBER_WORDS
        is_one = tok.text in ('1', 'one')
        is_ordinal = tok.text.lower() in ORDINALS
        if is_num and not is_one and not is_ordinal and tokens[i + 1].tag_ == 'NN':
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
    results = [(DOUBLE_NEG, tok.text) for tok in doc if tok.text.lower() in neg_indefinites]
    # also catch "no" used as a determiner alongside a separate negation ("didn't have no money")
    results += [(DOUBLE_NEG, tok.text) for tok in doc if tok.text.lower() == 'no' and tok.dep_ == 'det']
    return results


def check_incorrect_forms(doc) -> list[tuple[str, str]]:
    return [(IRREGULAR, tok.text) for tok in doc if tok.text.lower() in INCORRECT_FORMS]


def check_run_on(doc) -> list[tuple[str, str]]:
    results = []
    tokens = list(doc)
    for i, tok in enumerate(tokens[1:-1], start=1):
        if tok.text != ',':
            continue
        left  = tokens[:i]
        right = tokens[i + 1:]
        right_starts_conj = right[0].pos_ == 'CCONJ' if right else False
        left_is_sub = any(t.dep_ in ('mark', 'advcl', 'relcl') or t.pos_ == 'SCONJ' for t in left)
        if (
            not left_is_sub
            and any(t.pos_ == 'VERB' and t.dep_ != 'aux' for t in left)
            and any(t.dep_ == 'nsubj' for t in left)
            and any(t.pos_ == 'VERB' and t.dep_ != 'aux' for t in right)
            and any(t.dep_ == 'nsubj' for t in right)
            and not right_starts_conj
        ):
            results.append((RUN_ON, ','))
    return results


def check_preposition(doc) -> list[tuple[str, str]]:
    """
    Uses dependency structure (word → prep child) rather than consecutive token pairs,
    so 'accused him for' is caught even with an object between verb and preposition.
    Checks both lemma and surface form to handle inflected forms like 'married'/'marry'.
    """
    results = []
    for tok in doc:
        for child in tok.children:
            if child.dep_ != 'prep':
                continue
            prep = child.text.lower()
            if (tok.lemma_.lower(), prep) in PREPOSITION_ERRORS:
                results.append((PREPOSITION, tok.text))
                break
            elif (tok.text.lower(), prep) in PREPOSITION_ERRORS:
                results.append((PREPOSITION, tok.text))
                break
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
    """
    Catches dangling/misplaced participial phrase at sentence start:
    'Running to catch the bus, my bag fell' — the bag wasn't running.
    Skips when the main clause subject is a personal pronoun (likely correct).
    """
    tokens = list(doc)
    if len(tokens) <= 2 or tokens[0].tag_ != 'VBG':
        return []
    # allow comma up to position 10 to handle longer participial phrases
    if not any(tok.text == ',' for tok in tokens[:10]):
        return []
    personal = {'i', 'he', 'she', 'we', 'they', 'you'}
    for tok in doc:
        if tok.dep_ == 'nsubj' and tok.head.dep_ == 'ROOT':
            if tok.text.lower() in personal:
                return []  # "Looking out the window, I saw..." — correct
    return [(MODIFIER, tokens[0].text)]


def check_parallelism(doc) -> list[tuple[str, str]]:
    """
    Catches mismatched verb forms in coordinate lists:
    'swimming, running, and to dance' → all should be VBG.
    Also traverses into dobj/xcomp subtrees since spaCy may attach the first
    list item as dobj (NN) and the rest as conj children of that item.
    When head is itself a conj, includes head.head (the coord root) in items so
    spaCy parses like 'reading(VBG dobj), writing(NN conj), paint(VB conj)'
    still detect the VBG vs VB mismatch.
    """
    results = []
    for tok in doc:
        if tok.dep_ != 'cc':
            continue
        head = tok.head
        direct_conj = [c for c in head.children if c.dep_ == 'conj']
        # when head is itself a conj node, include the coordination root
        coord_root = head.head if head.dep_ == 'conj' else None
        # also look at the primary object and its conj chain
        primary_obj = next(
            (c for c in head.children if c.dep_ in ('dobj', 'xcomp') and c.pos_ in ('VERB', 'NOUN')),
            None
        )
        obj_conj = [c for c in primary_obj.children if c.dep_ == 'conj'] if primary_obj else []

        base = ([coord_root] if coord_root else []) + ([primary_obj] if primary_obj else [head])
        all_items = base + obj_conj + direct_conj
        if len(all_items) >= 2:
            tags = {item.tag_ for item in all_items}
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


def check_there_their_theyre(doc) -> list[tuple[str, str]]:
    """Dedicated check for there/their/they're confusion with specific detection logic."""
    results = []
    for tok in doc:
        lower = tok.text.lower()

        if lower == 'their':
            # "their" before a VBG/NN root → should be "they're" (e.g. "Their going...", "Their thinking...")
            if tok.dep_ in ('poss', 'nsubj') and tok.head.tag_ in ('VBG', 'NN') and tok.head.dep_ == 'ROOT':
                results.append((THERE_THEIR, tok.text))
            # "their" functioning as an adverb → should be "there"
            elif tok.pos_ == 'ADV':
                results.append((THERE_THEIR, tok.text))
            # "their" used as object of a preposition of location (e.g. "over their") → "there"
            elif tok.dep_ == 'pobj' and tok.head.dep_ == 'prep':
                results.append((THERE_THEIR, tok.text))

        elif lower == 'there':
            # "there" used as a possessive → should be "their"
            if tok.dep_ == 'poss' or (tok.pos_ == 'DET' and tok.dep_ != 'expl'):
                results.append((THERE_THEIR, tok.text))

        elif lower in ("they're", "theyre"):
            # contraction used as a possessive determiner → should be "their"
            if tok.dep_ == 'poss' or tok.pos_ == 'DET':
                results.append((THERE_THEIR, tok.text))

    return results


def check_similar_words(doc) -> list[tuple[str, str]]:
    """Confused word pairs — there/their/they're handled separately."""
    pos_errors = {
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

    # "Your going..." — spaCy may parse as poss or nsubj
    for tok in doc:
        if tok.text.lower() == 'your' and tok.dep_ in ('poss', 'nsubj'):
            if tok.head.tag_ == 'VBG' and tok.head.dep_ == 'ROOT' and tok.text not in seen:
                seen.add(tok.text)
                results.append((SIMILAR_WORDS, tok.text))

    return results


def check_adj_adv(doc) -> list[tuple[str, str]]:
    results = []
    for tok in doc:
        if tok.pos_ != 'ADJ':
            continue
        if tok.dep_ == 'advmod' and tok.head.pos_ == 'VERB':
            results.append((ADJ_ADV, tok.text))
        elif tok.dep_ == 'acomp' and tok.head.pos_ == 'VERB':
            if tok.head.lemma_.lower() not in LINKING_VERBS:
                results.append((ADJ_ADV, tok.text))
        # spaCy sometimes mislabels standalone adj-as-adverb as advcl
        # e.g. "speaks English very good" → 'good' dep=advcl of 'speaks'
        elif tok.dep_ == 'advcl' and tok.head.pos_ == 'VERB':
            results.append((ADJ_ADV, tok.text))
        # open predicative complement after non-linking verb
        # e.g. "sings beautiful" → 'beautiful' dep=oprd of 'sings'
        elif tok.dep_ == 'oprd' and tok.head.pos_ == 'VERB':
            if tok.head.lemma_.lower() not in LINKING_VERBS:
                results.append((ADJ_ADV, tok.text))
    return results


def check_comparative(doc) -> list[tuple[str, str]]:
    """Catches double comparatives ('more bigger'), double superlatives, and 'more good/bad'."""
    results = []
    tokens = list(doc)
    for i, tok in enumerate(tokens[:-1]):
        lower = tok.text.lower()
        nxt   = tokens[i + 1]
        if lower == 'more':
            if nxt.tag_ == 'JJR':                       # more + -er form
                results.append((COMPARATIVE, nxt.text))
            elif nxt.text.lower() in ('good', 'bad'):   # more good/bad → better/worse
                results.append((COMPARATIVE, tok.text))
        elif lower == 'most' and (nxt.tag_ == 'JJS' or nxt.text.lower().endswith('est')):
            results.append((COMPARATIVE, nxt.text))
    return results


def check_participial_adj(doc) -> list[tuple[str, str]]:
    """
    Catches '-ing' emotion adjectives (interesting, boring...) used as predicate
    adjectives with a personal pronoun subject — usually should be the '-ed' form.
    E.g., 'I am very interesting' → should be 'interested'.
    Handles elided subjects in conjoined clauses: 'I went and was very surprising'.
    For 'feel' (always experiential), flags all personal pronouns.
    For 'be', also flags when a temporal prep (after/before/during) is attached —
    'she was very tiring after the journey' signals experienced, not caused, state.
    """
    results = []
    personal = {'i', 'he', 'she', 'we', 'they'}
    for tok in doc:
        if tok.text.lower() not in EMOTION_ING:
            continue
        if tok.dep_ not in ('acomp', 'attr'):
            continue
        if tok.head.lemma_.lower() not in LINKING_VERBS:
            continue
        verb = tok.head
        subj = next((c for c in verb.children if c.dep_ == 'nsubj'), None)
        if subj is None and verb.dep_ == 'conj':
            subj = next((c for c in verb.head.children if c.dep_ == 'nsubj'), None)
        if subj is None or subj.text.lower() not in personal:
            continue
        if subj.text.lower() == 'i':
            results.append((PARTICIPIAL_ADJ, tok.text))
        elif verb.lemma_.lower() == 'feel':
            # 'feel' is inherently experiential — 'he felt boring' almost always means 'bored'
            results.append((PARTICIPIAL_ADJ, tok.text))
        else:
            # for he/she/we/they flag when there's an experiencer preposition
            # ('interested in X', 'excited about X') or a temporal context
            # ('very tiring after the journey' → person experienced tiredness)
            has_experiencer_prep = any(
                c.dep_ == 'prep' and c.text.lower() in ('in', 'about')
                for c in tok.children
            )
            has_temporal_context = any(
                c.dep_ == 'prep' and c.text.lower() in ('after', 'before', 'during')
                for c in verb.children
            )
            if has_experiencer_prep or has_temporal_context:
                results.append((PARTICIPIAL_ADJ, tok.text))
    return results


def check_since_for_ago(doc) -> list[tuple[str, str]]:
    """
    Catches 'since three years' (should be 'for three years') —
    'since' followed by a number + time unit signals a duration, not a point in time.
    """
    results = []
    tokens = list(doc)
    for i, tok in enumerate(tokens[:-2]):
        if tok.text.lower() != 'since':
            continue
        n1, n2 = tokens[i + 1], tokens[i + 2]
        if (n1.like_num or n1.text.lower() in NUMBER_WORDS) and n2.lemma_.lower() in TIME_UNITS:
            results.append((SINCE_FOR_AGO, tok.text))
    return results


def check_doubled_subject(doc) -> list[tuple[str, str]]:
    """
    Catches left-dislocated topics: 'The teacher, she explained'.
    Flags the redundant pronoun.
    """
    results = []
    tokens = list(doc)
    third_person = {'he', 'she', 'it', 'they'}
    hard_excluded = {'pobj', 'pcomp', 'advmod', 'prep'}

    for tok in tokens:
        if tok.dep_ != 'nsubj' or tok.text.lower() not in third_person:
            continue
        for j in range(tok.i):
            prev = tokens[j]
            if prev.pos_ not in ('NOUN', 'PROPN'):
                continue
            if prev.dep_ in hard_excluded:
                continue
            # skip temporal noun phrases only ("Last week", "This morning", etc.)
            if prev.dep_ in ('npadvmod', 'tmod', 'advmod'):
                is_temporal = (
                    prev.lemma_.lower() in TIME_NOUNS_PERIOD
                    or prev.lemma_.lower() in TIME_UNITS
                    or any(c.text.lower() in ('last', 'next', 'this', 'every', 'each')
                           for c in prev.children)
                )
                if is_temporal:
                    continue
            # bare time/manner nouns (Yesterday, Today...) have no determiner — skip them
            if prev.pos_ == 'NOUN':
                has_det = any(c.dep_ in ('det', 'poss', 'amod') for c in prev.children)
                if not has_det:
                    continue
            between = tokens[j + 1: tok.i]
            has_comma   = any(t.text == ',' for t in between)
            has_relpron = any(
                t.text.lower() in ('who', 'which', 'that', 'whom', 'whose')
                for t in between
            )
            # noun and pronoun must share the same governing verb
            if prev.head.i != tok.head.i:
                continue
            # flag if comma present, OR if noun and pronoun are immediately adjacent
            if (has_comma or not between) and not has_relpron:
                results.append((DOUBLED_SUBJ, tok.text))
                break

    return results


def check_subject_ordering(doc) -> list[tuple[str, str]]:
    """
    Catches first-person pronoun placed before 'and' in a compound subject/object:
    'me and my friend' → 'my friend and I'; 'I and John' → 'John and I'.
    """
    results = []
    tokens = list(doc)
    for i, tok in enumerate(tokens):
        if tok.text.lower() not in ('i', 'me'):
            continue
        if i + 2 >= len(tokens):
            continue
        if tokens[i + 1].text.lower() == 'and' and tokens[i + 2].pos_ in ('NOUN', 'PROPN', 'PRON', 'DET', 'ADJ'):
            results.append((SUBJ_ORDER, tok.text))
    return results


def check_redundant_pronoun(doc) -> list[tuple[str, str]]:
    """
    Catches redundant personal pronoun in a relative clause where 'who' or 'which'
    already fills the subject role: 'a man who he was arguing' → 'he' is redundant.
    Uses 'who'/'which' only (not 'that') to avoid false positives on correct
    relative clauses like 'the book that he read'.
    """
    personal = {'he', 'she', 'it', 'they', 'i', 'we'}
    results = []
    for tok in doc:
        if tok.dep_ != 'nsubj' or tok.text.lower() not in personal:
            continue
        # verb already has 'who' or 'which' as any child → personal pronoun is redundant
        if any(c.text.lower() in ('who', 'which') for c in tok.head.children):
            results.append((DOUBLED_SUBJ, tok.text))
    return results


def check_infinitive_form(doc) -> list[tuple[str, str]]:
    """
    Catches wrong verb form directly after 'to' (infinitive marker):
    'to accepts' → 'to accept', 'to going' → 'to go'.
    Also catches VBG as pcomp of ADP 'to' after causative verbs:
    'causing everyone to becoming' → 'to become'.
    """
    results = []
    for tok in doc:
        if tok.pos_ != 'VERB' or tok.tag_ == 'VB':
            continue
        if any(c.tag_ == 'TO' and c.dep_ == 'aux' for c in tok.children):
            results.append((INF_FORM, tok.text))
    # catch ADP "to" + VBG pcomp when "to" is attached to any VERB (not ADJ/ADV)
    # e.g. "tried to running", "plan to working" — "to" should be infinitive marker
    # Excludes verbs where "to + VBG" is correct (resort to lying, commit to doing)
    for tok in doc:
        if tok.tag_ != 'VBG' or tok.dep_ != 'pcomp':
            continue
        prep = tok.head
        if prep.pos_ != 'ADP' or prep.text.lower() != 'to':
            continue
        head = prep.head
        if head.pos_ not in ('VERB', 'AUX'):
            continue  # "looking forward to going" — "forward" is ADV, not VERB
        if head.lemma_.lower() in VERB_TO_GERUND:
            continue  # "resort to lying", "be used to going" — correct constructions
        results.append((INF_FORM, tok.text))
    return results


def check_modal_have(doc) -> list[tuple[str, str]]:
    """
    Catches 'should had', 'would had', 'could had' — after a modal, the auxiliary
    must be 'have' (base form), not 'had' (past tense).
    """
    results = []
    for tok in doc:
        auxi = list(tok.children)
        has_modal = any(c.tag_ == 'MD' for c in auxi)
        if not has_modal:
            continue
        for child in auxi:
            if child.lemma_ == 'have' and child.tag_ == 'VBD':
                results.append((MODAL_HAVE, child.text))
    return results


def check_gerund_after_verb(doc) -> list[tuple[str, str]]:
    """
    Catches bare infinitive where a gerund is required:
    'enjoy swim' → 'enjoy swimming', 'avoid talk' → 'avoid talking'.
    For GERUND_OR_INF_VERBS (stop, finish...), skips when TO is present —
    'stopped to rest' is a valid purpose infinitive.
    """
    results = []
    for tok in doc:
        if tok.dep_ != 'xcomp' or tok.tag_ != 'VB':
            continue
        head_lemma = tok.head.lemma_.lower()
        if head_lemma not in GERUND_VERBS:
            continue
        # For ambiguous verbs (stop/finish/...), TO+VB is a valid purpose infinitive
        has_to = any(c.tag_ == 'TO' for c in tok.children)
        if has_to and head_lemma in GERUND_OR_INF_VERBS:
            continue
        results.append((GERUND_VERB, tok.text))
    return results


def check_wrong_rel_pronoun(doc) -> list[tuple[str, str]]:
    """
    Catches two patterns:
    - 'what' used as a relative pronoun after a noun: 'the items what I bought' → that/which
    - 'which' used for a person antecedent: 'the man which called' → who
    Free relative clauses ('what I want is...') are not flagged.
    Handles both dep=relcl (normal) and dep=csubj (parser mis-attachment).
    """
    results = []
    for tok in doc:
        lower = tok.text.lower()
        # "what" after any explicit noun antecedent
        if lower == 'what' and tok.tag_ == 'WP':
            if tok.dep_ not in ('nsubj', 'dobj', 'pobj', 'attr'):
                continue
            verb = tok.head
            if verb.dep_ == 'relcl' and verb.head.pos_ in ('NOUN', 'PROPN'):
                results.append((WRONG_REL_PRON, tok.text))
            elif verb.dep_ == 'csubj':
                # spaCy sometimes parses embedded relative clause as csubj
                # e.g. "All the items what she bought were on sale"
                # Check that there's a noun nsubj earlier in the sentence (the antecedent)
                head_verb = verb.head
                antecedent = next(
                    (c for c in head_verb.children
                     if c.dep_ == 'nsubj' and c.pos_ in ('NOUN', 'PROPN') and c.i < tok.i),
                    None
                )
                if antecedent is not None:
                    results.append((WRONG_REL_PRON, tok.text))
        # "which" when the antecedent noun is a person (should be "who")
        elif lower == 'which' and tok.tag_ == 'WDT':
            if tok.dep_ not in ('nsubj', 'nsubjpass', 'dobj', 'pobj'):
                continue
            verb = tok.head
            if verb.dep_ == 'relcl' and verb.head.lemma_.lower() in PERSON_NOUNS:
                results.append((WRONG_REL_PRON, tok.text))
    return results


def check_modal_to(doc) -> list[tuple[str, str]]:
    """
    Catches 'can to swim', 'must to go' — modal verbs don't take 'to' before the infinitive.
    In spaCy's parse, 'can' is ROOT and 'speak' is xcomp, so the modal is the HEAD of
    the infinitive verb rather than a child — we check both directions.
    Flags the redundant 'to' token.
    """
    results = []
    for tok in doc:
        if tok.tag_ != 'TO' or tok.dep_ != 'aux':
            continue
        verb = tok.head
        # modal as sibling aux child of the same verb
        has_modal_sibling = any(c.tag_ == 'MD' and c.dep_ == 'aux' for c in verb.children)
        # modal as the head of the verb (e.g. "can" ROOT → "speak" xcomp, "to" aux of speak)
        head_is_modal = verb.head.tag_ == 'MD'
        if has_modal_sibling or head_is_modal:
            results.append((MODAL_TO_ERR, tok.text))
    return results


def check_much_many(doc) -> list[tuple[str, str]]:
    """
    Catches 'much' used with countable plural nouns: 'much books' → 'many books'.
    """
    results = []
    tokens = list(doc)
    for i, tok in enumerate(tokens[:-1]):
        if tok.text.lower() != 'much':
            continue
        nxt = tokens[i + 1]
        if nxt.tag_ == 'NNS' and nxt.lemma_.lower() not in UNCOUNTABLE_NOUNS:
            results.append((MUCH_MANY, tok.text))
    return results


def check_fewer_less(doc) -> list[tuple[str, str]]:
    """
    Catches 'less' used with countable plural nouns: 'less students' → 'fewer students'.
    """
    results = []
    tokens = list(doc)
    for i, tok in enumerate(tokens[:-1]):
        if tok.text.lower() != 'less':
            continue
        nxt = tokens[i + 1]
        if nxt.tag_ == 'NNS' and nxt.lemma_.lower() not in UNCOUNTABLE_NOUNS:
            results.append((FEWER_LESS, tok.text))
    return results


def check_have_participle(doc) -> list[tuple[str, str]]:
    """
    Catches bare VB where VBN (past participle) is needed after have/has/had:
    'prices has change' → 'prices have changed'.
    Excludes 'have to + VB' (obligation) constructions.
    """
    results = []
    for tok in doc:
        if tok.tag_ != 'VB' or tok.pos_ != 'VERB':
            continue
        for child in tok.children:
            if child.dep_ == 'aux' and child.lemma_ == 'have':
                has_to = any(c.tag_ == 'TO' for c in tok.children)
                if not has_to:
                    results.append((HAVE_PART, tok.text))
                    break
    return results


def check_causative_form(doc) -> list[tuple[str, str]]:
    """
    Catches inflected verb (VBZ) in the bare-infinitive slot after causative verbs:
    'make me feels' → 'make me feel'.
    Skips complement verbs whose subject is a relative pronoun (WDT) to avoid
    flagging embedded relative clauses like 'which is something I never do'.
    """
    results = []
    for tok in doc:
        if tok.lemma_.lower() not in CAUSATIVE_VERBS:
            continue
        for child in tok.children:
            if child.dep_ not in ('ccomp', 'xcomp') or child.tag_ != 'VBZ':
                continue
            nsubj = next((c for c in child.children if c.dep_ == 'nsubj'), None)
            if nsubj is not None and nsubj.tag_ == 'WDT':
                continue
            results.append((INF_FORM, child.text))
    return results


def check_infinitive_after_prep(doc) -> list[tuple[str, str]]:
    """
    Catches infinitive used after a preposition that requires a gerund:
    'thinking about to move' → 'thinking about moving'.
    Skips the 'be about to' (imminent future) construction.
    """
    results = []
    tokens = list(doc)
    for i, tok in enumerate(tokens[:-1]):
        if tok.pos_ != 'ADP' or tok.text.lower() not in GERUND_PREPS:
            continue
        if tokens[i + 1].tag_ != 'TO':
            continue
        # skip "be about to + VB" — "be" must be the ROOT, not just an auxiliary
        if tok.text.lower() == 'about':
            prev = tokens[max(0, i - 3): i]
            if any(t.lemma_ == 'be' and t.dep_ == 'ROOT' for t in prev):
                continue
        results.append((INF_AFTER_PREP, tok.text))
    return results


# ── tiered check lists ────────────────────────────────────────────────────────
# Tier 1 (always on): checks so structurally specific they cannot false-positive.
# Run regardless of model confidence.
CHECKS_TIER1 = [
    check_have_participle,    # have/has/had + bare VB — structurally unambiguous
    check_since_for_ago,      # since + NUMBER + TIME_UNIT — structurally unambiguous
    check_incorrect_forms,    # known wrong irregular forms (goed, buyed...)
]

# Tier 2 (prob_err ≥ 0.35): high-precision, pattern-based rules with low false-positive risk.
CHECKS_TIER2 = [
    check_spelling,
    check_double_negative,
    check_negated_verb,
    check_missing_s,
    check_present_continuous,
    check_comparative,
    check_pronoun_case,
    check_there_their_theyre,
    check_participial_adj,
    check_doubled_subject,
    check_redundant_pronoun,
    check_subject_ordering,
    check_infinitive_after_prep,
    check_infinitive_form,
    check_modal_have,
    check_gerund_after_verb,
    check_wrong_rel_pronoun,
    check_causative_form,
    check_modal_to,
    check_much_many,
    check_fewer_less,
    check_word_order,    # frequency adverb after non-be verb — precise enough for T2
]

# Tier 3 (prob_err ≥ 0.45): context-dependent rules with higher false-positive risk.
# Only run when the model is fairly confident there's an error.
CHECKS_TIER3 = [
    check_subject_verb_agreement,
    check_verb_tense,
    check_run_on,
    check_preposition,
    check_modifier,
    check_parallelism,
    check_vague_pronoun,
    check_similar_words,
    check_adj_adv,
]

TIER2_THRESHOLD = 0.35
TIER3_THRESHOLD = 0.45


def classify_errors_tiered(sentence: str, prob_err: float) -> list[tuple[str, str]]:
    """
    Run rule-based checks appropriate for the model's confidence level.
    Returns (error_type, span) pairs, or [(UNKNOWN, '')] if nothing specific fires.
    """
    doc = nlp(sentence)
    errors = []
    for check in CHECKS_TIER1:
        errors.extend(check(doc))
    if prob_err < TIER2_THRESHOLD:
        return errors  # only tier-1 results; no UNKNOWN fallback at this level
    for check in CHECKS_TIER2:
        errors.extend(check(doc))
    if prob_err >= TIER3_THRESHOLD:
        for check in CHECKS_TIER3:
            errors.extend(check(doc))
    return errors if errors else [(UNKNOWN, '')]
