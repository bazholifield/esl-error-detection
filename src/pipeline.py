"""
Full ESL analysis pipeline.
Accepts a sentence or paragraph, runs all checks, and returns
structured Error objects with the offending span and a brief lesson.
"""

import torch
import torch.nn.functional as F
import spacy
from dataclasses import dataclass
from spellchecker import SpellChecker

from article_checker import check_articles
from error_classifier import classify_errors_tiered, INCORRECT_FORMS, EMOTION_ING_TO_ED, DOUBLE_CONJ, DESPITE_CLAUSE

nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()


@dataclass
class Error:
    error_type: str
    span: str       # the offending word(s) — empty string if not pinpointable
    lesson: str     # brief layman explanation


LESSONS = {
    "Article error (a/an)":
        "Use 'a' before consonant sounds and 'an' before vowel sounds. "
        "E.g., 'a cat', 'an apple', 'an hour' (silent h), 'a university' (sounds like 'you-niversity').",

    "Unnecessary article":
        "Some nouns don't take 'a' or 'an'. Uncountable nouns (advice, information, "
        "furniture) and proper nouns usually don't need an article — say 'some advice' or just 'information'.",

    "Article omission":
        "",  # generated dynamically — uses the span (the noun) to say "did you mean a/an/the X?"

    "Wrong article (a/an vs the)":
        "Use 'the' when talking about something specific or unique — especially "
        "with superlatives ('the best') or relative clauses ('the car that I bought').",

    "Spelling error":
        "",  # generated dynamically

    "Subject-verb agreement":
        "The verb must match the subject. 'He/she/it' → verb+s (runs, goes, doesn't). "
        "'I/we/they/you' → base form (run, go, don't). "
        "Past tense: 'I/he/she/it was', 'we/you/they were'.",

    "Pronoun case error":
        "Use subject pronouns (I, he, she, we, they) as the subject of a sentence. "
        "Use object pronouns (me, him, her, us, them) after verbs and prepositions. "
        "E.g., 'She called me', not 'Her called I'.",

    "Incorrect verb form after auxiliary":
        "",  # generated dynamically — names the wrong word

    "Verb tense error":
        "Keep verb tenses consistent with the time marker in the sentence. "
        "'Yesterday/last week/ago' → past tense. 'Now/today/currently' → present tense. "
        "'Tomorrow/next week' → future.",

    "Missing -s ending":
        "After number words (two, three, four...), the noun should be plural. "
        "E.g., 'two cats', 'three dogs', not 'two cat'.",

    "Present simple vs continuous":
        "Stative verbs like 'know', 'like', 'want', 'believe', and 'understand' "
        "aren't normally used in the -ing form. Say 'I know', not 'I am knowing'.",

    "Double negative":
        "In English, two negatives cancel each other out and create a positive. "
        "Use only one: 'I don't know anything' or 'I know nothing' — not both together.",

    "Incorrect irregular form":
        "",  # generated dynamically

    "Run-on sentence":
        "This looks like two complete sentences joined with just a comma (a comma splice). "
        "Either split them with a period, or join with 'and', 'but', or 'so'.",

    "Preposition error":
        "This preposition doesn't fit the usual pattern. "
        "Common pairings: 'interested in', 'good at', 'married to', 'arrive at/in', 'depend on', 'afraid of'.",

    "Word order error":
        "Frequency adverbs (always, usually, often, never, sometimes) go before the "
        "main verb but after 'be'. E.g., 'She always runs' / 'He is always late'.",

    "Misplaced modifier":
        "An opening phrase should describe the subject of the sentence. "
        "'Running to catch the bus, my bag fell' is unclear — the bag wasn't running.",

    "Faulty parallelism":
        "Items in a list should use the same grammatical form. "
        "E.g., 'I like reading, writing, and drawing' — not '...and to draw'.",

    "Vague pronoun reference":
        "When there are multiple nouns in a sentence, a pronoun like 'he' or 'they' can be ambiguous. "
        "Try using the noun itself to make it clear who you mean.",

    "Confused similar words":
        "Some words look or sound alike but mean different things. "
        "Common ones: your/you're, its/it's, then/than, affect/effect, loose/lose.",

    "Adjective/adverb confusion":
        "Use an adverb (often ending in -ly) to describe a verb, and an adjective to describe a noun. "
        "E.g., 'She sings beautifully' (not 'beautiful'), 'It's a quick solution' (not 'quickly').",

    "there/their/they're confusion":
        "",  # generated dynamically — specific to which word was used

    "Comparative/superlative error":
        "Don't combine 'more/most' with -er/-est endings — that's a double form. "
        "Say 'bigger' OR 'more big', not 'more bigger'. "
        "Irregular forms: good → better → best, bad → worse → worst.",

    "Participial adjective confusion":
        "",  # generated dynamically — names the word and its corrected form

    "Since/for/ago error":
        "Use 'for' with a duration ('for three years', 'for a week'), "
        "'since' with a point in time ('since 2010', 'since Monday'), "
        "and 'ago' with simple past ('three years ago', 'an hour ago').",

    "Doubled subject":
        "In English, a sentence only needs one subject. Remove either the noun or the pronoun. "
        "E.g., 'The teacher explained' or 'She explained' — not 'The teacher, she explained'.",

    "Subject ordering":
        "When listing people including yourself, put yourself last. "
        "E.g., 'my friend and I went' (not 'I and my friend'), "
        "'between my friend and me' (not 'between me and my friend').",

    "Infinitive after preposition":
        "After a preposition, use the gerund (-ing form), not the infinitive (to + verb). "
        "E.g., 'thinking about moving' (not 'to move'), 'interested in learning' (not 'to learn'), "
        "'without knowing' (not 'to know').",

    "Gerund required after verb":
        "Some verbs must be followed by the -ing form, not a base verb. "
        "E.g., 'stop worrying' (not 'stop worry'), 'enjoy swimming' (not 'enjoy swim'), "
        "'avoid making' (not 'avoid make'), 'keep trying' (not 'keep try').",

    "Wrong verb form after 'to'":
        "After 'to', always use the base (infinitive) form of the verb. "
        "E.g., 'to go' (not 'to going' or 'to goes'), 'to accept' (not 'to accepts'), "
        "'to become' (not 'to becoming').",

    "Modal + 'had' error":
        "After modal verbs (should, would, could, might), use 'have' not 'had'. "
        "E.g., 'should have gone' (not 'should had gone'), 'would have known' (not 'would had known').",

    "Wrong relative pronoun":
        "Use 'who' for people and 'which'/'that' for things in relative clauses — not 'what'. "
        "'What' is used for free relative clauses with no antecedent ('I know what you mean'). "
        "E.g., 'the person who called' (not 'which'), 'the book that I read' (not 'what').",

    "Modal + infinitive error":
        "After modal verbs (can, could, will, would, should, must, might, may), "
        "use the bare infinitive — no 'to'. "
        "E.g., 'She can speak French' (not 'can to speak'), 'you must go' (not 'must to go').",

    "Much/many confusion":
        "Use 'many' with countable nouns (things you can count individually) "
        "and 'much' with uncountable nouns. "
        "E.g., 'many friends', 'many books', 'many students' — not 'much friends'. "
        "'Much' is correct for: 'much water', 'much advice', 'much information'.",

    "Fewer/less error":
        "Use 'fewer' with countable nouns and 'less' with uncountable nouns. "
        "E.g., 'fewer students', 'fewer cars', 'fewer mistakes' — not 'less students'. "
        "'Less' is correct for: 'less water', 'less time', 'less money'.",

    "Missing past participle":
        "After 'have', 'has', or 'had', use the past participle form of the verb (often ending in -ed or an irregular form). "
        "E.g., 'I have finished' (not 'have finish'), 'she has gone' (not 'has go'), "
        "'they had eaten' (not 'had eat').",

    "Double conjunction":
        "Don't combine a subordinating conjunction (although, because, though, whereas) with a "
        "coordinating conjunction (but, so, yet) or discourse connector (however) in the same clause — choose one structure. "
        "E.g., 'Although I was tired, I kept going.' OR 'I was tired, but I kept going.' — not both.",

    "'Despite' used as conjunction":
        "'Despite' is a preposition, so it needs a noun or gerund (-ing form) — not a full clause. "
        "Try 'Despite being tired, I kept going.' or swap it for 'although': 'Although I was tired, I kept going.'",

    "Grammatical error (unclassified)":
        "This sentence may contain a grammatical error that couldn't be automatically identified.",
}


def _lesson(error_type: str, span: str) -> str:
    """Return the lesson for an error. Dynamic for types where the span changes the message."""

    if error_type == "Spelling error":
        correction = spell.correction(span)
        if correction and correction.lower() != span.lower():
            return f"'{span}' may be misspelled. Did you mean '{correction}'?"
        return f"'{span}' may be misspelled. Check the spelling."

    if error_type == "Incorrect irregular form":
        correction = INCORRECT_FORMS.get(span.lower(), '')
        if correction:
            return f"'{span}' is not the standard form. Use '{correction}' instead."
        return f"'{span}' may be an incorrect verb or noun form."

    if error_type == "Article omission":
        return (
            f"'{span}' seems to be missing a determiner. "
            f"Did you mean 'a {span}', 'an {span}', or 'the {span}'?"
        )

    if error_type == "Incorrect verb form after auxiliary":
        return (
            f"After 'do/does/did', always use the base (infinitive) form of the verb. "
            f"'{span}' should be its base form here. "
            f"E.g., 'didn't go' (not 'went'), 'doesn't run' (not 'runs')."
        )

    if error_type == "there/their/they're confusion":
        lower = span.lower()
        if lower == "their":
            return (
                "'Their' is possessive — it means 'belonging to them' (their house, their idea). "
                "Did you mean 'they're' (= they are) or 'there' (a place, or 'there is/are')?"
            )
        if lower == "there":
            return (
                "'There' is used for locations or in 'there is/are'. "
                "Did you mean 'their' (possessive — belonging to them) or 'they're' (= they are)?"
            )
        return (
            "'They're' is a contraction of 'they are'. "
            "Did you mean 'their' (possessive) or 'there' (a place or 'there is/are')?"
        )

    if error_type == "Participial adjective confusion":
        ed_form = EMOTION_ING_TO_ED.get(span.lower(), '')
        if ed_form:
            return (
                f"'-ing' adjectives describe what causes a feeling, '-ed' adjectives describe how someone feels. "
                f"Since the subject is a person experiencing the feeling, try '{ed_form}' instead of '{span}'. "
                f"E.g., 'The lesson was {span}' (it caused the feeling) vs 'I was {ed_form}' (I felt it)."
            )
        return (
            "'-ing' adjectives describe what causes a feeling (boring film), "
            "'-ed' adjectives describe how someone feels (bored person)."
        )

    return LESSONS.get(error_type, '')


def _annotate_tokens(doc, errors: list[Error]) -> list[dict]:
    """Map errors onto spaCy tokens so the frontend can highlight individual words."""
    span_errors: dict[str, list[Error]] = {}
    for err in errors:
        if err.span:
            span_errors.setdefault(err.span, []).append(err)

    span_token_idx: dict[str, int] = {}
    for i, tok in enumerate(doc):
        if tok.text in span_errors and tok.text not in span_token_idx:
            span_token_idx[tok.text] = i

    token_errors: dict[int, list[Error]] = {
        span_token_idx[span]: errs
        for span, errs in span_errors.items()
        if span in span_token_idx
    }

    return [
        {
            'text': tok.text,
            'whitespace': tok.whitespace_,
            'errors': [{'type': e.error_type, 'lesson': e.lesson} for e in token_errors.get(i, [])]
        }
        for i, tok in enumerate(doc)
    ]


def _merge_sentences(sents: list[str]) -> list[str]:
    """Merge sentences that start with a coordinating conjunction into the previous one."""
    merged = []
    for s in sents:
        first_word = s.split()[0].lower() if s.split() else ''
        if merged and first_word in ('and', 'but', 'or', 'so', 'yet', 'nor'):
            merged[-1] = merged[-1].rstrip() + ' ' + s
        else:
            merged.append(s)
    return merged


def analyze(text: str, tokenizer, model) -> list[dict]:
    """
    Analyze a sentence or paragraph.
    Returns a list of per-sentence dicts:
      [{'sentence': str, 'errors': list[Error], 'tokens': list[dict]}, ...]
    """
    doc = nlp(text)
    raw_sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    results = []
    for sent_text in _merge_sentences(raw_sents):
        errors, tokens = _analyze_sentence(sent_text, tokenizer, model)
        results.append({'sentence': sent_text, 'errors': errors, 'tokens': tokens})
    return results


def _analyze_sentence(text: str, tokenizer, model) -> tuple[list[Error], list[dict]]:
    """Run all checks on a single sentence. Returns (errors, token_annotations)."""
    errors: list[Error] = []
    seen: set[tuple[str, str]] = set()

    def add(pairs: list[tuple[str, str]]):
        for error_type, span in pairs:
            key = (error_type, span)
            if key not in seen:
                seen.add(key)
                errors.append(Error(error_type, span, _lesson(error_type, span)))

    add(check_articles(text))

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    prob_err = F.softmax(logits, dim=-1)[0, 1].item()

    seen_types = {e.error_type for e in errors}
    add([(t, s) for t, s in classify_errors_tiered(text, prob_err) if t not in seen_types])

    doc = nlp(text)
    return errors, _annotate_tokens(doc, errors)
