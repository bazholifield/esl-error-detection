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
from error_classifier import classify_errors, INCORRECT_FORMS

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
        "furniture) and proper nouns usually don't need an article — say 'some advice' "
        "or just 'information'.",

    "Article omission":
        "This noun probably needs an article. Try adding 'a', 'an', or 'the' before it.",

    "Wrong article (a/an vs the)":
        "Use 'the' when talking about something specific or unique — especially "
        "with superlatives ('the best') or relative clauses ('the car that I bought').",

    "Spelling error":
        "",  # generated dynamically in _lesson()

    "Subject-verb agreement":
        "The verb must match the subject. 'He/she/it' → verb+s (runs, goes, doesn't). "
        "'I/we/they/you' → base form (run, go, don't). "
        "Past tense: 'I/he/she/it was', 'we/you/they were'.",

    "Pronoun case error":
        "Use subject pronouns (I, he, she, we, they) as the subject of a sentence. "
        "Use object pronouns (me, him, her, us, them) after verbs and prepositions. "
        "E.g., 'She called me', not 'Her called I'.",

    "Verb tense error":
        "Keep verb tenses consistent with the time being described. "
        "Words like 'today', 'now', and 'currently' signal present tense.",

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
        "",  # generated dynamically in _lesson()

    "Run-on sentence":
        "This looks like two complete sentences joined with just a comma (a comma splice). "
        "Either split them with a period, or join with 'and', 'but', or 'so'.",

    "Preposition error":
        "This preposition doesn't fit the usual pattern. "
        "Common pairings: 'interested in', 'good at', 'married to', "
        "'arrive at/in', 'depend on', 'afraid of'.",

    "Word order error":
        "Frequency adverbs (always, usually, often, never, sometimes) go before the "
        "main verb but after 'be'. E.g., 'She always runs' / 'He is always late'.",

    "Misplaced modifier":
        "An opening phrase should describe the subject of the sentence. "
        "'Running to catch the bus, my bag fell' is unclear — the bag wasn't running. "
        "Make sure the subject is the one doing the action in the opening phrase.",

    "Faulty parallelism":
        "Items in a list should use the same grammatical form. "
        "E.g., 'I like reading, writing, and drawing' — not '...and to draw'.",

    "Vague pronoun reference":
        "When there are multiple nouns in a sentence, a pronoun like 'he', 'she', or "
        "'they' can be ambiguous. Try using the noun itself to make it clear who you mean.",

    "Confused similar words":
        "Some words look or sound alike but mean different things. "
        "Common ones: their/there/they're, your/you're, its/it's, then/than, affect/effect.",

    "Adjective/adverb confusion":
        "Use an adverb (often ending in -ly) to describe a verb, and an adjective to "
        "describe a noun. E.g., 'She sings beautifully' (not 'beautiful'), "
        "'It's a quick solution' (not 'quickly').",

    "Grammatical error (unclassified)":
        "This sentence may contain a grammatical error that couldn't be automatically identified.",
}


def _lesson(error_type: str, span: str) -> str:
    """Return the lesson for an error, with dynamic text for spelling/irregular forms."""
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

    return LESSONS.get(error_type, '')


def analyze(text: str, tokenizer, model, threshold: float = 0.25) -> list[dict]:
    """
    Analyze a sentence or paragraph.
    Returns a list of per-sentence dicts: [{'sentence': str, 'errors': list[Error]}, ...]
    """
    doc = nlp(text)
    results = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        errors = _analyze_sentence(sent_text, tokenizer, model, threshold)
        results.append({'sentence': sent_text, 'errors': errors})
    return results


def _analyze_sentence(text: str, tokenizer, model, threshold: float) -> list[Error]:
    """Run all checks on a single sentence and return Error objects."""
    errors: list[Error] = []
    seen: set[tuple[str, str]] = set()

    def add(pairs: list[tuple[str, str]]):
        for error_type, span in pairs:
            key = (error_type, span)
            if key not in seen:
                seen.add(key)
                errors.append(Error(error_type, span, _lesson(error_type, span)))

    # article checks always run — high precision, no model needed
    add(check_articles(text))

    # binary model decides whether to run the full classifier
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    prob_err = F.softmax(logits, dim=-1)[0, 1].item()

    if prob_err >= threshold:
        seen_types = {e.error_type for e in errors}
        add([
            (t, s) for t, s in classify_errors(text)
            if t not in seen_types
        ])

    return errors
