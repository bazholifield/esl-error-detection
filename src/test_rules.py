"""
Comprehensive rule-based check tests.
Covers both the new conjunction checks and regression tests for all existing checks.

Run with:  pytest src/test_rules.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
import spacy
from article_checker import check_articles
from error_classifier import (
    nlp,
    # new checks
    check_double_conjunction, DOUBLE_CONJ,
    check_despite_clause, DESPITE_CLAUSE,
    # existing checks
    check_spelling, SPELLING,
    check_subject_verb_agreement, SVA,
    check_pronoun_case, PRONOUN,
    check_negated_verb, NEGATED_VERB,
    check_verb_tense, TENSE,
    check_missing_s, MISSING_S,
    check_present_continuous, PRES_CONT,
    check_double_negative, DOUBLE_NEG,
    check_incorrect_forms, IRREGULAR,
    check_run_on, RUN_ON,
    check_preposition, PREPOSITION,
    check_word_order, WORD_ORDER,
    check_modifier, MODIFIER,
    check_parallelism, PARALLELISM,
    check_there_their_theyre, THERE_THEIR,
    check_comparative, COMPARATIVE,
    check_participial_adj, PARTICIPIAL_ADJ,
    check_since_for_ago, SINCE_FOR_AGO,
    check_doubled_subject, DOUBLED_SUBJ,
    check_subject_ordering, SUBJ_ORDER,
    check_infinitive_after_prep, INF_AFTER_PREP,
    check_gerund_after_verb, GERUND_VERB,
    check_wrong_rel_pronoun, WRONG_REL_PRON,
    check_modal_have, MODAL_HAVE,
    check_modal_to, MODAL_TO_ERR,
    check_much_many, MUCH_MANY,
    check_fewer_less, FEWER_LESS,
    check_have_participle, HAVE_PART,
    check_infinitive_form, INF_FORM,
    check_similar_words, SIMILAR_WORDS,
    check_adj_adv, ADJ_ADV,
)
from article_checker import A_AN_ERROR, UNNECESSARY, OMISSION


def types(results):
    """Extract just the error type strings from (type, span) pairs."""
    return [r[0] for r in results]


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Double conjunction
# ─────────────────────────────────────────────────────────────────────────────

class TestDoubleConjunction:
    def test_although_but(self):
        doc = nlp("Although I was tired, but I kept going.")
        assert DOUBLE_CONJ in types(check_double_conjunction(doc))

    def test_although_but_no_comma(self):
        doc = nlp("Although she studied hard but she failed the exam.")
        assert DOUBLE_CONJ in types(check_double_conjunction(doc))

    def test_though_but(self):
        doc = nlp("Though it was raining, but we still went out.")
        assert DOUBLE_CONJ in types(check_double_conjunction(doc))

    def test_even_though_but(self):
        # 'even though' — spaCy marks 'though' as mark, so this is caught
        doc = nlp("Even though she tried her best, but she failed.")
        assert DOUBLE_CONJ in types(check_double_conjunction(doc))

    def test_whereas_but(self):
        doc = nlp("Whereas he prefers tea, but she prefers coffee.")
        assert DOUBLE_CONJ in types(check_double_conjunction(doc))

    def test_although_yet(self):
        doc = nlp("Although I tried, yet it was not enough.")
        assert DOUBLE_CONJ in types(check_double_conjunction(doc))

    def test_although_however(self):
        doc = nlp("Although I was tired, however I kept going.")
        assert DOUBLE_CONJ in types(check_double_conjunction(doc))

    def test_because_so(self):
        doc = nlp("Because it was raining, so we stayed inside.")
        assert DOUBLE_CONJ in types(check_double_conjunction(doc))

    def test_because_so_no_comma(self):
        doc = nlp("Because he was late so they started without him.")
        assert DOUBLE_CONJ in types(check_double_conjunction(doc))

    # ── negative cases (correct sentences — should NOT fire) ──────────────────

    def test_correct_although_no_but(self):
        doc = nlp("Although I was tired, I kept going.")
        assert DOUBLE_CONJ not in types(check_double_conjunction(doc))

    def test_correct_but_no_although(self):
        doc = nlp("I was tired, but I kept going.")
        assert DOUBLE_CONJ not in types(check_double_conjunction(doc))

    def test_correct_because_no_so(self):
        doc = nlp("Because it was cold, we stayed inside.")
        assert DOUBLE_CONJ not in types(check_double_conjunction(doc))

    def test_correct_so_no_because(self):
        doc = nlp("It was cold, so we stayed inside.")
        assert DOUBLE_CONJ not in types(check_double_conjunction(doc))

    def test_not_only_but_also(self):
        # 'not only...but also' is a valid correlative — should NOT be flagged
        doc = nlp("Not only did she sing, but she also danced.")
        assert DOUBLE_CONJ not in types(check_double_conjunction(doc))

    def test_not_only_but_also_inline(self):
        doc = nlp("She can sing not only well but also beautifully.")
        assert DOUBLE_CONJ not in types(check_double_conjunction(doc))

    def test_simple_but(self):
        doc = nlp("She was tired, but she kept going.")
        assert DOUBLE_CONJ not in types(check_double_conjunction(doc))

    def test_no_conjunction_at_all(self):
        doc = nlp("I went to the store.")
        assert DOUBLE_CONJ not in types(check_double_conjunction(doc))


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Despite + clause
# ─────────────────────────────────────────────────────────────────────────────

class TestDespiteClause:
    def test_despite_she(self):
        doc = nlp("Despite she was tired, she kept going.")
        assert DESPITE_CLAUSE in types(check_despite_clause(doc))

    def test_despite_he(self):
        doc = nlp("Despite he was wrong, they believed him.")
        assert DESPITE_CLAUSE in types(check_despite_clause(doc))

    def test_despite_i(self):
        doc = nlp("Despite I am tired, I will continue working.")
        assert DESPITE_CLAUSE in types(check_despite_clause(doc))

    def test_despite_we(self):
        doc = nlp("Despite we tried hard, we could not finish.")
        assert DESPITE_CLAUSE in types(check_despite_clause(doc))

    def test_despite_they(self):
        doc = nlp("Despite they warned us, we went anyway.")
        assert DESPITE_CLAUSE in types(check_despite_clause(doc))

    # ── negative cases ────────────────────────────────────────────────────────

    def test_correct_despite_gerund(self):
        doc = nlp("Despite being tired, she kept going.")
        assert DESPITE_CLAUSE not in types(check_despite_clause(doc))

    def test_correct_despite_noun(self):
        doc = nlp("Despite the rain, we went out.")
        assert DESPITE_CLAUSE not in types(check_despite_clause(doc))

    def test_correct_despite_possessive(self):
        doc = nlp("Despite her hard work, she failed.")
        assert DESPITE_CLAUSE not in types(check_despite_clause(doc))

    def test_correct_despite_it_being(self):
        # 'despite it being rainy' — accusative + gerund, correct
        doc = nlp("Despite it being rainy outside, I felt warm inside.")
        assert DESPITE_CLAUSE not in types(check_despite_clause(doc))

    def test_correct_despite_you_not_being(self):
        doc = nlp("Despite you not being there, we had fun.")
        assert DESPITE_CLAUSE not in types(check_despite_clause(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Article checks
# ─────────────────────────────────────────────────────────────────────────────

class TestArticleChecks:
    def test_a_before_vowel(self):
        results = check_articles("I have a apple.")
        assert A_AN_ERROR in types(results)

    def test_an_before_consonant(self):
        results = check_articles("She is an cat.")
        assert A_AN_ERROR in types(results)

    def test_a_before_silent_h(self):
        # "an hour" is correct — should NOT flag
        results = check_articles("It took an hour.")
        assert A_AN_ERROR not in types(results)

    def test_unnecessary_article_uncountable(self):
        results = check_articles("She gave me an advice.")
        assert UNNECESSARY in types(results)

    def test_correct_article(self):
        results = check_articles("I have a cat.")
        assert A_AN_ERROR not in types(results)
        assert UNNECESSARY not in types(results)


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Subject-verb agreement
# ─────────────────────────────────────────────────────────────────────────────

class TestSubjectVerbAgreement:
    def test_she_go(self):
        doc = nlp("She go to school every day.")
        assert SVA in types(check_subject_verb_agreement(doc))

    def test_they_was(self):
        doc = nlp("They was late to the meeting.")
        assert SVA in types(check_subject_verb_agreement(doc))

    def test_he_were(self):
        doc = nlp("He were happy.")
        assert SVA in types(check_subject_verb_agreement(doc))

    def test_correct_she_goes(self):
        doc = nlp("She goes to school every day.")
        assert SVA not in types(check_subject_verb_agreement(doc))

    def test_correct_they_were(self):
        doc = nlp("They were late.")
        assert SVA not in types(check_subject_verb_agreement(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Pronoun case
# ─────────────────────────────────────────────────────────────────────────────

class TestPronounCase:
    def test_object_pronoun_as_subject(self):
        doc = nlp("Him called me yesterday.")
        assert PRONOUN in types(check_pronoun_case(doc))

    def test_subject_pronoun_as_object(self):
        # "I" mid-sentence so it isn't fused with the period into a single NNP token
        doc = nlp("She called I yesterday.")
        assert PRONOUN in types(check_pronoun_case(doc))

    def test_correct_pronouns(self):
        doc = nlp("She called me yesterday.")
        assert PRONOUN not in types(check_pronoun_case(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Negated verb form
# ─────────────────────────────────────────────────────────────────────────────

class TestNegatedVerb:
    def test_didnt_understood(self):
        doc = nlp("She didn't understood the question.")
        assert NEGATED_VERB in types(check_negated_verb(doc))

    def test_doesnt_runs(self):
        doc = nlp("He doesn't runs every day.")
        assert NEGATED_VERB in types(check_negated_verb(doc))

    def test_correct_didnt_understand(self):
        doc = nlp("She didn't understand the question.")
        assert NEGATED_VERB not in types(check_negated_verb(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Verb tense
# ─────────────────────────────────────────────────────────────────────────────

class TestVerbTense:
    def test_yesterday_present(self):
        doc = nlp("Yesterday I go to the store.")
        assert TENSE in types(check_verb_tense(doc))

    def test_correct_yesterday_past(self):
        doc = nlp("Yesterday I went to the store.")
        assert TENSE not in types(check_verb_tense(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Missing plural -s
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingS:
    def test_two_cat(self):
        doc = nlp("I have two cat.")
        assert MISSING_S in types(check_missing_s(doc))

    def test_correct_two_cats(self):
        doc = nlp("I have two cats.")
        assert MISSING_S not in types(check_missing_s(doc))

    def test_one_cat_no_flag(self):
        # 'one' is singular — should not require plural
        doc = nlp("I have one cat.")
        assert MISSING_S not in types(check_missing_s(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Stative verb in progressive
# ─────────────────────────────────────────────────────────────────────────────

class TestPresentContinuous:
    def test_am_knowing(self):
        doc = nlp("I am knowing the answer.")
        assert PRES_CONT in types(check_present_continuous(doc))

    def test_correct_i_know(self):
        doc = nlp("I know the answer.")
        assert PRES_CONT not in types(check_present_continuous(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Double negative
# ─────────────────────────────────────────────────────────────────────────────

class TestDoubleNegative:
    def test_dont_know_nothing(self):
        doc = nlp("I don't know nothing.")
        assert DOUBLE_NEG in types(check_double_negative(doc))

    def test_correct_dont_know_anything(self):
        doc = nlp("I don't know anything.")
        assert DOUBLE_NEG not in types(check_double_negative(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Incorrect irregular forms
# ─────────────────────────────────────────────────────────────────────────────

class TestIncorrectForms:
    def test_goed(self):
        doc = nlp("She goed to the store.")
        assert IRREGULAR in types(check_incorrect_forms(doc))

    def test_buyed(self):
        doc = nlp("He buyed a new car.")
        assert IRREGULAR in types(check_incorrect_forms(doc))

    def test_correct_went(self):
        doc = nlp("She went to the store.")
        assert IRREGULAR not in types(check_incorrect_forms(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Run-on sentence (comma splice)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunOn:
    def test_comma_splice(self):
        doc = nlp("I went to the store, I bought some milk.")
        assert RUN_ON in types(check_run_on(doc))

    def test_correct_with_conjunction(self):
        doc = nlp("I went to the store and I bought some milk.")
        assert RUN_ON not in types(check_run_on(doc))

    def test_correct_with_period(self):
        # single clause — no comma splice
        doc = nlp("I went to the store.")
        assert RUN_ON not in types(check_run_on(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Preposition errors
# ─────────────────────────────────────────────────────────────────────────────

class TestPreposition:
    def test_married_with(self):
        doc = nlp("She is married with him.")
        assert PREPOSITION in types(check_preposition(doc))

    def test_interested_on(self):
        doc = nlp("I am interested on science.")
        assert PREPOSITION in types(check_preposition(doc))

    def test_correct_married_to(self):
        doc = nlp("She is married to him.")
        assert PREPOSITION not in types(check_preposition(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Word order (frequency adverbs)
# ─────────────────────────────────────────────────────────────────────────────

class TestWordOrder:
    def test_runs_always(self):
        doc = nlp("She runs always.")
        assert WORD_ORDER in types(check_word_order(doc))

    def test_correct_always_runs(self):
        doc = nlp("She always runs.")
        assert WORD_ORDER not in types(check_word_order(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Comparative/superlative
# ─────────────────────────────────────────────────────────────────────────────

class TestComparative:
    def test_more_taller(self):
        doc = nlp("She is more taller than him.")
        assert COMPARATIVE in types(check_comparative(doc))

    def test_most_tallest(self):
        doc = nlp("He is the most tallest person here.")
        assert COMPARATIVE in types(check_comparative(doc))

    def test_correct_taller(self):
        doc = nlp("She is taller than him.")
        assert COMPARATIVE not in types(check_comparative(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Participial adjective confusion
# ─────────────────────────────────────────────────────────────────────────────

class TestParticipialAdj:
    def test_i_am_interesting(self):
        doc = nlp("I am very interesting in science.")
        assert PARTICIPIAL_ADJ in types(check_participial_adj(doc))

    def test_i_feel_boring(self):
        doc = nlp("I feel boring.")
        assert PARTICIPIAL_ADJ in types(check_participial_adj(doc))

    def test_correct_i_am_interested(self):
        doc = nlp("I am very interested in science.")
        assert PARTICIPIAL_ADJ not in types(check_participial_adj(doc))

    def test_correct_the_film_was_boring(self):
        # film (non-personal subject) is the cause — correct use of -ing form
        doc = nlp("The film was boring.")
        assert PARTICIPIAL_ADJ not in types(check_participial_adj(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: there/their/they're
# ─────────────────────────────────────────────────────────────────────────────

class TestThereTheirTheyre:
    def test_their_as_adverb(self):
        doc = nlp("Put it over their.")
        assert THERE_THEIR in types(check_there_their_theyre(doc))

    def test_there_as_possessive(self):
        # spaCy reads "There house" as existential-there, so use a position where
        # "there" can't be existential — here as a determiner inside a prepositional phrase
        doc = nlp("I visited there country last year.")
        assert THERE_THEIR in types(check_there_their_theyre(doc))

    def test_correct_their_possessive(self):
        doc = nlp("Their house is big.")
        assert THERE_THEIR not in types(check_there_their_theyre(doc))

    def test_correct_there_location(self):
        doc = nlp("Put it over there.")
        assert THERE_THEIR not in types(check_there_their_theyre(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Since/for/ago
# ─────────────────────────────────────────────────────────────────────────────

class TestSinceForAgo:
    def test_since_three_years(self):
        doc = nlp("I have been here since three years.")
        assert SINCE_FOR_AGO in types(check_since_for_ago(doc))

    def test_correct_for_three_years(self):
        doc = nlp("I have been here for three years.")
        assert SINCE_FOR_AGO not in types(check_since_for_ago(doc))

    def test_correct_since_point_in_time(self):
        doc = nlp("I have been here since 2010.")
        assert SINCE_FOR_AGO not in types(check_since_for_ago(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Doubled subject
# ─────────────────────────────────────────────────────────────────────────────

class TestDoubledSubject:
    def test_teacher_she(self):
        doc = nlp("The teacher, she explained the lesson.")
        assert DOUBLED_SUBJ in types(check_doubled_subject(doc))

    def test_correct_no_double(self):
        doc = nlp("The teacher explained the lesson.")
        assert DOUBLED_SUBJ not in types(check_doubled_subject(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Subject ordering
# ─────────────────────────────────────────────────────────────────────────────

class TestSubjectOrdering:
    def test_me_and_friend(self):
        doc = nlp("Me and my friend went to school.")
        assert SUBJ_ORDER in types(check_subject_ordering(doc))

    def test_i_and_john(self):
        doc = nlp("I and John went to the park.")
        assert SUBJ_ORDER in types(check_subject_ordering(doc))

    def test_correct_friend_and_i(self):
        doc = nlp("My friend and I went to school.")
        assert SUBJ_ORDER not in types(check_subject_ordering(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Infinitive after preposition
# ─────────────────────────────────────────────────────────────────────────────

class TestInfinitiveAfterPrep:
    def test_about_to_move(self):
        doc = nlp("I am thinking about to move to a different city.")
        assert INF_AFTER_PREP in types(check_infinitive_after_prep(doc))

    def test_correct_about_moving(self):
        doc = nlp("I am thinking about moving to a different city.")
        assert INF_AFTER_PREP not in types(check_infinitive_after_prep(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Gerund required after verb
# ─────────────────────────────────────────────────────────────────────────────

class TestGerundAfterVerb:
    def test_enjoy_swim(self):
        # "swim" alone is parsed as NN (a swim = noun); use "avoid" + bare verb which
        # spaCy reliably parses as a verbal xcomp
        doc = nlp("I avoid eat junk food.")
        assert GERUND_VERB in types(check_gerund_after_verb(doc))

    def test_correct_enjoy_swimming(self):
        doc = nlp("I enjoy swimming.")
        assert GERUND_VERB not in types(check_gerund_after_verb(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Wrong relative pronoun
# ─────────────────────────────────────────────────────────────────────────────

class TestWrongRelPronoun:
    def test_which_for_person(self):
        doc = nlp("The man which called me was angry.")
        assert WRONG_REL_PRON in types(check_wrong_rel_pronoun(doc))

    def test_correct_who_for_person(self):
        doc = nlp("The man who called me was angry.")
        assert WRONG_REL_PRON not in types(check_wrong_rel_pronoun(doc))

    def test_correct_which_for_thing(self):
        doc = nlp("The book which I read was great.")
        assert WRONG_REL_PRON not in types(check_wrong_rel_pronoun(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Modal errors
# ─────────────────────────────────────────────────────────────────────────────

class TestModalHave:
    def test_should_had(self):
        doc = nlp("She should had gone to the meeting.")
        assert MODAL_HAVE in types(check_modal_have(doc))

    def test_correct_should_have(self):
        doc = nlp("She should have gone to the meeting.")
        assert MODAL_HAVE not in types(check_modal_have(doc))


class TestModalTo:
    def test_can_to_swim(self):
        doc = nlp("She can to swim very well.")
        assert MODAL_TO_ERR in types(check_modal_to(doc))

    def test_correct_can_swim(self):
        doc = nlp("She can swim very well.")
        assert MODAL_TO_ERR not in types(check_modal_to(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Much/many and fewer/less
# ─────────────────────────────────────────────────────────────────────────────

class TestMuchMany:
    def test_much_friends(self):
        doc = nlp("I have much friends.")
        assert MUCH_MANY in types(check_much_many(doc))

    def test_correct_many_friends(self):
        doc = nlp("I have many friends.")
        assert MUCH_MANY not in types(check_much_many(doc))


class TestFewerLess:
    def test_less_students(self):
        doc = nlp("There are less students than before.")
        assert FEWER_LESS in types(check_fewer_less(doc))

    def test_correct_fewer_students(self):
        doc = nlp("There are fewer students than before.")
        assert FEWER_LESS not in types(check_fewer_less(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Have + past participle
# ─────────────────────────────────────────────────────────────────────────────

class TestHaveParticiple:
    def test_has_finish(self):
        doc = nlp("She has finish her homework.")
        assert HAVE_PART in types(check_have_participle(doc))

    def test_correct_has_finished(self):
        doc = nlp("She has finished her homework.")
        assert HAVE_PART not in types(check_have_participle(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Wrong verb form after 'to' / wrong infinitive form
# ─────────────────────────────────────────────────────────────────────────────

class TestInfinitiveForm:
    def test_to_accepts(self):
        doc = nlp("She wants to accepts the offer.")
        assert INF_FORM in types(check_infinitive_form(doc))

    def test_correct_to_accept(self):
        doc = nlp("She wants to accept the offer.")
        assert INF_FORM not in types(check_infinitive_form(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Adjective/adverb confusion
# ─────────────────────────────────────────────────────────────────────────────

class TestAdjAdv:
    def test_speaks_beautiful(self):
        doc = nlp("She speaks English very beautiful.")
        assert ADJ_ADV in types(check_adj_adv(doc))

    def test_correct_speaks_beautifully(self):
        doc = nlp("She speaks English beautifully.")
        assert ADJ_ADV not in types(check_adj_adv(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Misplaced modifier
# ─────────────────────────────────────────────────────────────────────────────

class TestModifier:
    def test_dangling_participial(self):
        doc = nlp("Running to catch the bus, my bag fell.")
        assert MODIFIER in types(check_modifier(doc))

    def test_correct_with_person_subject(self):
        doc = nlp("Running to catch the bus, I dropped my bag.")
        assert MODIFIER not in types(check_modifier(doc))


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION: Faulty parallelism
# ─────────────────────────────────────────────────────────────────────────────

class TestParallelism:
    def test_mixed_forms_in_list(self):
        doc = nlp("I like swimming, running, and to dance.")
        assert PARALLELISM in types(check_parallelism(doc))

    def test_correct_parallel_list(self):
        doc = nlp("I like swimming, running, and dancing.")
        assert PARALLELISM not in types(check_parallelism(doc))
