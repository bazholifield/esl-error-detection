"""
Comprehensive ESL error detector test suite.
Positive tests: sentences WITH known errors — something should be flagged.
Negative tests: clean sentences — nothing should be flagged.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import CHECKPOINT_PATH
from pipeline import analyze

print("Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_PATH, local_files_only=True)
model.eval()
print("Model loaded.\n", flush=True)

def run(text):
    results = analyze(text, tokenizer, model)
    errors = []
    for r in results:
        errors.extend(r['errors'])
    return errors

# (sentence, description)
POSITIVE = [
    # ── Article errors ─────────────────────────────────────────────────────────
    ("She is a honest person.",                             "a/an: a→an"),
    ("I gave her a advice about the trip.",                 "unnecessary article: a advice"),
    ("Can you give me an information about the flight?",    "unnecessary article: an information"),
    ("I need a furniture for my new apartment.",            "unnecessary article: a furniture"),
    ("I have dog at home.",                                 "article omission: dog"),

    # ── SVA ────────────────────────────────────────────────────────────────────
    ("She don't like vegetables.",                          "SVA: don't→doesn't"),
    ("They was happy about the results.",                   "SVA: was→were"),
    ("My parents is coming tomorrow.",                      "SVA: is→are"),
    ("The man who have the keys is standing there.",        "relcl SVA: have→has"),
    ("The dogs that runs every morning are very fit.",      "relcl SVA: runs→run"),
    ("The store which were crowded had long lines.",        "relcl SVA: were→was"),
    ("There is many students in the class today.",          "SVA: is→are with existential there"),

    # ── Pronoun case ───────────────────────────────────────────────────────────
    ("Me went to the market alone yesterday.",              "pronoun case: Me→I"),
    ("John called she yesterday afternoon.",                "pronoun case: she→her"),
    ("Between you and I, this is a secret.",               "pronoun case: I→me"),
    ("The teacher gave the book to he.",                   "pronoun case: he→him"),

    # ── Subject ordering ───────────────────────────────────────────────────────
    ("Me and my friend went to the store.",                 "subject ordering: Me→My friend and I"),
    ("I and John went to the cinema last night.",           "subject ordering: I and John"),

    # ── Verb tense ────────────────────────────────────────────────────────────
    ("Yesterday, she is very happy about the news.",        "tense: is→was"),
    ("Last week, they go to the museum.",                   "tense: go→went"),
    ("I see him yesterday at the park.",                    "tense: see→saw"),

    # ── Negated verb form ─────────────────────────────────────────────────────
    ("I didn't understood the question.",                   "negated verb: understood→understand"),
    ("She doesn't runs in the morning.",                    "negated verb: runs→run"),
    ("He didn't went to school today.",                     "negated verb: went→go"),

    # ── Modal + had ───────────────────────────────────────────────────────────
    ("She should had finished her homework.",               "modal+had: had→have"),
    ("I could had done better on the test.",                "modal+had: had→have"),

    # ── Modal + to ────────────────────────────────────────────────────────────
    ("She can to speak three languages.",                   "modal+to: can to"),
    ("You must to submit your work today.",                 "modal+to: must to"),
    ("They will to arrive at noon.",                        "modal+to: will to"),
    ("He should to study harder for the exam.",             "modal+to: should to"),

    # ── Irregular forms ───────────────────────────────────────────────────────
    ("I buyed a new phone yesterday.",                      "irregular: buyed→bought"),
    ("She teached us English for years.",                   "irregular: teached→taught"),
    ("He goed to the store after work.",                    "irregular: goed→went"),
    ("I sitted next to her at the event.",                  "irregular: sitted→sat"),
    ("The pipe breaked during the night.",                  "irregular: breaked→broke"),

    # ── Wrong inf form after 'to' ─────────────────────────────────────────────
    ("She wants to goes to Paris.",                         "inf form: goes→go"),
    ("I need to accepted the terms.",                       "inf form: accepted→accept"),
    ("He tried to running away from the problem.",          "inf form: running→run"),
    ("They plan to working together on this.",              "inf form: working→work"),
    ("The rule is causing everyone to becoming confused.",  "inf form: to becoming"),

    # ── Have + participle ─────────────────────────────────────────────────────
    ("She has eat all the food already.",                   "have+participle: eat→eaten"),
    ("They have finish the project.",                       "have+participle: finish→finished"),
    ("He has write three novels.",                          "have+participle: write→written"),
    ("Prices has change significantly.",                    "have+participle+SVA"),

    # ── Gerund required after verb ────────────────────────────────────────────
    ("I enjoy to watch movies on weekends.",                "gerund after verb: enjoy→watching"),
    ("She avoid to talk about her past.",                   "gerund after verb: avoid→talking"),
    ("They keep to interrupt me during meetings.",          "gerund after verb: keep→interrupting"),
    ("I miss to see my family.",                            "gerund after verb: miss→seeing"),

    # ── Infinitive after preposition ──────────────────────────────────────────
    ("She is thinking about to move to London.",            "inf after prep: about to move"),
    ("He is interested in to learn new skills.",            "inf after prep: in to learn"),

    # ── Stative verbs in continuous ───────────────────────────────────────────
    ("I am knowing the answer to this question.",           "stative: knowing→know"),
    ("She is wanting to leave right now.",                  "stative: wanting→want"),
    ("He is believing in miracles these days.",             "stative: believing→believe"),

    # ── Double negative ───────────────────────────────────────────────────────
    ("I don't know nothing about it.",                      "double neg: don't/nothing"),
    ("She doesn't have no money left.",                     "double neg: doesn't/no"),
    ("He never says nothing useful in meetings.",           "double neg: never/nothing"),

    # ── Comparative/superlative ───────────────────────────────────────────────
    ("This is more better than the other option.",          "double comparative"),
    ("She is the most tallest girl in the school.",         "double superlative"),
    ("He is more good at sports than at studying.",         "more good→better"),

    # ── Participial adj ───────────────────────────────────────────────────────
    ("I am very interesting in this topic.",                "participial adj: interesting→interested"),
    ("She was very tiring after the long journey.",         "participial adj: tiring→tired"),
    ("He felt so boring at the lecture.",                   "participial adj: boring→bored"),

    # ── there/their/they're ───────────────────────────────────────────────────
    ("Their going to the store later today.",               "their→they're"),
    ("I left my bag over their.",                           "their→there"),
    ("They're house is very big and beautiful.",            "they're→their"),

    # ── Since/for/ago ─────────────────────────────────────────────────────────
    ("I have lived here since three years.",                "since→for duration"),
    ("She has worked there since ten years.",               "since→for duration"),

    # ── Doubled subject ───────────────────────────────────────────────────────
    ("The teacher, she explained everything very clearly.", "doubled subject"),
    ("My father, he always wakes up early in the morning.","doubled subject"),

    # ── Prepositions ──────────────────────────────────────────────────────────
    ("She is married with a doctor.",                       "preposition: married with→to"),
    ("I am interested on learning new things.",             "preposition: interested on→in"),
    ("She is good on mathematics.",                         "preposition: good on→at"),
    ("They accused him for the crime.",                     "preposition: accused for→of"),
    ("She is afraid from the dark.",                        "preposition: afraid from→of"),

    # ── Word order ────────────────────────────────────────────────────────────
    ("She goes always to the gym in the morning.",          "word order: goes always"),
    ("He drinks sometimes coffee in the afternoon.",        "word order: drinks sometimes"),

    # ── Wrong relative pronoun ────────────────────────────────────────────────
    ("The man what I met yesterday was very kind.",         "wrong rel pron: what→who/that"),
    ("All the items what she bought were on sale.",         "wrong rel pron: what→that"),
    ("The person which called me was rude.",                "wrong rel pron which→who: person"),
    ("The woman which lives next door is a nurse.",         "wrong rel pron which→who: woman"),
    ("The student which failed the exam was upset.",        "wrong rel pron which→who: student"),

    # ── Missing -s ────────────────────────────────────────────────────────────
    ("She has two cat at home.",                            "missing -s: cat→cats"),
    ("I bought three book from the library.",               "missing -s: book→books"),
    ("There are five bird on the fence outside.",           "missing -s: bird→birds"),

    # ── Spelling ──────────────────────────────────────────────────────────────
    ("I recieved your letter yesterday.",                   "spelling: recieved→received"),
    ("She is very inteligent and hardworking.",             "spelling: inteligent→intelligent"),

    # ── Run-on ────────────────────────────────────────────────────────────────
    ("I went to the store, I bought milk and bread.",       "run-on sentence"),
    ("She studies every day, she gets good grades.",        "run-on sentence"),

    # ── Misplaced modifier ────────────────────────────────────────────────────
    ("Running to catch the bus, my bag fell.",              "misplaced modifier"),

    # ── Faulty parallelism ────────────────────────────────────────────────────
    ("I like swimming, running, and to dance on weekends.", "faulty parallelism"),
    ("She enjoys reading, writing, and to paint pictures.", "faulty parallelism"),

    # ── Adjective/adverb ──────────────────────────────────────────────────────
    ("She speaks English very good.",                       "adj/adv: good→well"),
    ("He sings beautiful on stage.",                        "adj/adv: beautiful→beautifully"),

    # ── Much/many ─────────────────────────────────────────────────────────────
    ("I have much friends in this city.",                   "much/many: much→many"),
    ("There are much opportunities in this country.",       "much/many: much→many"),
    ("She has much problems with her work lately.",         "much/many: much→many"),

    # ── Fewer/less ────────────────────────────────────────────────────────────
    ("There are less students in the class this year.",     "fewer/less: less→fewer"),
    ("He made less mistakes on the second test.",           "fewer/less: less→fewer"),
    ("I have less friends than I used to have.",            "fewer/less: less→fewer"),

    # ── Redundant pronoun in relative clause ──────────────────────────────────
    ("A man who he was very tall entered the room.",        "redundant pronoun: he"),
    ("The woman who she called me is my neighbor.",         "redundant pronoun: she"),
]

NEGATIVE = [
    # Article
    "She is an honest person.",
    "I gave her some advice about the trip.",
    "Can you give me some information about the flight?",
    "I need some furniture for my new apartment.",
    "I have a dog at home.",
    # SVA
    "She doesn't like vegetables.",
    "They were happy about the results.",
    "My parents are coming tomorrow.",
    "The man who has the keys is standing there.",
    "The dogs that run every morning are very fit.",
    "The store which was crowded had long lines.",
    "There are many students in the class today.",
    # Pronouns
    "I went to the market alone yesterday.",
    "John called her yesterday afternoon.",
    "Between you and me, this is a secret.",
    "My friend and I went to the store.",
    "John and I went to the cinema last night.",
    # Verb tense
    "Yesterday, she was very happy about the news.",
    "Last week, they went to the museum.",
    "I saw him yesterday at the park.",
    # Negated verb
    "I didn't understand the question.",
    "She doesn't run in the morning.",
    "He didn't go to school today.",
    # Modal
    "She should have finished her homework.",
    "I could have done better on the test.",
    "She can speak three languages.",
    "You must submit your work today.",
    # Irregular
    "I bought a new phone yesterday.",
    "She taught us English for years.",
    "He went to the store after work.",
    "I sat next to her at the event.",
    # Inf form
    "She wants to go to Paris.",
    "I need to accept the terms.",
    "He tried to run away from the problem.",
    "The new policy is causing everyone to become confused.",
    # Have + participle
    "She has eaten all the food already.",
    "They have finished the project.",
    "Prices have changed significantly.",
    # Gerund after verb
    "I enjoy watching movies on weekends.",
    "She avoids talking about her past.",
    "They keep interrupting me during meetings.",
    # Inf after prep
    "She is thinking about moving to London.",
    "He is interested in learning new skills.",
    # Stative
    "I know the answer to this question.",
    "She wants to leave right now.",
    # Double neg
    "I don't know anything about it.",
    "She doesn't have any money left.",
    # Comparative
    "This is better than the other option.",
    "She is the tallest girl in the school.",
    "He is better at sports than at studying.",
    # Participial adj
    "I am very interested in this topic.",
    "She was very tired after the long journey.",
    "He felt so bored at the lecture.",
    # there/their/they're
    "They're going to the store later today.",
    "I left my bag over there.",
    "Their house is very big and beautiful.",
    # Since/for
    "I have lived here for three years.",
    "She has worked there since 2010.",
    # Doubled subject - no false positives
    "The teacher explained everything very clearly.",
    # Prepositions
    "She is married to a doctor.",
    "I am interested in learning new things.",
    "She is good at mathematics.",
    "They accused him of the crime.",
    "She is afraid of the dark.",
    # Word order
    "She always goes to the gym in the morning.",
    "He is always late to meetings.",
    "She sometimes drinks coffee in the afternoon.",
    # Relative pronouns - correct usage
    "The man who called me was very kind.",
    "The man that I met yesterday was kind.",
    "All the items that she bought were on sale.",
    "The person who called me was rude.",
    "The woman who lives next door is a nurse.",
    "What I want is a hot cup of tea.",
    "I know what you mean.",
    # Missing -s - no false positives on correct sentences
    "She has two cats at home.",
    "I bought three books from the library.",
    # Spelling - correct
    "I received your letter yesterday.",
    # Run-on - correct alternatives
    "I went to the store and bought milk and bread.",
    "She studies every day and gets good grades.",
    # Modifier - correct
    "Looking out the window, I saw a beautiful bird.",
    # Parallelism - correct
    "I like swimming, running, and dancing on weekends.",
    # Adj/adv - correct
    "She speaks English very well.",
    "He sings beautifully on stage.",
    # Much/many - correct
    "I have many friends in this city.",
    "There are many opportunities in this country.",
    "She has much water left in the bottle.",
    "We don't have much time.",
    # Fewer/less - correct
    "There are fewer students in the class this year.",
    "He made fewer mistakes on the second test.",
    "She has less water than she needs.",
    "I have less time than I thought.",
    # Relative pronoun - correct
    "A tall man entered the room.",
    # Stop + purpose infinitive (correct)
    "She stopped to look at the beautiful sunset.",
    "I stopped to rest for a moment.",
    # Used to (correct)
    "I used to go to school by bike.",
    "She used to live in London.",
    # Clean complex sentences
    "In a way that makes sense to me, you should try harder.",
    "She has lived there since 2010.",
    "I have been waiting here for three hours.",
    "He is the best student in the class.",
    "She was bored at the party.",
    "The movie was very boring.",
    "The news was surprising to everyone.",
]


def run_tests():
    print("=" * 60)
    print("POSITIVE TESTS (errors expected)")
    print("=" * 60)
    missed = []
    for sent, desc in POSITIVE:
        errs = run(sent)
        # filter out UNKNOWN catch-all
        specific = [e for e in errs if e.error_type != "Grammatical error (unclassified)"]
        if not specific:
            missed.append((sent, desc))
            print(f"  MISS  [{desc}]\n        {sent!r}")
        # uncomment to see all hits:
        # else:
        #     types = [e.error_type for e in specific]
        #     print(f"  HIT   [{desc}] → {types}")

    print(f"\n{len(missed)} / {len(POSITIVE)} positive tests missed\n")

    print("=" * 60)
    print("NEGATIVE TESTS (no errors expected)")
    print("=" * 60)
    false_positives = []
    for sent in NEGATIVE:
        errs = run(sent)
        specific = [e for e in errs if e.error_type != "Grammatical error (unclassified)"]
        if specific:
            types = [e.error_type for e in specific]
            false_positives.append((sent, types))
            print(f"  FP    {types}\n        {sent!r}")

    print(f"\n{len(false_positives)} / {len(NEGATIVE)} negative tests have false positives\n")

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Positive: {len(POSITIVE) - len(missed)}/{len(POSITIVE)} caught  ({len(missed)} missed)")
    print(f"  Negative: {len(NEGATIVE) - len(false_positives)}/{len(NEGATIVE)} clean  ({len(false_positives)} false positives)")


if __name__ == "__main__":
    run_tests()
