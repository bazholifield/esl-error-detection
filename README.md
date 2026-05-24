# ESL Error Detector

A tool that detects and explains grammar errors in ESL (English as a Second Language) writing. It combines a fine-tuned DistilBERT classifier with a set of rule-based checks to flag specific errors, highlight the offending word(s), and give a brief plain-English explanation of each one.

I started this while teaching English in Spain to work on my NLP skills and learn about language correction. A lot of the specific errors I targeted are ones I saw my students make all the time. I plan to incorporate it into my English teaching in the future, allowing students to quickly check for grammatical errors with clean explanations of why these errors are incorrect. I also hope to integrate it into a more comprehensive practice app, which I will offer to my students as a way to practice and learn on the go. 

## How to use it

**Web app** (the main way to use it):

```bash
python app.py
```

Then open `http://localhost:5000` in your browser. Type or paste a sentence or paragraph, hit Check, and hover over any highlighted word to see what's wrong and why.

**CLI demo** (quick terminal version):

```bash
python demo.py
```

![demo](screenshots/ui_screenshot.png)

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

To train the model yourself, run `python main.py` or use the Colab notebook at `notebook/train_colab.ipynb` (recommended — GPU makes it much faster). The trained checkpoint is not included in the repo due to file size.

## How it works

1. **DistilBERT classifier** — fine-tuned on learner English to decide whether a sentence likely contains an error (threshold: 0.25 confidence). Article errors always run regardless of the classifier score.
2. **Article checker** — rule-based checks for a/an confusion, unnecessary articles, article omission, and wrong article choice (always-on).
3. **Rule-based classifier** — if the model flags a sentence, ~25 targeted checks run to identify the specific error type and the exact word responsible.
4. **Token annotation** — errors are mapped back to individual tokens so the frontend can highlight them word by word.

## Error types detected

The rule-based layer currently covers:

- Article errors (a/an, unnecessary, omission, wrong choice)
- Subject-verb agreement (including compound subjects and elided subjects in conjoined clauses)
- Pronoun case errors
- Verb tense errors
- Missing plural -s after number words
- Present simple vs. continuous (stative verbs)
- Double negatives
- Incorrect irregular verb/noun forms
- Run-on sentences (comma splices)
- Preposition errors
- Word order (frequency adverbs)
- Misplaced modifiers
- Faulty parallelism
- Spelling errors
- Comparative/superlative errors (double forms)
- Participial adjective confusion (interesting vs. interested)
- there/their/they're confusion
- Since/for/ago errors
- Doubled subject (e.g. "The teacher, she explained")
- Subject ordering (I/me last in a list)
- Infinitive after preposition (should be gerund)
- Gerund required after certain verbs (stop, enjoy, avoid, etc.)
- Adjective/adverb confusion
- Confused similar words (your/you're, its/it's, etc.)

## Dataset

**W&I+LOCNESS** — Write & Improve + LOCNESS corpus of learner English, annotated with corrections (~38k sentence pairs).

## Model performance

Trained for 5 epochs on Colab (GPU), evaluated on a held-out 20% split:

| Metric | Score |
|--------|-------|
| Accuracy | 76.5% |
| F1 | 0.770 |
| Precision | 0.776 |
| Recall | 0.763 |

After training, a threshold sweep (0.30–0.50) was run on the eval set to find the F1-optimal decision boundary for classifying a sentence as erroneous.

The pipeline uses a threshold of **0.25** — lower than the F1-optimal value — which pushes recall close to 100% at the cost of some precision. The reasoning: it's better to run the rule-based checks on a sentence that turns out to be fine than to miss a real error entirely. False positives from the model don't matter much because if no rule fires on that sentence, nothing gets shown to the user anyway. The model is basically a first-pass filter: cast a wide net, then let the rules add specificity.

## Limitations

The rule-based layer depends on spaCy's dependency parser, which loses accuracy on short, heavily ungrammatical sentences — exactly the kind this tool is meant to handle. A few categories of error are out of reach as a result:

Errors that require context — tense mistakes with no explicit time marker, or word choice errors that depend on what the writer meant (e.g. peoples used to mean individual people rather than ethnic groups).
Coreference — detecting a misused reflexive pronoun (myself instead of me) requires knowing what the pronoun refers to, which spaCy's parser doesn't resolve.
Pragmatic/argument-structure errors — a missing preposition like "I was arguing the cashier" (should be arguing with) produces a syntactically valid parse, so there's no dependency signal to catch it.
Dialect variation — some constructions are standard in one dialect and non-standard in another (e.g. on line vs. in line), so flagging them would produce false positives without knowing the writer's background.
## Future work
The main planned direction is integrating this into a broader English practice app. Some specific things worth exploring:

Suggested corrections — currently the tool flags errors and explains them, but doesn't offer a corrected version of the sentence.
More error types — the rule set covers the most common patterns I observed in my students' writing, but there's plenty of room to expand it.
Larger model — a sequence-labelling model (e.g. fine-tuned on token-level annotations) could replace the binary classifier + rule-based approach and handle errors that are currently out of reach.

