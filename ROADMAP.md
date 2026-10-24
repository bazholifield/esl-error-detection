# Roadmap

**Goal:** a grammar checker that explains learner errors the way a teacher would, good enough to be lang-feed's writing-feedback engine and to stand alone as a portfolio piece.

**Now:** the web app and CLI work (101/101 known errors caught, 103/103 clean sentences on the test suite); next is closing known gaps and packaging it so lang-feed can import it properly.

## 1. Classifier + rules + web app · Done
- DistilBERT sentence classifier fine-tuned on learner English (gate threshold 0.25); article checks always run
- ~25 rule-based checks that name the error and the word responsible, with explanations
- Token-level highlighting in the web app (`app.py`) and a CLI demo
- `test_pipeline.py`: 101 sentences with known errors, 103 clean ones

**Done when:** the test suite passes and the web app explains each flagged word. Reached.

## 2. Close the known gaps · Next
- Gate opens but no rule fires: "I am agree with you." gets P(error) 0.55 and no flagged word (Sea trials case C-001)
- A listed rule that doesn't fire: "The film was very interested." (participial adjective, case C-002)
- Decide which checkpoint is the real one: the app uses the local `checkpoint-3000` (now backed up on Drive); Drive also holds a different run (`checkpoint-1585` plus a final model)
- Grow the test suite with every new miss, so fixes show up as numbers

**Done when:** C-001 and C-002 pass, the test suite has grown, and one checkpoint is documented as the one the app uses.

## 3. Package it for lang-feed · Next
- Turn the repo into an installable package (`pyproject.toml`, a stable `analyze()` entry point) so lang-feed's ML service imports it instead of adding it to `sys.path`
- Return lang-feed's span-based `Feedback` shape directly, or document the adapter
- Pin the spaCy model version the rules were written against
- Model location from config (or the Hugging Face Hub), not a path inside this repo

**Done when:** lang-feed installs it from a path or git URL and its ML tests pass without any `sys.path` change.

## 4. Suggested corrections and more errors · Later
- Suggest a corrected sentence, not only the explanation
- More error types from real learner writing (the README's "Limitations" lists the hard ones: context-dependent tense, coreference, missing prepositions, dialect variation)
- Explore a token-level sequence-labelling model as a replacement for classifier + rules, measured on the same suite

**Done when:** corrections are offered for the most common categories, and any model change is compared against the rules on the same test suite.

## 5. Portfolio polish · Later
- README for outsiders: the teaching story, how it works, results on the test suite, limits
- A short demo (GIF or recording); optionally a Hugging Face Space
- LICENSE and a note on the training data's licence

**Done when:** someone new understands what it does and how well in two minutes from the README.
