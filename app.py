from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from config import CHECKPOINT_PATH
from pipeline import analyze
from error_classifier import UNKNOWN

THRESHOLD = 0.25

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, local_files_only=True)
_model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_PATH, local_files_only=True)
_model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_route():
    data = request.get_json()
    text = (data or {}).get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    results = analyze(text, tokenizer, _model, THRESHOLD)

    sentences = []
    total_errors = 0
    for r in results:
        # drop unclassified catch-all — can't point to a specific word
        visible = [e for e in r['errors'] if e.error_type != UNKNOWN]
        total_errors += len(visible)
        sentences.append({'tokens': r['tokens']})

    return jsonify({'sentences': sentences, 'total_errors': total_errors})

if __name__ == '__main__':
    app.run(debug=True)
