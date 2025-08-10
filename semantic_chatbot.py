from sentence_transformers import SentenceTransformer, util
from spellchecker import SpellChecker
import json, re
from flask import Flask, request

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAQs
with open("faq.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

questions = list(faq_data.keys())
answers = list(faq_data.values())

# Precompute embeddings
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Helpers
spell = SpellChecker()

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text)  # normalize spaces
    return text

def correct_spelling(text):
    words = text.split()
    corrected = [spell.correction(w) or w for w in words]
    return ' '.join(corrected)

# Flask app
app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form.get('message', '')
    if not user_question:
        return "No question received", 400

    # Clean & correct spelling
    user_question = clean_text(correct_spelling(user_question))

    # Encode the cleaned question
    user_embedding = model.encode(user_question, convert_to_tensor=True)

    # Compare embeddings
    similarities = util.cos_sim(user_embedding, question_embeddings)[0]
    best_match_idx = similarities.argmax().item()
    best_score = similarities[best_match_idx].item()

    # Extract keywords for double-check
    user_keywords = set(user_question.split())
    matched_keywords = set(questions[best_match_idx].lower().split())

    # High-confidence match
    if best_score >= 0.65 and (user_keywords & matched_keywords):
        return answers[best_match_idx], 200

    # Fallback: keyword-based search
    for q, ans in faq_data.items():
        if any(word in q.lower() for word in user_keywords):
            return ans, 200

    # If still no good match, log it
    with open("unanswered.log", "a", encoding="utf-8") as log:
        log.write(user_question + "\n")

    # Suggest closest question if available
    suggested_q = questions[best_match_idx]
    return f"Sorry, I couldn't find a perfect match. Did you mean: '{suggested_q}'?", 200

if __name__ == '__main__':
    app.run(port=5000)
