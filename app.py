from flask import Flask, render_template, request, jsonify
import analysis

app = Flask(__name__)

MAX_LEN = 5000


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please enter a statement."}), 400
    if len(text) > MAX_LEN:
        return jsonify({"error": f"Statement is too long (max {MAX_LEN} characters)."}), 400
    return jsonify(analysis.predict_with_scores(text))


if __name__ == "__main__":
    app.run(debug=True)
