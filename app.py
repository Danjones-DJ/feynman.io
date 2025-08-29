# app.py
from flask import Flask, request, jsonify
import requests
import io
import av
from rapidfuzz import process as rf_process, fuzz as rf_fuzz

app = Flask(__name__, static_url_path="", static_folder="static")



@app.get("/")
def index():
    return app.send_static_file("index.html")

def build_initial_prompt(terms, context):
    # Short biasing prompt. Keep it concise or Whisper will ignore it.
    bits = []
    if terms:
        joined = ", ".join(terms[:50])
        bits.append(f"Use these terms when appropriate: {joined}.")
    if context:
        bits.append(context.strip())
    return (" ".join(bits)).strip() or None

def autocorrect_tokens(tokens, terms, prob_threshold=0.65, ratio_threshold=85):
    """
    tokens: [{"word": "fine", "prob": 0.42}, ...]
    terms: list of domain words. We only correct low-confidence tokens.
    """
    if not terms:
        return tokens

    canon_terms = list({t.strip().lower() for t in terms if t.strip()})
    corrected = []
    for tk in tokens:
        w = tk["word"]
        p = float(tk.get("prob", 1.0))
        if p < prob_threshold and any(c.isalpha() for c in w):
            match = rf_process.extractOne(w.lower(), canon_terms, scorer=rf_fuzz.ratio)
            if match and match[1] >= ratio_threshold:
                best = match[0]
                if w.istitle():
                    w = best.title()
                elif w.isupper():
                    w = best.upper()
                else:
                    w = best
        corrected.append({"word": w, "prob": p})
    return corrected

def stitch_tokens(tokens):
    out = []
    for i, tk in enumerate(tokens):
        w = tk["word"]
        if i == 0:
            out.append(w)
            continue
        if w in [".", ",", "!", "?", ":", ";", ")", "]", "}", "'s"]:
            out[-1] = out[-1] + w
        elif out[-1] in ["(", "[", "{", "£", "$"]:
            out[-1] = out[-1] + w
        else:
            out.append(" " + w)
    return "".join(out).strip()

@app.post("/transcribe")
def transcribe():
    """
    Server-first transcription. Client may auto-fallback to browser ASR on later recordings
    if this call is slow or the clip is long.
    """
    f = request.files.get("audio")
    context = (request.form.get("context") or "").strip()
    raw_terms = (request.form.get("terms") or "").replace("\n", ",")
    terms = [t.strip() for t in raw_terms.split(",") if t.strip()]
    auto = (request.form.get("auto_correct") == "true")

    if not f or f.filename == "":
        return jsonify({"text": "", "avg_logprob": None})

    raw = f.read()
    if not raw or len(raw) < 512:
        return jsonify({"text": "", "avg_logprob": None})

    buf = io.BytesIO(raw)
    buf.seek(0)

    # Example: Use OpenAI Whisper API (or another free/easy API)
    # Replace this with your actual API endpoint and key if needed
    api_url = "https://api.openai.com/v1/audio/transcriptions"
    api_key = "YOUR_OPENAI_API_KEY"  # Replace with your key or use env var
    files = {"file": (f.filename, buf, f.mimetype)}
    data = {"model": "whisper-1", "language": "en"}
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.post(api_url, files=files, data=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        text = result.get("text", "")
        avg_logprob = None  # Not available from API
        return jsonify({"text": text, "avg_logprob": avg_logprob})
    except Exception as e:
        return jsonify({"text": "", "avg_logprob": None, "error": str(e)})
    
if __name__ == "__main__":
    app.run(debug=True, port=5000)