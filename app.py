# app.py
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import io
import av
from rapidfuzz import process as rf_process, fuzz as rf_fuzz

app = Flask(__name__, static_url_path="", static_folder="static")

# Server-first: choose a bigger model but quantized for CPU.
# Bump to "medium.en" if you accept slower CPU inference for more accuracy.
MODEL_SIZE = "small.en"
model = WhisperModel(MODEL_SIZE, compute_type="int8")

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

    try:
        segments, info = model.transcribe(
            buf,
            language="en",
            initial_prompt=build_initial_prompt(terms, context),
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
            beam_size=5,
            patience=0.1,
            temperature=0.0,  # deterministic
            condition_on_previous_text=True,
            word_timestamps=True
        )
    except av.error.InvalidDataError:
        return jsonify({"text": "", "avg_logprob": None})
    except Exception:
        return jsonify({"text": "", "avg_logprob": None})

    segs = list(segments)

    # coarse segment-level confidence
    valid = [s.avg_logprob for s in segs if getattr(s, "avg_logprob", None) is not None]
    avg_logprob = (sum(valid) / len(valid)) if valid else None

    # tokenize with per-word probs (if available)
    tokens = []
    for s in segs:
        if s.words:
            for w in s.words:
                prob = getattr(w, "probability", None)
                if prob is None and getattr(s, "avg_logprob", None) is not None:
                    lp = s.avg_logprob  # ~ [-1,0]
                    prob = max(0.0, min(1.0, 1.0 + lp))
                if prob is None:
                    prob = 1.0
                tokens.append({"word": w.word, "prob": float(prob)})
        else:
            for w in s.text.split():
                tokens.append({"word": w, "prob": 1.0})

    if auto and terms:
        tokens = autocorrect_tokens(tokens, terms)

    text = stitch_tokens(tokens) if tokens else "".join(s.text for s in segs).strip()
    return jsonify({"text": text, "avg_logprob": avg_logprob})
    
if __name__ == "__main__":
    app.run(debug=True, port=5000)
