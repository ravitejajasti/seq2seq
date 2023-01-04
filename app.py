
# importing the sys module
import sys        
 
# appending the directory of mod.py
# in the sys.path list
sys.path.append('/Users/ravitejajasti/Library/CloudStorage/OneDrive-TheUniversityofTexasatDallas/Academic/Sem 4/NLP/seq2seq-polynomial-master-scale-ai/seq2seq-polynomial-master')

from flask import Flask,  request, render_template
from train import evaluate, load_model
from tqdm import tqdm

# Flask constructor
app = Flask(__name__, template_folder="templates")

model = load_model('seq2seq-polynomial-master/models/best')

def evaluate(model, test_pairs, batch_size=128):
    src_sentences, _ = zip(*test_pairs)

    prd_sentences, _, _ = model.predict(src_sentences, batch_size=batch_size)

    for i, (src, prd) in enumerate(
        tqdm(
            zip(src_sentences, prd_sentences),
            desc="scoring",
            total=len(src_sentences),
        )
    ):

        if i < 10:
            print(f"\n\n\n---- Example {i} ----")
            print(f"src = {src}")
            print(f"prd = {prd}")

    return prd_sentences

@app.route("/")
def hello():
    return render_template("form.html")

@app.route("/predict", methods=['POST'])
def predict():
    src= request.form.get("exprn")
    pairs = [(src,"")]
    prd = evaluate(model, pairs)

    return render_template("result.html", src=src, pred_exp=prd)

if __name__ == '__main__':
    app.run(host="localhost", port=8000)
