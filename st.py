
# importing the sys module
import sys
import streamlit as st
import logging
import time
import os
# appending the directory of mod.py
# in the sys.path list
sys.path.append(os.getcwd() + '/seq2seq-polynomial-master')

from flask import Flask,  request, render_template
from train import load_model
from tqdm import tqdm

# Configure logger
logging.basicConfig(format="\n%(asctime)s\n%(message)s", level=logging.INFO, force=True)

#load model
model = load_model('seq2seq-polynomial-master/models/best')

st.title("Solve the Complex Polynomial")
value = "(4-x)*(x+12)"
src = st.text_input("Enter the Polynomial Expression", value)
st.info('Some common expressions to use are (t+20)*(t-12), y**2, x*(x-22)', icon="ℹ️")





# Flask constructor
#app = Flask(__name__, template_folder="templates")

def eval(model, test_pairs, batch_size=128):
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
if "n_requests" not in st.session_state:
    st.session_state.n_requests = 0

def predict():
    if st.session_state.n_requests >= 5:
        st.session_state.text_error = "Too many requests. Please wait a few seconds before generating another text or image."
        logging.info(f"Session request limit reached: {st.session_state.n_requests}")
        st.error(f"{st.session_state.text_error} Please wait 20 seconds and try again.")
        time.sleep(20)
        st.session_state.n_requests = 1
        return
    st.session_state.n_requests += 1
    pairs = [(src,"")]
    prd = eval(model, pairs)
    return st.success(f'The simplied version is {prd[0]}', icon="✅")

btn = st.button("Simplify!!")
if btn:
    st.snow()
    predict()

# @app.route("/")
# def hello():
#     return render_template("form.html")

# @app.route("/predict", methods=['POST'])
# def predict():
#     src= request.form.get("exprn")
#     pairs = [(src,"")]
#     prd = evaluate(model, pairs)

#     return render_template("result.html", src=src, pred_exp=prd)

# if __name__ == '__main__':
#     app.run(host="localhost", port=8000)
