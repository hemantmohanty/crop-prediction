import numpy as np

from flask import Flask, request, jsonify, render_template

import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def Home():
    return render_template ("form.html")

@app.route("/predict",methods=["POST"])
def predict():
    float_feature = [float(x) for x in request.form.values()]
    feature = [np.array(float_feature)]
    prediction = model.predict(feature)
    return render_template ("form.html", prediction_text="The predicted crop is {} ".format(prediction))

if __name__=="__main__":
    app.run(debug=True)