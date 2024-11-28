import numpy as np
import pickle

from flask import Flask,request,render_template
import pickle

#flask app creation
app = Flask(__name__)

#pickle model loading
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template('index.html',prediction_text = 'Diagnosis is {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)