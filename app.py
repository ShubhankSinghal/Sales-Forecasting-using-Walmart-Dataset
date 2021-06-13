import numpy as np
from flask import Flask, render_template,request
import pickle
#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/home')
def home():
    return render_template('Home.html')
@app.route('/')
def prediction():
    return render_template('Prediction.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/predict',methods=['POST'])
def predict():
    #print(request.form.values())
    #For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('Prediction.html', prediction_text='Predicted Weekly Sales:{}'.format(output))
if __name__ == "__main__":
   app.run(debug=True, use_debugger=False, use_reloader=False)
