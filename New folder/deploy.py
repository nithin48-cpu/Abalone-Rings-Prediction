from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
# load the model
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    Sex = float(request.form['Sex'])
    Length = float(request.form['Length'])
    Diameter = float(request.form['Diameter'])
    Height = float(request.form['Height'])
    Whole_weight = float(request.form['Whole_weight'])
    Shucked_weight = float(request.form['Shucked_weight'])
    Viscera_weight = float(request.form['Viscera_weight'])
    Shell_weight = float(request.form['Shell_weight'])
    result = model.predict([[Sex, Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight]])[0]
    return render_template('index.html', **locals())


if __name__ == '__main__':
    app.run(debug=True)
