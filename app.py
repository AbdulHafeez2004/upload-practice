from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model-heart-disease.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)

    prediction = model.predict(final)
    prediction_proba = model.predict_proba(final)
    output = '{0:.{1}f}%'.format(prediction_proba[0][1]*100, 2)

    zero_value = prediction_proba[0][0] * 100

    one_value = prediction_proba[0][1] * 100

    if prediction == [0]:
        return render_template('heart_disease.html', pred=f'I am {zero_value:.2f}% sure you do NOT have heart disease any heart disease with the information supplied, do well to always keep your heart in good condition.')
    else:
        return render_template('heart_disease.html', pred=f' I am {one_value:.2f}% sure you have a heart disease from the information provided, do well to go visit your doctor as soon as possible.')


if __name__ == '__main__':
    app.run(debug=True)
