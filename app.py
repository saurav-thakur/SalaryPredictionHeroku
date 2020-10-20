import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
# model = pickle.load(open('model.pkl'), 'rb')
with open('model.pkl', 'rb') as f:
    # load using pickle de-serializer
    model = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The Predicted Salary is  $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=False)
