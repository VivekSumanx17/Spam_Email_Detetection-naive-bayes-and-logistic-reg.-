from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# load model and word_dictionary
with open("logistic_reg_model.pkl", "rb") as f:
    classifier = pickle.load(f)

with open("word_dict_1.pkl", "rb") as f:
    word_dict = pickle.load(f)

print("Model loaded successfully!")

#prediction function
def spam_detection(content):

    sample = []
    for i in word_dict:
        sample.append(content.split(" ").count(i[0]))

    sample = np.array(sample).reshape(1, -1)  # shape (1,3000)
    y_pred = classifier.predict(sample)[0]  # get single prediction (0 or 1)
    return y_pred

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/spam', methods=['POST'])
def spam():
    if request.method == 'POST':
        #collect email content from data
        content = request.form['content']

        y_pred = spam_detection(content)

        if y_pred == 1:
            result = "🚨 This email is Spam"
        else:
            result = "✅ This email is Not Spam"

        return render_template('index.html', result=result)
    return render_template('index.html')

#python main
if __name__ == '__main__':
    app.run(debug=True)
