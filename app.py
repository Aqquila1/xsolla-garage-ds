import pickle
from flask import Flask
from flask import request
from flask import jsonify
import json

from nltk.stem.porter import *
import pymorphy2

application = Flask(__name__)

# loading models
vec = pickle.load(open("./models/tfidf.pickle", "rb"))
with open("./models/mlp_model.pkl", 'rb') as file:
    model = pickle.load(file)

# test output
@application.route("/")
def hello():
    resp = {'message': "Hello World!"}
    response = jsonify(resp)
    return response


def get_label_id(proba_prediction):
    max_proba = proba_prediction.max()
    label_id = proba_prediction.argmax()
    proba_threshold = 0.5  # if max predicted probability bigger this value - return label id, otherwise - '-1'
    if (max_proba > proba_threshold):
        return int(label_id)
    else:
        return -1

def lemmatize_list_of_words(list_of_words):
    res = []
    for word in list_of_words:
        morph = pymorphy2.MorphAnalyzer()
        p = morph.parse(word)[0]
        new_word = p.normal_form
        res.append(new_word)
    return res

# cleaning text + lemmatization
def get_lemma(text):
    try:
        words = re.split(' ', text)
        true_words = []
        for word in words:
            m = re.search('(\w+)', word)
            if m is not None:
                good_word = m.group(0)
                true_words.append(good_word)
    except:
        pass

    lemma_list = lemmatize_list_of_words(true_words)
    lemma = ' '.join(lemma_list)
    return lemma


@application.route("/ds", methods=['GET', 'POST'])
def registration():
    resp = {
            'message': 'ok',
            'label': -1
            }

    try:
        # getting text from request
        getData = request.get_data()
        json_params = json.loads(getData)
        text_content = json_params['text_content']

        # preprocessing text for model
        text_content = text_content.lower()
        text_content = get_lemma(text_content)

        # predicting category
        proba_prediction = model.predict_proba(vec.transform([text_content]).toarray())
        resp['label'] = get_label_id(proba_prediction)

    except Exception as e:
        print(e)
        resp['message'] = e

    response = jsonify(resp)

    return response


if __name__ == "__main__":
    application.run(debug=True)



