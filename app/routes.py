# -*- coding: utf-8 -*-
import string
import numpy as np
import collections as co

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier


import pickle
from app import application
from flask import make_response
from flask import jsonify
from flask import request
import json

import re

from nltk.stem.porter import *
import pymorphy2

from nltk import WordPunctTokenizer
tokenizer = WordPunctTokenizer()

from datetime import datetime , timedelta
import time
from dateutil.relativedelta import relativedelta

import pymorphy2
pymorph = pymorphy2.MorphAnalyzer()
def my_lemming(word):
  return pymorph.parse(word)[0].normal_form


# loading models
vec = pickle.load(open("./models/tfidf.pickle", "rb"))
with open("./models/mlp_model.pkl", 'rb') as file:
    model = pickle.load(file)
    
    
def get_label_id(proba_prediction):
    max_proba = proba_prediction.max()
    label_id = proba_prediction.argmax()
    proba_threshold = 0.001  # if max predicted probability bigger this value - return label id, otherwise - '-1'
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

    



def date_processing(wordString, date_target=datetime.now().strftime("%Y-%m-%d %H:%M:%S")):
    
    #dicts for days, months and numbers
    numbers = {'тридцать':'30','двадцать':'20','девятнадцать':'19','восемнадцать':'18','семнадцать':'17','шесттнадцать':'16','пятнадцать':'15',
                      'четырнадцать':'14','тринадцать':'13','двенадцать':'12','одиннадцать':'11', 'десять':'10', 'девять':'09','восемь':'08',
                      'семь':'07','шесть':'06','пять':'05','четыре':'04','третье':'03','два':'02','первое':'01'}

    days = {'Mon':1, 'Tue':2, 'Wed':3, 'Thu':4, 'Fri':5, 'Sat':6, 'Sun':7, 'выходной':6,
            'понедельник':1, 'вторник':2, 'среда':3, 'четверг':4, 'пятница':5, 'суббота':6, 'воскресенье':7}
    
    months = {'январт':'01','февраль':'02','март':'03','апрель':'04','май':'05','июнь':'06','июль':'07',
                      'август':'08','сентябрь':'09','октябрь':'10','ноябрь':'11', 'декабрь':'12'}

    time_segments = {'вечер':'18:00:00', 'утро':'10:00:00', 'день':'14:00:00'}

    #deleting text before last token "на"
    if 'на' in wordString:
      wordString = wordString[wordString.rfind('на')+2:]
    
    #text clearing from garbage and word lemmatization
    wordString = wordString.lower()
    words = tokenizer.tokenize(re.sub('[-\’,·”–●•№~✅“=#—«"‚»|.?!:;()*^&%+/]', ' ' , wordString))
    words = [my_lemming(word) for word in words]

    #print(words)  

    #case for dates that starts with token 'следующий' 
    if 'следующий' in words:   
        if 'месяц' in words:
            date_target_dt = datetime(int(date_target[:4]), int(date_target[5:7]), int(date_target[8:10])) + relativedelta(months=1)
            date_target = date_target_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        elif 'день' in words  or 'вечер' in words or 'утро' in words :
            date_target_dt = datetime(int(date_target[:4]), int(date_target[5:7]), int(date_target[8:10])) + timedelta(days=1)
            date_target = date_target_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        elif 'неделя' in words:
            date_target_dt = datetime(int(date_target[:4]), int(date_target[5:7]), int(date_target[8:10])) + relativedelta(weeks=1)
            date_target = date_target_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        #case of day names/weekend after , 'следующие выходные' for example
        else:
            for day in days.keys():
                if day in words:
                    current_date_dt = datetime(int(date_target[:4]), int(date_target[5:7]), int(date_target[8:10]))
                    
                    current_day_name =  time.ctime(datetime.timestamp(current_date_dt))[:3]
                    current_day_num = days[current_day_name]
                    target_day_num = days[day]

                    days_delta = target_day_num + (7 - current_day_num)
                    
                    date_target = datetime(int(date_target[:4]), int(date_target[5:7]), int(date_target[8:10])) + relativedelta(days=days_delta)
                    date_target = date_target.strftime("%Y-%m-%d %H:%M:%S")        
                    break
    
    elif 'завтра' in words:
        date_target = datetime(int(date_target[:4]), int(date_target[5:7]), int(date_target[8:10])) + timedelta(days=1)
        date_target = date_target.strftime("%Y-%m-%d %H:%M:%S")
    elif 'послезавтра' in words:
        date_target = datetime(int(date_target[:4]), int(date_target[5:7]), int(date_target[8:10])) + timedelta(days=2)
        date_target = date_target.strftime("%Y-%m-%d %H:%M:%S")

    #case for days/weekends/defined dates
    else:
        #flag for case definition
        day_flag = 0
        
        #checking if there names of days in text
        for day in days.keys():
                if day in words:
                    date_dt = datetime(int(date_target[:4]), int(date_target[5:7]), int(date_target[8:10]))
                    
                    current_day_name =  time.ctime(datetime.timestamp(date_dt))[:3]
                    current_day_num = days[current_day_name]
                    target_day_num = days[day]

                    days_delta = (target_day_num  - current_day_num)%7
                    
                    date_target = datetime(int(date_target[:4]), int(date_target[5:7]), int(date_target[8:10])) + relativedelta(days=days_delta)
                    date_target = date_target.strftime("%Y-%m-%d %H:%M:%S")
                    
                    day_flag=1
                    break

        #if there are no any days of the week in text  
        if day_flag!=1:
            target_day = ''
            
            #checking for any numbers
            for number in numbers.keys():
              if number in words:
                if day_flag == 0:
                    target_day += numbers[number]
                    day_flag += 1
                else:
                    target_day = int(target_day) + int(numbers[number])
                    day_flag += 1
            
            if day_flag!=0:
                date_target = date_target[:8] + str(target_day) + date_target[10:]   
            #installing the next day to the current date as date_target in case there are no any numbers in text
            else:
                date_target = date_target[:8] + str(int(date_target[8:10])+1) + date_target[10:]
            
            #cheking for months names
            target_month = date_target[5:7]
            for month in months.keys():
              if month in words:
                target_month = months[month]
                break
            date_target = date_target[:5] + target_month + date_target[7:]

    #12:00 as default time settings
    date_target = date_target[:-8]+'12:00:00'
    for time_segment in time_segments.keys():
        if time_segment in words:
            date_target = date_target[:-8]+time_segments[time_segment]
            break

    return date_target        


def _build_cors_prelight_response():
    response = make_response()
    
    response.content_type = "application/json"
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response



@application.route("/", methods=['GET', 'POST'])  
def hello():
    resp = {'message':"Hello World!"}
    response = jsonify(resp)
    
    return response

@application.route("/date_tags" , methods=['POST', 'OPTIONS'])  
def date_and_tags():
    tags_list = ['Зимние виды', 'Мир', 'Coцсети', 'Деньги', 'Футбол', 'Бизнес', 'Музыка',
                 'Квартира', 'Бокс и ММА', 'Театр', 'Оружие', 'Дача', 'Прибалтика',
                 'Рынки', 'Звери', 'Техника', 'Интернет', 'Люди', 'Наука', 'Кино',
                 'ТВ и радио', 'Космос', 'Явления', 'Стиль', 'События', 'Деловой климат',
                 'Искусство', 'Книги', 'Закавказье', 'Летние виды', 'Пресса', 'Игры',
                 'Средняя Азия', 'Москва', 'Гаджеты', 'Город']
    
    
    resp = {"date_target":"yyyy-mm-dd hh:MM:ss"
           ,"tag": "Ukraine",
           'message':"ok"
           }
    
    if request.method == 'OPTIONS':
        return _build_cors_prelight_response()
    
    else:
        try:
            getData = request.get_data()
            json_params = json.loads(getData) 
            
            text_content = json_params['text_content']
            text_content = text_content.lower()
            text_content = get_lemma(text_content)
    
            # predicting category
            proba_prediction = model.predict_proba(vec.transform([text_content]).toarray())
            tag_id = get_label_id(proba_prediction)
            resp["tag"] = tags_list[tag_id]
            
            # date processing
            current_date = json_params['current_date']
            date_target = date_processing(json_params['text_content'], current_date)
            resp['date_target'] = date_target
         
        except Exception as e: 
            print(e)
            resp['message'] = e
          
        response = jsonify(resp)
        
        return _corsify_actual_response(response)
