# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:07:54 2020

@author: Amishkallan
"""

import json
import re
import nltk
from nltk.corpus import stopwords
all_stopwords = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import pickle
import pytesseract
from PIL import Image
import requests
from io import BytesIO

def pcmb(qns):
    with open("pickled//cv_pcmb","rb") as infile:
        cv=pickle.load(infile)
    infile.close()
    with open("pickled//classifier_pcmb","rb") as infile:
        classifier=pickle.load(infile)
    infile.close()
    X=cv.transform([qns]).toarray()
    a=classifier.predict(X)
    if a =='biology':
        return bio(qns)
    elif a =='chemistry':
        return chem(qns)
    elif a == 'physics' :
        return phy(qns)
    else:
        return {
        'statusCode': 200,
        'body': json.dumps(['maths','maths'])
        }

def bio(qns):
    with open("pickled//cv_b","rb") as infile:
        cv=pickle.load(infile)
    infile.close()
    with open("pickled//classifier_b","rb") as infile:
        classifier=pickle.load(infile)
    infile.close()
    X=cv.transform([qns]).toarray()
    a=classifier.predict(X)
    return {
        'statusCode': 200,
        'body': json.dumps(['biology',a[0]])
        }
    
def chem(qns):
    with open("pickled//cv_c","rb") as infile:
        cv=pickle.load(infile)
    infile.close()
    with open("pickled//classifier_c","rb") as infile:
        classifier=pickle.load(infile)
    infile.close()
    X=cv.transform([qns]).toarray()
    a=classifier.predict(X)
    return {
        'statusCode': 200,
        'body': json.dumps(['chemistry',a[0]])
        }
    
def phy(qns):
    with open("pickled//cv_p","rb") as infile:
        cv=pickle.load(infile)
    infile.close()
    with open("pickled//classifier_p","rb") as infile:
        classifier=pickle.load(infile)
    infile.close()
    X=cv.transform([qns]).toarray()
    a=classifier.predict(X)
    return {
        'statusCode': 200,
        'body': json.dumps(['physics',a[0]])
        }


def lambda_handler(event, context):
    #retrieving image from url
    response = requests.get(event['url'])
    img = Image.open(BytesIO(response.content))
    #convert image to text
    result = pytesseract.image_to_string(img)  
    #process text
    qns = re.sub('[^a-zA-Z]',' ', result)
    qns = qns.lower()
    qns = qns.split()
    qns = [ps.stem(word) for word in qns if word not in set(all_stopwords)]
    qns = ' '.join(qns)
    #fitting the processed image text
    return(pcmb(qns))
