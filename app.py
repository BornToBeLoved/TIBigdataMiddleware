#-*- coding:utf-8 -*-
import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from flask import Flask, jsonify, request, Response
from flask_restful import Resource, Api
from elasticsearch import Elasticsearch
from flask_cors import CORS, cross_origin
from collections import Counter
from operator import itemgetter
import time
import json
import sys
#from rcmdHelper import rcmd as rc

from common.cmm import INDEX

import os
if os.name == "nt":
    from eunjeon import Mecab
else:
    from konlpy.tag import Mecab
print("os system : ", os.name)

sys.path.insert(0, './common')

# Sentence-tokenizer
import re
#import tfidf
import tfidf_all
# Implement KR-Wordrank
# from krwordrank.hangle import normalize
# from krwordrank.word import KRWordRank
import datetime
import time

app = Flask(__name__)
# app.config['TESTING'] = True
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

isLocalEs = 0
#local es mode:
if(isLocalEs):
    # Url address of Elasticsearch
    import socket
    def get_ip_address():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]

    serverUrl = get_ip_address()  # '192.168.0.110'
    print(serverUrl)
    if(serverUrl != "http://203.252.112.15:9200"):
        serverUrl="http://localhost:9200"
    else:
        serverUrl = "http://elastic:epp2020@203.252.112.14"
else:
    serverUrl = "http://elastic:epp2020@203.252.112.14"
# ElasticSearch connection
es = Elasticsearch(serverUrl)
app = Flask(__name__)
api = Api(app)


# CORS(app, support_credentials=True)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

#################################################
#"""
#SVM
#2020.09.
#"""
#################################################
import SVM
import schedule
import time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.base import JobLookupError


#@app.route('/svm', methods=['GET'])#app객체로 라우팅 경로 설정
def svm():
    app=Flask(__name__)
    app.config['JSONIFY_PRETTYPRINT_REGULAR']=True

    now=datetime.datetime.now()
    date=now.strftime('%Y-%m-%d')

    filename='./log/svm.log'
    with open(filename, "a") as f:
        f.write(date)
        f.write("\n")
    print("svm.log에 모델예측 실행 날짜를 기록하였습니다.")
    result=SVM.MoEs(date)
    return json.dumps(result, ensure_ascii=False)
#@app.route('/train', methods=['GET'])#app객체로 라우팅 경로 설정
def svm_train():
    pp = Flask(__name__)
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

    now=datetime.datetime.now()
    date=now.strftime('%Y-%m-%d')

    filename='./log/svm_train.log'
    with open(filename, "a") as f:	
        f.write(date)
        f.write("\n")
    print("svm_train.log에 모델학습 실행 날짜를 기록하였습니다.")
    result=SVM.SVMTrain()
    print("SVM 모델 학습을 완료하였습니다.")
    return "SVM 모델 학습 완료"

#sched_train = BackgroundScheduler(daemon=True)
#sched_train.add_job(svm_train,'interval',hours=24)
#sched_train.add_job(svm_train)
#sched_train.start()

#sched = BackgroundScheduler(daemon=True)
#sched.add_job(svm)
#sched.start()

#################################################
#"""
#CNN
#2021.02.
#"""
#################################################
import CNN
import schedule
import time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.base import JobLookupError

#@app.route('/svm', methods=['GET'])#app객체로 라우팅 경로 설정
def cnn():
    pp = Flask(__name__)
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

    now = datetime.datetime.now()
    date = now.strftime('%Y-%m-%d')


    filename = './log/cnn.log'
    with open(filename,"a") as f:
        f.write(date)
        f.write("\n")
    print("날짜를 기록하였습니다")
    result=CNN.MoEs(date)
    return json.dumps(result, ensure_ascii=False)

#@app.route('/train', methods=['GET'])#app객체로 라우팅 경로 설정
def cnn_train():
    pp = Flask(__name__)
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    
    now=datetime.datetime.now()
    date=now.strftime('%Y-%m-%d')

    filename='./log/cnn_train.log'
    with open(filename, "a") as f:
        f.write(date)
        f.write("\n")
    print("cnn_train.log에 모델학습 실행 날짜를 기록하였습니다.")
    result=CNN. cnn_train()
    print("CNN 모델 학습을 완료하였습니다.")
    return "CNN 모델 학습 완료"   

#sched_train = BackgroundScheduler(daemon=True)
#sched_train.add_job(cnn_train,'interval',hours=24)
#sched_train.add_job(cnn_train)
#sched_train.start()

sched = BackgroundScheduler(daemon=True)
sched.add_job(cnn)
sched.start()

#################################################
#"""
#Multi-SVM
#2021.02
#"""
#################################################
import Multi_SVM
import schedule
import time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.base import JobLookupError

#@app.route('/svm', methods=['GET'])#app객체로 라우팅 경로 설정
def multi_SVM():
    app = Flask(__name__)
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

    now = datetime.datetime.now()
    date = now.strftime('%Y-%m-%d')

    filename = './log/multi_svm.log'
    with open(filename,"a") as f:
        f.write(date)
        f.write("\n")
    print("날짜를 기록하였습니다")
    result=multi_SVM.MoEs(date)
    return json.dumps(multi_SVM, ensure_ascii=False)

#@app.route('/train', methods=['GET'])#app객체로 라우팅 경로 설정
def multi_SVM_train():
    pp = Flask(__name__)
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    now=datetime.datetime.now()
    date=now.strftime('%Y-%m-%d')

    filename='./log/multi_svm_train.log'
    with open(filename, "a") as f:
        f.write(date)
        f.write("\n")
    print("multi_svm_train.log에 모델학습 실행 날짜를 기록하였습니다.")
    result=Multi_SVM.SVMTrain()
    print("MULTI_SVM 모델 학습을 완료하였습니다.")
    return "MULTI_SVM 모델 학습 완료"

#sched_train = BackgroundScheduler(daemon=True)
#sched_train.add_job(multi_SVM_train)
#sched_train.add_job(multi_SVM_train,'interval',hours=24)
#sched_train.start()

#sched = BackgroundScheduler(daemon=True)
#sched.add_job(multi_SVM)
#sched.start()

################################################################

#@app.route("/tfidfRaw",methods=['GET'])


"""
def tfidfRaw():
    app = Flask(__name__)
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

    Numdoc = 600

    contents = tfidf.getTfidfRaw(Numdoc)
    return json.dumps(contents, ensure_ascii=False)
"""
#@app.route("/allTfidfTable",methods=['GET'])

def allTfidfTable():
    app = Flask(__name__)
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    print("get request")
    tfidf_all.getAllTfidfTable()

    sys.exit()
#sched_train = BackgroundScheduler(daemon=True)
#sched_train.add_job(allTfidfTable)
#sched_train.add_job(multi_SVM_train,'interval',hours=24)
#sched_train.start()
#@app.route("/tfidfTable",methods=['GET'])
"""
def tfidfTable():
    app = Flask(__name__)
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

    Numdoc = 3

    contents = tfidf.getTfidfTable(Numdoc)
    #return json.dumps(contents, ensure_ascii=False)
    return json.dumps(contents, ensure_ascii=False)
"""
#########################################
# 191227 ES Test update : use esFunc module
"""
from common import esFunc
@app.route('/esTest', methods=['GET'])
def esTest():
    result = esFunc.esGetDocs(9)
    
    return json.dumps(result, ensure_ascii=False)

@app.route('/testRead', methods=['GET'])
def testRead():
    result = esFunc.dataLoad()

    return json.dumps(result, ensure_ascii=False)
"""
################################################
"""
LDA 잠재 디리클레 할당 모듈화
2019.12.27.
"""
################################################
# With LDA gensim library
# import LDA
"""
@app.route('/lda', methods=['GET'])
def lda():
    app = Flask(__name__)
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    
    result = LDA.LDA(10) ##문서 10개 돌림 
    # print
    return json.dumps(result, ensure_ascii=False)
"""
#recomandation function
#@app.route('/rcmd', methods=['GET', 'POST'])
#def rcmd():
#    print("recomandation function init.")

#    if request.method == 'POST':
#        req = request.json
#        idList = req["idList"]

#     # print("Get id list from Front-End: ",idList)

#    rcmdList = rc.getRcmd(idList)
#    print("rcmd function done!")
#    
#    return json.dumps(rcmdList, ensure_ascii=False)


"""
def textRank():
    import json
    from gensim.summarization import keywords
    DIR_FE = "../Front_KUBIC/src/assets/special_first/ctgRNNResult.json"

    with open(DIF_FE,'r',encoding='utf-8') as fp:
        data = json.load(fp)

    corpus = data[0]["doc"]


    isTokened = True

    if isTokened == True:
        tokenized_doc = []
        for doc in corpus:
            tokenized_doc.append(doc["words"])
    else:
        contents = []
        for doc in corpus:
            contents.append(doc["contents"])

        tagger = Mecab()
        print("데이터 전처리 중... It may takes few hours...")
        tokenized_doc = [tagger.nouns(contents[cnt]) for cnt in range(len(contents))]

    # with open("krWl.txt", "r" ,encoding='utf-8') as f:
    #     texts = f.read() 
        # print(f.read())
        # tokenized_doc = tagger.nouns(texts)
        # tokenized_doc = ' '.join(tokenized_doc) 
        # print("형태소 분석 이후 단어 토큰의 개수",len(tokenized_doc)) 

    # result = keywords(tokenized_doc, words = 15 , scores=True)
    result = keywords(tokenized_doc)

    # with open(DIR_FE, 'w', -1, "utf-8") as f:
    with open("./wrCul.json", 'w', -1, "utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    return json.dumps(result, ensure_ascii=False)
"""
"""
def wordRank():
    #Retreive text from elasticsearch
    results = es.get(index=INDEX, doc_type='nkdb', id='5dc9fc5033ec463330e97e94')
    texts = json.dumps(results['_source'], ensure_ascii=False)

    # split the text by sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', texts)

    # normalize the text
    texts = [normalize(text, number=True) for text in sentences]

    wordrank_extractor = KRWordRank(
        min_count=3,  # Minimum frequency of word
        max_length=10,  # Maximum length of word
        verbose=True
    )

    beta = 0.85  # Decaying factor beta of PageRank
    max_iter = 10

    keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter)

    result = []
    dic = {}
    # Make a dictionary [word, weight]
    for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:30]:
        dic["y"] = r
        dic["label"] = word
        result.append(dic)
        dic = {}

    return json.dumps(result, ensure_ascii=False)
    # return result
"""
"""
@app.route('/wordrank', methods=['GET'])
def chseAlg():
    try: 
        result = wordRank()
    except:
        result =  textRank()
    return result
"""
"""
@app.route('/keywordGraph', methods=['POST', 'GET'])
@cross_origin(app)
def draw():
    if request.method == 'POST':
        result = request.json
        keyword = result["keyword"]

    wholeDataArr = []
    searchDataArr = []

    resultArr = []

    startYear = 1950
    offset = 10

    # From 1950 ~ 2020
    for i in range(0, 8):

        allDocs = {
            "query": {
                "bool": {
                    "must": {
                        "match_all": {}
                    },
                     "filter" : [
                    {"range" : {
                        "post_date" : {
                                "gte" : "1950-01||/M",
                                "lte" : "1950-01||/M",
                                "format": "yyyy-MM"
                            }
                        }}
                    ]
                }
            }
        }

     
        allDocs["query"]["bool"]["filter"][0]["range"]["post_date"]["gte"]= str(startYear+(i*offset))+"-01||/M"
        allDocs["query"]["bool"]["filter"][0]["range"]["post_date"]["lte"]= str(startYear+((i+1) *offset))+"-01||/M"
        # print("query body has been built!")
        res = es.search(index=INDEX, body=allDocs)
        # print(res)

        numOfDocs = res["hits"]["total"]["value"]
        wholeDataArr.append(numOfDocs)
        print(numOfDocs)

        searchDocs = {
            "query" : {
                "bool" : {
                    "must" : [
                        {"match" : {"post_body" : ""}}
                    ],
                     "filter" : [
                    {"range" : {
                        "post_date" : {
                                "gte" : "1950-01||/M",
                                "lte" : "1950-01||/M",
                                "format": "yyyy-MM"
                            }
                        }}
                    ]
                }
            }
        }

        searchDocs["query"]["bool"]["filter"][0]["range"]["post_date"]["gte"]= str(startYear+(i*offset))+"-01||/M"
        searchDocs["query"]["bool"]["filter"][0]["range"]["post_date"]["lte"]= str(startYear+((i+1) *offset))+"-01||/M"
        searchDocs["query"]["bool"]["must"][0]["match"]["post_body"] = keyword
        # print("ready to search")
        res = es.search(index=INDEX, body=searchDocs)
        print(res)
        numOfDocs = res["hits"]["total"]["value"]
        searchDataArr.append(numOfDocs)

        # print(numOfDocs)

    dic = {}
    resultWholeArr = []
    resultSearchArr = []
    # Angular Data Format{ y: 150, label: "Dec" }
    for i in range(0, 8):
        dic["y"] = wholeDataArr[i]
        dic["label"] = str(startYear+(i*offset))

        resultWholeArr.append(dic)
        dic = {}
        dic["y"] = searchDataArr[i]
        dic["label"] = str(startYear+(i*offset))
        resultSearchArr.append(dic)

        dic = {}

    resultArr.append(resultWholeArr)
    resultArr.append(resultSearchArr)

    # print(resultArr)

    return json.dumps(resultArr, ensure_ascii=False)

"""
"""
@app.route('/test', methods=['POST', 'GET'])
def test():
    if request.method == 'POST':
        result = request.json
        keyword = result["keyword"]

    body = {
        "query": {
            "match_all": {}
        },
        "size": 1000,
    }

    res = es.search(index=INDEX, body=body)

    resultArr = res["hits"]["hits"]

    dateArr = []

    dayCntDic = {}

    for i in resultArr:
        dateArr.append(i["_source"]["post_date"])

    for i in dateArr:
        day = i[8:10]

        if day in dayCntDic:
            dayCntDic[day] += 1
        else:
            dayCntDic[day] = 1

    resultDic = []
    dic = {}

    for day, cnt in sorted(dayCntDic.items()):
        dic["y"] = cnt
        dic["label"] = day
        resultDic.append(dic)
        dic = {}

    return json.dumps(resultDic, ensure_ascii=False)
"""

# def rank(contents):
#      #split the text by sentences
#     sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', contents)


#     #normalize the text
#     contents = [normalize(text, number=True) for text in sentences]

#     wordrank_extractor = KRWordRank(
#         min_count = 7,  # Minimum frequency of word
#         max_length=10,  # Maximum length of word
#         verbose = True
#     )

#     beta = 0.85  # Decaying factor beta of PageRank
#     max_iter = 10

#     keywords, rank, graph = wordrank_extractor.extract(contents, beta, max_iter)

#     result=[]
#     dic={}
#     #Make a dictionary [word, weight]
#     for word, r in sorted(keywords.items(), key=lambda x:x[1], reverse=True)[:30]:
#         dic["y"]=r
#         dic["label"]=wordkeyword
#         result.append(dic)
#         dic={}

#     return result
"""
@app.after_request
def after_request(response):

   # response.headers.add('Access-Control-Allow-Origin', '*')
   # response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response
"""
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
