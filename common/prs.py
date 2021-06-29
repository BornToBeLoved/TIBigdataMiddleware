# KUBIC module : prs.py
# created Data : 어제 :(
# purpose : 데이터 전처리 모듈
import traceback
# from datetime import datetime
import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
print("called prs.py")
from cmm import showTime
from cmm import SAMP_DATA_DIR
import esFunc

# 운영체제에 따라 미캡 모듈이 다르다.
import os
if os.name == "nt":# 윈도우 운영체제
    from eunjeon import Mecab
else:# 현재 리눅스 서버 및 맥은 konlpy으로 미캡 모듈 import
    from konlpy.tag import Mecab

NUM_DOC = 0

#RANDOM_MODE
# 알고리즘 정확성 확인을 위해서 문서를 불러와서 순서를 섞는다.
RANDOM_MODE = False

#OFFLINE_MODE
# use sample data in ./raw data sample, and not connet to ES.
# without HGU-WLAN network, use raw data sample no matter this value
BACKEND_CONCT = True


# Phase 1 : ES에서 문서 쿼리 및 content와 title 분리 전처리
"""
function : loadData
purpose : 문서 로드 해준다.
input : 몇개의 문서의 로드?
output : dictionary : 
        {"id" : idList, "titles" : titles, "contents" : contents}

"""
def loadData(num_doc = NUM_DOC):
    #if internet connection failed to backend    
    import json
    import sys
    import traceback
    # print(N
    # UM_DOC)
    global NUM_DOC
    if NUM_DOC != num_doc:
        NUM_DOC = num_doc
        print("NUM_DOC updated to ", NUM_DOC)
    print("데이터 로드 중...")
    try :
        if BACKEND_CONCT == False:
            raise Exception("서버 연결 불가")
        corpus = esFunc.esGetDocs(NUM_DOC)
        print("connection to Backend server succeed!")
        print(len(corpus),"개의 문서를 가져옴")# 문서의 수... 내용 없으면 뺀다...

    except Exception as e:
        # traceback.print_exc()
        print('Error: {}. {}'.format(sys.exc_info()[0],
                sys.exc_info()[1]))


        print("current dir : " ,os.getcwd())
        print("대체 파일 로드 from ",SAMP_DATA_DIR)
        # print("\n\nDEBUG : 현재 실행 directory 위치",os.getcwd(),"\n\n")
        with open(SAMP_DATA_DIR, "rt", encoding="UTF8") as f:
            corpus = json.load(f)
        
        print("connection to Backend server failed!")
    showTime() 
    NUM_DOC = len(corpus) # 전체 사용 가능한 문서 수를 업데이트한다.
    print("NUM_DOC updated to ", NUM_DOC)
 
    print("문서 로드 완료!")
    print()


    # 알고리즘 정확성을 확인하기 위해 일부러 문서 순서를 섞는다.
    if RANDOM_MODE == True:
        import random
        random.shuffle(corpus)


    idList = []
    titles = []
    contents = []
    count = 0
    for idx, doc in enumerate(corpus):
        # print(doc["content"])
        if doc["content"] != "":
            idList.append(doc["_id"])
            titles.append(doc["post_title"])
            contents.append(doc["content"])
        else:
            count += 1

    NUM_DOC = len(contents)
    print("NUM_DOC updated to ", NUM_DOC)

    print(count,"개의 문서가 내용이 없음")
    print("투입된 문서의 수 : %d" %(NUM_DOC))

    corpusIdTtlCtt = {"id" : idList, "titles" : titles, "contents" : contents}
    return corpusIdTtlCtt


def loadAllData():
    #if internet connection failed to backend    
    import json
    import sys
    import traceback

    print("start getting all docs")
    corpus = esFunc.getAllDocQuery()

    print(len(corpus),"개의 문서를 가져옴")# 문서의 수... 내용 없으면 뺀다...

    showTime() 
    NUM_DOC = len(corpus) # 전체 사용 가능한 문서 수를 업데이트한다.
    print("문서 로드 완료!")
    # 알고리즘 정확성을 확인하기 위해 일부러 문서 순서를 섞는다.
    if RANDOM_MODE == True:
        import random
        random.shuffle(corpus)


    idList = []
    titles = []
    contents = []
    count = 0
    for idx, doc in enumerate(corpus):
        # print(doc["content"])
        if doc["content"] != "":
            idList.append(doc["_id"])
            titles.append(doc["post_title"])
            contents.append(doc["content"])
        else:
            count += 1

    corpusIdTtlCtt = {"id" : idList, "titles" : titles, "contents" : contents}
    return corpusIdTtlCtt

# phase 2 형태소 분석기 + 내용 없는 문서 지우기
def dataPrePrcs(corpus_with_id_title_content):
    cnt = 0
    idList = corpus_with_id_title_content["id"]
    titles = corpus_with_id_title_content["titles"]
    contents = corpus_with_id_title_content["contents"]
    #mecab test
    import re
    rex1 = re.compile('[^가-힣0-9*.?,!]')#한글 숫자 자주 쓰는 문자만 취급
    
    tagger = Mecab()
    for i,c in enumerate(contents):
        try:
            c = rex1.sub(" ",c)
        except Exception:
            # print("에러 : title : ", titles[i], ", content : ", c)# 문서 내용이 None
            cnt = cnt+1
            print(cnt)
    
    num_co = 0
    tokenized_doc = []
    failIdxList = []
    
    for i, c in enumerate(contents):
        num_co = num_co + 1
        print(str(num_co) + "번쨰 분석중")
        try:
            t = tagger.nouns(c)
            tokenized_doc.append(t)
        except:
            failIdxList.append(i)
    for idx in reversed(failIdxList):
        idList.pop(idx)
        titles.pop(idx)

    showTime()

    # 한글자 단어들 지우기!
    num_doc = len(tokenized_doc)
    for i in range(num_doc):
        tokenized_doc[i] = [word for word in tokenized_doc[i] if len(word) > 1]
    
    return idList, titles, tokenized_doc, contents


"""
# functin : readData(int)
# purpose : 데이터를 로드해서 형태소 분석 전처리까지 하고 데이터 반환
# input : int : num_doc : 준비할 데이터의 수
# output : [
                ["문서1 단어1", "문서1 단어 2"],
                ["문서1 단어1", "문서1 단어 2"],
                ...
           ]
"""
def readyData(num_doc, isCont = False):
    # NUM_DOC initialize
    global NUM_DOC
    NUM_DOC = num_doc
    idList = []
    titles = []
    contents = []

    # print("in readyData after if, ", NUM_DOC)

    # Phase 1 : ES에서 문서 쿼리 및 content와 title 분리 전처리
    
    print("\n\n#####Phase 1-1 : 데이터 로드 실행#####")
    print("Data Loda CURRENT OPTION : ",
          "\nBACKEND CONNECTION OPTION : ", str(BACKEND_CONCT),
          "\nRANDOM ORDER OPTION : ", str(RANDOM_MODE)
         )
    corpus_with_id_title_content = loadData(NUM_DOC)# load data and update NUM_DOC

    print("\n\n#####Phase 1-2 : 데이터 전처리 실행#####")
    (idList, titles, tokenized_doc, contents) = dataPrePrcs(corpus_with_id_title_content)

    return idList, titles, tokenized_doc, contents


def readyDataForAll():
    idList = []
    titles = []
    contents = []

    corpus_with_id_title_content = loadAllData()# load data and update NUM_DOC
    print("\n\n#####Phase 1-2 : 데이터 전처리 실행#####")
    (idList, titles, tokenized_doc, contents) = dataPrePrcs(corpus_with_id_title_content)

    return idList, titles, tokenized_doc, contents
