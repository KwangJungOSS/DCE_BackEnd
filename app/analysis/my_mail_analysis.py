#!/usr/bin/env python
# coding: utf-8

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from email import policy
import email
import imaplib
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import punkt
from pandas import DataFrame


# # 메인 함수

def MygetAnalysisResult():

    platform = input("메일 플랫폼 : ")
    user = input("아이디: ")
    password = input("비밀번호: ")
    
    if platform == "naver":
        imap = Naver_Login(user, password)
        concat_all_unseen = Calling_up_mail(imap)
        #Logout(imap)
    elif platform == "google":
        imap = Google_Login(user, password)
        concat_all_unseen= Calling_up_mail(imap)
        #Logout(imap)
    elif platform == "daum":
        imap = Daum_Login(user, password)
        concat_all_unseen = Calling_up_mail(imap)
        #Logout(imap)
    #concat_all_unseen = Calling_up_mail(imap)

    analysis_list = Mail_Analysis(concat_all_unseen)
    Ko_LDA_word_list = Ko_Mail_LDA(concat_all_unseen)
    En_LDA_word_list = En_Mail_LDA(concat_all_unseen)
    Mail_Naive_Bayes_list = Mail_Naive_Bayes(concat_all_unseen)

    
    LDA_word_list = []
    LDA_word_list.append(Ko_LDA_word_list)
    LDA_word_list.append(En_LDA_word_list)
    
    conclusion = []
    conclusion.append(analysis_list)
    conclusion.append(LDA_word_list)
    conclusion.append(Mail_Naive_Bayes_list)

    return conclusion


def Naver_Login(user, password):
    imap = imaplib.IMAP4_SSL('imap.naver.com')
    #user = "본인 아이디"
    #password="본인 비밀번호"
    imap.login(user,password)
    return imap



def Google_Login(user, password):
    imap = imaplib.IMAP4_SSL('imap.gmail.com')
    #user = "본인 메일@gmail.com"
    #password="구글 2차 비밀번호" -> 따로 설정 필요 & 설명필요
    imap.login(user,password)
    return imap


def Daum_Login(user, password):
    imap = imaplib.IMAP4_SSL('imap.daum.net')
    #user = "본인 메일주소"
    #password="본인 비밀번호"
    imap.login(user,password)
    return imap


# # 로그아웃

# # 메일 불러오기 및 데이터 프레임 형성

#get_key_from_mail과 세트입니다. 메일을 읽을 때 사용됩니다.
def findEncodingInfo(txt):    
    info = email.header.decode_header(txt)
    s, encoding = info[0]
    return s, encoding

def Calling_up_mail(imap):
    
    #메일 불러오면 담을 데이터프레임
    unseen_df = pd.DataFrame(columns={'FROM','TO','DATE','SUBJECT'})
    unseen_df=unseen_df[['FROM','TO','DATE','SUBJECT']]

    all_df = pd.DataFrame(columns={'FROM','TO','DATE','SUBJECT'})
    all_df=all_df[['FROM','TO','DATE','SUBJECT']]
    
    #메일함 선택
    imap.select("INBOX")
    
    status, msg_ids =imap.search(None, "UNSEEN")
    index = 0
    
    # 안읽은 메일 불러오기
    for num in msg_ids[0].split():
        unseen_mail_info=[]
        typ,data = imap.fetch(num,"(RFC822)")
    
        raw_email = data[0][1] 
        email_message = email.message_from_bytes(raw_email,policy=policy.default)
 
        b, encode = findEncodingInfo(email_message['Subject'])

        unseen_mail_info.append(email_message['From'])
        unseen_mail_info.append(email_message['To'])
        unseen_mail_info.append(email_message['Date'])
        unseen_mail_info.append(str(b))

        unseen_df.loc[index]=unseen_mail_info
        index+=1
    
    status, msg_ids =imap.search(None, "ALL")
    index = 0
    
    # 읽은 메일 불러오기
    for num in msg_ids[0].split():
        all_mail_info=[]
        typ,data = imap.fetch(num,"(RFC822)")
    
        raw_email = data[0][1] 
        email_message = email.message_from_bytes(raw_email,policy=policy.default)

        b, encode = findEncodingInfo(email_message['Subject'])

        all_mail_info.append(email_message['From'])
        all_mail_info.append(email_message['To'])
        all_mail_info.append(email_message['Date'])
        all_mail_info.append(str(b))

        all_df.loc[index]=all_mail_info
        index+=1
    
    # 메일 전처리
    all_df.dropna(inplace=True)
    all_df.reset_index(drop=True,inplace = True)


    unseen_df.dropna(inplace=True)
    unseen_df.reset_index(drop=True,inplace = True)
    
    unseen_df['DATE'] = unseen_df['DATE'].apply(lambda x: str(x))
    all_df['DATE'] = all_df['DATE'].apply(lambda x: str(x))
    
    unseen_df['DATE'] = pd.to_datetime(unseen_df['DATE'])
    all_df['DATE'] = pd.to_datetime(all_df['DATE'])
    
    unseen_df['from']=""
    all_df['from']=""
    
    a=0
    com =""
    for i in unseen_df['FROM']:
        split = i.split('<')
        unseen_df['from'][a]=split[0]
        a+=1
        
    a=0
    com =""
    for i in all_df['FROM']:
        split = i.split('<')
        all_df['from'][a]=split[0]
        a+=1
        
    all_df['unseen 여부']=0
    unseen_df['unseen 여부']=1
    
    concat_all_unseen = pd.concat([all_df,unseen_df])
    
    concat_all_unseen=concat_all_unseen.sort_values('DATE')
    
    concat_all_unseen.reset_index(drop=True,inplace = True)
    
    concat_all_unseen = concat_all_unseen.astype({'unseen 여부':'float'})
    
    for i in range(len(concat_all_unseen)-1):
        if (concat_all_unseen.iloc[i]['from'] == concat_all_unseen.iloc[i+1]['from']) & (concat_all_unseen.iloc[i]['DATE'] == concat_all_unseen.iloc[i+1]['DATE']):
            if (concat_all_unseen.iloc[i]['unseen 여부'] == 0.0):
                concat_all_unseen.loc[i, 'unseen 여부'] = np.nan
            elif (concat_all_unseen.iloc[i+1]['unseen 여부'] == 0.0):
                concat_all_unseen.loc[i+1, 'unseen 여부'] = np.nan
                
    concat_all_unseen=concat_all_unseen.dropna()
    concat_all_unseen.reset_index(drop=True,inplace = True)
    
    concat_all_unseen = concat_all_unseen.astype({'unseen 여부':'int'})
    
    return concat_all_unseen


def Mail_Analysis(concat_all_unseen):
    # 받은 메일 top 5
    freq = concat_all_unseen['from'].value_counts().to_frame()
    freq.reset_index(inplace=True)
    freq.columns = ['from','받은 메일 개수']
    freq_sort=freq.sort_values('from')
    freq_sort.reset_index(inplace=True)
    freq_sort.drop(['index'],axis=1, inplace=True)
    
    dict_top_5 = {
        freq.iloc[0][0] : freq.iloc[0][1],
        freq.iloc[1][0] : freq.iloc[1][1],
        freq.iloc[2][0] : freq.iloc[2][1],
        freq.iloc[3][0] : freq.iloc[3][1],
        freq.iloc[4][0] : freq.iloc[4][1]
    }
    
    
    # 전체 메일 개수, 읽은 메일개수, 안읽은 메일 개수, 전체 메일 대비 안읽은 메일 개수
    all_mail = len(concat_all_unseen)
    read_mail = len(concat_all_unseen.loc[concat_all_unseen['unseen 여부'] == 0])
    no_read_mail = len(concat_all_unseen.loc[concat_all_unseen['unseen 여부'] == 1])
    no_read_mail_ratio = round(no_read_mail/all_mail,2)
    dict_mail_analysis = {
        '전체 메일 개수' : all_mail,
        '읽은 메일 개수' : read_mail,
        '안 읽은 메일 개수' : no_read_mail,
        '전체 메일 대비 안읽은메일 비율' : no_read_mail_ratio
    }
    
    analysis_list = []
    analysis_list.append(dict_top_5)
    analysis_list.append(dict_mail_analysis)
    
    return analysis_list


# # 워드클라우드를 위한 LDA


def En_Mail_LDA(concat_all_unseen):
    import nltk
    from nltk.corpus import stopwords
    
    pos_df = concat_all_unseen.copy()
    
    pos_df['SUBJECT'] = pos_df['SUBJECT'].str.replace("[^a-zA-Z]"," ")

    #빈 문자열 NAN 값으로 바꾸기
    pos_df = pos_df.replace({'': np.nan})
    pos_df = pos_df.replace(r'^\s*$', None, regex=True)

    #NAN 이 있는 행은 삭제
    pos_df=pos_df.dropna()
    pos_df=pos_df.reset_index()
    pos_df = pos_df.drop(['index'], axis=1)
    
    
    subject_pos_df = pos_df['SUBJECT']
    subject_pos_df = DataFrame(subject_pos_df.str.lower())
    
    subject_pos_df['wordTokens'] ="zz"
    a=0
    for i in subject_pos_df['SUBJECT']:
        tokens = nltk.word_tokenize(i)
        subject_pos_df['wordTokens'][a] = tokens
        a+=1
        
    subject_pos_df = subject_pos_df.replace('zz', np.nan) # 빈 값 결측치로 처리
    
    subject_pos_df['posTag'] ="zz"
    a=0
    for i in subject_pos_df['wordTokens']:
        tokens_pos = nltk.pos_tag(i)
        subject_pos_df['posTag'][a] = tokens_pos
        a+=1
    
    # 명사는 NN을 포함하고 있음을 알 수 있음
    NN_words = []

    for i in range(len(subject_pos_df)):
        for word, pos in subject_pos_df['posTag'][i]:
            if 'NN' in pos:
                NN_words.append(word)

    # nltk에서 제공되는 WordNetLemmatizer을 이용
    # ex) 명사의 경우는 보통 복수 -> 단수 형태로 변형
    wlem = nltk.WordNetLemmatizer()
    lemmatized_words = []
    for word in NN_words:
        new_word = wlem.lemmatize(word)
        lemmatized_words.append(new_word)

    stopwords_list = stopwords.words('english') #nltk에서 제공하는 불용어사전 이용
    unique_NN_words = set(lemmatized_words)
    final_NN_words = lemmatized_words

    # 불용어 제거
    for word in unique_NN_words:
        if word in stopwords_list:
            while word in final_NN_words: final_NN_words.remove(word)
                

    c = Counter(final_NN_words) # input type should be a list of words (or tokens)
    k = 20
    
    import gensim
    from gensim.utils import simple_preprocess
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    def sent_to_words(sentences):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) 
                 if word not in stop_words] for doc in texts]
    
    data = subject_pos_df.SUBJECT.values.tolist()
    data_words = list(sent_to_words(data))
    # remove stop words
    data_words = remove_stopwords(data_words)
    
    import gensim.corpora as corpora
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # View
    
    # number of topics
    num_topics = 5
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics)
    # Print the Keyword in the 5 topics
    doc_lda = lda_model[corpus]
    
    
    topic_list = list(lda_model.print_topics())
    
    import re
    
    word_list=[]
    return_list=[]
    for i in range(5):
        a = topic_list[i][1]
        word = " ".join(re.findall("[a-zA-Z]+",a))
        word_list.append(word)
        return_list.append(word_list[i].split(' ')[0])
        
    return return_list


def Ko_Mail_LDA(concat_all_unseen):
    from konlpy.tag import Okt
    from gensim.models.ldamodel import LdaModel
    from gensim import corpora
    import numpy as np
    
    clean_Data = concat_all_unseen.copy()
    
    okt = Okt()
    
    #한글이 아니면 빈 문자열로 바꾸기
    clean_Data['SUBJECT'] = clean_Data['SUBJECT'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣]',' ',regex=True)

    #빈 문자열 NAN 값으로 바꾸기
    clean_Data = clean_Data.replace({'': np.nan})
    clean_Data = clean_Data.replace(r'^\s*$', None, regex=True)

    #NAN 이 있는 행은 삭제
    clean_Data.dropna(how='any', inplace=True)

    #인덱스 차곡차곡
    clean_Data = clean_Data.reset_index (drop = True)
    
    #데이터 프레임에 null 값이 있는지 확인
    
    #텍스트 데이터를 리스트로 변환
    Data_list=clean_Data['SUBJECT'].tolist()

    #리스트를 요소별로(트윗 하나) 가져와서 명사만 추출한 후 리스트로 저장
    data_word=[]
    for i in range(len(Data_list)):
        try:
            data_word.append(okt.nouns(Data_list[i]))
        except Exception as e:
            continue

    id2word=corpora.Dictionary(data_word)
    id2word.filter_extremes(no_below = 10) #10회 이하로 등장한 단어는 삭제
    texts = data_word
    corpus=[id2word.doc2bow(text) for text in texts]
    
    num_topics = 10
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = None


    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )
    
    topic_list = model.print_topics()
    import re
    
    word_list=[]
    return_list=[]
    for i in range(10):
        a = topic_list[i][1]
        word = " ".join(re.findall("[ㄱ-ㅎㅏ-ㅣ가-힣]+",a))
        word_list.append(word)
        return_list.append(word_list[i].split(' ')[0])
        
    return return_list


# # 삭제 메일주소 추천

def Mail_Naive_Bayes(concat_all_unseen):
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    
    #모델 생성
    data = pd.read_excel('app/analysis/concat_all_unseen.xlsx')
    data1 = data[data['label'].isnull()]
    
    data=data.dropna()
    data['label']=data['label'].astype('int')
    
    X = data['SUBJECT']
    y = data['label']
    
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=34)
    
    cv = CountVectorizer()
    
    x_traincv= cv.fit_transform(X_train)
    
    mnb = MultinomialNB(alpha=0.1, class_prior=None, fit_prior=True)

    mnb.fit(x_traincv,y_train)
    
    x_testcv = cv.transform(X_test)
    
    predictions = mnb.predict(x_testcv)

    # 모델에 이메일 데이터 테스트
    
    return_data = concat_all_unseen
    
    
    test = concat_all_unseen['SUBJECT']
    
    testcv= cv.transform(test)
    
    test_predictions = mnb.predict(testcv)
    
    predictions_df=pd.DataFrame(test_predictions)
    
    return_data['메일종류'] = predictions_df
    
    
    return_data['메일종류1']= return_data['메일종류']
    return_data.loc[return_data['메일종류'] == 2, '메일종류1'] = 1
    
    
    mail_sum = return_data['FROM'].value_counts().to_frame()
    mail_sum.reset_index(inplace=True)
    mail_sum=mail_sum.sort_values('index')
    mail_sum.reset_index(inplace=True)
    mail_sum.drop(['level_0'],axis=1, inplace=True)
    mail_sum.columns = ['FROM','받은 메일 개수']
    
    naive_sum = return_data.groupby('FROM')['메일종류1'].sum().to_frame()
    naive_sum.reset_index(inplace=True)
    naive_sum=naive_sum.sort_values('FROM')
    naive_sum.reset_index(inplace=True)
    naive_sum.drop(['index'],axis=1, inplace=True)
    naive_sum.columns = ['FROM','베이즈합']
    
    mail_sum['베이즈합'] = naive_sum['베이즈합'] 
    mail_sum['베이즈비율']=round(((mail_sum['베이즈합']/mail_sum['받은 메일 개수']) *100),2)
    mail_sum = mail_sum.sort_values(by=["베이즈비율", "받은 메일 개수"], ascending=[False, False]) 
    mail_sum.reset_index(inplace=True)
    mail_sum.drop(['index'],axis=1, inplace=True)
    
    mail_result = []
    
    for i in range(10):
        mail_result.append(mail_sum.iloc[i][0])

    
    return mail_result

conclusion = MygetAnalysisResult()
print(conclusion)

