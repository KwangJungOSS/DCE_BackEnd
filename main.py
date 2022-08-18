from fastapi import FastAPI,Request,Form
from fastapi.responses import HTMLResponse,PlainTextResponse
from fastapi.templating import Jinja2Templates
import imaplib
import email
from email import policy
import pandas as pd

app=FastAPI()

# html 파일이 있는 폴더 설정
templates = Jinja2Templates(directory="templates")

#get_key_from_mail과 세트입니다. 메일을 읽을 때 사용됩니다.
def findEncodingInfo(txt):    
    info = email.header.decode_header(txt)
    s, encoding = info[0]
    return s, encoding

# 루트 경로
@app.get("/",response_class=HTMLResponse)
async def read_root(request:Request):
    return templates.TemplateResponse("index.html",{"request":request})

#/test 경로
@app.post("/test", response_class=PlainTextResponse)
async def read_test(username: str = Form(), password: str = Form()): #form에서 보낸것을 받아옴.
    
    df = pd.DataFrame(columns={'FROM','TO','DATE','SUBJECT'})
    df=df[['FROM','TO','DATE','SUBJECT']]
    
    imap = imaplib.IMAP4_SSL('imap.naver.com')
   
    imap.login(username,password)  #메일서버 접속
    print("메일서버 접속성공")
    
    #메일 불러오는 코드인데 너무 오래걸려서 주석처리해놨슴
    '''
    imap.select("INBOX")
    status, msg_ids =imap.search(None,"ALL")

    index = 0
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
        df.loc[index]=unseen_mail_info
        index+=1
    top_=df["FROM"].value_counts(ascending=False)
    top5=top_.head(5).index.tolist()
    print(top5)
    '''

    #그냥 해놨어요
    return "Hello"
