from fastapi import APIRouter,Request,Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse,RedirectResponse
import pandas as pd
import imaplib
from enum import Enum
import email
from email import policy

IMAPADDRESS={"NAVER":"imap.naver.com","GOOGLE":"www","DAUM":"www"}

#get_key_from_mail과 세트입니다. 메일을 읽을 때 사용됩니다.
def findEncodingInfo(txt):    
    info = email.header.decode_header(txt)
    s, encoding = info[0]
    return s, encoding

router = APIRouter(
    prefix="/mails",
    tags=["mails"],
    responses={404: {"description": "Not found"}},
)

# html 파일이 있는 폴더 설정
templates = Jinja2Templates(directory="templates")


@router.post("/",response_class=HTMLResponse)
async def read_test(request:Request, platform:str = Form(), username: str = Form(), password: str = Form()): #form에서 보낸것을 받아옴.
    
    #imap 서버 주소 설정.
    imap = imaplib.IMAP4_SSL(IMAPADDRESS[platform])
   
    #예외처리
    try:
        imap.login(username,password)  #메일서버 접속
    
    #로그인 실패시,
    except imaplib.IMAP4.error as e:
        context={'request':request,'error':True}
        #루트로 리디렉트.
        return RedirectResponse(url="/")



    print("메일서버 접속성공")

    #메일 불러오는 코드인데 너무 오래걸려서 주석처리해놨슴
    
    imap.select("INBOX")
    status, msg_ids =imap.search(None,"ALL")

    df = pd.DataFrame(columns={'FROM','TO','DATE','SUBJECT'})
    df=df[['FROM','TO','DATE','SUBJECT']]

    '''
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
    context={'request':request}

    return templates.TemplateResponse("mid.html",context=context)
    #return {"top5":top5}
