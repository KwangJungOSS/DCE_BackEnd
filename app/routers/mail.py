from fastapi import APIRouter
from fastapi.responses import JSONResponse
import imaplib
import email
from email.header import decode_header

router = APIRouter(
    prefix="/mails",
    tags=["mails"]
)

@router.post("/remove",status_code=200)
async def read_maildelete(mailAddress:str):
    # account credentials
    username = ""
    password = ""

    imap = imaplib.IMAP4_SSL("imap.naver.com")
    try:
        imap.login(username,password)
      #로그인 실패시,
    except imaplib.IMAP4.error as e:
        return JSONResponse(status_code=404, content={"message":"User ID or Password is invalid"})

    delete_code= "FROM"+mailAddress

    try:
        imap.select("INBOX")
        status, messages = imap.search(None, delete_code)
        for mail_ID in messages:
            _, msg = imap.fetch(mail_ID, "(RFC822)")
            # you can delete the for loop for performance if you have a long list of emails
            # because it is only for printing the SUBJECT of target email to delete
            for response in msg:
                if isinstance(response, tuple):
                    msg = email.message_from_bytes(response[1])
                    # decode the email subject
                    subject = decode_header(msg["Subject"])[0][0]
                    if isinstance(subject, bytes):
                        # if it's a bytes type, decode to str
                        subject = subject.decode(encoding='utf-8',errors='ignore')
                    print("Deleting", subject)
            
        # mark the mail as deleted
        imap.store(mail_ID, "+FLAGS", "\\Deleted")
        imap.expunge()
        imap.close()
        imap.logout()

    except imaplib.IMAP4.error as e:
        return JSONResponse(status_code=404, content={"message":"delete-falied"})

    return 

'''
#get_key_from_mail과 세트입니다. 메일을 읽을 때 사용됩니다.
def findEncodingInfo(txt):    
    info = email.header.decode_header(txt)
    s, encoding = info[0]
    return s, encoding
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
    context={'request':request}

    return templates.TemplateResponse("mid.html",context=context)
    #return {"top5":top5}
'''