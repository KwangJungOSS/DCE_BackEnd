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

    imap = imaplib.IMAP4_SSL("imap.naver.com") # < 수정 필요 > IMAPADDRESS[item.socialId] - 이방법은 안 될 듯 : item은 안 받기 때문
    try:
        imap.login(username,password)
      #로그인 실패시,
    except imaplib.IMAP4.error as e:
        return JSONResponse(status_code=404, content={"message":"User ID or Password is invalid"})

    delete_code = "FROM"+mailAddress

    try:
        imap.select("INBOX")
        status, messages = imap.search(None, "FROM<mangonie@naver.com>") # < delete_code
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