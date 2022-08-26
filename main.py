#fastapi
from fastapi import FastAPI,Request,Form
from fastapi.responses import HTMLResponse,RedirectResponse
from fastapi.templating import Jinja2Templates

# CORS 문제
from fastapi.middleware.cors import CORSMiddleware

#router
from routers import mail

#response model, response body
from pydantic import BaseModel, EmailStr
from typing import Union,List,Optional

#email
import imaplib
import email

app=FastAPI()

# react client 주소
ORIGIN = "https://seo-inyoung.github.io/dce-client/"

# 모든 origin, 모든 cookie, 모든 method, 모든 header를 allow 한다는 얘기 
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGIN,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class UserIn(BaseModel):
    inputId : str
    inputPassword : str
    # full_name : Union[str, None] = None

class mailData(BaseModel):
    status: str
    data: Optional[dict]=None
    # full_name : Union[str, None] = None

IMAPADDRESS={"NAVER":"imap.naver.com","GOOGLE":"www","DAUM":"www"}

#/mail 라우터
app.include_router(mail.router)

# html 파일이 있는 폴더 설정
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root(request:Request):
    return {"kwang jeong"}

@app.post("/",response_model=mailData)
async def access_mail(item: UserIn):
    #imap 서버 주소 설정.
    imap = imaplib.IMAP4_SSL("imap.naver.com")
    
    #예외처리
    try:
        # 비밀번호에 . 들어가면 오류 이슈
        imap.login(item.inputId, item.inputPassword)  #메일서버 접속
    
    #로그인 실패시,
    except imaplib.IMAP4.error as e:
        msg={"status":"fail","data":None}
        return msg
    
    msg={"status":"success",
        "data":{
            "sender": {
                "rank": [
                    {"name": "sa","value":100},
                    {"name": "sb","value":600},
                    {"name": "sc","value":300},
                    {"name": "sd","value":250},
                    {"name": "se","value":100}]
            },
            "ratio": {
                "rank": [
                    {"name": "ra","value":100},
                    {"name": "rb","value":600},
                    {"name": "rc","value":300},
                    {"name": "rd","value":250},
                    {"name": "re","value":100}]
            },
            "topic": {
                "rank": [
                    {"name": "ta","value":100},
                    {"name": "tb","value":600},
                    {"name": "tc","value":300},
                    {"name": "td","value":250},
                    {"name": "te","value":100}]
            },
            "delete":["응암정보도서관<ealibsend@ealib.or.kr>","UPPITY<moneyletter@uppity.co.kr>",
            "Trip.com<kr_hotel@trip.com>"]
        }}
    return msg
