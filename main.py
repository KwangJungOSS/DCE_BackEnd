from fastapi import FastAPI,Request,Form
from fastapi.responses import HTMLResponse,RedirectResponse
from pydantic import BaseModel, EmailStr
from typing import Union
from fastapi.templating import Jinja2Templates
from routers import mail
import imaplib
from enum import Enum
import email
from email import policy
# CORS 문제
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()

# react client 주소
origins = "https://seo-inyoung.github.io/dce-client/"

# 모든 origin, 모든 cookie, 모든 method, 모든 header를 allow 한다는 얘기 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class UserIn(BaseModel):
    inputId : str
    inputPassword : str
    # full_name : Union[str, None] = None

class UserOut(BaseModel):
    inputId : str
    # full_name : Union[str, None] = None

#/mail 라우터
app.include_router(mail.router)

# html 파일이 있는 폴더 설정
templates = Jinja2Templates(directory="templates")

# 루트 경로
#@app.get("/",response_class=HTMLResponse)
#async def read_root(request:Request):
#    return templates.TemplateResponse("index.html",{"request":request,"error":False})

# 루트 경로
#@app.get("/")
#async def root(request:Request):
#    return {"kwang jeong"}

IMAPADDRESS={"NAVER":"imap.naver.com","GOOGLE":"www","DAUM":"www"}

@app.post("/")
async def create_Id(request:Request, item: UserIn):
    #imap 서버 주소 설정.
    imap = imaplib.IMAP4_SSL("imap.naver.com")
    
    #예외처리
    try:
        # 비밀번호에 . 들어가면 오류 이슈
        imap.login(item.inputId, item.inputPassword)  #메일서버 접속
    
    #로그인 실패시,
    except imaplib.IMAP4.error as e:
        return {"로그인 실패"}
        
    return {"로그인 성공"}

#메일 서버 접속 실패시
@app.post("/",response_class=HTMLResponse)
async def read_wrong_login(request:Request):
    return templates.TemplateResponse("index.html",{"request":request,"error":True})


#분석 시각화 
@app.get("/test",response_class=HTMLResponse)
async def read_mail(request:Request):

    #임시 json 파일
    my_json={
        "SendToYou": {
            "Rank": [
                {"name": "sa","value":100},
                {"name": "sb","value":600},
                {"name": "sc","value":300},
                {"name": "sd","value":250},
                {"name": "se","value":100}]
        },
        "Ratio": {
            "Rank": [
                {"name": "ra","value":100},
                {"name": "rb","value":600},
                {"name": "rc","value":300},
                {"name": "rd","value":250},
                {"name": "re","value":100}]
        },
        "Topic": {
            "Rank": [
                {"name": "ta","value":100},
                {"name": "tb","value":600},
                {"name": "tc","value":300},
                {"name": "td","value":250},
                {"name": "te","value":100}]
        },
        "Delete":["응암정보도서관<ealibsend@ealib.or.kr>","UPPITY<moneyletter@uppity.co.kr>",
        "Trip.com<kr_hotel@trip.com>"]
    }
    context={'request':request,'my_json':my_json}

    return templates.TemplateResponse("mail.html",context=context)
