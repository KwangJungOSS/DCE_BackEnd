from fastapi import APIRouter,Request,Form
import pandas as pd
import imaplib
from enum import Enum
import email
from email import policy

#response model, response body
from pydantic import BaseModel, EmailStr
from typing import Union,List,Optional

router = APIRouter()

# https://stackoverflow.com/questions/70302056/define-a-pydantic-nested-model <- 중첩된 json은 이런식으로 표현하래서 일케함.

class UserIn(BaseModel):
    inputId : str
    inputPassword : str
    # full_name : Union[str, None] = None

class value(BaseModel):
    name:str
    count:int

class mailData(BaseModel):
    sender: List[value]
    ratio: List[value]
    topic: List[value]
    delete: List[str]    
    # full_name : Union[str, None] = None

#response analysis 줄임말.
class responAna(BaseModel):
    status: str
    data: Optional[mailData]=None
 

 
@router.get("/")
async def root(request:Request):
    return {"kwang jeong"}


@router.post("/",response_model=responAna)
async def access_mail(request:Request, item: UserIn):
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

   #로그인 성공 -> Session에 정보 저장

    #request.session["id"]=item.inputId     
    #print(request.session["id"])

    msg={"status":"success",
        "data":{
            "sender": [
                    {"name": "링커리어","count":600},
                    {"name": "하나투어","count":400},
                    {"name": "한국SW산업협회","count":380},
                    {"name": "NEWNEEK","count":250},
                    {"name": "순살브리핑","count":100}]
            ,
            "ratio":  [
                    {"name": "광고","count":100},
                    {"name": "구독","count":600},
                    {"name": "A","count":300},
                    {"name": "B","count":250},
                    {"name": "C","count":100}]
            ,
            "topic":  [
                    {"name": "취업","count":500},
                    {"name": "개발","count":300},
                    {"name": "웹툰","count":200},
                    {"name": "광운대학교","count":119},
                    {"name": "유튜브","count":50}]
            ,
            "delete":["응암정보도서관<ealibsend@ealib.or.kr>","UPPITY<moneyletter@uppity.co.kr>",
            "Trip.com<kr_hotel@trip.com>"]
        }}
    return msg