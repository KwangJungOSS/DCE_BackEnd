from fastapi import HTTPException,APIRouter,Request,Response,Depends
from fastapi.responses import JSONResponse

#const
from common.consts import IMAPADDRESS

#email
import pandas as pd
import imaplib
from enum import Enum
import email
from email import policy

#response model, response body
from pydantic import BaseModel, EmailStr
from typing import Union,List,Optional

#Session Backend
from uuid import UUID,uuid4
from fastapi_sessions.backends.implementations import InMemoryBackend
from fastapi_sessions.session_verifier import SessionVerifier
from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters


router = APIRouter()

# https://stackoverflow.com/questions/70302056/define-a-pydantic-nested-model <- 중첩된 json은 이런식으로 표현하래서 일케함.

class UserIn(BaseModel):
    inputId : str
    inputPassword : str
    socialId: str
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
    data: Optional[mailData]=None


#session - 47 ~ 117 lines
class SessionData(BaseModel):
    platform : str
    #userinfo :UserIn
    
cookie_params = CookieParameters()

# Uses UUID
cookie = SessionCookie(
    cookie_name="cookie",
    identifier="general_verifier",
    auto_error=True,
    secret_key="DONOTUSE",
    cookie_params=cookie_params,
)

backend= InMemoryBackend[UUID,SessionData]()

class BasicVerifier(SessionVerifier[UUID, SessionData]):
    def __init__(
        self,
        *,
        identifier: str,
        auto_error: bool,
        backend: InMemoryBackend[UUID, SessionData],
        auth_http_exception: HTTPException,
    ):
        self._identifier = identifier
        self._auto_error = auto_error
        self._backend = backend
        self._auth_http_exception = auth_http_exception

    @property
    def identifier(self):
        return self._identifier

    @property
    def backend(self):
        return self._backend

    @property
    def auto_error(self):
        return self._auto_error

    @property
    def auth_http_exception(self):
        return self._auth_http_exception

    def verify_session(self, model: SessionData) -> bool:
        """If the session exists, it is valid"""
        return True


verifier = BasicVerifier(
    identifier="general_verifier",
    auto_error=True,
    backend=backend,
    auth_http_exception=HTTPException(status_code=403, detail="invalid session"),
)

@router.get("/whoami", dependencies=[Depends(cookie)])
async def whoami(session_data: SessionData = Depends(verifier)):
    return session_data

@router.get("/create")
async def create_session(response: Response):
    session = uuid4()
    data = SessionData(platform="hello")
    await backend.create(session, data)
    cookie.attach_to_response(response, session)
    return "created session for"


@router.post("/",response_model=responAna, status_code=200)
async def access_mail(item: UserIn,response:Response):


    #imap 서버 주소 설정.
    imap = imaplib.IMAP4_SSL(IMAPADDRESS[item.socialId])
    
    #예외처리
    try:
        # 비밀번호에 . 들어가면 오류 이슈
        imap.login(item.inputId, item.inputPassword)  #메일서버 접속
    
    #로그인 실패시,
    except imaplib.IMAP4.error as e:
        return JSONResponse(status_code=404, content={"message":"User ID or Password is invalid"})

   #로그인 성공 -> Session에 정보 저장
    session=uuid4()
    user_data = SessionData(userinfo=item,platform="naver")

    await backend.create(session,user_data)
    cookie.attach_to_response(response,session)

    #request.session["id"]=item.inputId     
    #print(request.session["id"])

    msg={"data":{
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