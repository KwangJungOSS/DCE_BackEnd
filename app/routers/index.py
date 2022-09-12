from fastapi import APIRouter,Response,Depends
from fastapi.responses import JSONResponse

#const
from common.consts import IMAPADDRESS
from models.mailAnalysis import UserIn,responAna

#email
import imaplib

router = APIRouter()

#서버 연결 Test용
@router.get("/")
async def read_test():
    return {"Hello":"World"}

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

   #로그인 성공, 분석 시작

    msg={"data":{
            #나에게 메일을 가장 많이 보내는 사람 Top5
            "sender": [
                    {"name": "링커리어","count":600},
                    {"name": "하나투어","count":400},
                    {"name": "한국SW산업협회","count":380},
                    {"name": "NEWNEEK","count":250},
                    {"name": "순살브리핑","count":100}]
            ,
            # 메일을 받는 비율 
            "ratio":  [
                    {"name": "광고","count":100},
                    {"name": "구독","count":600},
                    {"name": "A","count":300},
                    {"name": "B","count":250},
                    {"name": "C","count":100}]
            ,
            # WordCloud용 단어 최소 15개. 영어 5개, 한국어 10개
            "topic":  [
                    {"name": "취업","count":500},
                    {"name": "개발","count":300},
                    {"name": "웹툰","count":200},
                    {"name": "광운대학교","count":119},
                    {"name": "유튜브","count":50}]
            ,
            #삭제 권장 메일 ( 여러개 갈 수도 있음.)
            "delete":["응암정보도서관<ealibsend@ealib.or.kr>","UPPITY<moneyletter@uppity.co.kr>",
            "Trip.com<kr_hotel@trip.com>"]
        }}
    return msg