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
                    {"name": "받은 메일 Top 1","count":600},
                    {"name": "받은 메일 Top 2","count":400},
                    {"name": "받은 메일 Top 3","count":380},
                    {"name": "받은 메일 Top 4","count":250},
                    {"name": "받은 메일 Top 5","count":100}]
            ,
            # 메일을 받는 비율 
            "ratio":  [
                    {"name": "전체 메일 개수","count":500},
                    {"name": "읽은 메일 개수","count":100},
                    {"name": "안 읽은 메일 개수","count":400},
                    {"name": "전체 읽은 메일 대비 안 읽은 메일","count":80.00}]
            ,
            # WordCloud용 단어 15개. 영어 5개, 한국어 10개
            "topic":  [
                    {"text": "한글 1","value":100},
                    {"text": "한글 2","value":90},
                    {"text": "한글 3","value":80},
                    {"text": "한글 4","value":70},
                    {"text": "한글 5","value":60},
                    {"text": "한글 6","value":50},
                    {"text": "한글 7","value":40},
                    {"text": "한글 8","value":30},
                    {"text": "한글 9","value":20},
                    {"text": "한글 10","value":10},
                    {"text": "english 1","value":50},
                    {"text": "english 2","value":40},
                    {"text": "english 3","value":30},
                    {"text": "english 4","value":20},
                    {"text": "english 5","value":10}]
            ,
            # 삭제 권장 메일 ( 10개 )
            "delete":[
                {"name":"응암정보도서관","address":"<ealibsend@ealib.or.kr>"},
                {"name":"홍익정보도서관","address":"<ealibsend@ealib.or.kr>"},
                {"name":"석계정보도서관","address":"<ealibsend@ealib.or.kr>"},
                {"name":"태릉정보도서관","address":"<ealibsend@ealib.or.kr>"},
                {"name":"인천정보도서관","address":"<ealibsend@ealib.or.kr>"},
                {"name":"광운정보도서관","address":"<ealibsend@ealib.or.kr>"},
                {"name":"고려정보도서관","address":"<ealibsend@ealib.or.kr>"},
                {"name":"노원정보도서관","address":"<ealibsend@ealib.or.kr>"},
                {"name":"월계정보도서관","address":"<ealibsend@ealib.or.kr>"},
                {"name":"녹번정보도서관","address":"<ealibsend@ealib.or.kr>"}]
            }
        }
    return msg