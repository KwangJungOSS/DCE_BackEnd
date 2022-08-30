import uvicorn

from fastapi import FastAPI
# CORS 문제
from fastapi.middleware.cors import CORSMiddleware

#router
from common.consts import ORIGIN
from routers import index,mail

def create_app():
    
    app=FastAPI()

    #미들웨어 정의
    
    # 모든 origin, 모든 cookie, 모든 method, 모든 header를 allow 한다는 얘기 
    app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGIN,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
    )

    #라우터 정의
    app.include_router(index.router)
    app.include_router(mail.router)

    return app

app = create_app()

if __name__ =="__main__":
    uvicorn.run("main:app",host="127.0.0.1",port=8000,reload=True)
