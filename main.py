from fastapi import FastAPI,Request,Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


app=FastAPI()

# html 파일이 있는 폴더 설정
templates = Jinja2Templates(directory="templates")

# 루트 경로
@app.get("/",response_class=HTMLResponse)
async def read_root(request:Request):
    return templates.TemplateResponse("index.html",{"request":request})

#/test 경로
@app.post("/test")
async def read_test(username: str = Form(), password: str = Form()): #form에서 보낸것을 받아옴.
    return {"username":username}
