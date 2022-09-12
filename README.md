# RDCE(Reduce Digtal Carbon Emission) 소개 
지저분한 메일함! 뭘 지워야 할지 감이 안오시죠?  
DCE를 통해 나의 메일을 분석해보세요.  
내가 어떤 주제에 관심 있는지, 나를 귀찮게 하는 메일은 누구인지, 지워야할 메일은 누구인지 알 수 있어요!  
디지털 탄소 배출 절감을 실천해보세요^^

## 필수 설정
IMAP을 사용해주세요!

### Naver 
1.메일 왼쪽 아래의 환경설정 > POP3 / IMAP 설정 > IMAP / SMTP 설정 탭에서 IMAP/SMTP 사용 옵션을 사용함으로 설정  
![image](https://user-images.githubusercontent.com/71928522/189707903-733f1250-fde3-420f-816d-9a567a146250.png)

### Google
1.컴퓨터에서 Gmail을 엽니다.  
2.오른쪽 상단에서 설정 설정 다음 모든 설정 보기를 클릭합니다.  
3.전달 및 POP/IMAP 탭을 클릭합니다.  
4.'IMAP 액세스' 섹션에서 IMAP 사용을 선택합니다.  
5.변경사항 저장을 클릭합니다.  

## 주요 기능
### 메일 분석
Text-mining을 통한 메일 분석!  
+ ~~
+ ~~
+ ~~
  
  
## 시스템 구성 및 아키텍처
![image](https://user-images.githubusercontent.com/71928522/189708430-c80fc4c0-7318-4f8c-baf9-f3963c5e67c4.png)


## 환경 설정
Python version : 3.9.13  

### Backend
패키지
```
  pip install fastapi
  pip install "uvicorn[standard]"
  ```
  
### mail analysis
패키지
```
pip install numpy
pip install pandas
```

