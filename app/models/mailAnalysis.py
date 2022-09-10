#response model, response body
from pydantic import BaseModel
from typing import List,Optional

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


