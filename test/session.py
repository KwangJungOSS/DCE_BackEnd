'''
Session을 이용하는 코드입니다.
보류.
'''

from fastapi import HTTPException,Response,Depends
#Session Backend
from uuid import UUID,uuid4
from fastapi_sessions.backends.implementations import InMemoryBackend
from fastapi_sessions.session_verifier import SessionVerifier
from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters

#response model, response body
from pydantic import BaseModel

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
    #session_data.platform
    return session_data.platform

@router.get("/create")
async def create_session(response: Response):
    session = uuid4()
    data = SessionData(platform="hello")
    await backend.create(session, data)
    cookie.attach_to_response(response, session)
    return "created session for"