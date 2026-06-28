from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import User, get_db


bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthenticatedUser:
    id: str
    display_name: str


def create_api_token() -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    return token, hash_api_token(token)


def hash_api_token(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> AuthenticatedUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Требуется Bearer-токен")
    user = db.scalar(select(User).where(User.token_hash == hash_api_token(credentials.credentials)))
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Недействительный токен")
    return AuthenticatedUser(id=user.id, display_name=user.display_name)
