from functools import wraps
from fastapi import HTTPException
import jwt

def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_auth(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        token = kwargs.get('authorization')
        if not token:
            raise HTTPException(status_code=401, detail="No token provided")
        
        try:
            payload = verify_token(token)
            kwargs['user'] = payload
            return await func(*args, **kwargs)
        except Exception as e:
            raise HTTPException(status_code=401, detail=str(e))
    
    return wrapper