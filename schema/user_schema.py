from pydantic import BaseModel, EmailStr, validator,conint
from typing import List

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class  UserLog(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr  # Correct type and field for email

    class Config:
        orm_mode = True

class InfoCreate(BaseModel):
    
    email: List[EmailStr]  # Validate email format

class InfoResponse(BaseModel):

    email: List[EmailStr]
    photo_url: List[str]
    
    class Config:
        orm_mode = True

class PhotoPaths(BaseModel):
    paths: List[str]
