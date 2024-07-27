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

class EmergencyContactCreate(BaseModel):
    name: str
    email: EmailStr  # Validate email format
    relation: str

    @validator('email')
    def email_must_be_gmail(cls, v):
        if "@gmail.com" not in v:
            raise ValueError('Email must be a Gmail address (@gmail.com)')
        return v

class EmergencyContactResponse(BaseModel):
    id: int
    name: str
    email: EmailStr  # email for output
    relation: str

    class Config:
        orm_mode = True


class FamilyMemberCountUpdate(BaseModel):
    count: conint(ge=0)  # conint is a constrained int type ensuring the count is non-negative

class FamilyMemberCountResponse(BaseModel):
    user_id: int
    family_member_count: int

    class Config:
        orm_mode = True

class InfoCreate(BaseModel):
    
    email: List[EmailStr]  # Validate email format
    photo_url: List[str]  # List of base64 encoded strings

class InfoResponse(BaseModel):

    email: List[EmailStr]
    photo_url: List[str]
    
    class Config:
        orm_mode = True
