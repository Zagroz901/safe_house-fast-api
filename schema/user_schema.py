from pydantic import BaseModel, EmailStr, validator

class UserCreate(BaseModel):
    username: str
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
