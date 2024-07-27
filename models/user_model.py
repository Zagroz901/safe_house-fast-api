from sqlalchemy import Column, Integer, String, ForeignKey, create_engine
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

DATABASE_URL = "mysql+pymysql://root:zako0992417578@localhost/safe_house"

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)  # Add email field
    hashed_password = Column(String(255), nullable=False)
    emergency_contacts = relationship("EmergencyContact", back_populates="user")
    family_member_count = Column(Integer, default=0)  # New column to store the family member count
    family_photos= relationship("FamilyPhotos", back_populates="user")

class EmergencyContact(Base):
    __tablename__ = "emergency_contacts"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    email = Column(String(100), nullable=False)  # Assuming validation for email format will be handled
    user = relationship("User", back_populates="emergency_contacts")

class FamilyPhotos(Base):
    __tablename__= "family_photos"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    photo_url = Column(String(255), nullable=False)    
    user = relationship("User", back_populates="family_photos")
    
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)
