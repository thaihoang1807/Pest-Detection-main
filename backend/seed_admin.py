from sqlalchemy.orm import Session

from db.database import SessionLocal
from models.user import User, RoleEnum

from services.user_service import hash_password

db: Session = SessionLocal()

# Tìm xem tài khoản user đã có trong DB chưa
normal_user = db.query(User).filter(
    User.email == "user@pest.local"
).first()

if not normal_user:

    # Tạo tài khoản mới với quyền USER
    normal_user = User(
        name="Nong Dan",
        email="user@pest.local",
        password=hash_password("123456"),
        role=RoleEnum.USER, # Bắt buộc là quyền USER để hiện tính năng Upload
        is_verified=True
    )

    db.add(normal_user)
    db.commit()

    print("User account created successfully. Email: user@pest.local | Pass: 123456")

else:
    print("User already exists.")

db.close()