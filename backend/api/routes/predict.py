import os
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, Depends, Form, HTTPException
from sqlalchemy.orm import Session
from models.user import RoleEnum, User
from services.authentication_service import get_current_user
from models.detection import Detection
from models.batch_summary import BatchSummary
from schemas.detection import DetectionResponse, PredictCreateResponse
from schemas.batch import BatchCreateResponse, BatchStatusResponse, BatchSummaryResponse
from services.predict_service import dispatch_single, dispatch_batch
from services.inference_service import predict_video_file # IMPORT THÊM HÀM XỬ LÝ VIDEO
from api.deps import get_db
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["Predict"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/", response_model=PredictCreateResponse)
async def predict( 
    file: UploadFile = File(...),
    confidence_threshold: float = Form(default=0.25, ge=0.01, le=1.0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != RoleEnum.USER:
        raise HTTPException(status_code=403, detail="Only users can access this endpoint.")

    content = await file.read()
    record = dispatch_single(db, str(current_user.id), content, file.filename, confidence_threshold)
    return {"id": record.id, "status": "PENDING"}

@router.post("/batch", response_model=BatchCreateResponse)
async def predict_batch(
    files: list[UploadFile] = File(...),
    confidence_threshold: float = Form(default=0.25, ge=0.01, le=1.0),
    webhook_url: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != RoleEnum.USER:
        raise HTTPException(status_code=403, detail="Only users can access this endpoint.")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Not to upload over 10 images per batch.")
    
    files_data = [(await f.read(), f.filename) for f in files]
    batch_id = dispatch_batch(db, str(current_user.id), files_data, confidence_threshold, webhook_url)

    return {"batch_id": batch_id, "total": len(files), "status": "PROCESSING"}

# ==========================================================
# API MỚI: NHẬN DIỆN VÀ ĐẾM QUA VẠCH BẰNG VIDEO (MVP)
# ==========================================================
@router.post("/video")
async def predict_video(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(default=0.35, ge=0.01, le=1.0),
    current_user: User = Depends(get_current_user)
):
    # Kiểm tra quyền User
    if current_user.role != RoleEnum.USER:
        raise HTTPException(status_code=403, detail="Only users can access this endpoint.")

    # Kiểm tra định dạng video
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file video định dạng .mp4, .avi, .mov")

    # Lưu video vào thư mục tạm với tên ngẫu nhiên để tránh trùng lặp khi nhiều người up cùng lúc
    temp_filename = f"temp_{uuid.uuid4().hex[:8]}_{file.filename}"
    temp_input_path = os.path.join(UPLOAD_DIR, temp_filename)
    
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Gọi hàm AI xử lý Video từ inference_service
        logger.info(f"User {current_user.email} đang chạy AI trên Video: {file.filename}...")
        result = predict_video_file(temp_input_path, conf=confidence_threshold)
        
        # Trả về kết quả trực tiếp cho Frontend
        return {
            "status": "SUCCESS",
            "message": "Xử lý video thành công!",
            "total_pests": result["total_count"],
            "output_video_url": f"/{result['output_video_path']}" 
        }
    except Exception as e:
        logger.error(f"Lỗi khi xử lý video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ khi xử lý video: {str(e)}")
    finally:
        # Xóa file video gốc để giải phóng dung lượng ổ cứng cho máy chủ
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

# ==========================================================

@router.get("/{detection_id}", response_model=DetectionResponse)
async def get_predict_result(
    detection_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != RoleEnum.USER:
        raise HTTPException(status_code=403, detail="Only users can access this endpoint.")

    record = db.query(Detection).filter(
        Detection.id == detection_id,
        Detection.user_id == current_user.id
    ).first()
    
    if not record:
        raise HTTPException(status_code=404, detail="Request not found")

    return record

@router.get("/batch/{batch_id}", response_model=BatchStatusResponse)
def get_batch_status(
    batch_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != RoleEnum.USER:
        raise HTTPException(status_code=403, detail="Only users can access this endpoint.")

    records = db.query(Detection).filter(
        Detection.batch_id == batch_id,
        Detection.user_id == current_user.id
    ).all()

    if not records:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    finished = sum(1 for r in records if r.status == "FINISHED")

    return {
        "batch_id": batch_id,
        "total": len(records),
        "finished": finished,
        "failed": sum(1 for r in records if r.status == "FAILED"),
        "progress": f"{finished}/{len(records)}"
    }

@router.get("/batch/{batch_id}/summary", response_model=BatchSummaryResponse)
def get_batch_summary(
    batch_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != RoleEnum.USER:
        raise HTTPException(status_code=403, detail="Only users can access this endpoint.")

    summary = db.query(BatchSummary).filter(
        BatchSummary.batch_id == batch_id,
        BatchSummary.user_id == current_user.id
    ).first()
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found or batch not finished yet.")
    
    return summary