from fastapi import APIRouter, File, UploadFile
from app.models.schemas import FaceAnalysisResponse
from app.services.deepface_service import DeepFaceService

router = APIRouter(prefix="/analyze", tags=["Face Analysis"])

@router.post("/", response_model=FaceAnalysisResponse)
async def analyze_face(file: UploadFile = File(...)):
    """
    Анализ лица на загруженном изображении
    """
    # Проверяем тип файла
    if not file.content_type.startswith("image/"):
        return FaceAnalysisResponse(
            success=False,
            error="Файл должен быть изображением"
        )
    
    # Читаем байты
    image_bytes = await file.read()
    
    # Анализируем
    result = await DeepFaceService.analyze_face(image_bytes)
    
    if "error" in result:
        return FaceAnalysisResponse(
            success=False,
            error=result["error"]
        )
    
    return FaceAnalysisResponse(
        success=True,
        data={
            "emotion": result["emotion"],
            "emotion_scores": result["emotion_scores"],
            "age": result["age"],
            "gender": result["gender"],
            "gender_confidence": result["gender_confidence"]
        }
    )
