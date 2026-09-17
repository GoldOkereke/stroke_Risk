from fastapi import APIRouter

router = APIRouter()

@router.get("/test")
async def test_assessment():
    return {"message": "Assessment router works"}
