from fastapi import APIRouter

router = APIRouter()

@router.get("/login")
async def login():
    return {"message": "Auth router works"}
