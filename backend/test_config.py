# Create a test file test_config.py
from app.core.config import settings

print(f"App Name: {settings.APP_NAME}")
print(f"Debug: {settings.DEBUG}")
print(f"Port: {settings.PORT}")
print(f"CORS: {settings.CORS_ORIGINS}")