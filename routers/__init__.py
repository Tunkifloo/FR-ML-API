# routers/__init__.py
"""
Routers de la API REST para el sistema de reconocimiento facial
"""

from .users import router as users_router
from .recognition import router as recognition_router

__all__ = [
    "users_router",
    "recognition_router"
]