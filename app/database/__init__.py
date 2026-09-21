"""Database module for Supabase connectivity."""

from app.database.supabase import (
    get_supabase_client,
    get_supabase_service_client,
)

__all__ = [
    "get_supabase_client",
    "get_supabase_service_client",
]
