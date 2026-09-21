"""Supabase Database Client Factory.

Initializes and manages client instances for communicating with Supabase PostgreSQL.
Contains zero business logic.
"""

import logging
import socket
from typing import Optional
from supabase import Client, create_client
from app.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# Ensure IPv4 resolution on environments with unreachable IPv6/NAT64 routes
_original_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


try:
    socket.getaddrinfo = _ipv4_getaddrinfo
except Exception:
    pass

# Cached client instances
_supabase_client: Optional[Client] = None
_supabase_service_client: Optional[Client] = None


def get_supabase_client(settings: Optional[Settings] = None) -> Client:
    """Retrieve or initialize the standard Supabase client (using anon key).

    Args:
        settings: Optional Settings instance. If not provided, cached settings are used.

    Returns:
        Client: An active Supabase client instance.

    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_ANON_KEY are not configured.
    """
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    cfg = settings or get_settings()

    if not cfg.SUPABASE_URL or not cfg.SUPABASE_ANON_KEY:
        error_msg = (
            "Cannot initialize Supabase client: SUPABASE_URL and "
            "SUPABASE_ANON_KEY must be set in environment variables."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug("Initializing standard Supabase client for URL: %s", cfg.SUPABASE_URL)
    _supabase_client = create_client(cfg.SUPABASE_URL, cfg.SUPABASE_ANON_KEY)
    return _supabase_client


def get_supabase_service_client(settings: Optional[Settings] = None) -> Client:
    """Retrieve or initialize the privileged Supabase service-role client.

    Used for background worker tasks or administrative operations that bypass RLS.

    Args:
        settings: Optional Settings instance. If not provided, cached settings are used.

    Returns:
        Client: An active Supabase admin client instance.

    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY are not configured.
    """
    global _supabase_service_client

    if _supabase_service_client is not None:
        return _supabase_service_client

    cfg = settings or get_settings()
    service_key = cfg.get_service_role_key()

    if not cfg.SUPABASE_URL or not service_key:
        error_msg = (
            "Cannot initialize Supabase service client: SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY must be set in environment variables."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug("Initializing Supabase service-role client for URL: %s", cfg.SUPABASE_URL)
    _supabase_service_client = create_client(cfg.SUPABASE_URL, service_key)
    return _supabase_service_client


def reset_clients() -> None:
    """Reset cached clients. Useful during testing or reconfiguration."""
    global _supabase_client, _supabase_service_client
    _supabase_client = None
    _supabase_service_client = None
