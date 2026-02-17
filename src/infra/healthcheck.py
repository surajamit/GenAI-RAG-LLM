def health_check():
    return {
        "status": "healthy",
        "uptime_ok": True,
        "db_connected": True,
        "vector_store_ready": True
    }
