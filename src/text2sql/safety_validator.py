FORBIDDEN = {"DROP", "DELETE", "TRUNCATE"}


def is_safe_sql(query: str) -> bool:
    """
    Safety validation.
    """
    upper = query.upper()
    return not any(cmd in upper for cmd in FORBIDDEN)
