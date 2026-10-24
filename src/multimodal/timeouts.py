"""Timeout context supported on all declared Python versions."""
try:
    from asyncio import timeout
except ImportError:  # Python 3.10
    from async_timeout import timeout
