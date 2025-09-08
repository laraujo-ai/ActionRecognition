def is_rtsp_stream(media_link: str) -> bool:
    """Check if media link is an RTSP stream URL.

    Args:
        media_link: URL or file path to check

    Returns:
        bool: True if media_link is an RTSP stream, False otherwise
    """
    return isinstance(media_link, str) and media_link.startswith("rtsp://")
