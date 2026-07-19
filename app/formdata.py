"""Minimal multipart/form-data parser on the stdlib `email` package.

The approved dependency list for this service excludes `python-multipart`
(FastAPI's usual form parser), so /process reads the raw body and parses it
here. Handles exactly what the contract needs: one file field plus flat
string fields. Not a general-purpose parser.
"""

from email.parser import BytesParser
from email.policy import HTTP


def parse_multipart(content_type: str, body: bytes) -> dict[str, bytes | str]:
    """Return {field name: str for text fields, bytes for file fields}."""
    msg = BytesParser(policy=HTTP).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body
    )
    if not msg.is_multipart():
        raise ValueError("Body is not valid multipart/form-data")
    fields: dict[str, bytes | str] = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        if part.get_filename() is not None:
            fields[name] = payload
        else:
            fields[name] = payload.decode("utf-8", errors="replace").strip()
    return fields
