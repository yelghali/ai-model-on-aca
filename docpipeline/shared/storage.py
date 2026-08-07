"""Blob helpers.

THE RULE: orchestration history stores REFERENCES, never bytes.

Durable Functions serialises every activity input and output into Azure Storage
and replays that history on each checkpoint. Passing page images through it -
even base64 - inflates the history table, slows every replay, and eventually
trips the payload limits. So activities exchange blob paths (short strings) and
fetch/store the bytes themselves.

Layout:
  uploads/<file>.pdf                          <- user drops files here (trigger)
  work/<run_id>/<file>/pages/page-0001.png    <- rasterised pages
  work/<run_id>/<file>/figures/p0007-f00.png  <- figures extracted from pages
  results/<file>.json                         <- one merged result per file
"""
from __future__ import annotations

import os

from azure.storage.blob import BlobServiceClient

UPLOAD_CONTAINER = os.getenv("UPLOAD_CONTAINER", "uploads")
WORK_CONTAINER = os.getenv("WORK_CONTAINER", "work")
RESULT_CONTAINER = os.getenv("RESULT_CONTAINER", "results")

_client: BlobServiceClient | None = None


def _service() -> BlobServiceClient:
    global _client
    if _client is None:
        conn = os.getenv("AzureWebJobsStorage")
        if not conn:
            raise RuntimeError("AzureWebJobsStorage is not set")
        _client = BlobServiceClient.from_connection_string(conn)
    return _client


def ensure_containers() -> None:
    for name in (UPLOAD_CONTAINER, WORK_CONTAINER, RESULT_CONTAINER):
        try:
            _service().create_container(name)
        except Exception:
            pass  # already exists


def read_blob(container: str, path: str) -> bytes:
    return _service().get_blob_client(container, path).download_blob().readall()


def write_blob(container: str, path: str, data: bytes, content_type: str | None = None) -> str:
    from azure.storage.blob import ContentSettings

    settings = ContentSettings(content_type=content_type) if content_type else None
    _service().get_blob_client(container, path).upload_blob(
        data, overwrite=True, content_settings=settings
    )
    return path


def blob_exists(container: str, path: str) -> bool:
    return _service().get_blob_client(container, path).exists()
