"""Typed HTTP client for Sheaf deployments.

Hits the same ``/<deployment>/predict``, ``/health``, ``/ready`` endpoints
that ``ModelServer`` (Ray Serve) and ``ModalServer`` expose.  Decodes
responses into the correct Pydantic class via the ``AnyResponse``
discriminated union, so callers get a typed response object back instead of
a raw dict.

Usage (sync)::

    from sheaf.client import SheafClient
    from sheaf.api.time_series import Frequency, TimeSeriesRequest

    with SheafClient(base_url="http://localhost:8000") as client:
        resp = client.predict(
            "chronos",
            TimeSeriesRequest(
                model_name="chronos",
                history=[1.0, 2.0, 3.0],
                horizon=3,
                frequency=Frequency.HOURLY,
            ),
        )
    # resp is a TimeSeriesResponse
    print(resp.mean)

Usage (async)::

    from sheaf.client import AsyncSheafClient

    async with AsyncSheafClient(base_url="http://localhost:8000") as client:
        resp = await client.predict("chronos", req)

Errors:
  - :class:`ValidationError`  — server returned 422 (request shape didn't
    match the deployment's expected ``model_type`` or had a malformed field).
  - :class:`ServerError`      — server returned 5xx (backend exception).
  - :class:`SheafError`       — base class; also raised for unexpected
    status codes.

The client uses ``httpx`` under the hood; a custom ``transport`` can be
injected for tests or for hitting an in-process FastAPI app.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx
from pydantic import TypeAdapter

from sheaf.api.base import BaseRequest, BaseResponse
from sheaf.api.union import AnyResponse

__all__ = [
    "AsyncSheafClient",
    "ClientError",
    "ServerError",
    "SheafClient",
    "SheafError",
    "ValidationError",
]


# Cached once at import time — the TypeAdapter build is non-trivial and we
# call it on every successful predict() decode.
_ANYRESPONSE_ADAPTER: TypeAdapter[BaseResponse] = TypeAdapter(AnyResponse)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SheafError(Exception):
    """Base class for all sheaf-client errors.

    Attributes:
        status_code: HTTP status code returned by the server, or ``None``
            for transport-level failures.
        detail:      Server-supplied error detail (the FastAPI ``detail``
            field on the JSON response body), or the transport error message.
    """

    def __init__(self, detail: str, *, status_code: int | None = None) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class ValidationError(SheafError):
    """Raised when the server returns 422 (Unprocessable Entity).

    Common causes: ``model_type`` mismatch (e.g. sending a TabularRequest to
    a TIME_SERIES deployment), unknown LoRA adapter name, malformed payload.
    """


class ServerError(SheafError):
    """Raised when the server returns a 5xx status code.

    The backend raised an exception during inference; the server caught it
    and returned a structured 500 with the exception type + message in
    ``detail``.
    """


class ClientError(SheafError):
    """Raised for transport-level failures: connection refused, timeout,
    JSON decode failure on a 200 response, etc.

    Distinct from :class:`SheafError` only by intent — the error originated
    on the client side or in transit, not from a server-supplied response.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_detail(resp: httpx.Response) -> str:
    """Pull the ``detail`` field out of a FastAPI error response.

    FastAPI's HTTPException returns ``{"detail": "..."}``.  Falls back to
    the raw response text if the body isn't JSON or doesn't have ``detail``.
    """
    try:
        body = resp.json()
    except ValueError:
        return resp.text
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])
    return resp.text


def _raise_for_status(resp: httpx.Response) -> None:
    """Map an httpx response to the right SheafError subclass and raise.

    No-op if the response is 2xx.
    """
    if resp.is_success:
        return
    detail = _extract_detail(resp)
    if resp.status_code == 422:
        raise ValidationError(detail, status_code=422)
    if 500 <= resp.status_code < 600:
        raise ServerError(detail, status_code=resp.status_code)
    raise SheafError(detail, status_code=resp.status_code)


def _decode_predict_response(resp: httpx.Response) -> BaseResponse:
    """Decode a successful /predict response into the correct typed class."""
    _raise_for_status(resp)
    try:
        body = resp.json()
    except ValueError as e:
        raise ClientError(
            f"Server returned {resp.status_code} but body wasn't valid JSON: "
            f"{resp.text[:200]!r}"
        ) from e
    try:
        return _ANYRESPONSE_ADAPTER.validate_python(body)
    except Exception as e:
        raise ClientError(
            f"Could not decode response into AnyResponse: {type(e).__name__}: {e}"
        ) from e


def _request_payload(request: BaseRequest) -> dict[str, Any]:
    """Serialise a request to the JSON-compatible dict shape the server expects."""
    return request.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


class SheafClient:
    """Synchronous HTTP client for sheaf deployments.

    Args:
        base_url:  Root of the sheaf server, e.g. ``"http://localhost:8000"``
            or a Modal app URL.  Per-deployment paths (``/<name>/predict``,
            etc.) are appended automatically.
        timeout:   Per-request timeout in seconds.  Default 30.
        headers:   Optional headers to send with every request (auth, etc.).
        transport: Optional ``httpx.BaseTransport`` override.  Set this to
            ``httpx.MockTransport(...)`` in tests, or to
            ``httpx.WSGITransport(app=fastapi_app)`` to drive an in-process
            FastAPI app without spinning up a real HTTP server.

    Use as a context manager (recommended) so the underlying connection
    pool is closed cleanly::

        with SheafClient(base_url="...") as client:
            resp = client.predict("my-model", req)
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "base_url": base_url,
            "timeout": timeout,
            "headers": headers or {},
        }
        if transport is not None:
            kwargs["transport"] = transport
        self._http = httpx.Client(**kwargs)

    # Context manager protocol --------------------------------------------

    def __enter__(self) -> SheafClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying httpx connection pool."""
        self._http.close()

    # Endpoints -----------------------------------------------------------

    def predict(self, deployment: str, request: BaseRequest) -> BaseResponse:
        """POST a request to ``/<deployment>/predict`` and decode the response.

        Args:
            deployment: Name of the target deployment (matches ``ModelSpec.name``).
            request:    Any subclass of ``BaseRequest`` — typically the typed
                request class for the deployment's model type.

        Returns:
            The decoded response, as the correct Pydantic class for the
            request's model type (e.g. ``TimeSeriesResponse`` for a
            ``TimeSeriesRequest``).

        Raises:
            ValidationError: 422 from server (model_type mismatch, bad payload,
                unknown adapter, …).
            ServerError:     5xx from server (backend raised).
            SheafError:      Other non-2xx status codes.
            ClientError:     Transport / JSON decode failures.
        """
        try:
            resp = self._http.post(
                f"/{deployment}/predict", json=_request_payload(request)
            )
        except httpx.HTTPError as e:
            raise ClientError(f"{type(e).__name__}: {e}") from e
        return _decode_predict_response(resp)

    def health(self, deployment: str) -> dict[str, Any]:
        """GET ``/<deployment>/health``.  Returns the parsed JSON body."""
        try:
            resp = self._http.get(f"/{deployment}/health")
        except httpx.HTTPError as e:
            raise ClientError(f"{type(e).__name__}: {e}") from e
        _raise_for_status(resp)
        return cast(dict[str, Any], resp.json())

    def ready(self, deployment: str) -> dict[str, Any]:
        """GET ``/<deployment>/ready``.  Returns the parsed JSON body."""
        try:
            resp = self._http.get(f"/{deployment}/ready")
        except httpx.HTTPError as e:
            raise ClientError(f"{type(e).__name__}: {e}") from e
        _raise_for_status(resp)
        return cast(dict[str, Any], resp.json())


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


class AsyncSheafClient:
    """Async HTTP client for sheaf deployments.

    Mirror of :class:`SheafClient` with ``async`` methods on top of
    ``httpx.AsyncClient``.  Use as an async context manager so the
    connection pool closes cleanly::

        async with AsyncSheafClient(base_url="...") as client:
            resp = await client.predict("my-model", req)

    See :class:`SheafClient` for argument and error semantics.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "base_url": base_url,
            "timeout": timeout,
            "headers": headers or {},
        }
        if transport is not None:
            kwargs["transport"] = transport
        self._http = httpx.AsyncClient(**kwargs)

    # Context manager protocol --------------------------------------------

    async def __aenter__(self) -> AsyncSheafClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying httpx async connection pool."""
        await self._http.aclose()

    # Endpoints -----------------------------------------------------------

    async def predict(self, deployment: str, request: BaseRequest) -> BaseResponse:
        """POST a request to ``/<deployment>/predict`` and decode the response."""
        try:
            resp = await self._http.post(
                f"/{deployment}/predict", json=_request_payload(request)
            )
        except httpx.HTTPError as e:
            raise ClientError(f"{type(e).__name__}: {e}") from e
        return _decode_predict_response(resp)

    async def health(self, deployment: str) -> dict[str, Any]:
        """GET ``/<deployment>/health``."""
        try:
            resp = await self._http.get(f"/{deployment}/health")
        except httpx.HTTPError as e:
            raise ClientError(f"{type(e).__name__}: {e}") from e
        _raise_for_status(resp)
        return cast(dict[str, Any], resp.json())

    async def ready(self, deployment: str) -> dict[str, Any]:
        """GET ``/<deployment>/ready``."""
        try:
            resp = await self._http.get(f"/{deployment}/ready")
        except httpx.HTTPError as e:
            raise ClientError(f"{type(e).__name__}: {e}") from e
        _raise_for_status(resp)
        return cast(dict[str, Any], resp.json())

    async def stream(
        self, deployment: str, request: BaseRequest
    ) -> AsyncIterator[dict[str, Any]]:
        """POST to ``/<deployment>/stream`` and yield SSE events as dicts.

        Each event is a parsed JSON object from a ``data: {...}\\n\\n`` line.
        Two event shapes the server may emit:

        - ``{"type": "progress", "step": N, "total_steps": N, "done": False}``
        - ``{"type": "result", "done": True, ...response_fields}``

        Backend exceptions raised mid-stream become an in-band error event
        (``{"type": "error", "error": "..."}``) — the HTTP status is still 200
        in that case, so callers must check ``event["type"]`` to distinguish.

        Pre-stream HTTP errors (422 for model_type mismatch / unknown adapter,
        404 for missing deployment, etc.) raise the usual :class:`SheafError`
        subclass before any events are yielded.

        Args:
            deployment: Name of the target deployment.
            request:    A request whose backend supports ``stream_predict``
                (FLUX is the canonical example).

        Yields:
            Event dicts in arrival order.
        """
        try:
            stream_ctx = self._http.stream(
                "POST",
                f"/{deployment}/stream",
                json=_request_payload(request),
            )
        except httpx.HTTPError as e:
            raise ClientError(f"{type(e).__name__}: {e}") from e

        async with stream_ctx as resp:
            if not resp.is_success:
                # Need to read the body before letting the context exit so
                # _extract_detail can pull the FastAPI ``detail`` field.
                await resp.aread()
                _raise_for_status(resp)
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :]
                try:
                    yield cast(dict[str, Any], json.loads(payload))
                except json.JSONDecodeError as e:
                    raise ClientError(
                        f"Malformed SSE payload: {payload[:200]!r}"
                    ) from e
