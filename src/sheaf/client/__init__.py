"""Typed HTTP client for Sheaf deployments.

Hits the same ``/<deployment>/predict``, ``/health``, ``/ready``, and
``/stream`` endpoints that ``ModelServer`` (Ray Serve) and ``ModalServer``
expose.  Decodes responses into the correct Pydantic class via the
``AnyResponse`` discriminated union, so callers get a typed response object
back instead of a raw dict.

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

Retry config (opt-in, exponential backoff)::

    from sheaf.client import RetryConfig, SheafClient

    retry = RetryConfig(
        max_attempts=3,
        backoff_factor=0.5,                       # 0.5s, 1.0s, 2.0s, ...
        retry_on_status=(502, 503, 504),
        retry_on_connection_errors=True,
    )
    client = SheafClient(base_url="...", retry=retry)

Errors:
  - :class:`ValidationError`  — server returned 422 (request shape didn't
    match the deployment's expected ``model_type`` or had a malformed field).
  - :class:`ServerError`      — server returned 5xx (backend exception).
  - :class:`SheafError`       — base class; also raised for unexpected
    status codes.
  - :class:`ClientError`      — transport / decode failures.

All raised errors carry ``request_id`` (the UUID the client minted on the
``BaseRequest``) so callers can correlate a failed call with server-side
log lines and metrics without holding onto the original request.

The client uses ``httpx`` under the hood; a custom ``transport`` can be
injected for tests or for hitting an in-process FastAPI app.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

import httpx
from pydantic import TypeAdapter

from sheaf.api.base import BaseRequest, BaseResponse
from sheaf.api.union import AnyResponse

__all__ = [
    "AsyncSheafClient",
    "ClientError",
    "RetryConfig",
    "ServerError",
    "SheafClient",
    "SheafError",
    "ValidationError",
]


# Cached once at import time — the TypeAdapter build is non-trivial and we
# call it on every successful predict() decode.
_ANYRESPONSE_ADAPTER: TypeAdapter[BaseResponse] = TypeAdapter(AnyResponse)


# ---------------------------------------------------------------------------
# Retry config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryConfig:
    """Retry policy for client-side requests.

    The default (``max_attempts=1``) is no retry — same behavior as a client
    constructed without a retry config.  Opt in by passing a ``RetryConfig``
    with ``max_attempts > 1``.

    Attributes:
        max_attempts: Total number of attempts including the first.  ``1``
            disables retrying entirely.  Must be at least 1.
        backoff_factor: Base for exponential backoff between attempts, in
            seconds.  Sleep before attempt ``n`` (``n >= 1``) is
            ``backoff_factor * 2**(n - 1)`` — i.e. for ``backoff_factor=0.5``
            the gaps are 0.5s, 1.0s, 2.0s, …
        retry_on_status: HTTP status codes that should be retried.  Default
            is the standard transient-failure set ``(502, 503, 504)``.  4xx
            codes are never sensible to retry — a 422 will fail again.
        retry_on_connection_errors: When ``True`` (default), retry on any
            ``httpx.HTTPError`` raised from transport (connection refused,
            read timeout, etc.).
    """

    max_attempts: int = 1
    backoff_factor: float = 0.5
    retry_on_status: tuple[int, ...] = (502, 503, 504)
    retry_on_connection_errors: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.backoff_factor < 0:
            raise ValueError("backoff_factor must be >= 0")

    def sleep_seconds(self, attempt_index: int) -> float:
        """Return the backoff sleep before the (attempt_index)th attempt.

        ``attempt_index=0`` is the first attempt — never sleeps.  For 1, 2, 3,
        … the gap is ``backoff_factor * 2**(attempt_index - 1)``.
        """
        if attempt_index <= 0:
            return 0.0
        return self.backoff_factor * (2 ** (attempt_index - 1))


# Default: no retry.  Shared so we don't allocate a new instance per client.
_NO_RETRY = RetryConfig()


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
        request_id:  UUID of the request that triggered this error, when
            known.  Lifted from the calling ``BaseRequest`` so callers can
            log-correlate a failure without holding onto the request object.
    """

    def __init__(
        self,
        detail: str,
        *,
        status_code: int | None = None,
        request_id: UUID | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.request_id = request_id


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


def _raise_for_status(resp: httpx.Response, *, request_id: UUID | None = None) -> None:
    """Map an httpx response to the right SheafError subclass and raise.

    No-op if the response is 2xx.
    """
    if resp.is_success:
        return
    detail = _extract_detail(resp)
    if resp.status_code == 422:
        raise ValidationError(detail, status_code=422, request_id=request_id)
    if 500 <= resp.status_code < 600:
        raise ServerError(detail, status_code=resp.status_code, request_id=request_id)
    raise SheafError(detail, status_code=resp.status_code, request_id=request_id)


def _decode_predict_response(
    resp: httpx.Response, *, request_id: UUID | None = None
) -> BaseResponse:
    """Decode a successful /predict response into the correct typed class."""
    _raise_for_status(resp, request_id=request_id)
    try:
        body = resp.json()
    except ValueError as e:
        raise ClientError(
            f"Server returned {resp.status_code} but body wasn't valid JSON: "
            f"{resp.text[:200]!r}",
            request_id=request_id,
        ) from e
    try:
        return _ANYRESPONSE_ADAPTER.validate_python(body)
    except Exception as e:
        raise ClientError(
            f"Could not decode response into AnyResponse: {type(e).__name__}: {e}",
            request_id=request_id,
        ) from e


def _request_payload(request: BaseRequest) -> dict[str, Any]:
    """Serialise a request to the JSON-compatible dict shape the server expects."""
    return request.model_dump(mode="json")


def _retry_sync(
    do: Callable[[], httpx.Response],
    retry: RetryConfig,
    sleep: Callable[[float], None] | None = None,
) -> httpx.Response:
    """Run *do* with retry/backoff per *retry*.  Returns the final response.

    The final response may itself be a non-2xx — it's the caller's job to
    map that to the appropriate SheafError.  Connection errors are
    re-raised after exhausting attempts.

    ``sleep`` is resolved to ``time.sleep`` at call time (not as a default
    argument) so tests can monkeypatch ``time.sleep`` to assert backoff
    behavior without real waits.
    """
    if sleep is None:
        sleep = time.sleep
    last_exc: httpx.HTTPError | None = None
    for attempt in range(retry.max_attempts):
        wait = retry.sleep_seconds(attempt)
        if wait > 0:
            sleep(wait)
        try:
            resp = do()
        except httpx.HTTPError as e:
            if not retry.retry_on_connection_errors:
                raise
            last_exc = e
            continue
        if (
            resp.status_code in retry.retry_on_status
            and attempt < retry.max_attempts - 1
        ):
            continue
        return resp
    # All attempts exhausted; only reachable when every attempt raised.
    assert last_exc is not None
    raise last_exc


async def _retry_async(
    do: Callable[[], Awaitable[httpx.Response]],
    retry: RetryConfig,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> httpx.Response:
    """Async mirror of :func:`_retry_sync`."""
    if sleep is None:
        sleep = asyncio.sleep
    last_exc: httpx.HTTPError | None = None
    for attempt in range(retry.max_attempts):
        wait = retry.sleep_seconds(attempt)
        if wait > 0:
            await sleep(wait)
        try:
            resp = await do()
        except httpx.HTTPError as e:
            if not retry.retry_on_connection_errors:
                raise
            last_exc = e
            continue
        if (
            resp.status_code in retry.retry_on_status
            and attempt < retry.max_attempts - 1
        ):
            continue
        return resp
    assert last_exc is not None
    raise last_exc


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
        retry:     Optional :class:`RetryConfig`.  Default is no retry.
        transport: Optional ``httpx.BaseTransport`` override.  Set this to
            ``httpx.MockTransport(...)`` in tests.

    Use as a context manager so the underlying connection pool is closed
    cleanly::

        with SheafClient(base_url="...") as client:
            resp = client.predict("my-model", req)
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        retry: RetryConfig | None = None,
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
        self._retry = retry or _NO_RETRY

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
            request's model type.

        Raises:
            ValidationError: 422 from server.
            ServerError:     5xx from server.
            SheafError:      Other non-2xx status codes.
            ClientError:     Transport / JSON decode failures.

            All carry ``e.request_id`` set to ``request.request_id``.
        """
        rid = request.request_id
        payload = _request_payload(request)
        try:
            resp = _retry_sync(
                lambda: self._http.post(f"/{deployment}/predict", json=payload),
                self._retry,
            )
        except httpx.HTTPError as e:
            raise ClientError(f"{type(e).__name__}: {e}", request_id=rid) from e
        return _decode_predict_response(resp, request_id=rid)

    def health(self, deployment: str) -> dict[str, Any]:
        """GET ``/<deployment>/health``.  Returns the parsed JSON body."""
        try:
            resp = _retry_sync(
                lambda: self._http.get(f"/{deployment}/health"), self._retry
            )
        except httpx.HTTPError as e:
            raise ClientError(f"{type(e).__name__}: {e}") from e
        _raise_for_status(resp)
        return cast(dict[str, Any], resp.json())

    def ready(self, deployment: str) -> dict[str, Any]:
        """GET ``/<deployment>/ready``.  Returns the parsed JSON body."""
        try:
            resp = _retry_sync(
                lambda: self._http.get(f"/{deployment}/ready"), self._retry
            )
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

    See :class:`SheafClient` for argument and error semantics.  Streaming
    (``client.stream(...)``) does NOT retry — streams are stateful and
    re-running them mid-flight would yield interleaved progress events.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        retry: RetryConfig | None = None,
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
        self._retry = retry or _NO_RETRY

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
        rid = request.request_id
        payload = _request_payload(request)
        try:
            resp = await _retry_async(
                lambda: self._http.post(f"/{deployment}/predict", json=payload),
                self._retry,
            )
        except httpx.HTTPError as e:
            raise ClientError(f"{type(e).__name__}: {e}", request_id=rid) from e
        return _decode_predict_response(resp, request_id=rid)

    async def health(self, deployment: str) -> dict[str, Any]:
        """GET ``/<deployment>/health``."""
        try:
            resp = await _retry_async(
                lambda: self._http.get(f"/{deployment}/health"), self._retry
            )
        except httpx.HTTPError as e:
            raise ClientError(f"{type(e).__name__}: {e}") from e
        _raise_for_status(resp)
        return cast(dict[str, Any], resp.json())

    async def ready(self, deployment: str) -> dict[str, Any]:
        """GET ``/<deployment>/ready``."""
        try:
            resp = await _retry_async(
                lambda: self._http.get(f"/{deployment}/ready"), self._retry
            )
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

        Pre-stream HTTP errors (422, 5xx, 404, …) raise the usual
        :class:`SheafError` subclass before any events are yielded.

        Streaming bypasses :class:`RetryConfig` — re-running a partial stream
        would yield interleaved progress events from two backend invocations.
        Configure timeouts on the client itself if you need a stream-level
        upper bound.

        Args:
            deployment: Name of the target deployment.
            request:    A request whose backend supports ``stream_predict``
                (FLUX is the canonical example).

        Yields:
            Event dicts in arrival order.
        """
        rid = request.request_id
        try:
            stream_ctx = self._http.stream(
                "POST",
                f"/{deployment}/stream",
                json=_request_payload(request),
            )
        except httpx.HTTPError as e:
            raise ClientError(f"{type(e).__name__}: {e}", request_id=rid) from e

        async with stream_ctx as resp:
            if not resp.is_success:
                # Need to read the body before letting the context exit so
                # _extract_detail can pull the FastAPI ``detail`` field.
                await resp.aread()
                _raise_for_status(resp, request_id=rid)
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :]
                try:
                    yield cast(dict[str, Any], json.loads(payload))
                except json.JSONDecodeError as e:
                    raise ClientError(
                        f"Malformed SSE payload: {payload[:200]!r}",
                        request_id=rid,
                    ) from e
