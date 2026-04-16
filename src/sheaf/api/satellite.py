"""API contract for Earth observation / satellite imagery foundation models.

Supports Prithvi (IBM/NASA), Clay, SatMAE, and similar architectures.

Encoding convention
-------------------
``pixels_b64`` is a base64-encoded little-endian float32 byte string.

  Shape: (n_time, n_bands, height, width)

For single-time input set n_time=1.  Values are typically surface
reflectance in [0, 1] (after dividing sensor DN by 10 000 for
Landsat/Sentinel-2) or raw DN if normalize=False.

Encode:  base64.b64encode(arr.astype(np.float32).tobytes()).decode()
Decode:  np.frombuffer(base64.b64decode(s), dtype=np.float32)
         .reshape(n_time, n_bands, height, width)

Band names (examples)
---------------------
HLS (Harmonized Landsat-Sentinel) 6-band subset used by Prithvi:
  ["blue", "green", "red", "nir08", "swir16", "swir22"]

Sentinel-2 L2A 10-band (used by Clay):
  ["coastal", "blue", "green", "red", "rededge1", "rededge2",
   "rededge3", "nir08", "nir09", "swir16", "swir22"]

Wavelengths (μm) for Clay (examples, Sentinel-2):
  [0.443, 0.490, 0.560, 0.665, 0.704, 0.740, 0.783, 0.842, 0.865,
   1.610, 2.190]
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class SatelliteRequest(BaseRequest):
    """Request contract for Earth observation foundation models.

    Args:
        pixels_b64:    Base64 float32 pixel array of shape
                       (n_time, n_bands, height, width).
        n_time:        Number of time steps in the input stack.
        n_bands:       Number of spectral bands.
        height:        Spatial height in pixels.
        width:         Spatial width in pixels.
        band_names:    Human-readable band labels, length n_bands.
        wavelengths:   Center wavelengths in micrometers, length n_bands.
                       Required by Clay; ignored by Prithvi.
        gsd:           Ground sample distance in metres (default 10 m for
                       Sentinel-2).
        lat:           Centre latitude in degrees (optional metadata).
        lon:           Centre longitude in degrees (optional metadata).
        timestamps:    ISO-8601 timestamp per time step (optional metadata).
        pooling:       "mean" pools all output tokens; "cls" uses the first
                       token (CLS or register token).
        normalize:     Apply the model's per-band mean/std normalization
                       using statistics stored in the image processor config.
                       Disable if you have already normalized your data.
    """

    model_type: Literal[ModelType.GEOSPATIAL] = ModelType.GEOSPATIAL

    # Pixel data
    pixels_b64: str
    n_time: int = 1
    n_bands: int
    height: int
    width: int

    # Band metadata
    band_names: list[str]
    wavelengths: list[float] | None = None  # μm — used by Clay

    # Geospatial metadata
    gsd: float = 10.0  # metres
    lat: float | None = None
    lon: float | None = None
    timestamps: list[str] | None = None  # ISO-8601, one per time step

    # Inference options
    pooling: Literal["mean", "cls"] = "mean"
    normalize: bool = True

    @model_validator(mode="after")
    def validate_metadata(self) -> SatelliteRequest:
        if len(self.band_names) != self.n_bands:
            raise ValueError(
                f"band_names has {len(self.band_names)} entries "
                f"but n_bands={self.n_bands}."
            )
        if self.wavelengths is not None and len(self.wavelengths) != self.n_bands:
            raise ValueError(
                f"wavelengths has {len(self.wavelengths)} entries "
                f"but n_bands={self.n_bands}."
            )
        if self.timestamps is not None and len(self.timestamps) != self.n_time:
            raise ValueError(
                f"timestamps has {len(self.timestamps)} entries "
                f"but n_time={self.n_time}."
            )
        return self


class SatelliteResponse(BaseResponse):
    """Response contract for Earth observation foundation models.

    embedding — scene-level float vector of length ``dim``.
                For multi-temporal input the tokens from all time steps
                are pooled together into a single vector.
    dim       — embedding dimensionality.
    n_time    — number of time steps from the request (passed through for
                bookkeeping).
    """

    model_type: Literal[ModelType.GEOSPATIAL] = ModelType.GEOSPATIAL

    embedding: list[float]
    dim: int
    n_time: int
