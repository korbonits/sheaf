# API contracts

Typed request / response schemas — one module per model type. Validation
runs at the request boundary; backends receive validated objects.

## Base + discriminated unions

::: sheaf.api.base

::: sheaf.api.union

## Time series

::: sheaf.api.time_series

## Tabular

::: sheaf.api.tabular

## Audio (ASR + TTS + audio generation)

::: sheaf.api.audio

::: sheaf.api.audio_generation

## Vision

::: sheaf.api.embedding

::: sheaf.api.segmentation

::: sheaf.api.depth

::: sheaf.api.detection

::: sheaf.api.pose

::: sheaf.api.optical_flow

::: sheaf.api.video

## Diffusion / multimodal generation

::: sheaf.api.diffusion

::: sheaf.api.multimodal_generation

## Cross-modal embedding

::: sheaf.api.multimodal_embedding

## Molecular / genomics / materials

::: sheaf.api.molecular

::: sheaf.api.genomic

::: sheaf.api.small_molecule

::: sheaf.api.materials

## Earth / weather

::: sheaf.api.weather

::: sheaf.api.satellite

## LiDAR / point cloud

::: sheaf.api.point_cloud
