# ThatOne

**Find that one photo, just by describing it.**

[简体中文](README.md) | [English](README_EN.md)

ThatOne is a local-first semantic photo search application for macOS. Instead of browsing through folders or manually tagging every image, describe a scene you remember and search your personal photo library using natural language.

Powered by Chinese-CLIP, ThatOne maps images and text into a shared embedding space and performs similarity search entirely on your Mac. Original photographs stay where they are; indexes and previews are managed separately.

## Features

- **Natural-language search** — Find photos by describing scenes, subjects, colors, or visual details. Chinese is the primary validated language; English queries may also produce useful matches.
- **Folder-based import** — Recursively discover JPG, JPEG, and PNG images in selected local directories.
- **Incremental indexing** — Detect new and modified images using content hashes, while reusing embeddings for unchanged files.
- **Multiple photo sources** — Search across independently imported folders.
- **Native macOS interface** — Browse ranked results with thumbnails, similarity scores, and service status.
- **Finder integration** — Reveal the original photograph directly from a search result.
- **Offline inference** — Run image and text encoding locally without relying on cloud inference.
- **Separate user data** — Store embeddings, metadata, and previews independently from original image files.

## Architecture

| Component | Technology |
|---|---|
| Desktop application | Swift, SwiftUI, AppKit |
| Local backend | Python, FastAPI, Uvicorn |
| Vision-language model | Chinese-CLIP ViT-B/16 |
| Model runtime | PyTorch, Transformers |
| Metadata and indexing | SQLite |
| Vector search | NumPy, normalized 512-dimensional embeddings |
| Local communication | `127.0.0.1`, Bearer Token |

ThatOne separates the native user interface from the inference runtime. SwiftUI handles user interaction and macOS integration, while the local Python service manages scanning, embedding generation, index updates, and search.

## Requirements

ThatOne is currently developed and validated primarily on Apple Silicon Macs.

Minimum supported macOS versions, supported architectures, and distribution-signing status will be specified for each published release.

## How It Works

The intended workflow for an installable release is:

1. Launch ThatOne.
2. Select **Import Folder** and choose a directory containing your photographs.
3. Let ThatOne build the initial image index.
4. Enter a natural-language description to retrieve ranked matches.
5. Select **Show in Finder** to locate the original file.
6. Use **Update Index** after making changes to your photo folders.

Initial indexing time depends on image count, resolution, and hardware performance. Subsequent scans reuse existing embeddings for unchanged images.

## Privacy and Data

ThatOne is designed around local-first processing.

- Original photographs are not uploaded for inference.
- Source images are not moved, overwritten, or edited.
- Embeddings, databases, and previews remain on the local device.
- Photo folders may reside on internal or accessible external drives.
- Temporarily disconnected drives should not cause an entire source library to be treated as deleted.
- No cloud account or automatic photo synchronization is required.

Model resources and user-generated library data are kept separate. Updating the application should not overwrite an existing photo index.

## Current Scope

Supported formats: JPG, JPEG, PNG.

RAW, HEIC, direct Apple Photos integration, image editing, and calibrated no-match rejection are not currently included.

Search results are ranked by embedding similarity. Similarity scores are not probabilities that every condition in a query has been satisfied, and the application does not guarantee a relevant match for every query.

## Versions and Releases

The v1.0.0 release target includes a complete local model, folder import, incremental indexing, and a standalone macOS application.

Installable builds, checksums, supported platforms, and known limitations will be published under [GitHub Releases](https://github.com/arofish/ThatOne/releases). Development builds should not be assumed to be signed, notarized, or validated across different Macs.

## Source Structure

```text
photoindex/    Indexing, retrieval, and local HTTP services
macos/         Native macOS SwiftUI application
scripts/       Model management, build, and verification tools
tests/         Python automated tests
```

Python environments, personal photographs, user databases, and generated model files are not stored in Git source history.

Building from source requires a compatible Python and Xcode environment, along with verified model files.

## Model and Acknowledgments

ThatOne uses the [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) pretrained model `OFA-Sys/chinese-clip-vit-base-patch16`.

Thanks to the Chinese-CLIP authors and the open-source projects behind PyTorch, Transformers, FastAPI, and NumPy.

Chinese-CLIP model resources remain subject to their respective rights and distribution terms. Whether pretrained weights are included in a particular release asset depends on the authorization review and release documentation for that version.

## Project Status

ThatOne is undergoing v1.0.0 release validation. Officially published builds and their support status are documented on GitHub Releases.