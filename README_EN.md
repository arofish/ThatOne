# ThatOne

**Remember it. Find that photo.**

[简体中文](README.md) | [English](README_EN.md)

ThatOne is a local semantic photo search application for macOS. Simply describe a scene you remember to find the corresponding photos in your personal library.

ThatOne uses Chinese-CLIP to generate image and text embeddings and performs similarity search locally. Original photographs stay in their existing locations, while the application maintains indexes and previews separately.

## Features

- **Natural-language search**: Describe people, landscapes, colors, and scenes to retrieve results ranked by semantic similarity. Chinese remains the primary validated language, while English queries have also performed well in several tests.
- **Local folder import**: Select a photo directory and scan its files. Currently supports JPG, JPEG, and PNG only.
- **Multiple folders**: Import separate photo directories and search across them in one place.
- **Native macOS interface**
- **Offline inference**
- **Separate storage for user data**

## System Support

Development and validation currently focus primarily on Apple Silicon Macs.

## Usage

The intended workflow for the installable release is:

1. Launch ThatOne.
2. Click **Import Folder** and select a local directory containing your photographs.
3. Wait for the initial image encoding and index construction to finish.
4. Enter a scene description in the search field and browse results ranked by similarity.
5. Click **Show in Finder** to locate the original photo.
6. After your photo library changes, use **Update Index** to synchronize added, modified, and deleted files.

Initial indexing time depends on the number of images, their resolution, and your device's performance. Subsequent scans reuse indexes for unchanged images.

## Data and Privacy

ThatOne follows a local-first approach.

- Original photos are not uploaded for inference.
- Original files are not moved, overwritten, or modified.
- Image embeddings, databases, and thumbnails are stored locally.
- Selected folders may be located on your Mac's internal drive or an accessible external drive.
- A temporarily disconnected external drive should not cause its entire directory to be treated as deleted.
- No cloud account or automatic photo synchronization is provided.

Model weights and user library data are kept separate. Updating the application should not overwrite an existing photo index.

## Versions and Releases

The v1.0.0 release aims to include a complete local model, photo-folder import, incremental indexing, and a standalone macOS application.

Installable builds, checksums, verified system support, and known limitations will be published under [GitHub Releases](https://github.com/arofish/ThatOne/releases).

## Development and Source Code

The repository is organized primarily as follows:

```text
photoindex/    Local indexing, vector search, and HTTP service
macos/         macOS SwiftUI client
scripts/       Model management, build, and verification utilities
tests/         Python automated tests
```

Python model environments, personal photographs, user databases, and generated model files are excluded from source control.

Building locally requires compatible Python and Xcode environments and fully verified model files.

## Model and Acknowledgments

ThatOne uses the pretrained [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) model `OFA-Sys/chinese-clip-vit-base-patch16`.

Thanks to the Chinese-CLIP project and open-source projects including PyTorch, Transformers, FastAPI, and NumPy.

Chinese-CLIP and its associated resources remain the property of their respective rights holders. Whether model weights are included in a particular release asset depends on the authorization review and release notes for that version.