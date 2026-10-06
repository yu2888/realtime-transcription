# Streaming Speech Recognition

This project uses NVIDIA's Nemotron streaming speech recognition model to transcribe audio in real time. It includes scripts for microphone input and system audio.

## Setup

Install Python 3.14 and [uv](https://docs.astral.sh/uv/), then install the project dependencies:

```powershell
uv sync
```

## Run

Transcribe microphone input:

```powershell
uv run python stream.py
```

Transcribe audio playing through the default speaker:

```powershell
uv run python stream_system_audio.py
```

Press `Ctrl+C` to stop transcription.