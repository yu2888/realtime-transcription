# Streaming Speech Recognition

This project uses NVIDIA's Nemotron streaming speech recognition model to transcribe audio in real time. It includes scripts for microphone input and system audio.

## Setup

Install [uv](https://docs.astral.sh/uv/), then install the project dependencies:

```powershell
uv sync
```

## Demo


https://github.com/user-attachments/assets/6fb955b0-f966-4b15-bd44-c76956c2610c



## Run
Transcribe audio playing through the default speaker:

```powershell
uv run stream_system_audio.py
```

Press `Ctrl+C` to stop transcription.
