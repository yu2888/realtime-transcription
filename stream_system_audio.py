from queue import Empty, Full, Queue
from threading import Event, Thread
import warnings

import numpy as np
import soundcard as sc
from soundcard.mediafoundation import SoundcardRuntimeWarning
from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer

warnings.filterwarnings(
    "ignore",
    message=r"data discontinuity in recording",
    category=SoundcardRuntimeWarning,
)

model_id = "nvidia/nemotron-3.5-asr-streaming-0.6b"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForRNNT.from_pretrained(model_id, device_map="auto")

processor.set_num_lookahead_tokens(6)
print(f"Streaming latency: {processor.streaming_latency_ms} ms")

language = "en-US"
sampling_rate = processor.feature_extractor.sampling_rate
audio_queue: Queue[np.ndarray] = Queue(maxsize=64)
capture_errors: Queue[Exception] = Queue()
generation_errors: Queue[Exception] = Queue()
stop_capture = Event()
audio_buffer = np.empty(0, dtype=np.float32)
buffer_start_idx = 0
received_samples = 0


def capture_system_audio(loopback_microphone, blocksize: int) -> None:
    try:
        with loopback_microphone.recorder(
            samplerate=sampling_rate, blocksize=blocksize
        ) as recorder:
            while not stop_capture.is_set():
                audio = recorder.record(numframes=blocksize)
                mono_audio = audio.mean(axis=1, dtype=np.float32)
                try:
                    audio_queue.put_nowait(mono_audio)
                except Full:
                    raise RuntimeError(
                        "Audio processing fell behind; stopping system-audio capture."
                    )
    except Exception as error:
        if not stop_capture.is_set():
            capture_errors.put(error)
            stop_capture.set()


def get_audio_samples(start_idx: int, end_idx: int) -> np.ndarray | None:
    global audio_buffer, buffer_start_idx, received_samples

    while received_samples < end_idx:
        if stop_capture.is_set():
            if not capture_errors.empty():
                raise capture_errors.get()
            return None
        try:
            block = audio_queue.get(timeout=0.1)
        except Empty:
            continue
        audio_buffer = np.concatenate((audio_buffer, block))
        received_samples += block.size

    start = start_idx - buffer_start_idx
    end = end_idx - buffer_start_idx
    return audio_buffer[start:end]


def discard_audio_before(sample_idx: int) -> None:
    global audio_buffer, buffer_start_idx

    discard_until = min(sample_idx, received_samples)
    count = discard_until - buffer_start_idx
    if count > 0:
        audio_buffer = audio_buffer[count:]
        buffer_start_idx = discard_until


def input_features_generator(first_chunk_inputs):
    yield first_chunk_inputs.input_features[
        :, : processor.num_mel_frames_first_audio_chunk, :
    ]

    mel_frame_idx = processor.num_mel_frames_first_audio_chunk
    hop_length = processor.feature_extractor.hop_length
    n_fft = processor.feature_extractor.n_fft
    start_idx = mel_frame_idx * hop_length - n_fft // 2

    while True:
        end_idx = start_idx + processor.num_samples_per_audio_chunk
        audio_chunk = get_audio_samples(start_idx, end_idx)
        if audio_chunk is None:
            return

        inputs = processor(
            audio_chunk,
            sampling_rate=sampling_rate,
            is_streaming=True,
            is_first_audio_chunk=False,
            language=language,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device, dtype=model.dtype)
        yield inputs.input_features

        mel_frame_idx += processor.num_mel_frames_per_audio_chunk
        start_idx = mel_frame_idx * hop_length - n_fft // 2
        discard_audio_before(start_idx)


def run_generation(generate_kwargs) -> None:
    try:
        model.generate(**generate_kwargs)
    except Exception as error:
        generation_errors.put(error)
        streamer.end()


streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
blocksize = max(1, int(sampling_rate * 0.08))
speaker = sc.default_speaker()
loopback_microphone = sc.get_microphone(
    id=str(speaker.name), include_loopback=True
)
capture_thread = Thread(
    target=capture_system_audio, args=(loopback_microphone, blocksize)
)
generation_thread = None

print("Capturing audio from the default speaker output. Press Ctrl+C to stop.")
try:
    capture_thread.start()
    first_audio_chunk = get_audio_samples(
        0, processor.num_samples_first_audio_chunk
    )
    if first_audio_chunk is None:
        raise RuntimeError("System-audio capture stopped before the first chunk arrived.")

    first_chunk_inputs = processor(
        first_audio_chunk,
        sampling_rate=sampling_rate,
        is_streaming=True,
        is_first_audio_chunk=True,
        language=language,
        return_tensors="pt",
    )
    first_chunk_inputs = first_chunk_inputs.to(model.device, dtype=model.dtype)
    generate_kwargs = {
        **first_chunk_inputs,
        "input_features": input_features_generator(first_chunk_inputs),
        "streamer": streamer,
        "max_new_tokens": 100_000
    }
    generation_thread = Thread(
        target=run_generation, args=(generate_kwargs,)
    )
    generation_thread.start()

    print("Transcript: ", end="", flush=True)
    try:
        for text_chunk in streamer:
            print(text_chunk, end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopping system-audio capture...")
except KeyboardInterrupt:
    print("\nStopping system-audio capture...")
finally:
    stop_capture.set()
    if generation_thread is not None:
        generation_thread.join()
    if capture_thread.is_alive():
        capture_thread.join()

if not generation_errors.empty():
    raise generation_errors.get()
if not capture_errors.empty():
    raise capture_errors.get()
