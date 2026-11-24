
import asyncio
import json
import wave
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

try:
    import numpy as np
except ImportError as e:
    np = None
    print(f"❌ NumPy import failed: {e}")

try:
    from rtlsdr import RtlSdr
except Exception as exc:
    RtlSdr = None
    print(f"❌ RTL-SDR import failed: {exc}")

app = FastAPI()
BASE_DIR = Path(__file__).parent
RECORDINGS_DIR = BASE_DIR / "recordings"

SAMPLE_RATE = 1152000
DECIMATION = 24
IQ_CHUNK_SIZE = 96 * 1024
SPECTRUM_BINS = 256

sdr = None
audio_queue = asyncio.Queue(maxsize=10)
rssi_value = -100.0
current_frequency_mhz = 462.125
signal_detected = False
hardware_ready = False
samples_seen = 0
noise_floor_db = -120.0
signal_hits = 0
spectrum_points = [ -120.0 ] * SPECTRUM_BINS
signal_log_path = None
audio_wave_file = None
recording_started_at = None
recording_state = "idle"
recording_paused = False
recording_session_active = False


def ensure_recording_session() -> None:
    global signal_log_path, audio_wave_file, recording_started_at

    RECORDINGS_DIR.mkdir(exist_ok=True)
    if audio_wave_file is not None:
        audio_wave_file.close()
        audio_wave_file = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    signal_log_path = RECORDINGS_DIR / f"dashboard_{timestamp}.jsonl"
    audio_path = RECORDINGS_DIR / f"audio_{timestamp}.wav"

    audio_wave_file = wave.open(str(audio_path), "wb")
    audio_wave_file.setnchannels(1)
    audio_wave_file.setsampwidth(2)
    audio_wave_file.setframerate(48000)
    recording_started_at = datetime.now().isoformat(timespec="seconds")
    print(f"► Recording session started: {signal_log_path.name} and {audio_path.name}")


def set_recording_state(command: str) -> dict:
    global signal_log_path, audio_wave_file, recording_started_at, recording_state, recording_paused, recording_session_active

    if command == "start":
        if recording_paused and recording_session_active:
            recording_paused = False
            recording_state = "recording"
            print("► Recording resumed.")
        else:
            if audio_wave_file is not None:
                audio_wave_file.close()
                audio_wave_file = None
            ensure_recording_session()
            recording_session_active = True
            recording_paused = False
            recording_state = "recording"
        return {"recording_state": recording_state, "recording_session_active": recording_session_active}

    if command == "pause":
        if recording_session_active and not recording_paused:
            recording_paused = True
            recording_state = "paused"
            print("► Recording paused.")
        return {"recording_state": recording_state, "recording_session_active": recording_session_active}

    if command == "stop":
        if audio_wave_file is not None:
            audio_wave_file.close()
            audio_wave_file = None
        if signal_log_path is None:
            RECORDINGS_DIR.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            signal_log_path = RECORDINGS_DIR / f"dashboard_{timestamp}.jsonl"
        try:
            with signal_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "event": "stop"
                }, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"❌ Signal log write failed: {exc}")
        recording_session_active = False
        recording_paused = False
        recording_state = "stopped"
        print("► Recording stopped.")
        return {"recording_state": recording_state, "recording_session_active": recording_session_active}

    return {"recording_state": recording_state, "recording_session_active": recording_session_active}


def log_signal_event(payload: dict) -> None:
    global signal_log_path
    if not recording_session_active or recording_paused:
        return
    if signal_log_path is None:
        ensure_recording_session()
    if signal_log_path is None:
        return
    try:
        with signal_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"❌ Signal log write failed: {exc}")


def log_audio_chunk(pcm_audio: bytes) -> None:
    global audio_wave_file
    if not recording_session_active or recording_paused:
        return
    if audio_wave_file is None:
        ensure_recording_session()
    if audio_wave_file is None:
        return
    try:
        audio_wave_file.writeframes(pcm_audio)
    except Exception as exc:
        print(f"❌ Audio write failed: {exc}")


def should_detect_signal(rssi_db: float, noise_floor: float) -> bool:
    if np is None or not np.isfinite(rssi_db):
        return False
    if rssi_db < -80.0:
        return False
    margin_db = rssi_db - noise_floor
    return margin_db >= 10.0


@app.get("/")
async def serve_index():
    response = FileResponse(BASE_DIR / "index.html")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.on_event("startup")
def startup_event():
    global sdr, hardware_ready, signal_detected, samples_seen, noise_floor_db, signal_hits
    RECORDINGS_DIR.mkdir(exist_ok=True)
    try:
        if RtlSdr is None:
            raise RuntimeError("RTL-SDR support is unavailable in this environment.")
        if np is None:
            raise RuntimeError("NumPy is unavailable in this environment.")

        sdr = RtlSdr()
        sdr.sample_rate = SAMPLE_RATE
        sdr.center_freq = current_frequency_mhz * 1e6
        sdr.gain = "auto"
        hardware_ready = True
        signal_detected = False
        samples_seen = 0
        noise_floor_db = -120.0
        signal_hits = 0
        print("► RTL-SDR Hardware Initialized Cleanly!")
    except Exception as exc:
        print(f"❌ Hardware Error: {exc}")
        sdr = None
        hardware_ready = False
        signal_detected = False
        samples_seen = 0
        noise_floor_db = -120.0
        signal_hits = 0

    asyncio.create_task(sdr_hardware_loop())


async def sdr_hardware_loop():
    global rssi_value, signal_detected, hardware_ready, samples_seen, noise_floor_db, signal_hits, spectrum_points
    prev_sample = complex(1, 0)
    window_size = DECIMATION
    kernel = np.ones(window_size) / window_size if np is not None else None

    while True:
        if sdr is None or np is None:
            hardware_ready = False
            signal_detected = False
            samples_seen = 0
            noise_floor_db = -120.0
            signal_hits = 0
            spectrum_points = [ -120.0 ] * SPECTRUM_BINS
            if audio_queue.full():
                audio_queue.get_nowait()
            sample_rate = 48000
            frame_size = 2048
            tone = np.sin(2 * np.pi * 440 * np.arange(frame_size) / sample_rate) if np is not None else None
            if tone is None:
                pcm_audio = b"\x00" * (frame_size * 2)
            else:
                pcm_audio = (tone * 0.35 * 32767).astype(np.int16).tobytes()
            await audio_queue.put(pcm_audio)
            await asyncio.sleep(0.05)
            continue

        try:
            samples = await asyncio.to_thread(sdr.read_samples, IQ_CHUNK_SIZE)
            if len(samples) == 0:
                continue

            samples_seen += 1
            rssi_value = 10 * np.log10(np.mean(np.abs(samples[:1024]) ** 2) + 1e-10)
            noise_floor_db = noise_floor_db * 0.95 + rssi_value * 0.05

            # Compute a wide spectrum overview for the dashboard
            fft_size = min(len(samples), 4096)
            if fft_size >= 256:
                window = np.hamming(fft_size)
                spectrum = np.fft.fftshift(np.fft.fft(samples[:fft_size] * window))
                spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
                if len(spectrum_db) > SPECTRUM_BINS:
                    spectrum_db = spectrum_db.reshape(-1, len(spectrum_db) // SPECTRUM_BINS).mean(axis=1)
                spectrum_db = np.clip(spectrum_db, -120.0, 10.0)
                spectrum_points = spectrum_db.tolist()

            shifted_samples = np.empty_like(samples)
            shifted_samples[0] = samples[0] * np.conj(prev_sample)
            shifted_samples[1:] = samples[1:] * np.conj(samples[:-1])
            prev_sample = samples[-1]
            demodulated = np.angle(shifted_samples)

            filtered = np.convolve(demodulated, kernel, mode="same")
            audio_data = filtered[::DECIMATION]

            rms_power = np.sqrt(np.mean(audio_data**2))
            hardware_ready = True
            if should_detect_signal(rssi_value, noise_floor_db):
                signal_hits += 1
            else:
                signal_hits = max(0, signal_hits - 1)
            signal_detected = bool(signal_hits >= 3)
            if rms_power < 0.02:
                audio_data = np.zeros_like(audio_data)
            else:
                audio_data *= 5.0

            audio_data = np.clip(audio_data, -1.0, 1.0)
            pcm_audio = (audio_data * 32767).astype(np.int16).tobytes()

            if audio_queue.full():
                audio_queue.get_nowait()
            await audio_queue.put(pcm_audio)
        except Exception as exc:
            print(f"Hardware loop error: {exc}")
            await asyncio.sleep(0.1)


@app.websocket("/ws/signal")
async def signal_endpoint(websocket: WebSocket):
    global current_frequency_mhz
    await websocket.accept()
    try:
        while True:
            telemetry_payload = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "rssi": float(rssi_value),
                "detected": bool(signal_detected),
                "hardware_ready": bool(hardware_ready),
                "frequency": float(current_frequency_mhz),
                "status": "ready" if hardware_ready else "demo",
                "spectrum": [float(x) for x in spectrum_points],
                "span_mhz": float(SAMPLE_RATE / 1e6),
                "recording_state": recording_state
            }
            log_signal_event(telemetry_payload)
            await websocket.send_json(telemetry_payload)

            try:
                raw_data = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
                payload = json.loads(raw_data)
                if payload.get("action") == "tune":
                    current_frequency_mhz = float(payload.get("frequency", current_frequency_mhz))
                    if sdr is not None:
                        sdr.center_freq = current_frequency_mhz * 1e6
                    log_signal_event({
                        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                        "event": "tune",
                        "frequency": float(current_frequency_mhz)
                    })
                elif payload.get("action") == "recording":
                    command = payload.get("command", "")
                    if command in {"start", "pause", "stop"}:
                        set_recording_state(command)
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break
            except Exception as exc:
                print(f"Signal receive error: {exc}")

            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.websocket("/ws/audio")
async def audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            pcm_audio = await audio_queue.get()
            log_audio_chunk(pcm_audio)
            await websocket.send_bytes(pcm_audio)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.on_event("shutdown")
def shutdown():
    global sdr, audio_wave_file, recording_state, recording_paused, recording_session_active
    if sdr is not None:
        sdr.close()
        print("► RTL-SDR Hardware Released safely.")
    if audio_wave_file is not None:
        audio_wave_file.close()
        audio_wave_file = None
    recording_state = "stopped"
    recording_paused = False
    recording_session_active = False


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

