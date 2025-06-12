import numpy as np
import pyaudio
import torch
import matplotlib.pyplot as plt
from scipy.signal import stft
from matplotlib.colors import LogNorm

from src.examples.loading_pretrained_models import load_pretrained_CleanUMamba

# --- PyTorch Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# --- Load Model ---
model_path = "checkpoints/pruned/CleanUMamba-3N-E8_pruned-5M.pkl"  # Make sure this path is correct
try:
    model = load_pretrained_CleanUMamba(model_path)
    model.to(device)
    model.eval()
    print(f"Model '{model_path}' loaded successfully.")
except Exception as e:
    print(f"Error loading model from '{model_path}': {e}")
    print("Please ensure the model file exists and the loading function is correct.")
    raise


# --- Audio Parameters ---
CHUNK = 256 * 16  # Number of audio frames per buffer
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000

# --- STFT Parameters ---
NPERSEG = 256
NOVERLAP = 128

# --- History Parameters ---
HISTORY_SECONDS = 5
TOTAL_HISTORY_SAMPLES = int(RATE * HISTORY_SECONDS)

# --- Initialize PyAudio ---
p = pyaudio.PyAudio()
stream = None

print("test")

try:
    default_input_device_info = p.get_default_input_device_info()
    default_input_device_index = default_input_device_info['index']
    print(f"Using default input device: {default_input_device_info['name']} (Index: {default_input_device_index})")

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=default_input_device_index,
                    frames_per_buffer=CHUNK)
except OSError as e:
    print(f"Error opening audio stream: {e}")
    print("Please check your microphone and sound settings.")
    print("Available input devices:")
    for i in range(p.get_device_count()):
        try:
            dev_info = p.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:
                print(f"  Device index {i}: {dev_info['name']}")
        except Exception as dev_e:
            print(f"  Could not get info for device index {i}: {dev_e}")
    if p: p.terminate()
    raise


original_audio_history = np.zeros(TOTAL_HISTORY_SAMPLES, dtype=np.float32)
denoised_audio_history = np.zeros(TOTAL_HISTORY_SAMPLES, dtype=np.float32)

# --- Visualization Plot ---
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

f_coords, t_coords, Zxx_initial_complex = stft(
    original_audio_history, fs=RATE, nperseg=NPERSEG, noverlap=NOVERLAP
)
Zxx_mag_initial = np.abs(Zxx_initial_complex)

log_epsilon = 1e-9
vmin_spectro, vmax_spectro = 1e-5, 1e-1  # Initial guess, might need tuning

im1 = ax1.imshow(Zxx_mag_initial, aspect='auto', origin='lower', cmap='viridis',
                 extent=[t_coords.min(), t_coords.max(), f_coords.min(), f_coords.max()],
                 interpolation='nearest',
                 norm=LogNorm(vmin=vmin_spectro, vmax=vmax_spectro, clip=True))
im2 = ax2.imshow(Zxx_mag_initial, aspect='auto', origin='lower', cmap='viridis',
                 extent=[t_coords.min(), t_coords.max(), f_coords.min(), f_coords.max()],
                 interpolation='nearest',
                 norm=LogNorm(vmin=vmin_spectro, vmax=vmax_spectro, clip=True))

fig.colorbar(im1, ax=ax1, label='Magnitude')
fig.colorbar(im2, ax=ax2, label='Magnitude')
ax1.set_title(f'Original Audio Spectrogram ({HISTORY_SECONDS}s history)')
ax2.set_title(f'Denoised Audio Spectrogram ({HISTORY_SECONDS}s history)')
ax1.set_ylabel('Frequency [Hz]')
ax2.set_ylabel('Frequency [Hz]')
ax2.set_xlabel('Time [sec]')

fig.tight_layout()
plt.show(block=False)

fig.canvas.draw()
axbackground = fig.canvas.copy_from_bbox(ax1.bbox)
ax2background = fig.canvas.copy_from_bbox(ax2.bbox)



print("Starting real-time audio processing. Press Ctrl+C to stop.")

try:
    while True:
        raw_data = stream.read(CHUNK, exception_on_overflow=False)
        audio_chunk_raw = np.frombuffer(raw_data, dtype=np.float32)

        # Ensure audio_chunk is exactly CHUNK size
        if len(audio_chunk_raw) < CHUNK:
            audio_chunk = np.pad(audio_chunk_raw, (0, CHUNK - len(audio_chunk_raw)), 'constant')
        elif len(audio_chunk_raw) > CHUNK:
            audio_chunk = audio_chunk_raw[:CHUNK]
        else:
            audio_chunk = audio_chunk_raw

        # --- Denoising Process ---
        tensor_chunk = torch.from_numpy(audio_chunk.copy()).unsqueeze(0).to(device)

        with torch.no_grad():
            model_output = model.feed(tensor_chunk)

        denoised_chunk_from_model = model_output.squeeze(0).cpu().numpy()


        if len(denoised_chunk_from_model) < CHUNK:
            padding_needed = CHUNK - len(denoised_chunk_from_model)
            # Pad with zeros at the end.
            denoised_chunk_processed = np.pad(denoised_chunk_from_model, (0, padding_needed), 'constant')
        elif len(denoised_chunk_from_model) > CHUNK:
            denoised_chunk_processed = denoised_chunk_from_model[:CHUNK]
        else:
            denoised_chunk_processed = denoised_chunk_from_model

        # --- Update Audio History Buffers ---
        original_audio_history = np.roll(original_audio_history, -CHUNK, axis=0)
        original_audio_history[-CHUNK:] = audio_chunk

        denoised_audio_history = np.roll(denoised_audio_history, -CHUNK, axis=0)
        denoised_audio_history[-CHUNK:] = denoised_chunk_processed  # Use the processed (padded/truncated) chunk

        # --- Update Spectrogram Data ---
        _, _, Zxx1_complex = stft(original_audio_history, fs=RATE, nperseg=NPERSEG, noverlap=NOVERLAP)
        Zxx1_mag = np.abs(Zxx1_complex) + log_epsilon
        im1.set_data(Zxx1_mag)

        _, _, Zxx2_complex = stft(denoised_audio_history, fs=RATE, nperseg=NPERSEG, noverlap=NOVERLAP)
        Zxx2_mag = np.abs(Zxx2_complex) + log_epsilon
        im2.set_data(Zxx2_mag)

        # --- Update Matplotlib ---
        fig.canvas.restore_region(axbackground)
        fig.canvas.restore_region(ax2background)
        ax1.draw_artist(im1)
        ax2.draw_artist(im2)
        fig.canvas.blit(ax1.bbox)
        fig.canvas.blit(ax2.bbox)
        fig.canvas.flush_events()

except KeyboardInterrupt:
    print("\nStopping the stream and closing application...")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback

    traceback.print_exc()

finally:
    if hasattr(model, 'time_per_frame') and hasattr(model, 'frame_length') and hasattr(model, 'total_stride'):
        print(f"Inference speed was {1000 * model.time_per_frame:.2f} ms per frame "
              f"or {(model.total_stride / RATE / model.time_per_frame):.2f}x real time "
              f"(for model's processed frames of length {model.frame_length} with a total stride of {model.total_stride})")
    else:
        print("Model does not have 'time_per_frame', 'frame_length' or 'total_stride' attributes for speed reporting.")

    if stream is not None:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
    if p: p.terminate()

    plt.ioff()
    if 'fig' in locals() and fig.canvas.manager.window:  # Check if fig exists and window is open
        plt.close(fig)
    print("Application closed.")