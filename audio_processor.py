"""
1-Audio Compression Project: MP3 Encoder Implementation
Objective: Implement a basic MP3-like audio compression system
Steps:
1. Audio Input Handling
o Create a raw PCM audio file (e.g., WAV format). That contains noise
and silence
o Plot audio signal without noise
o Plot audio signal that includes noise and silence
2. Transform to Frequency Domain
o Apply Short-Time Fourier Transform (STFT) or MDCT.
o Split the audio into frequency bands.
3. Quantization & Bit Allocation
.
4. Encoding
o Use Huffman or run-length encoding to compress quantized values.
5. Testing & Evaluation
o Compare original vs. decompressed audio.
o Use Signal-to-Noise Ratio (SNR)
"""

import numpy as np
import soundfile as sf
import librosa
import pyaudio
import threading
from scipy.signal import stft, istft
import zlib
import pickle
import os


class AudioProcessor:
    def __init__(self):
        self.signal = None
        self.time = None
        self.fs = None
        self.reconstructed = None
        self.snr = None
        self.filename = None
        
        self._pa = pyaudio.PyAudio()
        self._play_thread = None

        self.metadata = {}
        self.compressed_data = None
        self.phase = None
        self.win_size = None
        self.bits = None

    def load(self, filepath):
        try:
            data, sr = sf.read(filepath)
        except Exception:
            try:
                y, sr = librosa.load(filepath, sr=None, mono=False)
            except Exception as e:
                raise ValueError(
                    f"Could not open '{filepath}'. "
                    "Use WAV/FLAC/MP3 or install ffmpeg for M4A support."
                ) from e
            data = y.T if y.ndim > 1 else y
        self.filename = os.path.basename(filepath)
        # Convert to mono if stereo
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        self.signal = data
        self.fs = sr
        length = data.shape[0]
        self.time = np.linspace(0, length / sr, num=length)

    def stop(self):
        if self._play_thread is not None:
            # Create a signal to stop the stream
            self._stop_signal = True
            
            # Wait for thread to finish
            self._play_thread.join(timeout=0.5)  # Add timeout to prevent hanging
            self._play_thread = None
            
    def denoise(self, threshold: float) -> np.ndarray:
        # print("Denoising...")
        if self.signal is None:
            raise ValueError("Load audio first")
        
        # Method 1: Simple threshold-based denoising (current implementation)
        # clean = self.signal.copy()
        # clean[np.abs(clean) < threshold] = 0.0
        
        # Method 2: Spectral subtraction denoising
        # Apply STFT
        n_fft = 2048
        hop_length = n_fft // 4
        
        # Compute spectrogram
        S = librosa.stft(self.signal, n_fft=n_fft, hop_length=hop_length)
        
        # Get magnitude and phase
        mag = np.abs(S)
        phase = np.angle(S)
        
        # Estimate noise floor (assuming first few frames are noise)
        noise_frames = int(len(self.signal) * 0.1 / hop_length)  # use first 10% of signal
        noise_frames = max(5, min(noise_frames, 20))  # at least 5, at most 20 frames
        noise_estimate = np.mean(mag[:, :noise_frames], axis=1)
        
        # Apply spectral subtraction
        gain = np.maximum(0, 1 - threshold * noise_estimate[:, np.newaxis] / (mag + 1e-10))
        mag_clean = mag * gain
        
        # Reconstruct signal
        S_clean = mag_clean * np.exp(1j * phase)
        clean = librosa.istft(S_clean, hop_length=hop_length, length=len(self.signal))
        self.reconstructed = clean.astype(np.float32)
        return clean

    def compress(self, win_size, bits, method):
        self.win_size = win_size
        self.bits = bits
        self.metadata["method"] = method

        # STFT
        f, t, Z = stft(self.signal, fs=self.fs, nperseg=win_size)
        mag = np.abs(Z)
        self.phase = np.angle(Z)

        # Quantize magnitude
        mag_q = np.round(mag / mag.max() * (2**bits - 1)).astype(np.uint8)

        # Serialize + compress using zlib
        serialized = pickle.dumps(mag_q)
        self.compressed_data = zlib.compress(serialized, level=9)

        # Save for metadata
        self.metadata.update(
            {
                "original_shape": mag_q.shape,
                "max_value": float(mag.max()),
                "fs": self.fs,
            }
        )

        # Reconstruct audio
        mag_d = mag_q / (2**bits - 1) * mag.max()
        Z_rec = mag_d * np.exp(1j * self.phase)
        _, x_rec = istft(Z_rec, fs=self.fs, nperseg=win_size)
        self.reconstructed = x_rec.astype(np.float32)

        # SNR
        N = min(len(self.signal), len(self.reconstructed))
        sig = self.signal[:N]
        rec = self.reconstructed[:N]
        noise = sig - rec
        p_sig = np.sum(sig**2)
        p_noise = np.sum(noise**2)
        self.snr = 10 * np.log10(p_sig / p_noise) if p_noise > 0 else float("inf")

    def play(self, data):
        self._stop_signal = False  # Reset stop signal
        
        def _worker():
            stream = self._pa.open(
                format=pyaudio.paFloat32, channels=1, rate=self.fs, output=True
            )
            
            # Process data in chunks to allow interruption
            chunk_size = 1024
            data_bytes = data.tobytes()
            for i in range(0, len(data_bytes), chunk_size * 4): 
                if self._stop_signal:
                    break
                chunk = data_bytes[i:i + chunk_size * 4]
                stream.write(chunk)
                
            stream.stop_stream()
            stream.close()

        self._play_thread = threading.Thread(target=_worker, daemon=True)
        self._play_thread.start()

    def play_original(self):
        if self.signal is None:
            raise ValueError("Load audio first")
        self.play(self.signal.astype(np.float32))

    def play_processed(self):
        if self.reconstructed is None:
            raise ValueError("Run compress first")
        self.play(self.reconstructed)

    def save_wav(self, path):
        if self.reconstructed is None:
            raise ValueError("Nothing to save")

        # If path is a directory, use original filename + "_constructed"
        if os.path.isdir(path):
            if self.filename:
                base, ext = os.path.splitext(self.filename)
                new_path = os.path.join(path, f"{base}_constructed{ext}")
                sf.write(new_path, self.reconstructed, self.fs)
                return new_path
        
        # Otherwise use provided path
        sf.write(path, self.reconstructed, self.fs)
        return path
    
    def save_bitstream(self, path):
        if self.compressed_data is None:
            raise ValueError("Run compress() first")
        np.savez(
            path,
            compressed_data=self.compressed_data,
            phase=self.phase,
            win_size=self.win_size,
            bits=self.bits,
            fs=self.fs,
            metadata=self.metadata,
        )

    def load_bitstream(self, path):
        data = np.load(path, allow_pickle=True)
        self.compressed_data = data["compressed_data"].item()
        self.phase = data["phase"]
        self.win_size = int(data["win_size"])
        self.bits = int(data["bits"])
        self.fs = int(data["fs"])
        self.metadata = data["metadata"].item()

        # Decompress + deserialize
        mag_q = pickle.loads(zlib.decompress(self.compressed_data))

        # Reconstruct
        mag_d = mag_q / (2**self.bits - 1) * self.metadata["max_value"]
        Z_rec = mag_d * np.exp(1j * self.phase)
        _, x_rec = istft(Z_rec, fs=self.fs, nperseg=self.win_size)
        self.reconstructed = x_rec.astype(np.float32)
        length = len(self.reconstructed)
        self.time = np.linspace(0, length / self.fs, num=length)
        return True