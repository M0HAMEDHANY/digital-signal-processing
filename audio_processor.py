import numpy as np
import soundfile as sf
import librosa
import pyaudio
import threading
from scipy.signal import stft, istft
import zlib
import pickle


class AudioProcessor:
    def __init__(self):
        self.signal = None
        self.time = None
        self.fs = None
        self.reconstructed = None
        self.snr = None

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

        # Convert to mono if stereo
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        self.signal = data
        self.fs = sr
        length = data.shape[0]
        self.time = np.linspace(0, length / sr, num=length)

    def denoise(self, threshold: float) -> np.ndarray:
        if self.signal is None:
            raise ValueError("Load audio first")
        clean = self.signal.copy()
        clean[np.abs(clean) < threshold] = 0.0
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
        def _worker():
            stream = self._pa.open(
                format=pyaudio.paFloat32, channels=1, rate=self.fs, output=True
            )
            stream.write(data.tobytes())
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
            raise ValueError("Run compress() first")
        self.play(self.reconstructed)

    def save_wav(self, path):
        if self.reconstructed is None:
            raise ValueError("Nothing to save")
        sf.write(path, self.reconstructed, self.fs)

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
