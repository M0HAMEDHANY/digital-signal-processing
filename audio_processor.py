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
    """
    Core audio processing class that handles loading, compression, decompression,
    playback and analysis of audio signals.

    The compression algorithm uses Short-Time Fourier Transform (STFT) to convert
    the signal to frequency domain, followed by quantization and entropy coding.
    """

    def __init__(self):
        # Audio data containers
        self.signal = None  # Original audio signal
        self.time = None  # Time axis for plotting
        self.fs = None  # Sampling frequency in Hz
        self.reconstructed = None  # Reconstructed signal after compression
        self.snr = None  # Signal-to-noise ratio after compression
        self.filename = None  # Original filename for reference

        # Audio playback resources
        self._pa = pyaudio.PyAudio()  # PyAudio instance for playback
        self._play_thread = None  # Thread for non-blocking audio playback

        # Compression data
        self.metadata = {}  # Metadata for compression parameters
        self.compressed_data = None  # Compressed binary data
        self.phase = None  # Phase information preserved during compression
        self.win_size = None  # Window size for STFT
        self.bits = None  # Bit depth for quantization

    def load(self, filepath):
        """
        Load audio file using librosa's built-in functions which handles
        various formats and conversions automatically.

        Args:
            filepath (str): Path to the audio file (.wav, .flac, .mp3, etc.)

        Raises:
            ValueError: If the file can't be opened or is in an unsupported format
        """
        try:
            # Load audio file and convert to mono if needed
            self.signal, self.fs = librosa.load(filepath, sr=None, mono=True)
            self.filename = os.path.basename(filepath)

            # Create time axis for visualization
            length = self.signal.shape[0]
            self.time = np.linspace(0, length / self.fs, num=length)

        except Exception as e:
            raise ValueError(
                f"Could not open '{filepath}'. "
                "Use WAV/FLAC/MP3 or install ffmpeg for M4A support."
            ) from e

    def stop(self):
        """
        Stop any currently playing audio by signaling the playback thread to stop
        and joining it to ensure clean termination.
        """
        if self._play_thread is not None:
            # Signal the playback thread to stop
            self._stop_signal = True

            # Wait for the thread to finish
            self._play_thread.join(timeout=0.5)
            self._play_thread = None

    def denoise(self, threshold: float) -> np.ndarray:
        """
        Apply spectral gating noise reduction to the audio signal.

        This method performs noise reduction by:
        1. Converting to frequency domain using STFT
        2. Estimating noise from the first few frames
        3. Applying a gain function based on the noise threshold
        4. Reconstructing the signal with reduced noise

        Args:
            threshold (float): Noise reduction threshold. Higher values
                              result in more aggressive noise reduction.

        Returns:
            np.ndarray: The cleaned audio signal

        Raises:
            ValueError: If no audio is loaded
        """
        if self.signal is None:
            raise ValueError("Load audio first")

        # Parameters for the STFT
        n_fft = 2048
        hop_length = n_fft // 4

        # Compute the Short-Time Fourier Transform
        S = librosa.stft(self.signal, n_fft=n_fft, hop_length=hop_length)

        # Separate magnitude and phase
        mag = np.abs(S)
        phase = np.angle(S)

        # Estimate noise profile from the beginning of the signal
        # (assumes the beginning has representative noise)
        noise_frames = int(len(self.signal) * 0.1 / hop_length)
        noise_frames = max(5, min(noise_frames, 20))
        noise_estimate = np.mean(mag[:, :noise_frames], axis=1)

        # Calculate a spectral gain function based on the noise estimate
        gain = np.maximum(
            0, 1 - threshold * noise_estimate[:, np.newaxis] / (mag + 1e-10)
        )
        mag_clean = mag * gain

        # Reconstruct the signal using the original phase
        S_clean = mag_clean * np.exp(1j * phase)
        clean = librosa.istft(S_clean, hop_length=hop_length, length=len(self.signal))
        self.reconstructed = clean.astype(np.float32)
        return clean

    def compress(self, win_size, bits, method):
        """
        Compress the audio signal using frequency domain transformation,
        quantization, and entropy coding.

        Process:
        1. Transform signal to frequency domain using STFT
        2. Quantize magnitude values to specified bit depth
        3. Use entropy coding (default is Huffman via zlib)
        4. Store phase information separately for reconstruction
        5. Calculate SNR between original and reconstructed signals

        Args:
            win_size (int): Window size for the STFT
            bits (int): Number of bits for quantization (4, 8, 16)
            method (str): Compression method (currently only "Huffman" is implemented)

        Raises:
            ValueError: If compression fails for any reason
        """
        # Store parameters for later reconstruction
        self.win_size = win_size
        self.bits = bits
        self.metadata["method"] = method

        # Apply Short-Time Fourier Transform
        f, t, Z = stft(self.signal, fs=self.fs, nperseg=win_size)
        mag = np.abs(Z)  # Magnitude component
        self.phase = np.angle(Z)  # Phase component (stored without compression)

        # Quantize magnitude values to reduce precision
        mag_q = np.round(mag / mag.max() * (2**bits - 1)).astype(np.uint8)

        # Apply entropy coding using zlib (which includes Huffman coding)
        serialized = pickle.dumps(mag_q)
        self.compressed_data = zlib.compress(serialized, level=9)

        # Store metadata needed for reconstruction
        self.metadata.update(
            {
                "original_shape": mag_q.shape,
                "max_value": float(mag.max()),
                "fs": self.fs,
            }
        )

        # Reconstruct the signal for playback and quality assessment
        mag_d = mag_q / (2**bits - 1) * mag.max()
        Z_rec = mag_d * np.exp(1j * self.phase)
        _, x_rec = istft(Z_rec, fs=self.fs, nperseg=win_size)
        self.reconstructed = x_rec.astype(np.float32)

        # Calculate Signal-to-Noise Ratio (SNR) as quality metric
        N = min(len(self.signal), len(self.reconstructed))
        sig = self.signal[:N]
        rec = self.reconstructed[:N]
        noise = sig - rec
        p_sig = np.sum(sig**2)  # Signal power
        p_noise = np.sum(noise**2)  # Noise power
        self.snr = 10 * np.log10(p_sig / p_noise) if p_noise > 0 else float("inf")

    def play(self, data):
        """
        Play audio data in a separate thread to avoid blocking the UI.

        Args:
            data (np.ndarray): Audio data to play (should be float32 format)
        """
        self._stop_signal = False

        def _worker():
            """Worker thread function that handles audio playback"""
            stream = self._pa.open(
                format=pyaudio.paFloat32, channels=1, rate=self.fs, output=True
            )

            chunk_size = 1024
            data_bytes = data.tobytes()

            # Fixed the duplicate loop issue in the original code
            for i in range(0, len(data_bytes), chunk_size * 4):
                if self._stop_signal:
                    break
                chunk = data_bytes[i : i + chunk_size * 4]
                stream.write(chunk)

            stream.stop_stream()
            stream.close()

        self._play_thread = threading.Thread(target=_worker, daemon=True)
        self._play_thread.start()

    def play_original(self):
        """
        Play the original, unprocessed audio signal.

        Raises:
            ValueError: If no audio has been loaded
        """
        if self.signal is None:
            raise ValueError("Load audio first")
        self.play(self.signal.astype(np.float32))

    def play_processed(self):
        """
        Play the processed (denoised or compressed) audio signal.

        Raises:
            ValueError: If no processing has been done yet
        """
        if self.reconstructed is None:
            raise ValueError("Run compress first")
        self.play(self.reconstructed)

    def save_wav(self, path):
        """
        Save the processed audio as a WAV file.

        Args:
            path (str): Directory or file path to save the audio

        Returns:
            str: Path to the saved file

        Raises:
            ValueError: If there is no processed audio to save
        """
        if self.reconstructed is None:
            raise ValueError("Nothing to save")

        # Handle both directory and file path inputs
        if os.path.isdir(path):
            if self.filename:
                base, ext = os.path.splitext(self.filename)
                new_path = os.path.join(path, f"{base}_constructed{ext}")
                sf.write(new_path, self.reconstructed, self.fs)
                return new_path

        sf.write(path, self.reconstructed, self.fs)
        return path

    def save_bitstream(self, path):
        """
        Save the compressed audio data and metadata to a binary file.
        Useful for later decompression without needing the original file.

        Args:
            path (str): Path to save the compressed data file

        Raises:
            ValueError: If compression hasn't been run
        """
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
        """
        Load previously compressed audio data from a file and reconstruct the audio.

        Args:
            path (str): Path to the compressed data file

        Returns:
            bool: True if loading was successful
        """
        data = np.load(path, allow_pickle=True)
        self.compressed_data = data["compressed_data"].item()
        self.phase = data["phase"]
        self.win_size = int(data["win_size"])
        self.bits = int(data["bits"])
        self.fs = int(data["fs"])
        self.metadata = data["metadata"].item()

        # Decompress the quantized magnitude data
        mag_q = pickle.loads(zlib.decompress(self.compressed_data))

        # Reconstruct the audio signal
        mag_d = mag_q / (2**self.bits - 1) * self.metadata["max_value"]
        Z_rec = mag_d * np.exp(1j * self.phase)
        _, x_rec = istft(Z_rec, fs=self.fs, nperseg=self.win_size)
        self.reconstructed = x_rec.astype(np.float32)
        length = len(self.reconstructed)
        self.time = np.linspace(0, length / self.fs, num=length)
        return True
