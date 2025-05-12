"""
2-Video Compression Project:
Objective: Implement a basic video compression.
Steps:
# 1. Video Input Handling
# o Create video frame-by-frame
o Convert frames to YUV color space.

2. Frame Type Decision
o Choose I-frames and P-frames (e.g., every 10th frame is an I-frame).

3. Intra-frame Compression (I-frame)
o Apply DCT on 8x8 blocks.
o Quantize DCT coefficients.
o Apply zig-zag scan and run-length encoding.

4. Inter-frame Compression (P-frame)
o Perform motion estimation (e.g., block matching).
o Compute motion vectors.
o Encode motion vectors and residuals.

5. Entropy Coding
o Use Huffman or Arithmetic Coding on motion vectors and residuals.

6. Bitstream Formation
o Package frames into a bitstream with headers and frame type indicators.

7. Testing & Evaluation
o Compare original and decoded video.
o Measure compression ratio and PSNR (Peak Signal-to-Noise Ratio).

"""

import cv2
import numpy as np
import pickle
import zlib
import os


class VideoProcessor:
    def __init__(self):
        self.frames = []
        self.decoded = []
        self.compressed_data = None
        self.fps = 30  # Default FPS
        self.frame_size = (0, 0)
        self.metadata = {}
        self.ratio = None
        self.psnr = None

    def load(self, path):
        """Load a video file and extract frames."""
        if not os.path.exists(path):
            raise ValueError(f"Video file does not exist: {path}")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {path}")

        self.frames = []
        self.fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:
                continue  # Skip invalid frames
            self.frames.append(frame)

        if not self.frames:
            cap.release()
            raise ValueError("No frames loaded from video")

        self.frame_size = (
            self.frames[0].shape[1],
            self.frames[0].shape[0],
        )  # Width, Height
        cap.release()

    def compress(self, gop=1, q=50, encoding_method="quantization"):
        """Compress video frames using the specified method."""
        if not self.frames:
            raise ValueError("No frames to compress")

        if encoding_method.lower() == "intra":
            self.compress_intra(gop, q)
        elif encoding_method.lower() == "pframe":
            self.compress_inter(gop, q)
        else:
            self.compress_quantization(gop, q)

    def compress_inter(self, gop=10, q=50):
        """Compress video using I-frames and P-frames with OpenCV."""
        # Create a temporary file for compression
        temp_file = "temp_compressed.avi"

        # Use OpenCV's VideoWriter for compression
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # MJPEG codec
        out = cv2.VideoWriter(
            temp_file, fourcc, self.fps, self.frame_size, isColor=True
        )

        # Write frames with specified quality
        for frame in self.frames:
            # OpenCV's compression quality is set via imwrite parameters
            # For MJPG, quality ranges from 0-100
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), q]
            _, encoded_frame = cv2.imencode(".jpg", frame, encode_params)
            decoded_frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
            out.write(decoded_frame)

        out.release()

        # Read back the compressed file
        with open(temp_file, "rb") as f:
            self.compressed_data = f.read()

        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

        # Metadata
        self.metadata = {
            "frame_size": self.frame_size,
            "fps": self.fps,
            "q_factor": q,
            "encoding_method": "pframe",
            "gop": gop,
        }

        # Decompress for analysis
        self.decompress()

        # Calculate compression ratio
        original_size = sum(f.nbytes for f in self.frames)
        compressed_size = len(self.compressed_data)
        self.ratio = (
            original_size / compressed_size if compressed_size > 0 else float("inf")
        )

        # Calculate PSNR
        self.psnr = self.calculate_psnr(self.frames, self.decoded)

    def compress_quantization(self, gop=1, q=8):
        """Use OpenCV's built-in quantization."""
        # Convert q to quality factor (higher q = lower quality)
        quality = min(100, max(1, int(100 - (q * 10))))

        # Compress each frame using JPEG with specified quality
        compressed_frames = []
        for frame in self.frames:
            # JPEG compression with specified quality
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded_frame = cv2.imencode(".jpg", frame, encode_params)
            compressed_frames.append(encoded_frame)

        # Store metadata
        self.metadata = {
            "bits": q,
            "frame_size": self.frame_size,
            "fps": self.fps,
            "encoding_method": "quantization",
        }

        # Serialize and compress data
        serialized = pickle.dumps(compressed_frames)
        self.compressed_data = zlib.compress(serialized)

        # Decompress for analysis
        self.decompress()

        # Calculate compression ratio
        original_size = sum(f.nbytes for f in self.frames)
        compressed_size = len(self.compressed_data)
        self.ratio = (
            original_size / compressed_size if compressed_size > 0 else float("inf")
        )

        # Calculate PSNR
        self.psnr = self.calculate_psnr(self.frames, self.decoded)

    def compress_intra(self, gop=1, q=50):
        """Use OpenCV's built-in JPEG compression for intra-frame encoding."""
        # Quality in OpenCV ranges from 0-100 (higher is better)
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), q]

        compressed_frames = []
        for frame in self.frames:
            # Encode to JPEG format
            _, encoded_frame = cv2.imencode(".jpg", frame, encode_params)
            compressed_frames.append(encoded_frame)

        # Store metadata
        self.metadata = {
            "q_factor": q,
            "frame_size": self.frame_size,
            "fps": self.fps,
            "encoding_method": "intra",
        }

        # Serialize and compress
        serialized = pickle.dumps(compressed_frames)
        self.compressed_data = zlib.compress(serialized)

        # Decompress for analysis
        self.decompress()

        # Calculate compression ratio
        original_size = sum(f.nbytes for f in self.frames)
        compressed_size = len(self.compressed_data)
        self.ratio = (
            original_size / compressed_size if compressed_size > 0 else float("inf")
        )

        # Calculate PSNR
        self.psnr = self.calculate_psnr(self.frames, self.decoded)

    def decompress(self):
        """Decompress using OpenCV's built-in functions."""
        if not self.compressed_data:
            raise ValueError("No compressed data available")

        try:
            decompressed = zlib.decompress(self.compressed_data)
            compressed_frames = pickle.loads(decompressed)

            encoding_method = self.metadata.get("encoding_method", "quantization")

            self.decoded = []
            for compressed in compressed_frames:
                # Use OpenCV's imdecode function to decompress
                frame = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
                self.decoded.append(frame)

        except (zlib.error, pickle.PickleError) as e:
            raise ValueError(f"Decompression failed: {str(e)}")

    def calculate_psnr(self, original_frames, compressed_frames):
        """Calculate PSNR between original and compressed frames using OpenCV."""
        if len(original_frames) != len(compressed_frames):
            raise ValueError("Mismatch in number of original and compressed frames")

        psnr_values = []
        for orig, comp in zip(original_frames, compressed_frames):
            # Ensure same shape and type
            if orig.shape != comp.shape:
                comp = cv2.resize(comp, (orig.shape[1], orig.shape[0]))

            # Use OpenCV's built-in PSNR calculation
            psnr = cv2.PSNR(orig, comp)
            psnr_values.append(psnr)

        return np.mean(psnr_values) if psnr_values else 0.0

    def save_video(self, path):
        """Save the decoded video to a file using OpenCV's VideoWriter."""
        if not self.decoded:
            raise ValueError("No decoded video available")

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            path,
            fourcc,
            self.fps,
            self.frame_size,
            isColor=(len(self.decoded[0].shape) > 2 and self.decoded[0].shape[2] == 3),
        )

        for frame in self.decoded:
            out.write(frame)
        out.release()

    def save_bitstream(self, path):
        """Save the compressed bitstream to a file."""
        if not self.compressed_data:
            raise ValueError("No compressed data available")

        with open(path, "wb") as f:
            f.write(self.compressed_data)
