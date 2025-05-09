"""
2-Video Compression Project:
Objective: Implement a basic video compression.
Steps:
1. Video Input Handling
o Create video frame-by-frame
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
            self.frames.append(frame)
        
        if not self.frames:
            cap.release()
            raise ValueError("No frames loaded from video")
        
        self.frame_size = self.frames[0].shape[:2]
        cap.release()

    def compress(self, gop=1, q=50, encoding_method="quantization"):
        """Compress video frames using the specified method."""
        if not self.frames:
            raise ValueError("No frames to compress")
        
        if encoding_method.lower() == "intra":
            self.compress_intra(gop, q)
        else:
            self.compress_quantization(gop, q)

    def compress_quantization(self, gop=1, q=8):
        """Original quantization-based compression method."""
        # if q < 1 or q > 8:
        #     raise ValueError("Quantization bits must be between 1 and 8")

        quantized_frames = []
        for frame in self.frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            normalized = gray.astype(np.float32) / 255.0
            levels = 2 ** q
            quantized = np.round(normalized * (levels - 1)).astype(np.uint8)
            quantized_frames.append(quantized)

        self.metadata["bits"] = q
        self.metadata["frame_size"] = self.frame_size
        self.metadata["fps"] = self.fps
        self.metadata["encoding_method"] = "quantization"

        serialized = pickle.dumps(quantized_frames)
        self.compressed_data = zlib.compress(serialized)

        self.decompress()

        original_size = sum(f.nbytes for f in self.frames)
        compressed_size = len(self.compressed_data)
        self.ratio = original_size / compressed_size if compressed_size > 0 else float('inf')

        self.psnr = self.calculate_psnr(self.frames, self.decoded)

    def compress_intra(self, gop=1, q=50):
        """Compress frames using intra-frame compression (I-frame) with DCT, quantization, zigzag, and RLE using libraries."""
        if q < 1 or q > 100:
            raise ValueError("Quantization factor must be between 1 and 100")

        # Standard JPEG quantization matrix for 8x8 blocks
        quantization_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32)

        # Scale the quantization matrix based on q (quality factor: 1-100)
        if q < 50:
            scale = 5000 / q
        else:
            scale = 200 - 2 * q
        scale = max(1, min(200, scale))  # Clamp scale between 1 and 200
        q_matrix = np.clip(np.round(quantization_matrix * scale / 100), 1, 255)

        compressed_frames = []
        for frame in self.frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) - 128  # Center around 0
            height, width = gray.shape
            # Pad the frame to be divisible by 8x8 blocks
            pad_height = (8 - height % 8) % 8
            pad_width = (8 - width % 8) % 8
            padded = np.pad(gray, ((0, pad_height), (0, pad_width)), mode='constant')

            # Process 8x8 blocks
            compressed_blocks = []
            for i in range(0, padded.shape[0], 8):
                for j in range(0, padded.shape[1], 8):
                    block = padded[i:i+8, j:j+8]
                    # Apply DCT using cv2.dct
                    dct_block = cv2.dct(block)
                    # Quantize using the scaled matrix
                    quantized_block = np.round(dct_block / q_matrix).astype(np.int16)
                    # Zigzag scan using numpy indexing
                    zigzag = self.zigzag_scan(quantized_block)
                    # Run-length encoding
                    rle = self.run_length_encode(zigzag)
                    compressed_blocks.append(rle)

            compressed_frames.append(compressed_blocks)

        # Store metadata
        self.metadata["q_factor"] = q
        self.metadata["frame_size"] = self.frame_size
        self.metadata["padded_size"] = padded.shape
        self.metadata["fps"] = self.fps
        self.metadata["encoding_method"] = "intra"

        # Serialize and compress
        serialized = pickle.dumps(compressed_frames)
        self.compressed_data = zlib.compress(serialized)

        # Decompress for PSNR calculation
        self.decompress()

        # Calculate compression ratio
        original_size = sum(f.nbytes for f in self.frames)
        compressed_size = len(self.compressed_data)
        self.ratio = original_size / compressed_size if compressed_size > 0 else float('inf')

        # Calculate PSNR
        self.psnr = self.calculate_psnr(self.frames, self.decoded)

    def zigzag_scan(self, block):
        """Convert an 8x8 block into a 1D array using zigzag order with numpy."""
        if block.shape != (8, 8):
            raise ValueError("Block must be 8x8")
        
        # Define zigzag pattern indices
        indices = np.array([
            [0, 1, 5, 6, 14, 15, 27, 28],
            [2, 4, 7, 13, 16, 26, 29, 42],
            [3, 8, 12, 17, 25, 30, 41, 43],
            [9, 11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63]
        ])
        return block[indices]

    def run_length_encode(self, data):
        """Apply run-length encoding to a 1D array."""
        if not data:
            return []

        encoded = []
        count = 1
        current = data[0]
        for i in range(1, len(data)):
            if data[i] == current and count < 255:  # Limit count to 255 for byte encoding
                count += 1
            else:
                encoded.append((current, count))
                current = data[i]
                count = 1
        encoded.append((current, count))
        return encoded

    def inverse_zigzag_scan(self, data):
        """Convert a 1D zigzag array back into an 8x8 block."""
        if len(data) != 64:
            raise ValueError("Data must have 64 elements for an 8x8 block")
        
        indices = np.array([
            [0, 1, 5, 6, 14, 15, 27, 28],
            [2, 4, 7, 13, 16, 26, 29, 42],
            [3, 8, 12, 17, 25, 30, 41, 43],
            [9, 11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63]
        ])
        block = np.zeros((8, 8), dtype=np.float32)
        block[indices] = data
        return block

    def run_length_decode(self, rle_data):
        """Decode run-length encoded data into a 1D array."""
        decoded = []
        for value, count in rle_data:
            decoded.extend([value] * count)
        return decoded

    def calculate_psnr(self, original_frames, compressed_frames):
        """Calculate PSNR between original and compressed frames."""
        if len(original_frames) != len(compressed_frames):
            raise ValueError("Mismatch in number of original and compressed frames")
        
        psnr_values = []
        max_pixel_value = 255.0
        for orig, comp in zip(original_frames, compressed_frames):
            if len(comp.shape) == 2:  # Grayscale
                comp = cv2.cvtColor(comp, cv2.COLOR_GRAY2BGR)
            mse = np.mean((orig.astype(np.float32) - comp.astype(np.float32)) ** 2)
            if mse == 0:
                psnr_values.append(100.0)
            else:
                psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
                psnr_values.append(psnr)
        return np.mean(psnr_values) if psnr_values else 0.0

    def decompress(self):
        """Decompress the compressed video data."""
        if not self.compressed_data:
            raise ValueError("No compressed data available")

        try:
            decompressed = zlib.decompress(self.compressed_data)
            compressed_frames = pickle.loads(decompressed)
        except (zlib.error, pickle.PickleError) as e:
            raise ValueError(f"Decompression failed: {str(e)}")

        encoding_method = self.metadata.get("encoding_method", "quantization")
        if encoding_method == "quantization":
            bits = self.metadata.get("bits", 8)
            self.decoded = [
                np.uint8(f * 255 / (2 ** bits - 1)) for f in compressed_frames
            ]
        else:  # Intra-frame compression
            q = self.metadata.get("q_factor", 50)
            quantization_matrix = np.array([
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]
            ], dtype=np.float32)

            if q < 50:
                scale = 5000 / q
            else:
                scale = 200 - 2 * q
            scale = max(1, min(200, scale))
            q_matrix = np.clip(np.round(quantization_matrix * scale / 100), 1, 255)

            self.decoded = []
            padded_shape = self.metadata["padded_size"]
            for compressed_blocks in compressed_frames:
                frame = np.zeros(padded_shape, dtype=np.float32)
                block_idx = 0
                for i in range(0, padded_shape[0], 8):
                    for j in range(0, padded_shape[1], 8):
                        rle = compressed_blocks[block_idx]
                        zigzag = self.run_length_decode(rle)
                        quantized_block = self.inverse_zigzag_scan(zigzag)
                        dct_block = quantized_block * q_matrix
                        block = cv2.idct(dct_block)
                        frame[i:i+8, j:j+8] = block
                        block_idx += 1
                frame = frame[:self.frame_size[0], :self.frame_size[1]] + 128
                self.decoded.append(np.clip(frame, 0, 255).astype(np.uint8))

    def save_video(self, path):
        """Save the decoded video to a file."""
        if not self.decoded:
            raise ValueError("No decoded video available")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(path, fourcc, self.fps, self.frame_size, isColor=False)
        
        for frame in self.decoded:
            out.write(frame)
        out.release()

    def save_bitstream(self, path):
        """Save the compressed bitstream to a file."""
        if not self.compressed_data:
            raise ValueError("No compressed data available")

        with open(path, 'wb') as f:
            f.write(self.compressed_data)