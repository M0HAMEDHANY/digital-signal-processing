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
from multiprocessing import Pool


def blockify(img, block_size=8):
    """
    Divide an image into non-overlapping blocks for block-based processing.

    This is a fundamental operation in image/video compression where each block
    is processed independently (e.g., with DCT transform).

    Args:
        img (np.ndarray): Input image (2D array)
        block_size (int): Size of square blocks (default: 8)

    Returns:
        np.ndarray: Array of blocks with shape (n_blocks, block_size, block_size)

    Raises:
        ValueError: If image dimensions aren't divisible by block_size
    """
    h, w = img.shape

    if h % block_size != 0 or w % block_size != 0:
        raise ValueError(
            f"Image dimensions ({h}, {w}) are not divisible by block size {block_size}. "
            "Resize the image or adjust the block size."
        )
    img = img[:h, :w]
    return (
        img.reshape(h // block_size, block_size, -1, block_size)
        .swapaxes(1, 2)
        .reshape(-1, block_size, block_size)
    )


def unblockify(blocks, h, w, block_size=8):
    """
    Reconstruct an image from its block representation.

    This is the inverse operation of blockify(), used during decompression
    to restore the spatial arrangement of pixel blocks.

    Args:
        blocks (np.ndarray): Array of blocks with shape (n_blocks, block_size, block_size)
        h (int): Original image height
        w (int): Original image width
        block_size (int): Size of square blocks (default: 8)

    Returns:
        np.ndarray: Reconstructed image with shape (h, w)

    Raises:
        ValueError: If blocks can't be reshaped into the specified dimensions
    """
    if blocks.size != (h // block_size) * (w // block_size) * block_size * block_size:
        raise ValueError(
            f"Cannot reshape blocks of size {blocks.size} into shape ({h}, {w}) "
            f"with block size {block_size}. Check input dimensions."
        )
    blocks = blocks.reshape(
        h // block_size, w // block_size, block_size, block_size
    ).swapaxes(1, 2)
    return blocks.reshape(h, w)


def quantize(blocks, q=10):
    """
    Quantize DCT coefficients to reduce precision and increase compressibility.

    Higher q values result in more aggressive quantization (lower quality, higher compression).
    This is a lossy operation that reduces the precision of transform coefficients.

    Args:
        blocks (np.ndarray): Array of DCT coefficient blocks
        q (int): Quantization factor (higher = more compression, lower quality)

    Returns:
        np.ndarray: Quantized coefficient blocks as int16
    """
    return np.round(blocks / q).astype(np.int16)


def dequantize(blocks, q=10):
    """
    Dequantize coefficients during decompression.

    This reverses the quantization step but cannot recover the original precision
    (information is permanently lost during quantization).

    Args:
        blocks (np.ndarray): Array of quantized coefficient blocks
        q (int): The same quantization factor used during compression

    Returns:
        np.ndarray: Dequantized coefficient blocks as float32
    """
    return (blocks * q).astype(np.float32)


def dct2(block):
    """
    Apply 2D Discrete Cosine Transform to a block.

    DCT concentrates image energy in the upper-left (low frequency) coefficients,
    making it ideal for compression as many high-frequency coefficients
    can be quantized heavily or discarded.

    Args:
        block (np.ndarray): Input image block

    Returns:
        np.ndarray: DCT coefficients
    """
    return cv2.dct(block.astype(np.float32))


def idct2(block):
    """
    Apply inverse 2D Discrete Cosine Transform.

    Converts DCT coefficients back to pixel values during decompression.

    Args:
        block (np.ndarray): DCT coefficient block

    Returns:
        np.ndarray: Reconstructed image block
    """
    return cv2.idct(block.astype(np.float32))


def zigzag(block):
    """
    Reorder a block using zigzag scanning pattern.

    Zigzag scanning creates a 1D sequence by traversing the 2D block in a zigzag pattern,
    which groups low-frequency coefficients (typically non-zero) together, improving
    entropy coding efficiency.

    Args:
        block (np.ndarray): 8x8 input block

    Returns:
        np.ndarray: 1D array of coefficients in zigzag order
    """
    index_order = sorted(
        ((x, y) for x in range(8) for y in range(8)),
        key=lambda s: s[0] + s[1] if (s[0] + s[1]) % 2 == 0 else -s[0],
    )
    return np.array([block[i, j] for i, j in index_order])


def izigzag(array):
    """
    Reverse zigzag scanning to reconstruct a 2D block.

    Converts the 1D sequence back to a 2D block during decompression.

    Args:
        array (np.ndarray): 1D array of coefficients in zigzag order

    Returns:
        np.ndarray: Reconstructed 8x8 block
    """
    index_order = sorted(
        ((x, y) for x in range(8) for y in range(8)),
        key=lambda s: s[0] + s[1] if (s[0] + s[1]) % 2 == 0 else -s[0],
    )
    block = np.zeros((8, 8), dtype=array.dtype)
    for val, (i, j) in zip(array, index_order):
        block[i, j] = val
    return block


def motion_estimate(ref, target, block_size=8, search_range=32):
    """
    Perform motion estimation using a hierarchical search algorithm.

    Finds the best matching block in a reference frame for each block in the target frame.
    This implementation uses a logarithmic step search to efficiently find matches.

    Args:
        ref (np.ndarray): Reference frame
        target (np.ndarray): Target frame
        block_size (int): Size of blocks for motion estimation
        search_range (int): Maximum pixel distance to search

    Returns:
        tuple: (motion_vectors, residuals) where motion_vectors is a list of (dy, dx)
              displacement pairs and residuals contains the difference blocks
    """
    h, w = ref.shape
    mv = []
    residuals = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = target[i : i + block_size, j : j + block_size]
            if block.shape != (block_size, block_size):
                continue
            best_match = (0, 0)
            min_error = float("inf")

            # Hierarchical search with logarithmic step sizes
            step = search_range // 2
            while step >= 1:
                for dy in range(-step, step + 1, step):
                    for dx in range(-step, step + 1, step):
                        y, x = i + best_match[0] + dy, j + best_match[1] + dx
                        if y < 0 or x < 0 or y + block_size > h or x + block_size > w:
                            continue
                        ref_block = ref[y : y + block_size, x : x + block_size]
                        if ref_block.shape != (block_size, block_size):
                            continue
                        error = np.sum(np.abs(ref_block - block))
                        if error < min_error:
                            min_error = error
                            best_match = (best_match[0] + dy, best_match[1] + dx)
                step //= 2

            dy, dx = best_match
            y, x = i + dy, j + dx
            ref_block = ref[y : y + block_size, x : x + block_size]
            if ref_block.shape == block.shape:
                residual = block - ref_block
                mv.append((dy, dx))
                residuals.append(residual)
    return mv, residuals


def motion_estimate_block(args):
    """
    Process a single block for motion estimation (used in parallel processing).

    Args:
        args (tuple): Contains (reference frame, target frame, block y, block x,
                      block size, search range)

    Returns:
        tuple: ((dy, dx), residual) - motion vector and residual block
    """
    ref, target, i, j, block_size, search_range = args
    block = target[i : i + block_size, j : j + block_size]
    best_match = (0, 0)
    min_error = float("inf")

    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            y, x = i + dy, j + dx
            if (
                y < 0
                or x < 0
                or y + block_size > ref.shape[0]
                or x + block_size > ref.shape[1]
            ):
                continue
            ref_block = ref[y : y + block_size, x : x + block_size]
            error = np.sum(np.abs(ref_block - block))
            if error < min_error:
                min_error = error
                best_match = (dy, dx)

    dy, dx = best_match
    y, x = i + dy, j + dx
    ref_block = ref[y : y + block_size, x : x + block_size]
    residual = block - ref_block
    return (dy, dx), residual


def motion_estimate_parallel(ref, target, block_size=8, search_range=16):
    """
    Parallel implementation of motion estimation using multiple CPU cores.

    This function distributes blocks across available CPU cores to speed up
    the computationally intensive motion estimation process.

    Args:
        ref (np.ndarray): Reference frame
        target (np.ndarray): Target frame
        block_size (int): Size of blocks for motion estimation
        search_range (int): Maximum pixel distance to search

    Returns:
        tuple: (motion_vectors, residuals)
    """
    h, w = ref.shape
    args = [
        (ref, target, i, j, block_size, search_range)
        for i in range(0, h, block_size)
        for j in range(0, w, block_size)
    ]
    with Pool() as pool:
        results = pool.map(motion_estimate_block, args)
    mv, residuals = zip(*results)
    return list(mv), list(residuals)


def motion_compensate(ref, mv, residuals, block_size=8):
    """
    Reconstruct a frame using motion compensation.

    Given a reference frame, motion vectors, and residuals, this function
    reconstructs the target frame during decompression of inter-frames.

    Args:
        ref (np.ndarray): Reference frame
        mv (list): List of motion vectors (dy, dx)
        residuals (list): List of residual blocks
        block_size (int): Size of blocks

    Returns:
        np.ndarray: Reconstructed frame
    """
    h, w = ref.shape
    rec = np.zeros_like(ref, dtype=np.float32)
    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            dy, dx = mv[idx]
            y, x = i + dy, j + dx
            ref_block = ref[y : y + block_size, x : x + block_size]
            residual = residuals[idx]
            if ref_block.shape == residual.shape:
                rec[i : i + block_size, j : j + block_size] = ref_block + residual
            idx += 1
    return rec


class VideoProcessor:
    """
    Main class for video compression and processing.

    Implements both intra-frame (I-frame) and inter-frame (P-frame) compression
    similar to modern video codecs but in a simplified form for educational purposes.
    """

    def __init__(self):
        """Initialize the video processor with empty frames and default parameters"""
        self.frames = []  # Original frames (BGR format)
        self.frames_yuv = []  # Frames converted to YUV color space
        self.decoded = []  # Reconstructed frames after compression/decompression
        self.compressed_data = None  # Binary compressed data
        self.fps = 30  # Default frames per second
        self.frame_size = (0, 0)  # Width x height
        self.metadata = {}  # Additional information about the video
        self.ratio = None  # Compression ratio
        self.psnr = None  # Peak Signal-to-Noise Ratio (quality metric)

    def load(self, path, scale=0.5):
        """
        Load a video file and convert frames to appropriate format.

        Args:
            path (str): Path to the video file
            scale (float): Scale factor for resizing (default: 0.5)
        """
        cap = cv2.VideoCapture(path)
        self.frames = []
        self.frames_yuv = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if scale != 1.0:
                frame = cv2.resize(
                    frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                )
            h, w, _ = frame.shape
            if h % 8 != 0 or w % 8 != 0:
                # Ensure dimensions are divisible by 8 for block processing
                h = (h // 8) * 8
                w = (w // 8) * 8
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            self.frames.append(frame)
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            self.frames_yuv.append(yuv)
        cap.release()
        self.frame_size = (self.frames[0].shape[1], self.frames[0].shape[0])

    def compress(self, gop=10, q=10, encoding_method="intra"):
        """
        Compress the video using the specified method.

        Args:
            gop (int): Group of Pictures size - how many frames between I-frames
            q (int): Quantization parameter (higher = more compression)
            encoding_method (str): "intra" for I-frame only, "pframe" for P-frame encoding

        Raises:
            ValueError: If no frames are loaded or invalid encoding method
        """
        if not self.frames_yuv:
            raise ValueError("No frames loaded. Please load a video first.")

        if encoding_method.lower() == "intra":
            self.compress_intra(q)
        elif encoding_method.lower() == "pframe":
            self.compress_inter(gop, q)
        else:
            raise ValueError("Invalid encoding method. Use 'intra' or 'pframe'.")

        self.decoded = self.convert_yuv_to_bgr(self.decoded)
        self.ratio = self._compression_ratio()
        self.psnr = self.calculate_psnr(self.frames_yuv, self.decoded)

    def compress_intra(self, q):
        """
        Compress using intra-frame coding only (similar to MJPEG).

        Each frame is compressed independently using DCT transform,
        quantization, and zigzag scanning.

        Args:
            q (int): Quantization parameter
        """
        compressed = []
        for frame in self.frames_yuv:
            # Process only the Y channel (luminance) for compression
            y = frame[:, :, 0]
            h, w = y.shape
            blocks = blockify(y)
            dct_blocks = np.array([dct2(b) for b in blocks])
            q_blocks = quantize(dct_blocks, q)
            zz_blocks = np.array([zigzag(b) for b in q_blocks])
            compressed.append(zz_blocks)

        # Serialize and compress the data using zlib
        serialized = pickle.dumps({"type": "intra", "data": compressed, "q": q})
        self.compressed_data = zlib.compress(serialized, level=9)

        # Decompress for quality assessment
        raw = zlib.decompress(self.compressed_data)
        decoded_blocks = pickle.loads(raw)["data"]
        self.decoded = []
        for zz_blocks, frame in zip(decoded_blocks, self.frames_yuv):
            # Reverse the compression steps
            blocks = np.array([izigzag(b) for b in zz_blocks])
            dq_blocks = dequantize(blocks, q)
            idct_blocks = np.array([idct2(b) for b in dq_blocks])
            try:
                y_rec = unblockify(idct_blocks, frame.shape[0], frame.shape[1])
            except ValueError as e:
                raise ValueError(f"Error during decompression: {e}")
            # Reconstruct using compressed Y and original UV channels
            rec = np.stack([y_rec, frame[:, :, 1], frame[:, :, 2]], axis=2)
            self.decoded.append(rec.astype(np.uint8))

    def compress_inter(self, gop, q, search_range=8):
        """
        Compress using inter-frame prediction (P-frames).

        This method compresses the video using a combination of I-frames and P-frames:
        - I-frames are compressed independently (like JPEG)
        - P-frames are encoded as motion vectors + residuals from a reference frame

        Args:
            gop (int): Group of Pictures size (distance between I-frames)
            q (int): Quantization parameter
            search_range (int): Pixel range for motion search
        """
        compressed = []
        for i in range(0, len(self.frames_yuv), gop):
            # Compress I-frame (key frame)
            y = self.frames_yuv[i][:, :, 0]
            blocks = blockify(y)
            dct_blocks = np.array([dct2(b) for b in blocks])
            q_blocks = quantize(dct_blocks, q)
            zz_blocks = np.array([zigzag(b) for b in q_blocks])
            compressed.append(("I", zz_blocks))

            # Use this I-frame as reference for following P-frames
            ref = y.copy()
            for j in range(i + 1, min(i + gop, len(self.frames_yuv))):
                y_p = self.frames_yuv[j][:, :, 0]
                # Compute motion vectors and residuals
                mv, res = motion_estimate(ref, y_p, search_range=search_range)
                res_dct = [dct2(r) for r in res]
                q_res = [quantize(b, q) for b in res_dct]
                zz_res = [zigzag(b) for b in q_res]
                compressed.append(("P", mv, zz_res))

        # Serialize and compress the data
        serialized = pickle.dumps(
            {"type": "inter", "data": compressed, "gop": gop, "q": q}
        )
        self.compressed_data = zlib.compress(serialized, level=9)

        # Decompress for quality assessment
        raw = zlib.decompress(self.compressed_data)
        data = pickle.loads(raw)["data"]
        self.decoded = []
        ref = None

        for item, frame in zip(data, self.frames_yuv):
            if item[0] == "I":  # I-frame
                zz_blocks = item[1]
                blocks = np.array([izigzag(b) for b in zz_blocks])
                dq_blocks = dequantize(blocks, q)
                idct_blocks = np.array([idct2(b) for b in dq_blocks])
                y_rec = unblockify(idct_blocks, frame.shape[0], frame.shape[1])
                ref = y_rec.copy()
            else:  # P-frame
                mv, zz_res = item[1], item[2]
                res_blocks = [izigzag(b) for b in zz_res]
                dq_res = [dequantize(b, q) for b in res_blocks]
                res_idct = [idct2(b) for b in dq_res]
                y_rec = motion_compensate(ref, mv, res_idct)
                ref = y_rec.copy()

            # Reconstruct frame with compressed Y and original UV
            rec = np.stack([ref, frame[:, :, 1], frame[:, :, 2]], axis=2)
            self.decoded.append(rec.astype(np.uint8))

    def convert_yuv_to_bgr(self, yuv_frames):
        """
        Convert YUV frames to BGR color space for display.

        Args:
            yuv_frames (list): List of frames in YUV color space

        Returns:
            list: List of frames in BGR color space
        """
        return [cv2.cvtColor(f, cv2.COLOR_YUV2BGR) for f in yuv_frames]

    def calculate_psnr(self, originals, recs):
        """
        Calculate Peak Signal-to-Noise Ratio between original and reconstructed frames.

        PSNR is a standard quality metric for lossy compression.
        Higher values indicate better quality.

        Args:
            originals (list): Original video frames
            recs (list): Reconstructed video frames

        Returns:
            float: Average PSNR across all frames in dB
        """
        psnr_vals = []
        for o, r in zip(originals, recs):
            o_bgr = cv2.cvtColor(o, cv2.COLOR_YUV2BGR)
            r_bgr = cv2.cvtColor(r, cv2.COLOR_YUV2BGR) if r.shape[2] == 3 else r
            if o_bgr.shape != r_bgr.shape:
                r_bgr = cv2.resize(r_bgr, (o_bgr.shape[1], o_bgr.shape[0]))
            psnr_vals.append(cv2.PSNR(o_bgr, r_bgr))
        return np.mean(psnr_vals)

    def _compression_ratio(self):
        """
        Calculate the compression ratio between original and compressed video.

        Returns:
            float: Compression ratio (original size / compressed size)
        """
        orig_size = sum(f.nbytes for f in self.frames_yuv)
        comp_size = len(self.compressed_data)
        return orig_size / comp_size if comp_size > 0 else float("inf")

    def save_video(self, path):
        """
        Save compressed video to a file.

        Args:
            path (str): Output file path

        Raises:
            ValueError: If codec is not supported
        """
        bgr_frames = self.convert_yuv_to_bgr(self.decoded)

        fourcc = (
            cv2.VideoWriter_fourcc(*"MP4V")
            if cv2.VideoWriter_fourcc(*"MP4V") != -1
            else cv2.VideoWriter_fourcc(*"XVID")
        )
        out = cv2.VideoWriter(path, fourcc, self.fps, self.frame_size, isColor=True)
        if not out.isOpened():
            raise ValueError("Could not create video file. Check codec support.")
        for f in bgr_frames:
            out.write(f)
        out.release()

    def save_bitstream(self, path):
        """
        Save the compressed bitstream and metadata to files.

        This allows for future decompression without recomputing.

        Args:
            path (str): Output file path
        """
        with open(path, "wb") as f:
            f.write(self.compressed_data)
        np.savez(
            path + ".metadata",
            metadata=self.metadata,
            fps=self.fps,
            frame_size=self.frame_size,
        )
