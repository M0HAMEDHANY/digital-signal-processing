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

# Updated video_processor.py with DCT, quantization, zig-zag scan (I-frames) and motion estimation (P-frames)
import cv2
import numpy as np
import pickle
import zlib
import os


def blockify(img, block_size=8):
    h, w = img.shape
    return img.reshape(h // block_size, block_size, -1, block_size).swapaxes(1, 2).reshape(-1, block_size, block_size)

def unblockify(blocks, h, w, block_size=8):
    blocks = blocks.reshape(h // block_size, w // block_size, block_size, block_size).swapaxes(1, 2)
    return blocks.reshape(h, w)

def quantize(blocks, q=10):
    return np.round(blocks / q).astype(np.int16)

def dequantize(blocks, q=10):
    return (blocks * q).astype(np.float32)

def dct2(block):
    return cv2.dct(block.astype(np.float32))

def idct2(block):
    return cv2.idct(block.astype(np.float32))

def zigzag(block):
    index_order = sorted(((x, y) for x in range(8) for y in range(8)), key=lambda s: s[0]+s[1] if (s[0]+s[1]) % 2 == 0 else -s[0])
    return np.array([block[i, j] for i, j in index_order])

def izigzag(array):
    index_order = sorted(((x, y) for x in range(8) for y in range(8)), key=lambda s: s[0]+s[1] if (s[0]+s[1]) % 2 == 0 else -s[0])
    block = np.zeros((8, 8), dtype=array.dtype)
    for val, (i, j) in zip(array, index_order):
        block[i, j] = val
    return block


def motion_estimate(ref, target, block_size=8, search_range=4):
    h, w = ref.shape
    mv = []
    residuals = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            best_match = (0, 0)
            min_error = float('inf')
            block = target[i:i+block_size, j:j+block_size]
            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    y, x = i + dy, j + dx
                    if y < 0 or x < 0 or y + block_size > h or x + block_size > w:
                        continue
                    ref_block = ref[y:y+block_size, x:x+block_size]
                    error = np.sum(np.abs(ref_block - block))
                    if error < min_error:
                        min_error = error
                        best_match = (dy, dx)
            dy, dx = best_match
            y, x = i + dy, j + dx
            ref_block = ref[y:y+block_size, x:x+block_size]
            residual = block - ref_block
            mv.append((dy, dx))
            residuals.append(residual)
    return mv, residuals


def motion_compensate(ref, mv, residuals, block_size=8):
    h, w = ref.shape
    rec = np.zeros_like(ref)
    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            dy, dx = mv[idx]
            y, x = i + dy, j + dx
            ref_block = ref[y:y+block_size, x:x+block_size]
            rec[i:i+block_size, j:j+block_size] = ref_block + residuals[idx]
            idx += 1
    return rec


class VideoProcessor:
    def __init__(self):
        self.frames = []
        self.frames_yuv = []
        self.decoded = []
        self.compressed_data = None
        self.fps = 30
        self.frame_size = (0, 0)
        self.metadata = {}
        self.ratio = None
        self.psnr = None

    def load(self, path):
        cap = cv2.VideoCapture(path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            self.frames_yuv.append(yuv)
        cap.release()
        self.frame_size = (self.frames[0].shape[1], self.frames[0].shape[0])

    def compress(self, gop=10, q=10, encoding_method="intra"):
        if encoding_method.lower() == "intra":
            self.compress_intra(q)
        elif encoding_method.lower() == "pframe":
            self.compress_inter(gop, q)

        self.decoded = self.convert_yuv_to_bgr(self.decoded)
        self.ratio = self._compression_ratio()
        self.psnr = self.calculate_psnr(self.frames_yuv, self.decoded)

    def compress_intra(self, q):
        compressed = []
        for frame in self.frames_yuv:
            y = frame[:, :, 0]
            h, w = y.shape
            blocks = blockify(y)
            dct_blocks = np.array([dct2(b) for b in blocks])
            q_blocks = quantize(dct_blocks, q)
            zz_blocks = np.array([zigzag(b) for b in q_blocks])
            compressed.append(zz_blocks)

        serialized = pickle.dumps({"type": "intra", "data": compressed})
        self.compressed_data = zlib.compress(serialized)

        # Decompress
        raw = zlib.decompress(self.compressed_data)
        decoded_blocks = pickle.loads(raw)["data"]
        self.decoded = []
        for zz_blocks, frame in zip(decoded_blocks, self.frames_yuv):
            blocks = np.array([izigzag(b) for b in zz_blocks])
            dq_blocks = dequantize(blocks, q)
            idct_blocks = np.array([idct2(b) for b in dq_blocks])
            y_rec = unblockify(idct_blocks, frame.shape[0], frame.shape[1])
            rec = np.stack([y_rec, frame[:, :, 1], frame[:, :, 2]], axis=2)
            self.decoded.append(rec.astype(np.uint8))

    def compress_inter(self, gop, q):
        compressed = []
        ref = self.frames_yuv[0][:, :, 0]
        h, w = ref.shape

        # Intra compress first frame
        blocks = blockify(ref)
        dct_blocks = np.array([dct2(b) for b in blocks])
        q_blocks = quantize(dct_blocks, q)
        zz_blocks = np.array([zigzag(b) for b in q_blocks])
        compressed.append(("I", zz_blocks))

        for i in range(1, len(self.frames_yuv)):
            y = self.frames_yuv[i][:, :, 0]
            mv, res = motion_estimate(ref, y)
            res_dct = [dct2(r) for r in res]
            q_res = [quantize(b, q) for b in res_dct]
            zz_res = [zigzag(b) for b in q_res]
            compressed.append(("P", mv, zz_res))
            ref = y

        serialized = pickle.dumps({"type": "inter", "data": compressed})
        self.compressed_data = zlib.compress(serialized)

        # Decompress
        raw = zlib.decompress(self.compressed_data)
        data = pickle.loads(raw)["data"]
        self.decoded = []
        ref = None

        for item, frame in zip(data, self.frames_yuv):
            if item[0] == "I":
                zz_blocks = item[1]
                blocks = np.array([izigzag(b) for b in zz_blocks])
                dq_blocks = dequantize(blocks, q)
                idct_blocks = np.array([idct2(b) for b in dq_blocks])
                y_rec = unblockify(idct_blocks, frame.shape[0], frame.shape[1])
                ref = y_rec.copy()
            else:
                mv, zz_res = item[1], item[2]
                res_blocks = [izigzag(b) for b in zz_res]
                dq_res = [dequantize(b, q) for b in res_blocks]
                res_idct = [idct2(b) for b in dq_res]
                y_rec = motion_compensate(ref, mv, res_idct)
                ref = y_rec.copy()

            rec = np.stack([ref, frame[:, :, 1], frame[:, :, 2]], axis=2)
            self.decoded.append(rec.astype(np.uint8))

    def convert_yuv_to_bgr(self, yuv_frames):
        return [cv2.cvtColor(f, cv2.COLOR_YUV2BGR) for f in yuv_frames]

    def calculate_psnr(self, originals, recs):
        psnr_vals = []
        for o, r in zip(originals, recs):
            o_bgr = cv2.cvtColor(o, cv2.COLOR_YUV2BGR)
            psnr_vals.append(cv2.PSNR(o_bgr, r))
        return np.mean(psnr_vals)

    def _compression_ratio(self):
        orig_size = sum(f.nbytes for f in self.frames_yuv)
        comp_size = len(self.compressed_data)
        return orig_size / comp_size if comp_size > 0 else float("inf")

    def save_video(self, path):
        bgr_frames = self.convert_yuv_to_bgr(self.decoded)
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"XVID"), self.fps, self.frame_size)
        for f in bgr_frames:
            out.write(f)
        out.release()

    def save_bitstream(self, path):
        with open(path, "wb") as f:
            f.write(self.compressed_data)
        np.savez(path + ".metadata", metadata=self.metadata, fps=self.fps, frame_size=self.frame_size)
