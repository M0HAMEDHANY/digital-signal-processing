import cv2
import numpy as np
import pickle
import zlib
import os


def blockify(img, block_size=8):
    h, w = img.shape
    # Ensure dimensions are multiples of block_size
    h_pad = h - (h % block_size) if h % block_size != 0 else h
    w_pad = w - (w % block_size) if w % block_size != 0 else w
    img_padded = img[:h_pad, :w_pad]
    return img_padded.reshape(h_pad // block_size, block_size, -1, block_size).swapaxes(1, 2).reshape(-1, block_size, block_size)

def unblockify(blocks, h, w, block_size=8):
    # Ensure dimensions are multiples of block_size
    h_pad = h - (h % block_size) if h % block_size != 0 else h
    w_pad = w - (w % block_size) if w % block_size != 0 else w
    blocks = blocks.reshape(h_pad // block_size, w_pad // block_size, block_size, block_size).swapaxes(1, 2)
    result = blocks.reshape(h_pad, w_pad)
    # If original dimensions were larger, pad with zeros
    if h_pad < h or w_pad < w:
        full_result = np.zeros((h, w), dtype=result.dtype)
        full_result[:h_pad, :w_pad] = result
        return full_result
    return result

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
    h, w = target.shape  # Use target dimensions to ensure proper size
    # Ensure ref and target have the same dimensions
    ref = cv2.resize(ref, (w, h)) if ref.shape != target.shape else ref
    
    mv = []
    residuals = []
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
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
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            if idx >= len(mv):  # Safety check
                break
            dy, dx = mv[idx]
            y, x = i + dy, j + dx
            # Ensure coordinates are within bounds
            if y < 0 or x < 0 or y + block_size > h or x + block_size > w:
                # Use zero motion vector if out of bounds
                y, x = i, j
            ref_block = ref[y:y+block_size, x:x+block_size]
            if idx < len(residuals):  # Safety check
                rec[i:i+block_size, j:j+block_size] = ref_block + residuals[idx]
            else:
                rec[i:i+block_size, j:j+block_size] = ref_block
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
        if self.frames:
            self.frame_size = (self.frames[0].shape[1], self.frames[0].shape[0])

    def compress(self, gop=10, q=10, encoding_method="intra"):
        if not self.frames_yuv:
            raise ValueError("No frames loaded. Please load a video first.")

        # Convert encoding method to lowercase for case-insensitive comparison
        encoding_method = encoding_method.lower()
        
        if encoding_method == "intra":
            self.compress_intra(q)
        elif encoding_method in ["p-frame", "pframe"]:
            self.compress_inter(gop, q)
        else:
            raise ValueError(f"Unsupported encoding method: {encoding_method}")

        # Convert decoded YUV frames back to BGR
        bgr_decoded = self.convert_yuv_to_bgr(self.decoded)
        self.decoded = bgr_decoded
        
        # Calculate compression ratio and PSNR
        self.ratio = self._compression_ratio()
        self.psnr = self.calculate_psnr()

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
        if not self.frames_yuv:
            raise ValueError("No video loaded. Please load a video first.")

        compressed = []

        # Process frames in groups of pictures (GOP)
        for i in range(0, len(self.frames_yuv), gop):
            # I-frame: the first frame in each GOP
            i_frame = self.frames_yuv[i]
            y = i_frame[:, :, 0].copy()  # Y channel
            h, w = y.shape
            
            # Compress I-frame
            blocks = blockify(y)
            dct_blocks = np.array([dct2(b) for b in blocks])
            q_blocks = quantize(dct_blocks, q)
            zz_blocks = np.array([zigzag(b) for b in q_blocks])
            compressed.append(("I", zz_blocks))
            
            # Decode I-frame to use as reference
            blocks_dec = np.array([izigzag(b) for b in zz_blocks])
            dq_blocks = dequantize(blocks_dec, q)
            idct_blocks = np.array([idct2(b) for b in dq_blocks])
            ref = unblockify(idct_blocks, h, w)
            
            # Process P-frames
            end_idx = min(i + gop, len(self.frames_yuv))
            for j in range(i + 1, end_idx):
                curr_frame = self.frames_yuv[j]
                curr_y = curr_frame[:, :, 0].copy()
                
                # Motion estimation and compensation
                mv, res = motion_estimate(ref, curr_y)
                
                # Process residuals
                res_dct = [dct2(r) for r in res]
                q_res = [quantize(r, q) for r in res_dct]
                zz_res = [zigzag(r) for r in q_res]
                
                compressed.append(("P", mv, zz_res))
                
                # Update reference frame for next P-frame
                # Decode current frame
                res_dec = [izigzag(r) for r in zz_res]
                dq_res = [dequantize(r, q) for r in res_dec]
                idct_res = [idct2(r) for r in dq_res]
                
                # Motion compensation to get reconstructed frame
                ref = motion_compensate(ref, mv, idct_res)

        # Store compressed data
        serialized = pickle.dumps({"type": "inter", "gop": gop, "data": compressed})
        self.compressed_data = zlib.compress(serialized)

        # Decode all frames
        self.decoded = self.decode_compressed_data(q)

    def decode_compressed_data(self, q):
        if not self.compressed_data:
            raise ValueError("No compressed data available")
            
        raw = zlib.decompress(self.compressed_data)
        data = pickle.loads(raw)["data"]
        decoded_frames = []
        ref = None
        
        for item in data:
            frame_type = item[0]
            
            if frame_type == "I":
                # Decode I-frame
                zz_blocks = item[1]
                blocks = np.array([izigzag(b) for b in zz_blocks])
                dq_blocks = dequantize(blocks, q)
                idct_blocks = np.array([idct2(b) for b in dq_blocks])
                
                # Get original frame dimensions from YUV frame
                frame_idx = len(decoded_frames)
                if frame_idx < len(self.frames_yuv):
                    h, w = self.frames_yuv[frame_idx][:, :, 0].shape
                else:
                    h, w = self.frame_size[1], self.frame_size[0]
                
                y_rec = unblockify(idct_blocks, h, w)
                ref = y_rec.copy()
                
                # Create full YUV frame
                if frame_idx < len(self.frames_yuv):
                    orig_frame = self.frames_yuv[frame_idx]
                    u_channel = orig_frame[:, :, 1].copy()
                    v_channel = orig_frame[:, :, 2].copy()
                    rec = np.stack([y_rec, u_channel, v_channel], axis=2)
                    decoded_frames.append(rec.astype(np.uint8))
                
            elif frame_type == "P":
                # Decode P-frame
                if ref is None:
                    raise RuntimeError("Missing reference frame for P-frame decoding")
                    
                mv, zz_res = item[1], item[2]
                
                # Decode residual
                res_blocks = [izigzag(b) for b in zz_res]
                dq_res = [dequantize(b, q) for b in res_blocks]
                idct_res = [idct2(b) for b in dq_res]
                
                # Apply motion compensation
                y_rec = motion_compensate(ref, mv, idct_res)
                ref = y_rec.copy()
                
                # Create full YUV frame
                frame_idx = len(decoded_frames)
                if frame_idx < len(self.frames_yuv):
                    orig_frame = self.frames_yuv[frame_idx]
                    u_channel = orig_frame[:, :, 1].copy()
                    v_channel = orig_frame[:, :, 2].copy()
                    rec = np.stack([y_rec, u_channel, v_channel], axis=2)
                    decoded_frames.append(rec.astype(np.uint8))
        
        return decoded_frames

    def convert_yuv_to_bgr(self, yuv_frames):
        return [cv2.cvtColor(f, cv2.COLOR_YUV2BGR) for f in yuv_frames]

    def calculate_psnr(self):
        if not self.frames or not self.decoded:
            return 0.0
            
        # Make sure both lists have the same length for comparison
        min_len = min(len(self.frames), len(self.decoded))
        psnr_vals = []
        
        for i in range(min_len):
            # Both should be in BGR format
            orig = self.frames[i]
            rec = self.decoded[i]
            
            # Ensure same dimensions
            if orig.shape != rec.shape:
                rec = cv2.resize(rec, (orig.shape[1], orig.shape[0]))
                
            # Calculate PSNR
            try:
                psnr = cv2.PSNR(orig, rec)
                psnr_vals.append(psnr)
            except Exception:
                # Skip if PSNR calculation fails
                continue
                
        return np.mean(psnr_vals) if psnr_vals else 0.0

    def _compression_ratio(self):
        if not self.compressed_data or not self.frames_yuv:
            return 0.0
            
        orig_size = sum(f.nbytes for f in self.frames_yuv)
        comp_size = len(self.compressed_data)
        return orig_size / comp_size if comp_size > 0 else float("inf")

    def save_video(self, path):
        if not self.decoded:
            raise ValueError("No decoded frames available. Run compression first.")
            
        # Ensure all frames have the same dimensions as frame_size
        h, w = self.frame_size[1], self.frame_size[0]
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"XVID"), self.fps, (w, h))
        
        for frame in self.decoded:
            # Resize if necessary
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            out.write(frame)
            
        out.release()

    def save_bitstream(self, path):
        if not self.compressed_data:
            raise ValueError("No compressed data available. Run compression first.")
            
        with open(path, "wb") as f:
            f.write(self.compressed_data)
            
        # Save metadata
        metadata = {
            "fps": self.fps,
            "frame_size": self.frame_size,
            "ratio": self.ratio,
            "psnr": self.psnr
        }
        np.savez(path + ".metadata", **metadata)