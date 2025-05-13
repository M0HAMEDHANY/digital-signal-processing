import cv2
import numpy as np
import pickle
import zlib
from multiprocessing import Pool


def blockify(img, block_size=8):
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
    return np.round(blocks / q).astype(np.int16)


def dequantize(blocks, q=10):
    return (blocks * q).astype(np.float32)


def dct2(block):
    return cv2.dct(block.astype(np.float32))


def idct2(block):
    return cv2.idct(block.astype(np.float32))


def zigzag(block):
    index_order = sorted(
        ((x, y) for x in range(8) for y in range(8)),
        key=lambda s: s[0] + s[1] if (s[0] + s[1]) % 2 == 0 else -s[0],
    )
    return np.array([block[i, j] for i, j in index_order])


def izigzag(array):
    index_order = sorted(
        ((x, y) for x in range(8) for y in range(8)),
        key=lambda s: s[0] + s[1] if (s[0] + s[1]) % 2 == 0 else -s[0],
    )
    block = np.zeros((8, 8), dtype=array.dtype)
    for val, (i, j) in zip(array, index_order):
        block[i, j] = val
    return block


def motion_estimate(ref, target, block_size=8, search_range=32):
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

    def load(self, path, scale=0.5):
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

                h = (h // 8) * 8
                w = (w // 8) * 8
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            self.frames.append(frame)
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            self.frames_yuv.append(yuv)
        cap.release()
        self.frame_size = (self.frames[0].shape[1], self.frames[0].shape[0])

    def compress(self, gop=10, q=10, encoding_method="intra"):
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
        compressed = []
        for frame in self.frames_yuv:
            y = frame[:, :, 0]
            h, w = y.shape
            blocks = blockify(y)
            dct_blocks = np.array([dct2(b) for b in blocks])
            q_blocks = quantize(dct_blocks, q)
            zz_blocks = np.array([zigzag(b) for b in q_blocks])
            compressed.append(zz_blocks)

        serialized = pickle.dumps({"type": "intra", "data": compressed, "q": q})
        self.compressed_data = zlib.compress(serialized, level=9)

        raw = zlib.decompress(self.compressed_data)
        decoded_blocks = pickle.loads(raw)["data"]
        self.decoded = []
        for zz_blocks, frame in zip(decoded_blocks, self.frames_yuv):
            blocks = np.array([izigzag(b) for b in zz_blocks])
            dq_blocks = dequantize(blocks, q)
            idct_blocks = np.array([idct2(b) for b in dq_blocks])
            try:
                y_rec = unblockify(idct_blocks, frame.shape[0], frame.shape[1])
            except ValueError as e:
                raise ValueError(f"Error during decompression: {e}")
            rec = np.stack([y_rec, frame[:, :, 1], frame[:, :, 2]], axis=2)
            self.decoded.append(rec.astype(np.uint8))

    def compress_inter(self, gop, q, search_range=8):
        compressed = []
        for i in range(0, len(self.frames_yuv), gop):

            y = self.frames_yuv[i][:, :, 0]
            blocks = blockify(y)
            dct_blocks = np.array([dct2(b) for b in blocks])
            q_blocks = quantize(dct_blocks, q)
            zz_blocks = np.array([zigzag(b) for b in q_blocks])
            compressed.append(("I", zz_blocks))

            ref = y.copy()
            for j in range(i + 1, min(i + gop, len(self.frames_yuv))):
                y_p = self.frames_yuv[j][:, :, 0]
                mv, res = motion_estimate(ref, y_p, search_range=search_range)
                res_dct = [dct2(r) for r in res]
                q_res = [quantize(b, q) for b in res_dct]
                zz_res = [zigzag(b) for b in q_res]
                compressed.append(("P", mv, zz_res))

        serialized = pickle.dumps(
            {"type": "inter", "data": compressed, "gop": gop, "q": q}
        )
        self.compressed_data = zlib.compress(serialized, level=9)

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
            r_bgr = cv2.cvtColor(r, cv2.COLOR_YUV2BGR) if r.shape[2] == 3 else r
            if o_bgr.shape != r_bgr.shape:
                r_bgr = cv2.resize(r_bgr, (o_bgr.shape[1], o_bgr.shape[0]))
            psnr_vals.append(cv2.PSNR(o_bgr, r_bgr))
        return np.mean(psnr_vals)

    def _compression_ratio(self):
        orig_size = sum(f.nbytes for f in self.frames_yuv)
        comp_size = len(self.compressed_data)
        return orig_size / comp_size if comp_size > 0 else float("inf")

    def save_video(self, path):
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
        with open(path, "wb") as f:
            f.write(self.compressed_data)
        np.savez(
            path + ".metadata",
            metadata=self.metadata,
            fps=self.fps,
            frame_size=self.frame_size,
        )
