import cv2
import numpy as np
from scipy.fftpack import dct, idct
from utils import calculate_psnr


def _block_process(channel: np.ndarray, Q: int):
    """
    Apply 8×8 block DCT, quantization by Q, inverse DCT.
    """
    h, w = channel.shape
    out = np.zeros_like(channel, dtype=np.float32)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = channel[i : i + 8, j : j + 8].astype(np.float32)
            B = dct(dct(block.T, norm="ortho").T, norm="ortho")
            Bq = np.round(B / Q)
            Bi = idct(idct(Bq * Q.T, norm="ortho").T, norm="ortho")
            out[i : i + 8, j : j + 8] = Bi
    return out


class VideoProcessor:
    def __init__(self):
        self.frames = []
        self.decoded = []
        self.psnr = None
        self.ratio = None
        self.metadata = {}
        self.fps = None

    def load(self, filepath: str):
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise ValueError(f"Could not open video '{filepath}'")
        self.frames = []
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
        cap.release()

    def compress(self, gop_size: int, Q: int, method: str):
        """
        Simple I-/P-frame codec: every gop_size-th frame is I-frame.
        """
        self.decoded = []
        total_original = 0
        total_compressed = 0

        prev_dec = None
        for idx, fr in enumerate(self.frames):
            yuv = cv2.cvtColor(fr, cv2.COLOR_BGR2YCrCb)
            channels = cv2.split(yuv)
            rec_ch = []

            if idx % gop_size == 0:
                # I-frame: DCT quant per block
                for ch in channels:
                    rec_ch.append(_block_process(ch, Q))
                # rough bit‐estimate
                total_compressed += fr.size  # placeholder
            else:
                # P-frame: simple copy previous decoded
                if prev_dec is None:
                    rec_ch = channels
                else:
                    rec_ch = cv2.split(prev_dec)
                total_compressed += 1  # placeholder

            # merge and convert back
            merged = cv2.merge([np.clip(c, 0, 255).astype(np.uint8) for c in rec_ch])
            bgr = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
            self.decoded.append(bgr)
            prev_dec = merged
            total_original += fr.size

        # PSNR per frame averaged
        psnrs = [calculate_psnr(o, d) for o, d in zip(self.frames, self.decoded)]
        self.psnr = float(np.mean(psnrs))
        self.ratio = total_original / max(total_compressed, 1)

        self.metadata.update(
            {
                "gop_size": gop_size,
                "Q": Q,
                "method": method,
                "original_frames": len(self.frames),
            }
        )

    def save_bitstream(self, path: str):
        np.savez(path, metadata=self.metadata, psnr=self.psnr, ratio=self.ratio)

    def save_video(self, path: str):
        if not self.decoded:
            raise ValueError("No decoded video to save")
        h, w, _ = self.decoded[0].shape
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"XVID"), self.fps, (w, h)
        )
        for frame in self.decoded:
            writer.write(frame)
        writer.release()
        
