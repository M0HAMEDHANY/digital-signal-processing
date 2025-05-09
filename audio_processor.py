"""
2-Video Compression Project:
Objective: Implement a basic video compression.
Steps:

1. Video Input Handling ===============>> Done
o Create video frame-by-frame
o Convert frames to YUV color space.

# Y: Brightness, U: Blue - Brightness, V: Red - Brightness -> !!!!!!!!IMPORTANT!!!!!!!!!!

2. Frame Type Decision ===============> Done
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
from scipy.fftpack import dct, idct
from utils import calculate_psnr



class VideoProcessor:
    def __init__(self):

        self.frames = []
        self.resolution = ()
        self.decoded = []
        self.psnr = None
        self.ratio = None
        self.metadata = {}
        self.fps = 0

    def load(self, filepath: str):
        
        # Validating video loading

        capture = cv2.VideoCapture(filepath)

        if not capture.isOpened():
            raise ValueError(f"Could not open video '{filepath}'")
        
        # Getting video resolution and FPS

        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)

        self.resolution = (height, width)
        self.fps = capture.get(cv2.CAP_PROP_FPS)

        # Extracting Frames (images) from the video

        capture = cv2.VideoCapture(filepath)
        ret_value, image = capture.read()

        while ret_value == True:

            # Converting from colored image to YUV color space
            image = cv2.cvtColor(image, cv2.BGR2YUV)   

            self.frames.append(image)
            ret_value, image = capture.read()

        # Releasing the video
        capture.release()

    def compress(self, gop_size: int, Q: int, method: str):
        """
        Simple I-/P-frame codec: every gop_size-th frame is I-frame.
        """
        self.decoded = []
        total_original = 0
        total_compressed = 0
        prev_dec = None

        for idx, frame in enumerate(self.frames):
            
            channels = cv2.split(frame)
            rec_ch = []

            # I-frame: DCT quant per block
            if idx % gop_size == 0:

                for ch in channels:
                    rec_ch.append(self.block_process(ch, Q))
                
                total_compressed += frame.size

            # P-frame: simple copy previous decoded
            else:
                
                if prev_dec is None:
                    rec_ch = channels

                else:
                    rec_ch = cv2.split(prev_dec)
                    
                total_compressed += 1 

            # merge and convert back
            merged = cv2.merge([np.clip(c, 0, 255).astype(np.uint8) for c in rec_ch])
            bgr = cv2.cvtColor(merged, cv2.COLOR_YUV2BGR_I420)
            self.decoded.append(bgr)
            prev_dec = merged
            total_original += frame.size

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

    def block_process(channel: np.ndarray, Q: int):
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
