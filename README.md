# Multimedia Compression Suite

A comprehensive desktop application for audio and video compression, analysis, and processing built with Python and PyQt5.


## Features

### Audio Processing
- Load and visualize audio waveforms
- Apply noise reduction with adjustable threshold
- Compress audio using various bit depths (4, 8, 16 bit)
- Huffman encoding for further compression
- Calculate and display Signal-to-Noise Ratio (SNR)
- Real-time audio playback of original and processed signals
- Save processed audio in WAV format
  ![Screenshot 2025-05-13 021226](https://github.com/user-attachments/assets/0abe4b66-42ef-4d00-94cf-f65e295fc456)


### Video Processing
- Load and preview video files
- Compress video using different encoding methods:
  - Intra-frame coding (similar to JPEG)
  - P-frame coding (motion compensation)
  - Huffman and Arithmetic entropy coding
- Adjustable quality factor and GOP (Group of Pictures) size
- Calculate and display PSNR and compression ratio
- Preview original and compressed video
- Save processed video in MP4 or AVI format
- Save raw bitstream for further analysis
- Create videos from sequences of image frames
- ![Screenshot 2025-05-13 021317](https://github.com/user-attachments/assets/bb16de07-19bd-4e73-807a-f62b1966553f)


## Installation

### Requirements
- Python 3.7+
- PyQt5
- NumPy
- OpenCV (cv2)
- Matplotlib
- SciPy
- Librosa
- SoundFile
- PyAudio

### Setup
1. Clone this repository:
   ```
   git clone https://github.com/M0HAMEDHANY/digital-signal-processing.git
   cd digital-signal-processing
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python main.py
   ```

## Usage

### Audio Compression
1. Click "Load Audio" to open an audio file (WAV, MP3, FLAC, M4A)
2. Adjust window size, bit depth, and encoding method
3. Click "Run Compression" to process the audio
4. Use playback buttons to compare original and processed audio
5. Save the processed audio using the "Save" button

### Audio Denoising
1. Load an audio file with noise
2. Adjust the noise threshold slider 
3. Click "Plot Clean Signal" to apply denoising
4. Play the denoised audio and adjust threshold as needed

### Video Compression
1. Click "Load Video" to open a video file (MP4, AVI, MOV)
2. Select encoding method (Intra/P-frame) and adjust quality settings
3. Click "Run Compression" to process the video
4. Preview original and compressed video using the preview buttons
5. Save the processed video using the "Save Decoded" button

### Creating Videos from Frames
1. Click "Create Video from Frames"
2. Select a folder containing image frames (named sequentially)
3. Choose an output location for the compiled video

## Technical Details

### Audio Compression
- Uses Short-Time Fourier Transform (STFT) to convert to frequency domain
- Quantizes frequency components to reduce bit depth
- Applies Huffman coding for entropy encoding
- Reconstructs audio using inverse STFT

### Video Compression
- Converts frames to YUV colorspace
- Applies DCT (Discrete Cosine Transform) on 8x8 blocks
- Quantizes DCT coefficients based on quality factor
- Uses motion compensation for P-frames
- Zigzag scanning and entropy coding for bitstream compression

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
