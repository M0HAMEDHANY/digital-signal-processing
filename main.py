import os
import sys
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QSlider,
    QComboBox,
    QTabWidget,
    QMessageBox,
    QHBoxLayout,
    QGroupBox,
    QProgressDialog,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor

# Add matplotlib imports for plotting
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from audio_processor import AudioProcessor
from video_processor import VideoProcessor
from create_video import CreateVideo


class MainWindow(QMainWindow):
    """
    Main application window that provides a tabbed interface for audio and video compression.
    Contains UI elements and logic for processing multimedia content.
    """

    def __init__(self):
        """Initialize the main application window with styling and tabs"""
        super().__init__()
        self.setWindowTitle("Multimedia Compression Suite")
        self.resize(1000, 800)

        # Apply styling
        self.apply_styles()

        # Track last processed audio buffer (denoised or reconstructed)
        self.last_audio = None

        # Create tabbed interface
        tabs = QTabWidget()
        tabs.addTab(self.build_audio_tab(), "Audio")
        tabs.addTab(self.build_video_tab(), "Video")
        self.setCentralWidget(tabs)

    def apply_styles(self):
        """
        Apply stylesheet to the application for consistent and attractive UI.
        Sets colors, borders, paddings and other visual elements.
        """
        stylesheet = """
        QMainWindow {
            background-color: #f0f0f0;
        }
        
        QTabWidget::pane {
            border: 1px solid #cccccc;
            background-color: #ffffff;
            border-radius: 4px;
        }
        
        QTabBar::tab {
            background-color: #e0e0e0;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        
        QTabBar::tab:selected {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-bottom-color: #ffffff;
        }
        
        QPushButton {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #2980b9;
        }
        
        QPushButton:pressed {
            background-color: #1c6ea4;
        }
        
        QLabel {
            color: #333333;
            font-size: 12px;
        }
        
        QSlider {
            height: 24px;
        }
        
        QSlider::groove:horizontal {
            border: 1px solid #999999;
            height: 8px;
            background: #cccccc;
            margin: 2px 0;
            border-radius: 4px;
        }
        
        QSlider::handle:horizontal {
            background: #3498db;
            border: 1px solid #5c5c5c;
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }
        
        QSlider::handle:horizontal:hover {
            background: #2980b9;
        }
        
        QComboBox {
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 5px;
            background-color: white;
            selection-background-color: #3498db;
        }
        
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left-width: 1px;
            border-left-color: #cccccc;
            border-left-style: solid;
            border-top-right-radius: 4px;
            border-bottom-right-radius: 4px;
        }
        
        QGroupBox {
            border: 1px solid #cccccc;
            border-radius: 4px;
            margin-top: 16px;
            padding-top: 16px;
            font-weight: bold;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
            color: #333333;
        }
        """

        # Set the application-wide stylesheet
        self.setStyleSheet(stylesheet)

        # Set a darker background for the plot canvases
        for canvas_name in ["origCan", "cleanCan", "procCan"]:
            if hasattr(self, canvas_name):
                canvas = getattr(self, canvas_name)
                canvas.figure.patch.set_facecolor("#f5f5f5")
                canvas.figure.tight_layout(pad=3.0)

    def build_audio_tab(self):
        """
        Create and configure the audio processing tab with controls and visualizations.

        Returns:
            QWidget: The configured audio tab widget
        """
        w = QWidget()
        layout = QVBoxLayout(w)

        # Set spacing and margins for better layout
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # Initialize audio processor
        self.ap = AudioProcessor()

        # Buttons and controls
        btn_load = QPushButton("Load Audio")
        btn_load.clicked.connect(self.load_audio)
        btn_run = QPushButton("Run Compression")
        btn_run.clicked.connect(self.run_audio)
        btn_play_o = QPushButton("Play Original")
        btn_play_o.clicked.connect(self.ap.play_original)
        btn_play_p = QPushButton("Play Processed")
        btn_play_p.clicked.connect(self.play_last)
        btn_stop = QPushButton("Stop")
        btn_stop.clicked.connect(self.stop_audio)
        btn_save = QPushButton("Save")
        btn_save.clicked.connect(self.save_audio)

        # Window size slider for FFT
        self.winSlider = QSlider(Qt.Horizontal)
        self.winSlider.setRange(256, 4096)
        self.winSlider.setValue(1024)
        self.winValueLabel = QLabel("1024")
        self.winSlider.valueChanged.connect(
            lambda: self.winValueLabel.setText(str(self.winSlider.value()))
        )

        # Bit depth and encoding options
        self.bitsCombo = QComboBox()
        self.bitsCombo.addItems(["4", "8", "16"])
        self.encCombo = QComboBox()
        self.encCombo.addItems(["Huffman"])

        # Noise threshold controls for denoising
        self.noiseSlider = QSlider(Qt.Horizontal)
        self.noiseSlider.setRange(100, 1000)
        self.noiseSlider.setValue(10)
        self.noiseValueLabel = QLabel("10")
        self.noiseSlider.valueChanged.connect(
            lambda: self.noiseValueLabel.setText(str(self.noiseSlider.value()))
        )

        btn_clean = QPushButton("Plot Clean Signal")
        btn_clean.clicked.connect(self.plot_clean)

        # Matplotlib canvases for signal visualization
        self.origCan = FigureCanvas(Figure(figsize=(5, 2)))
        self.cleanCan = FigureCanvas(Figure(figsize=(5, 2)))
        self.procCan = FigureCanvas(Figure(figsize=(5, 2)))

        # Layout assembly
        hl = QHBoxLayout()
        for wgt in (
            btn_load,
            btn_run,
            QLabel("Win:"),
            self.winSlider,
            self.winValueLabel,
            QLabel("Bits:"),
            self.bitsCombo,
            QLabel("Enc:"),
            self.encCombo,
        ):
            hl.addWidget(wgt)
        layout.addLayout(hl)

        layout.addWidget(self.origCan)

        hl2 = QHBoxLayout()
        hl2.addWidget(QLabel("Noise Threshold %:"))
        hl2.addWidget(self.noiseSlider)
        hl2.addWidget(self.noiseValueLabel)
        hl2.addWidget(btn_clean)
        layout.addLayout(hl2)

        layout.addWidget(self.cleanCan)
        layout.addWidget(self.procCan)

        hl3 = QHBoxLayout()
        for wgt in (btn_play_o, btn_play_p, btn_save, btn_stop):
            hl3.addWidget(wgt)
        layout.addLayout(hl3)

        return w

    def build_video_tab(self):
        """
        Create and configure the video processing tab with controls and preview area.

        Returns:
            QWidget: The configured video tab widget
        """
        w = QWidget()
        layout = QVBoxLayout(w)

        # Set spacing and margins for better layout
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # Initialize video processor
        self.vp = VideoProcessor()

        # Button controls for video operations
        btn_load = QPushButton("Load Video")
        btn_load.clicked.connect(self.load_video)
        btn_run = QPushButton("Run Compression")
        btn_run.clicked.connect(self.run_video)
        btn_play_o = QPushButton("Preview Original")
        btn_play_o.clicked.connect(self.preview_original)
        btn_play_p = QPushButton("Preview Decoded")
        btn_play_p.clicked.connect(self.preview_decoded)
        btn_save_vid = QPushButton("Save Decoded")
        btn_save_vid.clicked.connect(self.save_video)
        btn_save_bs = QPushButton("Save Bitstream")
        btn_save_bs.clicked.connect(self.save_bs)
        btn_create_vid = QPushButton("Create Video from Frames")
        btn_create_vid.clicked.connect(self.create_from_frames)

        # GOP (Group of Pictures) slider for video compression
        self.gopSlider = QSlider(Qt.Horizontal)
        self.gopSlider.setRange(1, 30)
        self.gopSlider.setValue(10)
        self.gopValueLabel = QLabel("10")
        self.gopSlider.valueChanged.connect(
            lambda: self.gopValueLabel.setText(str(self.gopSlider.value()))
        )

        # Quality factor slider for compression
        self.qSlider = QSlider(Qt.Horizontal)
        self.qSlider.setRange(1, 100)
        self.qSlider.setValue(20)
        self.qValueLabel = QLabel("20")
        self.qSlider.valueChanged.connect(
            lambda: self.qValueLabel.setText(str(self.qSlider.value()))
        )

        # Encoding methods dropdown
        self.encVCombo = QComboBox()
        self.encVCombo.addItems(["Huffman", "Arithmetic", "Intra", "P-frame"])

        # Video frame display area
        self.frameLbl = QLabel()
        self.frameLbl.setFixedHeight(360)
        self.frameIdx = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self._show_frame)

        # Control layout
        hl = QHBoxLayout()
        for wgt in (
            btn_load,
            btn_run,
            QLabel("GOP:"),
            self.gopSlider,
            self.gopValueLabel,
            QLabel("Q:"),
            self.qSlider,
            self.qValueLabel,
            QLabel("Enc:"),
            self.encVCombo,
        ):
            hl.addWidget(wgt)
        layout.addLayout(hl)
        layout.addWidget(self.frameLbl)

        # Save Options Group Box
        save_group = QGroupBox("Save Options")
        save_layout = QHBoxLayout()
        save_layout.addWidget(btn_save_vid)
        save_layout.addWidget(btn_save_bs)
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)

        # Preview buttons layout
        hl2 = QHBoxLayout()
        hl2.addWidget(btn_play_o)
        hl2.addWidget(btn_play_p)
        hl2.addWidget(btn_create_vid)
        layout.addLayout(hl2)

        return w

    # --- Audio methods ---

    def load_audio(self):
        """
        Open a file dialog to select and load an audio file.
        Displays the audio waveform and updates the window title.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio", "", "Audio (*.wav *.flac *.mp3 *.m4a)"
        )
        if not path:
            return
        try:
            self.ap.load(path)
            self.last_audio = None
            self._plot(self.origCan, self.ap.time, self.ap.signal, "Original Signal")

            filename = os.path.basename(path)
            self.setWindowTitle(f"Multimedia Compression Suite - Audio: {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def plot_clean(self):
        """
        Apply denoising to the loaded audio signal and display the cleaned signal.
        Uses the noise threshold from the slider.
        """
        try:
            threshold = self.noiseSlider.value()
            clean = self.ap.denoise(threshold)
            self.last_audio = clean.astype(np.float32)
            self._plot(self.cleanCan, self.ap.time, clean, "Cleaned Signal")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_audio(self):
        """
        Compress the audio using the selected parameters (window size, bit depth, encoding).
        Displays the reconstructed signal and SNR after compression.
        """
        try:
            self.ap.compress(
                self.winSlider.value(),
                int(self.bitsCombo.currentText()),
                self.encCombo.currentText(),
            )
        except Exception as e:
            return QMessageBox.critical(self, "Compression Error", str(e))

        L = len(self.ap.reconstructed)
        self.ap.time = np.linspace(0, L / self.ap.fs, num=L)
        self.last_audio = self.ap.reconstructed
        self._plot(
            self.procCan,
            self.ap.time,
            self.ap.reconstructed,
            f"Reconstructed (SNR: {self.ap.snr:.2f} dB)",
        )

    def stop_audio(self):
        """Stop any currently playing audio."""
        if self.ap is not None:
            self.ap.stop()
        else:
            QMessageBox.warning(self, "No Audio", "No audio is currently playing.")

    def play_last(self):
        """Play the last processed audio buffer (denoised or reconstructed)."""
        if self.last_audio is None:
            QMessageBox.warning(
                self, "No Processed Audio", "Run compression or clean first."
            )
        else:
            self.ap.play(self.last_audio)

    def save_audio(self):
        """
        Save the processed audio to a file.
        Opens a file dialog to choose save location.
        """
        if hasattr(self.ap, "filename") and self.ap.filename:
            path = QFileDialog.getExistingDirectory(self, "Select Directory to Save")
            if not path:
                return
            try:
                saved_path = self.ap.save_wav(path)
                QMessageBox.information(
                    self, "Saved", f"Reconstructed audio saved to:\n{saved_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Save Error", str(e))
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Save Audio", "", "WAV (*.wav)")
            if not path:
                return
            try:
                self.ap.save_wav(path)
                QMessageBox.information(
                    self, "Saved", f"Reconstructed audio saved to:\n{path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Save Error", str(e))

    def _plot(self, canvas, x, y, title):
        """
        Plot signal data on the specified matplotlib canvas.

        Args:
            canvas: The FigureCanvas to plot on
            x: X-axis data (typically time)
            y: Y-axis data (signal amplitude)
            title: Plot title to display
        """
        canvas.figure.clf()
        ax = canvas.figure.add_subplot(111)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlabel("Time [s]", fontweight="bold")
        ax.set_ylabel("Amplitude", fontweight="bold")
        ax.set_facecolor("#f9f9f9")
        ax.grid(True, linestyle="--", alpha=0.7)
        if y.ndim == 1:
            N = min(len(x), len(y))
            ax.plot(x[:N], y[:N], linewidth=1.5, color="#3498db")
        else:
            colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
            for ch in range(y.shape[1]):
                N = min(len(x), y.shape[0])
                color = colors[ch % len(colors)]
                ax.plot(
                    x[:N], y[:N, ch], linewidth=1.5, color=color, label=f"Ch {ch+1}"
                )
            ax.legend()
        canvas.draw()

    # --- Video methods ---

    def load_video(self):
        """
        Open a file dialog to select and load a video file.
        Initializes the video processor and displays information about the loaded video.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )

        if not path:
            return

        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Video file not found: {path}")

            self.vp = VideoProcessor()
            self.vp.load(path)

            if not hasattr(self.vp, "frames") or not self.vp.frames:
                raise ValueError("No frames were loaded. The video might be corrupted.")

            self.frameIdx = 0
            self._play_list = self.vp.frames

            filename = os.path.basename(path)
            self.setWindowTitle(f"Multimedia Compression Suite - Video: {filename}")

            QMessageBox.information(
                self,
                "Video Loaded",
                f"Successfully loaded {len(self.vp.frames)} frames\n"
                f"Resolution: {self.vp.frame_size[0]}x{self.vp.frame_size[1]}\n"
                f"FPS: {self.vp.fps:.2f}",
            )

        except FileNotFoundError as e:
            QMessageBox.critical(self, "File Error", str(e))
        except Exception as e:
            QMessageBox.critical(
                self,
                "Loading Error",
                f"Failed to load video:\n{str(e)}\n"
                "Please make sure the file is a valid video.",
            )

    def run_video(self):
        """
        Compress the video using the selected parameters (GOP size, quality, encoding).
        Displays compression metrics (PSNR and ratio) after processing.
        """
        try:
            encoding_method = self.encVCombo.currentText().lower().replace("-", "")
            self.vp.compress(
                self.gopSlider.value(), self.qSlider.value(), encoding_method
            )
            QMessageBox.information(
                self, "Done", f"PSNR: {self.vp.psnr:.2f} dB\nRatio: {self.vp.ratio:.2f}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _show_frame(self):
        """
        Display a single frame from the current video playlist.
        Called by the timer to animate video playback.
        Handles format conversion and scaling for display.
        """
        try:
            # Check if we have a valid playlist
            if not hasattr(self, "_play_list") or not self._play_list:
                self.timer.stop()
                return

            # Check if we've reached the end of the frames
            if self.frameIdx >= len(self._play_list):
                self.timer.stop()
                if hasattr(self, "playback_looping") and self.playback_looping:
                    self.frameIdx = 0
                    return
                else:
                    self.frameIdx = 0
                    return

            # Get the current frame
            current_frame = self._play_list[self.frameIdx]

            # Convert to proper format if needed
            if len(current_frame.shape) == 2:
                frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)
            elif current_frame.shape[2] == 4:
                frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGRA2BGR)
            else:
                frame_rgb = current_frame

            h, w, _ = frame_rgb.shape
            bytes_per_line = 3 * w

            # Create QImage and properly scaled pixmap
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_BGR888)
            scaled_pixmap = QPixmap.fromImage(q_img).scaled(
                self.frameLbl.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

            # Display the frame
            self.frameLbl.setAlignment(Qt.AlignCenter)
            self.frameLbl.setPixmap(scaled_pixmap)

            # Update progress if available
            if hasattr(self, "playback_progress"):
                self.playback_progress.setValue(self.frameIdx)

            # Move to next frame
            self.frameIdx += 1

        except AttributeError as e:
            self.timer.stop()
            QMessageBox.warning(
                self, "Playback Error", f"Missing required attributes: {str(e)}"
            )
        except IndexError:
            self.timer.stop()
            if hasattr(self, "_play_list"):
                self.frameIdx = 0
        except Exception as e:
            self.timer.stop()
            QMessageBox.critical(
                self,
                "Playback Error",
                f"Error displaying frame {self.frameIdx}:\n{str(e)}\n\n"
                f"Frame size: {current_frame.shape if 'current_frame' in locals() else 'N/A'}",
            )

    def preview_original(self):
        """Preview the original unprocessed video frames."""
        if not self.vp.frames:
            QMessageBox.warning(self, "No Video", "Load a video first.")
            return
        self._play_list = self.vp.frames
        self.frameIdx = 0
        self.timer.start(int(1000 / self.vp.fps))

    def preview_decoded(self):
        """Preview the decoded (compressed/decompressed) video frames."""
        if not self.vp.decoded:
            QMessageBox.warning(self, "No Decoded Video", "Run compression first.")
            return
        self._play_list = self.vp.decoded
        self.frameIdx = 0
        self.timer.start(int(1000 / self.vp.fps))

    def save_video(self):
        """
        Save the processed video to a file.
        Supports MP4 and AVI formats with appropriate codecs.
        """
        if not hasattr(self.vp, "decoded") or not self.vp.decoded:
            QMessageBox.warning(
                self, "No Video", "Please load and process a video first before saving."
            )
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Compressed Video",
            "",
            "MP4 Video (*.mp4);;AVI Video (*.avi);;All Files (*)",
        )

        if not path:
            return

        try:
            if selected_filter.startswith("MP4"):
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                if not path.endswith(".mp4"):
                    path += ".mp4"
            else:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                if not path.endswith(".avi"):
                    path += ".avi"

            out = cv2.VideoWriter(path, fourcc, self.vp.fps, self.vp.frame_size)

            if not out.isOpened():
                raise IOError("Could not create video file. Check codec support.")

            for frame in self.vp.decoded:
                out.write(frame)

            out.release()

            QMessageBox.information(
                self,
                "Success",
                f"Video saved successfully!\n\n"
                f"Location: {path}\n"
                f"Size: {os.path.getsize(path)//1024} KB\n"
                f"Frames: {len(self.vp.decoded)}\n"
                f"Resolution: {self.vp.frame_size[0]}x{self.vp.frame_size[1]}",
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Failed",
                f"Failed to save video:\n{str(e)}\n\n"
                "Possible reasons:\n"
                "- Unsupported codec\n"
                "- Invalid file path\n"
                "- Disk full or write permissions",
            )

    def save_bs(self):
        """
        Save the compressed video bitstream to a file.
        Useful for later decompression or analysis.
        """
        path, _ = QFileDialog.getSaveFileName(self, "Save Bitstream", "", "NPZ (*.npz)")
        if not path:
            return
        try:
            self.vp.save_bitstream(path)
            QMessageBox.information(self, "Saved", path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def create_from_frames(self):
        """
        Create a video from a folder of image frames.
        Opens dialogs to select the source folder and output file.
        """
        folder = QFileDialog.getExistingDirectory(self, "Select Frame Folder")

        if not folder:
            return

        if not os.path.exists(folder):
            QMessageBox.critical(self, "Error", f"Path '{folder}' does not exist.")
            return

        try:
            creator = CreateVideo()
            creator.create_video(folder)

            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Video As", "Video.mp4", "MP4 Video (*.mp4)"
            )
            if not save_path:
                return

            creator.save_video(save_path)
            QMessageBox.information(
                self, "Success", f"Video created and saved as:\n{save_path}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Video Creation Failed", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set application-wide font
    app.setFont(QApplication.font("QApplication"))

    # Apply styling to QMessageBox globally
    app.setStyleSheet(
        """
    QMessageBox {
        background-color: #ffffff;
    }
    QMessageBox QLabel {
        color: #333333;
    }
    QMessageBox QPushButton {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 6px 12px;
        border-radius: 3px;
        font-weight: bold;
        min-width: 80px;
    }
    QMessageBox QPushButton:hover {
        background-color: #2980b9;
    }
    """
    )

    # Create and display the main window
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
