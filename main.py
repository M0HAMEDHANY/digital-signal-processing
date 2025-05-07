# main.py

import sys
import numpy as np
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
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from audio_processor import AudioProcessor
from video_processor import VideoProcessor  # unchanged


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multimedia Compression Suite")
        self.resize(1000, 700)

        # Track last processed audio buffer (denoised or reconstructed)
        self.last_audio = None

        tabs = QTabWidget()
        tabs.addTab(self.build_audio_tab(), "Audio")
        tabs.addTab(self.build_video_tab(), "Video")
        self.setCentralWidget(tabs)

    def build_audio_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

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
        btn_save = QPushButton("Save WAV")
        btn_save.clicked.connect(self.save_audio)

        self.winSlider = QSlider(Qt.Horizontal)
        self.winSlider.setRange(256, 4096)
        self.winSlider.setValue(1024)
        self.bitsCombo = QComboBox()
        self.bitsCombo.addItems(["4", "8", "16"])
        self.encCombo = QComboBox()
        self.encCombo.addItems(["Huffman"])

        # Noise threshold controls
        self.noiseSlider = QSlider(Qt.Horizontal)
        self.noiseSlider.setRange(0, 100)
        self.noiseSlider.setValue(5)
        btn_clean = QPushButton("Plot Clean Signal")
        btn_clean.clicked.connect(self.plot_clean)

        # Canvases
        self.origCan = FigureCanvas(Figure(figsize=(5, 2)))
        self.cleanCan = FigureCanvas(Figure(figsize=(5, 2)))  # for denoised
        self.procCan = FigureCanvas(Figure(figsize=(5, 2)))

        # Layout assembly
        hl = QHBoxLayout()
        for wgt in (
            btn_load,
            btn_run,
            QLabel("Win:"),
            self.winSlider,
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
        hl2.addWidget(btn_clean)
        layout.addLayout(hl2)

        layout.addWidget(self.cleanCan)
        layout.addWidget(self.procCan)

        hl3 = QHBoxLayout()
        for wgt in (btn_play_o, btn_play_p, btn_save):
            hl3.addWidget(wgt)
        layout.addLayout(hl3)

        return w

    def build_video_tab(self):
        # unchanged
        w = QWidget()
        layout = QVBoxLayout(w)
        self.vp = VideoProcessor()

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

        self.gopSlider = QSlider(Qt.Horizontal)
        self.gopSlider.setRange(1, 30)
        self.gopSlider.setValue(10)
        self.qSlider = QSlider(Qt.Horizontal)
        self.qSlider.setRange(1, 50)
        self.qSlider.setValue(10)
        self.encVCombo = QComboBox()
        self.encVCombo.addItems(["Huffman", "Arithmetic"])

        self.frameLbl = QLabel()
        self.frameLbl.setFixedHeight(360)
        self.frameIdx = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self._show_frame)

        hl = QHBoxLayout()
        for wgt in (
            btn_load,
            btn_run,
            QLabel("GOP:"),
            self.gopSlider,
            QLabel("Q:"),
            self.qSlider,
            QLabel("Enc:"),
            self.encVCombo,
        ):
            hl.addWidget(wgt)
        layout.addLayout(hl)
        layout.addWidget(self.frameLbl)
        hl2 = QHBoxLayout()
        for wgt in (btn_play_o, btn_play_p, btn_save_vid, btn_save_bs):
            hl2.addWidget(wgt)
        layout.addLayout(hl2)
        return w

    # --- Audio methods ---

    def load_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio", "", "Audio (*.wav *.flac *.mp3 *.m4a)"
        )
        if not path:
            return
        try:
            self.ap.load(path)
            self.last_audio = None
            self._plot(self.origCan, self.ap.time, self.ap.signal, "Original Signal")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def plot_clean(self):
        try:
            pct = self.noiseSlider.value() / 100.0
            threshold = pct * np.max(np.abs(self.ap.signal))
            clean = self.ap.denoise(threshold)
            self.last_audio = clean.astype(np.float32)
            self._plot(self.cleanCan, self.ap.time, clean, "Cleaned Signal")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_audio(self):
        try:
            self.ap.compress(
                self.winSlider.value(),
                int(self.bitsCombo.currentText()),
                self.encCombo.currentText(),
            )
        except Exception as e:
            return QMessageBox.critical(self, "Compression Error", str(e))

        # rebuild time axis
        L = len(self.ap.reconstructed)
        self.ap.time = np.linspace(0, L / self.ap.fs, num=L)
        self.last_audio = self.ap.reconstructed
        self._plot(
            self.procCan,
            self.ap.time,
            self.ap.reconstructed,
            f"Reconstructed (SNR: {self.ap.snr:.2f} dB)",
        )

    def play_last(self):
        if self.last_audio is None:
            QMessageBox.warning(
                self, "No Processed Audio", "Run compression or clean first."
            )
        else:
            self.ap.play(self.last_audio)

    def save_audio(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save WAV", "", "WAV (*.wav)")
        if not path:
            return
        try:
            # always save the reconstructed version
            self.ap.save_wav(path)
            QMessageBox.information(
                self, "Saved", f"Reconstructed audio saved to:\n{path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _plot(self, canvas, x, y, title):
        canvas.figure.clf()
        ax = canvas.figure.add_subplot(111)
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        if y.ndim == 1:
            N = min(len(x), len(y))
            ax.plot(x[:N], y[:N])
        else:
            for ch in range(y.shape[1]):
                N = min(len(x), y.shape[0])
                ax.plot(x[:N], y[:N, ch], label=f"Ch {ch+1}")
            ax.legend()
        canvas.draw()

    # --- Video methods (unchanged) ---

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video (*.mp4 *.avi *.mov)"
        )
        if not path:
            return
        try:
            self.vp.load(path)
            self.frameIdx = 0
            QMessageBox.information(self, "Loaded", f"{len(self.vp.frames)} frames")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_video(self):
        try:
            self.vp.compress(
                self.gopSlider.value(),
                self.qSlider.value(),
                self.encVCombo.currentText(),
            )
            QMessageBox.information(
                self, "Done", f"PSNR: {self.vp.psnr:.2f} dB\nRatio: {self.vp.ratio:.2f}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _show_frame(self):
        lst = self._play_list
        if self.frameIdx >= len(lst):
            self.timer.stop()
            return
        fr = lst[self.frameIdx]
        h, w, _ = fr.shape
        img = QImage(fr.data, w, h, 3 * w, QImage.Format_BGR888)
        self.frameLbl.setPixmap(
            QPixmap.fromImage(img).scaled(self.frameLbl.size(), Qt.KeepAspectRatio)
        )
        self.frameIdx += 1

    def preview_original(self):
        if not self.vp.frames:
            return
        self._play_list = self.vp.frames
        self.frameIdx = 0
        self.timer.start(int(1000 / self.vp.fps))

    def preview_decoded(self):
        if not self.vp.decoded:
            return
        self._play_list = self.vp.decoded
        self.frameIdx = 0
        self.timer.start(int(1000 / self.vp.fps))

    def save_video(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "AVI (*.avi)")
        if not path:
            return
        try:
            self.vp.save_video(path)
            QMessageBox.information(self, "Saved", path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def save_bs(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Bitstream", "", "NPZ (*.npz)")
        if not path:
            return
        try:
            self.vp.save_bitstream(path)
            QMessageBox.information(self, "Saved", path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
