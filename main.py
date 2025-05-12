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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multimedia Compression Suite")
        self.resize(1000, 800)

        # Apply styling
        self.apply_styles()

        # Track last processed audio buffer (denoised or reconstructed)
        self.last_audio = None

        tabs = QTabWidget()
        tabs.addTab(self.build_audio_tab(), "Audio")
        tabs.addTab(self.build_video_tab(), "Video")
        self.setCentralWidget(tabs)

    def apply_styles(self):
        """Apply stylesheet to the application"""
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
        w = QWidget()
        layout = QVBoxLayout(w)

        # Set spacing and margins for better layout
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

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

        self.winSlider = QSlider(Qt.Horizontal)
        self.winSlider.setRange(256, 4096)
        self.winSlider.setValue(1024)
        self.winValueLabel = QLabel("1024")  # Add value label for window size
        self.winSlider.valueChanged.connect(
            lambda: self.winValueLabel.setText(str(self.winSlider.value()))
        )

        self.bitsCombo = QComboBox()
        self.bitsCombo.addItems(["4", "8", "16"])
        self.encCombo = QComboBox()
        self.encCombo.addItems(["Huffman"])

        # Noise threshold controls
        self.noiseSlider = QSlider(Qt.Horizontal)
        self.noiseSlider.setRange(100, 1000)
        self.noiseSlider.setValue(10)
        self.noiseValueLabel = QLabel("10")  # Add value label for noise threshold
        self.noiseSlider.valueChanged.connect(
            lambda: self.noiseValueLabel.setText(str(self.noiseSlider.value()))
        )

        btn_clean = QPushButton("Plot Clean Signal")
        btn_clean.clicked.connect(self.plot_clean)

        # Canvases
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
            self.winValueLabel,  # Add window size value label to layout
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
        hl2.addWidget(self.noiseValueLabel)  # Add noise threshold value label to layout
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
        w = QWidget()
        layout = QVBoxLayout(w)

        # Set spacing and margins for better layout
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

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
        btn_create_vid = QPushButton("Create Video from Frames")
        btn_create_vid.clicked.connect(self.create_from_frames)

        self.gopSlider = QSlider(Qt.Horizontal)
        self.gopSlider.setRange(1, 30)
        self.gopSlider.setValue(10)
        self.gopValueLabel = QLabel("10")  # Add value label for GOP size
        self.gopSlider.valueChanged.connect(
            lambda: self.gopValueLabel.setText(str(self.gopSlider.value()))
        )

        self.qSlider = QSlider(Qt.Horizontal)
        self.qSlider.setRange(1, 100)
        self.qSlider.setValue(20)
        self.qValueLabel = QLabel("20")  # Label to display current Q value
        self.qSlider.valueChanged.connect(
            lambda: self.qValueLabel.setText(str(self.qSlider.value()))
        )
        self.encVCombo = QComboBox()
        self.encVCombo.addItems(["Huffman", "Arithmetic", "Intra", "P-frame"])

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
            self.gopValueLabel,  # Add GOP size value label to layout
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
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio", "", "Audio (*.wav *.flac *.mp3 *.m4a)"
        )
        if not path:
            return
        try:
            self.ap.load(path)
            self.last_audio = None
            self._plot(self.origCan, self.ap.time, self.ap.signal, "Original Signal")

            import os

            filename = os.path.basename(path)
            self.setWindowTitle(f"Multimedia Compression Suite - Audio: {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def plot_clean(self):
        try:
            # Map slider value (0–100) to 0.0–1.0
            threshold = self.noiseSlider.value() / 1000.0
            clean = self.ap.denoise(prop_decrease=threshold)

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
        if self.ap is not None:
            self.ap.stop()
        else:
            QMessageBox.warning(self, "No Audio", "No audio is currently playing.")

    def play_last(self):
        if self.last_audio is None:
            QMessageBox.warning(
                self, "No Processed Audio", "Run compression or clean first."
            )
        else:
            self.ap.play(self.last_audio)

    def save_audio(self):
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
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        
        if not path:
            return
        
        try:
            # التحقق من وجود الملف أولاً
            if not os.path.exists(path):
                raise FileNotFoundError(f"Video file not found: {path}")
            
            # تحميل الفيديو
            self.vp = VideoProcessor()  # إعادة تهيئة المعالج لتفادي مشاكل التحميل المتكرر
            self.vp.load(path)
            
            # التحقق من تحميل الإطارات بنجاح
            if not hasattr(self.vp, 'frames') or not self.vp.frames:
                raise ValueError("No frames were loaded. The video might be corrupted.")
            
            # تهيئة متغيرات العرض
            self.frameIdx = 0
            self._play_list = self.vp.frames  # تهيئة قائمة التشغيل
            
            # تحديث واجهة المستخدم
            filename = os.path.basename(path)
            self.setWindowTitle(f"Multimedia Compression Suite - Video: {filename}")
            
            # تنسيق عرض الفيديو
            self.frameLbl.setStyleSheet(
                "border: 2px solid #3498db; border-radius: 4px; background-color: #000000;"
            )
            
            # عرض رسالة تأكيد مع معلومات الفيديو
            QMessageBox.information(
                self,
                "Video Loaded",
                f"Successfully loaded {len(self.vp.frames)} frames\n"
                f"Resolution: {self.vp.frame_size[0]}x{self.vp.frame_size[1]}\n"
                f"FPS: {self.vp.fps:.2f}"
            )
            
        except FileNotFoundError as e:
            QMessageBox.critical(self, "File Error", str(e))
        except Exception as e:
            QMessageBox.critical(
                self,
                "Loading Error",
                f"Failed to load video:\n{str(e)}\n"
                "Please make sure the file is a valid video."
            )
            
            # عرض رسالة تأكيد مع معلومات الفيديو
            QMessageBox.information(
                self,
                "Video Loaded",
                f"Successfully loaded {len(self.vp.frames)} frames\n"
                f"Resolution: {self.vp.frame_size[0]}x{self.vp.frame_size[1]}\n"
                f"FPS: {self.vp.fps:.2f}"
            )
            
        except FileNotFoundError as e:
            QMessageBox.critical(self, "File Error", str(e))
        except Exception as e:
            QMessageBox.critical(
                self,
                "Loading Error",
                f"Failed to load video:\n{str(e)}\n"
                "Please make sure the file is a valid video."
            )

    def run_video(self):
        try:
            encoding_method = self.encVCombo.currentText().lower().replace("-", "")
            self.vp.compress(
                self.gopSlider.value(),
                self.qSlider.value(),
                encoding_method
            )
            QMessageBox.information(
                self, "Done", f"PSNR: {self.vp.psnr:.2f} dB\nRatio: {self.vp.ratio:.2f}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _show_frame(self):
        try:

            # Check if we've reached the end of the frames
            if self.frameIdx >= len(self._play_list):
                self.timer.stop()
                return

            # Get the current frame
            frame = self._play_list[self.frameIdx]

            # Convert grayscale to BGR if needed
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Create QImage from frame data
            height, width = frame.shape[:2]
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_BGR888)

            # Create a pixmap and scale it while maintaining aspect ratio
            pixmap = QPixmap.fromImage(qImg)

            # Create a properly scaled pixmap that fits within the label
            scaled_pixmap = pixmap.scaled(
                self.frameLbl.width(),
                self.frameLbl.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

            # Center the pixmap in the label
            self.frameLbl.setAlignment(Qt.AlignCenter)
            self.frameLbl.setPixmap(scaled_pixmap)

            # Move to next frame
            self.frameIdx += 1


            # التحقق من وجود قائمة تشغيل صالحة
            if not hasattr(self, '_play_list') or not self._play_list:
                self.timer.stop()
                return

            # التحقق من أن الفهرس ضمن النطاق الصحيح
            if self.frameIdx >= len(self._play_list):
                self.timer.stop()
                if hasattr(self, 'playback_looping') and self.playback_looping:
                    self.frameIdx = 0  # إعادة التشغيل إذا كان الوضع التكرار مفعل
                    return
                else:
                    self.frameIdx = 0  # إعادة التعيين للبدء من الأول
                    return

            # الحصول على الإطار الحالي
            current_frame = self._play_list[self.frameIdx]

            # معالجة الإطار (للتأكد من أنه ملون)
            if len(current_frame.shape) == 2:  # إذا كان إطاراً رمادياً
                frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)
            elif current_frame.shape[2] == 4:  # إذا كان فيه قناة ألفا
                frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGRA2BGR)
            else:
                frame_rgb = current_frame

            h, w, _ = frame_rgb.shape
            bytes_per_line = 3 * w

            # إنشاء QImage من الإطار
            q_img = QImage(
                frame_rgb.data,
                w, h,
                bytes_per_line,
                QImage.Format_BGR888
            )

            # تحجيم الصورة مع الحفاظ على النسبة
            scaled_pixmap = QPixmap.fromImage(q_img).scaled(
                self.frameLbl.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation  # لجودة تحجيم أفضل
            )

            # عرض الإطار في الواجهة
            self.frameLbl.setPixmap(scaled_pixmap)

            # تحديث شريط التقدم إذا كان موجوداً
            if hasattr(self, 'playback_progress'):
                self.playback_progress.setValue(self.frameIdx)

            # زيادة العداد للانتقال للإطار التالي
            self.frameIdx += 1

        except AttributeError as e:
            self.timer.stop()
            QMessageBox.warning(
                self,
                "Playback Error",
                f"Missing required attributes: {str(e)}"
            )
        except IndexError:
            self.timer.stop()
            if hasattr(self, '_play_list'):
                self.frameIdx = 0  # إعادة التعيين عند الوصول لنهاية الفيديو

        except Exception as e:
            self.timer.stop()
            QMessageBox.critical(
                self,
                "Playback Error",
                f"Error displaying frame {self.frameIdx}:\n{str(e)}\n\n"
                f"Frame size: {current_frame.shape if 'current_frame' in locals() else 'N/A'}"
            )

    def preview_original(self):
        if not self.vp.frames:
            QMessageBox.warning(self, "No Video", "Load a video first.")
            return
        self._play_list = self.vp.frames
        self.frameIdx = 0
        self.timer.start(int(1000 / self.vp.fps))

    def preview_decoded(self):
        if not self.vp.decoded:
            QMessageBox.warning(self, "No Decoded Video", "Run compression first.")
            return
        self._play_list = self.vp.decoded
        self.frameIdx = 0
        self.timer.start(int(1000 / self.vp.fps))

    def save_video(self):
        # التحقق من وجود فيديو معالج أولاً
        if not hasattr(self.vp, 'decoded') or not self.vp.decoded:
            QMessageBox.warning(
                self, 
                "No Video", 
                "Please load and process a video first before saving."
            )
            return

        # عرض مربع حوار الحفظ مع خيارات الصيغ
        path, selected_filter = QFileDialog.getSaveFileName(
            self, 
            "Save Compressed Video",
            "",
            "MP4 Video (*.mp4);;AVI Video (*.avi);;All Files (*)"
        )
        
        if not path:
            return
        
        try:
            # تحديد الكودك بناء على الامتداد المختار
            if selected_filter.startswith("MP4"):
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                if not path.endswith('.mp4'):
                    path += '.mp4'
            else:  # AVI كبديل
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                if not path.endswith('.avi'):
                    path += '.avi'

            # إعداد كاتب الفيديو مع التحقق من نجاح الإعداد
            out = cv2.VideoWriter(
                path,
                fourcc,
                self.vp.fps,
                self.vp.frame_size
            )
            
            if not out.isOpened():
                raise IOError("Could not create video file. Check codec support.")

            # كتابة الإطارات
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
                f"Resolution: {self.vp.frame_size[0]}x{self.vp.frame_size[1]}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Failed",
                f"Failed to save video:\n{str(e)}\n\n"
                "Possible reasons:\n"
                "- Unsupported codec\n"
                "- Invalid file path\n"
                "- Disk full or write permissions"
            )

    def save_bs(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Bitstream", "", "NPZ (*.npz)")
        if not path:
            return
        try:
            self.vp.save_bitstream(path)
            QMessageBox.information(self, "Saved", path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def create_from_frames(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Frame Folder")

        if not folder:
            return

        # Debugging: Print the selected folder path
        print(f"Selected folder: {folder}")

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


# Apply the styles to QMessageBox globally
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set application-wide font
    app.setFont(QApplication.font("QApplication"))

    # Style QMessageBox
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

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
