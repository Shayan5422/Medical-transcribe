import sys
import os
import json
import tempfile
import logging
import time
import fnmatch
import psutil
import shutil
import keyboard
import sounddevice as sd
import numpy as np
import requests
import wave
import pyperclip
from PyQt5.QtWidgets import (
    QApplication, QSystemTrayIcon, QMenu, QMessageBox, QDialog, QLabel,
    QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor

# نام برنامه برای پوشه‌ی داده‌ها
APP_NAME = "AudioRecorder"

# مسیر پوشه داده‌ها با توجه به حالت exe یا اسکریپت پایتون
if getattr(sys, 'frozen', False):
    # در حالت بسته‌بندی‌شده (exe)
    application_path = sys._MEIPASS
    # در ویندوز از AppData استفاده کنیم
    app_data_dir = os.path.join(os.environ.get('APPDATA', ''), APP_NAME)
else:
    # در حالت اجرای عادی پایتون
    application_path = os.path.dirname(os.path.abspath(__file__))
    app_data_dir = os.path.join(application_path, 'data')

# ساخت پوشه داده‌ها در صورت نبود آن
os.makedirs(app_data_dir, exist_ok=True)

# فایل لاگ در پوشه داده
log_file = os.path.join(app_data_dir, 'audio_recorder.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file
)

# فایل JSON برای ذخیرهٔ یوزرنیم/پسورد USB
CREDENTIALS_FILE = os.path.join(app_data_dir, "usb_credentials.json")

class AudioRecorder(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    chunk_processed = pyqtSignal(str)

    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.recording = False
        self.audio_data = []
        self.token = None
        
        # یوزرنیم/پسورد ثابت (برای ضبط‌های کمتر از ۵ ثانیه)
        # self.usb_username = "word"
        # self.usb_password = "1234"

        # یوزرنیم/پسورد USB (از فایل JSON بارگذاری می‌شود)
        self.usb_username = None
        self.usb_password = None

        # تنظیمات چانک (هر ۳۰ ثانیه یک چانک)
        self.chunk_duration = 30  # بر حسب ثانیه
        self.chunk_samples = self.chunk_duration * self.sample_rate
        self.current_chunk = []
        self.all_transcriptions = []
        self.last_chunk_time = 0
        self.recording_start_time = 0
        self.recording_duration = 0
        self.session_id = None

    def run(self):
        """حلقه اصلی ضبط صدا"""
        try:
            self.recording_start_time = time.time()
            self.last_chunk_time = time.time()
            self.session_id = str(int(time.time() * 1000))  # ایجاد شناسه منحصر به فرد برای جلسه ضبط

            # شروع ضبط صدا در حالت استریم
            with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self.audio_callback):
                while self.recording:
                    current_time = time.time()
                    # اگر از آخرین چانک ۳۰ ثانیه گذشته و داده‌ای در چانک هست
                    if current_time - self.last_chunk_time >= self.chunk_duration and len(self.current_chunk) > 0:
                        self.process_current_chunk(is_final=False)
                    sd.sleep(100)

            # وقتی حلقه تمام شد، ضبط متوقف شده است:
            self.recording_duration = time.time() - self.recording_start_time

            # اگر هنوز داده‌ای در current_chunk وجود دارد:
            if self.current_chunk:
                self.process_current_chunk(is_final=True)

            # اگر چند تکه متن جمع شده باشد:
            if self.all_transcriptions:
                final_transcription = " ".join(self.all_transcriptions)
                self.finished.emit(final_transcription)

        except Exception as e:
            logging.error(f"Recording error: {str(e)}")
            self.error.emit(f"Recording error: {str(e)}")

    def audio_callback(self, indata, frames, time_info, status):
        """تابع callback برای ضبط صدا (هر بافر ورودی صدا را اینجا می‌گیریم)"""
        if status:
            logging.warning(f"Audio callback status: {status}")
        if self.recording:
            self.current_chunk.append(indata.copy())
            # اگر تعداد نمونه‌های صوتی از حد ۳۰ ثانیه بیشتر شود:
            total_samples = sum(len(chunk) for chunk in self.current_chunk)
            if total_samples >= self.chunk_samples:
                self.process_current_chunk(is_final=False)

    def process_current_chunk(self, is_final=False):
        """پردازش یک چانک (ترکیب داده‌ها + ساخت فایل موقت WAV + ارسال به سرور)"""
        try:
            if not self.current_chunk:
                return

            # ترکیب داده‌های چانک فعلی
            audio_array = np.concatenate(self.current_chunk, axis=0)

            # ساخت فایل موقت WAV
            temp_file = tempfile.mktemp(suffix='.wav')
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes((audio_array * 32767).astype(np.int16).tobytes())

            # ارسال به سرور
            transcription = self.send_chunk_to_backend(temp_file, is_final=is_final)
            if transcription:
                self.all_transcriptions.append(transcription)
                self.chunk_processed.emit(transcription)

            # پاک‌کردن فایل موقت و خالی‌کردن چانک
            os.remove(temp_file)
            self.current_chunk = []
            self.last_chunk_time = time.time()

        except Exception as e:
            logging.error(f"Chunk processing error: {str(e)}")
            self.error.emit(f"Chunk processing error: {str(e)}")

    def send_chunk_to_backend(self, audio_file, is_final=False):
        """ارسال فایل ضبط شده به سرور"""
        try:
            # اگر هنوز توکنی نگرفتیم، لاگین کنیم
            if not self.token:
                if not self.login():
                    self.error.emit("Authentication failed")
                    return None

            url = "https://backend.shaz.ai/process-chunk/"
            headers = {"Authorization": f"Bearer {self.token}"}
            
            with open(audio_file, 'rb') as f:
                files = {'file': ('chunk.wav', f, 'audio/wav')}
                data = {
                    'chunk_number': str(len(self.all_transcriptions)),
                    'session_id': self.session_id,
                    'is_final': str(is_final).lower(),
                    'model': 'openai/whisper-large-v3-turbo'
                }
                response = requests.post(url, files=files, data=data, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if is_final:
                    return data.get('transcription', '')
                else:
                    return data.get('chunk_transcription', '')
            elif response.status_code == 401:
                # اگر توکن منقضی شده دوباره لاگین کنیم
                if self.login():
                    return self.send_chunk_to_backend(audio_file, is_final=is_final)
            
            return None

        except Exception as e:
            logging.error(f"Chunk upload error: {str(e)}")
            return None

    def login(self):
        """لاگین به سرور با اعتبارنامه USB یا ثابت"""
        try:
            url = "https://backend.shaz.ai/token/"
            if self.usb_username and self.usb_password:
                data = {"username": self.usb_username, "password": self.usb_password}
            else:
                data = {"username": self.usb_username, "password": self.usb_password}
                
            response = requests.post(url, data=data)
            if response.status_code == 200:
                self.token = response.json().get('access_token')
                return True
            else:
                logging.error(f"Login failed: {response.status_code}")
                return False
        except Exception as e:
            logging.error(f"Login error: {str(e)}")
            return False

    def start_recording(self):
        """شروع ضبط صدا"""
        self.recording = True
        self.current_chunk = []
        self.all_transcriptions = []
        self.last_chunk_time = time.time()
        self.recording_start_time = time.time()
        self.recording_duration = 0
        self.start()

    def stop_recording(self):
        """توقف ضبط صدا"""
        self.recording = False

    def send_file_to_backend(self, file_path):
        """ارسال یک فایل صوتی موجود در فلش به سرور با یوزرنیم/پسورد USB"""
        try:
            if not self.login():
                return False

            url = "https://backend.shaz.ai/upload-audio/"
            headers = {"Authorization": f"Bearer {self.token}"}
            
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'audio/wav')}
                response = requests.post(url, files=files, headers=headers)
            
            return response.status_code == 200

        except Exception as e:
            logging.error(f"File upload error: {str(e)}")
            return False

class USBWatcher(QThread):
    """نخ (Thread) جداگانه برای مانیتورکردن فلش و ارسال فایل‌های خاص"""
    def __init__(self, recorder, poll_interval=5):
        super().__init__()
        self.recorder = recorder
        self.poll_interval = poll_interval
        self.already_processed = set()
        self.running = True

    def run(self):
        logging.info("USB monitoring started")
        while self.running:
            try:
                partitions = psutil.disk_partitions(all=False)
                for partition in partitions:
                    # تشخیص درایوهای قابل حمل
                    if 'removable' in partition.opts.lower() or partition.fstype == '':
                        mountpoint = partition.mountpoint
                        try:
                            for entry in os.listdir(mountpoint):
                                if fnmatch.fnmatch(entry, '*DJI_Audio_*'):
                                    folder_path = os.path.join(mountpoint, entry)
                                    if folder_path not in self.already_processed and os.path.isdir(folder_path):
                                        logging.info(f"Processing USB folder: {folder_path}")
                                        if not self.recorder.usb_username or not self.recorder.usb_password:
                                            logging.error("USB credentials not set")
                                            continue
                                        self.process_usb_folder(folder_path)
                                        self.already_processed.add(folder_path)
                        except Exception as e:
                            logging.error(f"Drive check error: {str(e)}")
            except Exception as e:
                logging.error(f"USB monitoring error: {str(e)}")
            time.sleep(self.poll_interval)

    def process_usb_folder(self, folder_path):
        """ارسال تمام فایل‌های wav موجود در پوشهٔ موردنظر به سرور"""
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    logging.info(f"Processing file: {file_path}")
                    success = self.recorder.send_file_to_backend(file_path)
                    if success:
                        logging.info(f"File {file} processed successfully")
                    else:
                        logging.error(f"File {file} processing failed")

    def stop(self):
        self.running = False

class CredentialDialog(QDialog):
    """پنجره تنظیم یوزرنیم/پسورد USB برای ضبط طولانی"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("USB Credential Configuration")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowModality(Qt.ApplicationModal)
        self.usb_username = None
        self.usb_password = None
        self.resize(300, 150)
        self.init_ui()

    def init_ui(self):
        try:
            layout = QVBoxLayout()
            
            user_label = QLabel("Username:")
            self.user_input = QLineEdit()
            self.user_input.setPlaceholderText("Enter username")
            
            pass_label = QLabel("Password:")
            self.pass_input = QLineEdit()
            self.pass_input.setPlaceholderText("Enter password")
            self.pass_input.setEchoMode(QLineEdit.Password)
            
            btn_box = QHBoxLayout()
            save_btn = QPushButton("Save")
            cancel_btn = QPushButton("Cancel")
            
            save_btn.clicked.connect(self.save_credentials)
            cancel_btn.clicked.connect(self.reject)
            
            btn_box.addWidget(save_btn)
            btn_box.addWidget(cancel_btn)
            
            layout.addWidget(user_label)
            layout.addWidget(self.user_input)
            layout.addWidget(pass_label)
            layout.addWidget(self.pass_input)
            layout.addSpacing(10)
            layout.addLayout(btn_box)
            
            self.setLayout(layout)
            
        except Exception as e:
            logging.error(f"Error initializing credential dialog: {str(e)}")
            QMessageBox.critical(self, "Error", "Error initializing dialog")

    def save_credentials(self):
        """ذخیره اطلاعات واردشده توسط کاربر در خود آبجکت دیالوگ"""
        try:
            username = self.user_input.text().strip()
            password = self.pass_input.text().strip()
            
            if not username or not password:
                QMessageBox.warning(self, "Error", "Please enter both username and password")
                return
            
            self.usb_username = username
            self.usb_password = password
            self.accept()
            
        except Exception as e:
            logging.error(f"Error saving credentials: {str(e)}")
            QMessageBox.critical(self, "Error", "Failed to save credentials")

class SystemTrayApp:
    """کلاس اصلی برنامه که نماد Tray را ایجاد می‌کند و ضبط صدا و غیره را کنترل می‌نماید."""
    def __init__(self):
        # ساخت QApplication
        self.app = QApplication(sys.argv)
        self.recorder = AudioRecorder()

        # بارگذاری یوزرنیم/پسورد USB اگر قبلاً ذخیره شده باشد
        self.load_usb_credentials()
        
        # اگر وجود ندارد، یک دیالوگ برای تنظیم آن نشان می‌دهیم
        if not self.recorder.usb_username or not self.recorder.usb_password:
            self.configure_usb_credentials()
        
        # اتصال سیگنال‌های ضبط به اسلات‌ها
        self.recorder.finished.connect(self.on_transcription_received)
        self.recorder.error.connect(self.show_error)
        self.recorder.chunk_processed.connect(self.on_chunk_processed)
        
        # شروع به کار مانیتور فلش
        self.usb_watcher = USBWatcher(self.recorder)
        self.usb_watcher.start()
        
        # آماده‌سازی وضعیت کلید F12
        self.last_key_press = 0
        self.key_press_interval = 0.5
        self.is_recording = False
        
        # ساخت نماد Tray
        self.tray = QSystemTrayIcon()
        self.tray.setIcon(self.create_icon())
        self.tray.setVisible(True)
        
        # ایجاد منوی کلیک راست روی Tray
        self.accumulated_text = ""
        self.setup_menu()
        
        # شنود کلید F12
        keyboard.on_press_key("f12", self.handle_key_press)
        
        # پیام اولیه
        self.tray.showMessage(
            "Audio Recorder",
            "Double-press F12 key to start/stop recording",
            QSystemTrayIcon.Information,
            3000
        )

    def create_icon(self):
        """آیکن پیشفرض (دایره قرمز)"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setBrush(QColor(255, 0, 0))  # قرمز
        painter.drawEllipse(0, 0, 32, 32)
        painter.end()
        return QIcon(pixmap)

    def create_recording_icon(self):
        """آیکن حین ضبط"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setBrush(QColor(255, 0, 0))  # قرمز
        painter.drawEllipse(0, 0, 32, 32)
        painter.end()
        return QIcon(pixmap)

    def create_idle_icon(self):
        """آیکن وقتی ضبط متوقف است"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setBrush(QColor(0, 128, 0))  # سبز
        painter.drawEllipse(0, 0, 32, 32)
        painter.end()
        return QIcon(pixmap)

    def setup_menu(self):
        """ساخت منوی راست‌کلیک Tray"""
        self.menu = QMenu()
        self.menu.addAction("Recording Status: Double-press F12 to start/stop")
        self.menu.addSeparator()
        self.menu.addAction("Configure USB Credentials", self.configure_usb_credentials)
        self.menu.addSeparator()
        self.menu.addAction("Exit", self.quit_app)
        self.tray.setContextMenu(self.menu)

    def handle_key_press(self, event):
        """شناسایی دوبار فشار دادن کلید F12 برای استارت/استاپ ضبط"""
        current_time = time.time()
        time_diff = current_time - self.last_key_press
        
        if time_diff < self.key_press_interval:
            # اگر فاصله دو کلیک کمتر از ۰٫۵ ثانیه باشد، دستور دوبار فشردن است
            if self.is_recording:
                self.stop_recording()
            else:
                self.start_recording()
        
        self.last_key_press = current_time

    def start_recording(self):
        """شروع ضبط با کلیک دوبار F12"""
        if not self.is_recording:
            self.is_recording = True
            self.accumulated_text = ""
            self.tray.setIcon(self.create_recording_icon())
            self.tray.showMessage(
                "Audio Recording",
                "Recording started...",
                QSystemTrayIcon.Information,
                1000
            )
            self.recorder.start_recording()

    def stop_recording(self):
        """توقف ضبط با کلیک دوبار F12"""
        if self.is_recording:
            self.is_recording = False
            self.tray.setIcon(self.create_idle_icon())
            self.tray.showMessage(
                "Audio Recording",
                "Recording stopped...",
                QSystemTrayIcon.Information,
                1000
            )
            self.recorder.stop_recording()

    def on_chunk_processed(self, text):
        """هر بار که یک تکه صوتی ارسال و متن آن تشخیص داده شد"""
        self.accumulated_text += " " + text
        self.tray.showMessage(
            "Chunk Processed",
            "New segment transcribed",
            QSystemTrayIcon.Information,
            1000
        )

    def on_transcription_received(self, text):
        """در پایان کل ضبط، نتیجهٔ متن نهایی"""
        pyperclip.copy(self.accumulated_text.strip())
        keyboard.press_and_release('ctrl+v')
        self.accumulated_text = ""
        self.tray.showMessage(
            "Success",
            "Recording transcribed and pasted!",
            QSystemTrayIcon.Information,
            2000
        )

    def show_error(self, error_msg):
        """نمایش خطا به‌صورت پیام سیستم تری"""
        self.tray.showMessage(
            "Error",
            error_msg,
            QSystemTrayIcon.Critical,
            3000
        )

    def configure_usb_credentials(self):
        """بازکردن دیالوگی برای تنظیم یوزرنیم/پسورد USB"""
        try:
            dialog = CredentialDialog(None)  # parent=None
            # اگر از قبل مقداری داریم، در فیلدها قرار دهیم
            if self.recorder.usb_username and self.recorder.usb_password:
                dialog.user_input.setText(self.recorder.usb_username)
                dialog.pass_input.setText(self.recorder.usb_password)
            
            if dialog.exec_() == QDialog.Accepted and dialog.username and dialog.password:
                # ذخیره در شی recorder
                self.recorder.usb_username = dialog.username
                self.recorder.usb_password = dialog.password
                
                # ذخیره در فایل
                if self.save_usb_credentials():
                    self.tray.showMessage(
                        "Success",
                        "USB credentials saved successfully",
                        QSystemTrayIcon.Information,
                        3000
                    )
                    return True
                else:
                    self.show_error("Failed to save credentials")
            return False
        except Exception as e:
            logging.error(f"Error in configure_usb_credentials: {str(e)}")
            self.show_error(f"Error configuring credentials: {str(e)}")
            return False

    def load_usb_credentials(self):
        """خواندن یوزرنیم/پسورد USB از فایل JSON"""
        try:
            if os.path.exists(CREDENTIALS_FILE):
                with open(CREDENTIALS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.recorder.usb_username = data.get("usb_username")
                    self.recorder.usb_password = data.get("usb_password")
                    logging.info("USB credentials loaded successfully")
                    return True
            return False
        except Exception as e:
            logging.error(f"Error loading USB credentials: {str(e)}")
            return False

    def save_usb_credentials(self):
        """ذخیرهٔ یوزرنیم/پسورد USB در فایل JSON"""
        try:
            data = {
                "usb_username": self.recorder.usb_username,
                "usb_password": self.recorder.usb_password
            }
            os.makedirs(os.path.dirname(CREDENTIALS_FILE), exist_ok=True)
            with open(CREDENTIALS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            logging.info("USB credentials saved successfully")
            return True
        except Exception as e:
            logging.error(f"Error saving USB credentials: {str(e)}")
            self.show_error(f"Failed to save credentials: {str(e)}")
            return False

    def quit_app(self):
        """خروج تمیز از برنامه"""
        if self.is_recording:
            self.stop_recording()
        self.usb_watcher.stop()
        self.usb_watcher.wait()
        self.tray.hide()
        QApplication.quit()

    def run(self):
        """اجرای برنامه تا پایان"""
        os.makedirs(app_data_dir, exist_ok=True)
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    try:
        main_app = SystemTrayApp()
        main_app.run()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        app = QApplication(sys.argv)
        QMessageBox.critical(None, "Error", f"Application error: {str(e)}")
        sys.exit(1)
