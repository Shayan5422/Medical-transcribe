import sys
import os
import json
import time
import keyboard
import sounddevice as sd
import requests
import pyperclip
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QSystemTrayIcon,
    QMenu, QMessageBox, QFrame, QHBoxLayout, QProgressBar,
    QDialog, QListWidget, QListWidgetItem, QComboBox, 
    QDialogButtonBox, QMenuBar, QAction, QFormLayout,
    QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QStandardPaths
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor, QPixmap, QPainter
import numpy as np
import wave
import tempfile

# Fonctions auxiliaires pour gérer les chemins de configuration
def get_config_directory():
    """Retourne le chemin du répertoire de configuration approprié en fonction du système d'exploitation."""
    config_dir = QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    return config_dir

def get_config_path(filename):
    """Retourne le chemin complet du fichier de configuration."""
    return os.path.join(get_config_directory(), filename)

# Composants UI de base
class ModernQLineEdit(QLineEdit):
    def __init__(self, placeholder="", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setMinimumHeight(40)
        self.setStyleSheet("""
            QLineEdit {
                padding: 8px 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                background-color: white;
                font-size: 14px;
                selection-background-color: #4a90e2;
            }
            QLineEdit:focus {
                border: 2px solid #4a90e2;
                background-color: #f8f9fa;
            }
            QLineEdit:hover {
                border: 2px solid #4a90e2;
            }
        """)

class ModernQPushButton(QPushButton):
    def __init__(self, text, color="#4a90e2", parent=None):
        super().__init__(text, parent)
        self.setMinimumHeight(40)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self._adjust_color(color, 1.1)};
            }}
            QPushButton:pressed {{
                background-color: {self._adjust_color(color, 0.9)};
            }}
            QPushButton:disabled {{
                background-color: #cccccc;
                color: #666666;
            }}
        """)
    
    def _adjust_color(self, hex_color, factor):
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        rgb = tuple(min(255, max(0, int(x * factor))) for x in rgb)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

class ModernQProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(15)
        self.setMaximumHeight(15)
        self.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 7px;
                background-color: #f0f0f0;
                text-align: center;
            }
            QProgressBar::chunk {
                border-radius: 7px;
                background-color: #4a90e2;
            }
        """)

class ModernQFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #e0e0e0;
            }
        """)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

# Dialogues
class SettingsDialog(QDialog):
    def __init__(self, parent=None, current_backend="", current_model="openai/whisper-large-v3-turbo", models=None):
        super().__init__(parent)
        self.setWindowTitle("Paramètres")
        self.setFixedSize(400, 200)  # Augmenter la taille pour inclure le modèle
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f6fa;
            }
        """)

        self.current_backend = current_backend
        self.current_model = current_model
        self.models = models if models else {'Rapide': 'openai/whisper-large-v3-turbo', 'Précis': 'openai/whisper-large-v3'}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        # Entrée de l'URL du backend
        url_label = QLabel("URL du Backend :")
        url_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.backend_input = ModernQLineEdit()
        self.backend_input.setText(self.current_backend)
        form_layout.addRow(url_label, self.backend_input)

        # Sélection du modèle
        model_label = QLabel("Modèle de Transcription :")
        model_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.model_combo = QComboBox()
        for name in self.models:
            self.model_combo.addItem(name)
        self.model_combo.setCurrentText(self.get_model_name(self.current_model))
        form_layout.addRow(model_label, self.model_combo)

        layout.addLayout(form_layout)

        # Boutons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.setStyleSheet("""
            QPushButton {
                min-width: 80px;
                min-height: 30px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton[text="OK"] {
                background-color: #4a90e2;
                color: white;
                border: none;
            }
            QPushButton[text="OK"]:hover {
                background-color: #357abd;
            }
            QPushButton[text="Cancel"] {
                background-color: #e0e0e0;
                border: none;
            }
            QPushButton[text="Cancel"]:hover {
                background-color: #cccccc;
            }
        """)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_model_name(self, model_value):
        for name, value in self.models.items():
            if value == model_value:
                return name
        return 'Rapide'  # Par défaut

    def get_backend_url(self):
        return self.backend_input.text()

    def get_selected_model(self):
        model_name = self.model_combo.currentText()
        return self.models.get(model_name, 'openai/whisper-large-v3-turbo')

class AutoShareDialog(QDialog):
    def __init__(self, parent=None, users=None, current_configs=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration AutoShare")
        self.setMinimumWidth(600)
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f6fa;
            }
            QListWidget {
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #4a90e2;
                color: white;
            }
            QComboBox {
                padding: 8px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                min-height: 40px;
            }
        """)
        
        self.users = users if users else []
        self.current_configs = current_configs if current_configs else []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Section de la liste des utilisateurs
        user_frame = ModernQFrame()
        user_layout = QVBoxLayout(user_frame)
        user_layout.setSpacing(10)
        
        user_label = QLabel("Sélectionnez les utilisateurs pour AutoShare :")
        user_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        user_layout.addWidget(user_label)

        self.user_list = QListWidget()
        for user in self.users:
            item = QListWidgetItem(user['username'])
            item.setData(Qt.UserRole, user['id'])
            self.user_list.addItem(item)
        user_layout.addWidget(self.user_list)
        layout.addWidget(user_frame)

        # Section du type d'accès
        access_frame = ModernQFrame()
        access_layout = QVBoxLayout(access_frame)
        access_layout.setSpacing(10)

        access_label = QLabel("Type d'accès :")
        access_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        access_layout.addWidget(access_label)

        self.access_type_combo = QComboBox()
        self.access_type_combo.addItems(['viewer', 'editor'])
        access_layout.addWidget(self.access_type_combo)
        layout.addWidget(access_frame)

        # Bouton pour ajouter une configuration
        add_button = ModernQPushButton("Ajouter une Configuration", "#27ae60")
        add_button.clicked.connect(self.add_config)
        layout.addWidget(add_button)

        # Section des configurations actuelles
        config_frame = ModernQFrame()
        config_layout = QVBoxLayout(config_frame)
        config_layout.setSpacing(10)

        config_label = QLabel("Configurations AutoShare Actuelles :")
        config_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        config_layout.addWidget(config_label)

        self.config_list = QListWidget()
        self.refresh_config_list()
        config_layout.addWidget(self.config_list)
        layout.addWidget(config_frame)

        # Bouton pour supprimer une configuration
        remove_button = ModernQPushButton("Supprimer la Sélection", "#e74c3c")
        remove_button.clicked.connect(self.remove_config)
        layout.addWidget(remove_button)

        # Boutons du dialogue
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def refresh_config_list(self):
        self.config_list.clear()
        for config in self.current_configs:
            username = next((user['username'] for user in self.users if user['id'] == config['userId']), "Utilisateur Inconnu")
            item_text = f"{username} - {config['accessType']}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, config['userId'])
            self.config_list.addItem(item)

    def add_config(self):
        selected_items = self.user_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Avertissement", "Veuillez sélectionner un utilisateur")
            return

        user_id = selected_items[0].data(Qt.UserRole)
        access_type = self.access_type_combo.currentText()

        if any(config['userId'] == user_id for config in self.current_configs):
            QMessageBox.warning(self, "Avertissement", "Cet utilisateur a déjà une configuration")
            return

        self.current_configs.append({
            'userId': user_id,
            'accessType': access_type
        })
        self.refresh_config_list()

    def remove_config(self):
        selected_items = self.config_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Avertissement", "Veuillez sélectionner une configuration à supprimer")
            return

        for item in selected_items:
            user_id = item.data(Qt.UserRole)
            self.current_configs = [config for config in self.current_configs if config['userId'] != user_id]
        self.refresh_config_list()

    def get_configs(self):
        return self.current_configs

class RecorderThread(QThread):
    finished = pyqtSignal(str, int)  # Signal for final text and upload ID
    error = pyqtSignal(str)
    transcription_update = pyqtSignal(str)
    recording_level = pyqtSignal(float)

    def __init__(self, api_url, token, model):
        super().__init__()
        self.api_url = api_url
        self.token = token
        self.model = model
        self.is_recording = False
        self.audio_data = []
        self.chunk_duration = 10  # Duration in seconds
        self.session_id = str(int(time.time()))
        self.chunk_number = 0
        self.selectedUploadId = None
        self.retry_max = 3  # Maximum retry attempts

    def process_chunk(self, is_final=False, retry_count=0):
        if len(self.audio_data) > 0:
            try:
                # Create WAV file from audio data
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                    audio_concat = np.concatenate(self.audio_data)
                    with wave.open(temp_wav.name, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(16000)
                        wav_file.writeframes(audio_concat.tobytes())

                    # Prepare form data
                    files = {
                        'file': ('chunk.wav', open(temp_wav.name, 'rb'), 'audio/wav')
                    }
                    data = {
                        'chunk_number': self.chunk_number,
                        'session_id': self.session_id,
                        'is_final': str(is_final).lower(),
                        'model': self.model
                    }
                    headers = {'Authorization': f'Bearer {self.token}'}

                    # Send request to server
                    response = requests.post(
                        f"{self.api_url}/process-chunk/",
                        files=files,
                        data=data,
                        headers=headers
                    )

                    if response.status_code == 200:
                        result = response.json()
                        
                        if is_final:
                            final_trans = result.get('transcription', '')
                            self.selectedUploadId = result.get('upload_id', None)
                            if final_trans and self.selectedUploadId:
                                self.finished.emit(final_trans, self.selectedUploadId)
                        else:
                            chunk_trans = result.get('chunk_transcription', '')
                            if chunk_trans:
                                # Emit only the current chunk transcription
                                self.transcription_update.emit(chunk_trans)
                    else:
                        raise Exception(f"Server error: {response.status_code}")

            except Exception as e:
                # Implement retry logic
                if retry_count < self.retry_max and not is_final:
                    time.sleep(1 * (retry_count + 1))  # Exponential backoff
                    self.process_chunk(is_final, retry_count + 1)
                else:
                    self.error.emit(str(e))
                    if is_final:
                        self.finished.emit("", None)  # Signal completion with error

            finally:
                # Cleanup
                self.audio_data = []
                self.chunk_number += 1

    def run(self):
        try:
            samplerate = 16000
            channels = 1
            dtype = np.int16
            samples_per_chunk = samplerate * self.chunk_duration
            current_samples = 0

            def audio_callback(indata, frames, time_info, status):
                nonlocal current_samples
                if self.is_recording:
                    self.audio_data.append(indata.copy())
                    current_samples += frames

                    # Calculate and emit audio level
                    if len(indata) > 0:
                        audio_level = np.max(np.abs(indata)) / 32768.0
                        self.recording_level.emit(audio_level)

                    # Process chunk when duration is reached
                    if current_samples >= samples_per_chunk:
                        self.process_chunk(False)
                        current_samples = 0

            with sd.InputStream(
                samplerate=samplerate,
                channels=channels,
                dtype=dtype,
                callback=audio_callback
            ):
                self.is_recording = True
                while self.is_recording:
                    sd.sleep(100)

                # Process final chunk
                if self.audio_data:
                    self.process_chunk(True)

        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialisations de base
        self.recorder_thread = None
        self.current_transcription = ""
        self.is_recording = False
        self.f12_count = 0
        self.last_f12_time = 0
        self.api_url = "https://backend.shaz.ai"
        self.autoshare_configs = []
        self.token = None  # Initialiser l'attribut token
        
        # Liste des modèles disponibles
        self.models = {
            'Rapide': 'openai/whisper-large-v3-turbo',
            'Précis': 'openai/whisper-large-v3'
        }
        self.selected_model = 'openai/whisper-large-v3-turbo'  # Modèle par défaut
        
        # Créer et configurer l'icône de la barre des tâches avant tout
        self.setup_tray()

        # Initialiser les composants de l'interface utilisateur
        self.init_ui()
        
        # Charger les configurations et autres configurations
        self.cleanup_temp_files()
        self.load_config()
        self.load_autoshare_configs()
        self.setup_menu_bar()
        self.setup_global_hotkey()

    def setup_tray(self):
        """Configurer l'icône de la barre des tâches et le menu"""
        self.tray_icon = QSystemTrayIcon(self)
        
        # Créer les icônes
        self.normal_icon = self.create_normal_icon()
        self.recording_icon = self.create_recording_icon()
        
        # Définir l'icône initiale
        self.tray_icon.setIcon(self.normal_icon)
        self.setWindowIcon(self.normal_icon)
        self.tray_icon.setToolTip('Transcription Vocale')
        
        # Créer le menu de la barre des tâches
        tray_menu = QMenu()
        tray_menu.setStyleSheet("""
            QMenu {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                padding: 5px;
            }
            QMenu::item {
                padding: 8px 20px;
                border-radius: 3px;
                font-size: 13px;
            }
            QMenu::item:selected {
                background-color: #f0f0f0;
            }
            QMenu::separator {
                height: 1px;
                background-color: #e0e0e0;
                margin: 5px 15px;
            }
        """)

        # Ajouter les éléments du menu
        show_window_action = tray_menu.addAction("Ouvrir la Fenêtre")
        show_window_action.triggered.connect(self.show_main_window)
        
        tray_menu.addSeparator()
        
        settings_action = tray_menu.addAction("Paramètres")
        settings_action.triggered.connect(self.open_backend_dialog)
        
        autoshare_action = tray_menu.addAction("AutoShare")
        autoshare_action.triggered.connect(self.open_autoshare_dialog)
        
        tray_menu.addSeparator()
        
        self.toggle_recording_action = tray_menu.addAction("Démarrer l'Enregistrement")
        self.toggle_recording_action.triggered.connect(self.toggle_recording)
        
        tray_menu.addSeparator()
        
        quit_action = tray_menu.addAction("Quitter")
        quit_action.triggered.connect(self.quit_application)

        # Définir le menu et afficher l'icône
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.tray_icon_activated)
        self.tray_icon.show()

    def create_normal_icon(self):
        """Create normal state icon"""
        size = 64
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw microphone body
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor('#4a90e2'))
            
            # Main circle
            painter.drawEllipse(12, 12, 40, 40)
            
            # Inner microphone
            painter.setBrush(QColor('white'))
            painter.drawRoundedRect(24, 16, 16, 32, 8, 8)
        finally:
            painter.end()
            
        return QIcon(pixmap)

    def create_recording_icon(self):
        """Create recording state icon"""
        size = 64
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw recording indicator
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor('#e74c3c'))  # Red for recording
            
            # Main circle
            painter.drawEllipse(12, 12, 40, 40)
            
            # Inner recording symbol
            painter.setBrush(QColor('white'))
            painter.drawEllipse(24, 24, 16, 16)
        finally:
            painter.end()
            
        return QIcon(pixmap)

    def update_recording_status(self, is_recording):
        """Mettre à jour l'icône de la barre des tâches et le menu en fonction du statut d'enregistrement"""
        icon = self.recording_icon if is_recording else self.normal_icon
        self.tray_icon.setIcon(icon)
        
        action_text = "Arrêter l'Enregistrement" if is_recording else "Démarrer l'Enregistrement"
        self.toggle_recording_action.setText(action_text)
        
        status = "Enregistrement..." if is_recording else "Prêt"
        self.tray_icon.setToolTip(f'Transcription Vocale - {status}')

    def show_main_window(self):
        """Afficher et activer la fenêtre principale"""
        self.show()
        self.activateWindow()
        self.raise_()

    def quit_application(self):
        """Quitter proprement l'application"""
        self.save_config()
        self.save_autoshare_configs()
        QApplication.quit()

    def tray_icon_activated(self, reason):
        """Gérer l'activation de l'icône de la barre des tâches"""
        if reason == QSystemTrayIcon.DoubleClick:
            self.show_main_window()

    def init_ui(self):
        # Créer le widget central et la mise en page principale
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # Ajouter le titre
        title_label = QLabel("Transcription Vocale")
        title_label.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 20px;
            padding: 10px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Créer les sections principales
        self.create_login_section(main_layout)
        self.create_recording_section(main_layout)
        self.create_autoshare_section(main_layout)
        
        # Ajouter la section de sélection du modèle
        self.create_model_selection_section(main_layout)

    def create_login_section(self, parent_layout):
        login_frame = ModernQFrame()
        login_layout = QVBoxLayout(login_frame)
        login_layout.setSpacing(15)
        login_layout.setContentsMargins(20, 20, 20, 20)

        # Nom d'utilisateur
        username_label = QLabel("Nom d'utilisateur :")
        username_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.username_input = ModernQLineEdit(placeholder="Entrez votre nom d'utilisateur")
        login_layout.addWidget(username_label)
        login_layout.addWidget(self.username_input)

        # Mot de passe
        password_label = QLabel("Mot de passe :")
        password_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.password_input = ModernQLineEdit(placeholder="Entrez votre mot de passe")
        self.password_input.setEchoMode(QLineEdit.Password)
        login_layout.addWidget(password_label)
        login_layout.addWidget(self.password_input)

        # Bouton de connexion
        self.login_button = ModernQPushButton("Connexion")
        self.login_button.clicked.connect(self.login)
        login_layout.addWidget(self.login_button)

        # Bouton de configuration AutoShare
        self.autoshare_button = ModernQPushButton("Configurer AutoShare", "#2ecc71")
        self.autoshare_button.clicked.connect(self.open_autoshare_dialog)
        self.autoshare_button.setEnabled(False)
        login_layout.addWidget(self.autoshare_button)

        parent_layout.addWidget(login_frame)

    def create_recording_section(self, parent_layout):
        recording_frame = ModernQFrame()
        recording_layout = QVBoxLayout(recording_frame)
        recording_layout.setSpacing(15)
        recording_layout.setContentsMargins(20, 20, 20, 20)

        # Label de statut
        self.status_label = QLabel("Appuyez deux fois rapidement sur F12 pour démarrer/arrêter l'enregistrement")
        self.status_label.setStyleSheet("""
            font-size: 14px;
            color: #2c3e50;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        recording_layout.addWidget(self.status_label)

        # Barre de progression du niveau audio
        self.level_bar = ModernQProgressBar()
        recording_layout.addWidget(self.level_bar)

        parent_layout.addWidget(recording_frame)

    def create_autoshare_section(self, parent_layout):
        autoshare_frame = ModernQFrame()
        autoshare_layout = QVBoxLayout(autoshare_frame)
        autoshare_layout.setSpacing(10)
        autoshare_layout.setContentsMargins(20, 20, 20, 20)

        self.autoshare_status_label = QLabel("AutoShare : Désactivé")
        self.autoshare_status_label.setStyleSheet("""
            font-size: 14px;
            color: #e74c3c;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
            font-weight: bold;
        """)
        self.autoshare_status_label.setAlignment(Qt.AlignCenter)
        autoshare_layout.addWidget(self.autoshare_status_label)

        parent_layout.addWidget(autoshare_frame)

    def create_model_selection_section(self, parent_layout):
        model_frame = ModernQFrame()
        model_layout = QVBoxLayout(model_frame)
        model_layout.setSpacing(15)
        model_layout.setContentsMargins(20, 20, 20, 20)

        # Label pour la sélection du modèle
        model_label = QLabel("Sélectionnez le Modèle de Transcription :")
        model_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        model_layout.addWidget(model_label)

        # ComboBox pour les modèles
        self.model_combo = QComboBox()
        for name in self.models:
            self.model_combo.addItem(name)
        self.model_combo.setCurrentText(self.get_model_name(self.selected_model))
        self.model_combo.currentIndexChanged.connect(self.change_model)
        model_layout.addWidget(self.model_combo)

        parent_layout.addWidget(model_frame)

    def get_model_name(self, model_value):
        for name, value in self.models.items():
            if value == model_value:
                return name
        return 'Rapide'  # Par défaut

    def change_model(self, index):
        model_name = self.model_combo.currentText()
        self.selected_model = self.models.get(model_name, 'openai/whisper-large-v3-turbo')
        self.save_config()
        self.show_notification("Succès", f"Modèle changé en {model_name}")

    def setup_menu_bar(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #ffffff;
                border-bottom: 1px solid #e0e0e0;
            }
            QMenuBar::item {
                padding: 8px 12px;
                background-color: transparent;
            }
            QMenuBar::item:selected {
                background-color: #f0f0f0;
            }
        """)

        # Menu Paramètres
        settings_menu = menubar.addMenu("Paramètres")
        backend_action = QAction("Modifier l'URL du Backend", self)
        backend_action.triggered.connect(self.open_backend_dialog)
        settings_menu.addAction(backend_action)

    def setup_global_hotkey(self):
        keyboard.on_press_key("F12", self.handle_f12)

    def handle_f12(self, event):
        current_time = time.time()
        if current_time - self.last_f12_time < 0.5:
            self.f12_count += 1
        else:
            self.f12_count = 1

        self.last_f12_time = current_time

        if self.f12_count == 2:
            self.f12_count = 0
            self.toggle_recording()

    def toggle_recording(self):
        if not self.token:
            self.show_error_message("Veuillez vous connecter d'abord")
            return

        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.update_recording_status(True)
        self.update_status("Enregistrement en cours...", "recording")
        self.recorder_thread = RecorderThread(self.api_url, self.token, self.selected_model)
        self.recorder_thread.finished.connect(self.handle_transcription)
        self.recorder_thread.error.connect(self.handle_error)
        self.recorder_thread.transcription_update.connect(self.handle_transcription_update)
        self.recorder_thread.recording_level.connect(self.update_level_bar)
        self.recorder_thread.start()

    def stop_recording(self):
        if self.recorder_thread:
            self.is_recording = False
            self.recorder_thread.is_recording = False
            self.update_recording_status(False)
            self.update_status("Traitement...", "processing")
            self.level_bar.setValue(0)

    def update_level_bar(self, level):
        self.level_bar.setValue(int(level * 100))

    def update_status(self, message, status_type="normal"):
        self.status_label.setText(message)
        if status_type == "recording":
            self.status_label.setStyleSheet("""
                font-size: 14px;
                color: #e74c3c;
                padding: 10px;
                background-color: #fdf0f0;
                border-radius: 5px;
                font-weight: bold;
            """)
        elif status_type == "processing":
            self.status_label.setStyleSheet("""
                font-size: 14px;
                color: #f39c12;
                padding: 10px;
                background-color: #fdf6e9;
                border-radius: 5px;
                font-weight: bold;
            """)
        elif status_type == "error":
            self.status_label.setStyleSheet("""
                font-size: 14px;
                color: #e74c3c;
                padding: 10px;
                background-color: #fdf0f0;
                border-radius: 5px;
                font-weight: bold;
            """)
        else:
            self.status_label.setStyleSheet("""
                font-size: 14px;
                color: #2c3e50;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
            """)

    def remove_duplicates(self, text):
        # Split into sentences (roughly)
        sentences = text.split('.')
        cleaned_sentences = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if this sentence is a duplicate or part of another sentence
            is_duplicate = False
            for j, other_sentence in enumerate(sentences):
                if i != j and sentence and other_sentence.strip():
                    # Check if this sentence is contained within another
                    if sentence in other_sentence:
                        is_duplicate = True
                        break
                        
            if not is_duplicate:
                cleaned_sentences.append(sentence)
                
        return '. '.join(cleaned_sentences)

    def handle_transcription_update(self, transcription):
        # Just store the transcription without pasting
        self.current_transcription = transcription
        self.update_status("Mise à jour de la transcription...", "processing")

    def handle_transcription(self, transcription, upload_id):
        if not transcription:
            return
            
        # Clean up any duplicate content
        cleaned_transcription = self.remove_duplicates(transcription)
        
        # Add a space and paste the final transcription
        final_text = cleaned_transcription + " "
        time.sleep(0.5)
        pyperclip.copy(final_text)
        keyboard.send('ctrl+v')
        self.current_transcription = final_text
            
        self.update_status("Transcription terminée", "normal")
        self.show_notification("Transcription terminée", "Le texte final a été copié")

        self.selectedUploadId = upload_id
        if self.autoshare_configs:
            for config in self.autoshare_configs:
                self.share_with_user(config['userId'], config['accessType'])

    def handle_error(self, error_msg):
        self.show_error_message(error_msg)
        self.update_status("Une erreur est survenue", "error")
        self.level_bar.setValue(0)

    def open_backend_dialog(self):
        dialog = SettingsDialog(self, current_backend=self.api_url, current_model=self.selected_model, models=self.models)
        if dialog.exec_() == QDialog.Accepted:
            new_url = dialog.get_backend_url().strip()
            new_model = dialog.get_selected_model()
            if new_url:
                self.api_url = new_url
            self.selected_model = new_model
            self.save_config()
            QMessageBox.information(self, "Succès", "URL du Backend et modèle mis à jour avec succès")

    def show_error_message(self, message):
        QMessageBox.warning(self, "Erreur", message, QMessageBox.Ok)

    def show_notification(self, title, message):
        self.tray_icon.showMessage(title, message, QSystemTrayIcon.Information, 2000)

    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        if not username or not password:
            self.show_error_message("Veuillez remplir tous les champs")
            return

        try:
            response = requests.post(
                f"{self.api_url}/token/",
                data={"username": username, "password": password}
            )

            if response.status_code == 200:
                self.token = response.json().get('access_token')
                self.save_config()
                self.update_status("Connexion réussie", "normal")
                self.login_button.setEnabled(False)
                self.username_input.setEnabled(False)
                self.password_input.setEnabled(False)
                self.autoshare_button.setEnabled(True)
                self.fetch_users()
                self.update_autoshare_status()
            else:
                self.show_error_message("Nom d'utilisateur ou mot de passe invalide")

        except Exception as e:
            self.show_error_message(f"Erreur de connexion : {str(e)}")

    def fetch_users(self):
        try:
            headers = {'Authorization': f'Bearer {self.token}'}
            response = requests.get(f"{self.api_url}/users/", headers=headers)
            if response.status_code == 200:
                self.users = response.json().get('users', [])
            else:
                self.show_error_message("Échec de la récupération des utilisateurs")
        except Exception as e:
            self.show_error_message(f"Erreur lors de la récupération des utilisateurs : {str(e)}")

    def open_autoshare_dialog(self):
        if not hasattr(self, 'users') or not self.users:
            self.show_error_message("Veuillez vous connecter d'abord pour configurer AutoShare")
            return

        dialog = AutoShareDialog(self, users=self.users, current_configs=self.autoshare_configs.copy())
        if dialog.exec_() == QDialog.Accepted:
            self.autoshare_configs = dialog.get_configs()
            self.save_autoshare_configs()
            self.update_autoshare_status()
            self.show_notification("Succès", "Configurations AutoShare mises à jour")

    def update_autoshare_status(self):
        if self.autoshare_configs:
            self.autoshare_status_label.setText(f"AutoShare : Actif pour {len(self.autoshare_configs)} utilisateur(s)")
            self.autoshare_status_label.setStyleSheet("""
                font-size: 14px;
                color: #27ae60;
                padding: 10px;
                background-color: #f0f9f4;
                border-radius: 5px;
                font-weight: bold;
            """)
        else:
            self.autoshare_status_label.setText("AutoShare : Désactivé")
            self.autoshare_status_label.setStyleSheet("""
                font-size: 14px;
                color: #e74c3c;
                padding: 10px;
                background-color: #fdf0f0;
                border-radius: 5px;
                font-weight: bold;
            """)

    def load_config(self):
        config_path = get_config_path('config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.username_input.setText(config.get('username', ''))
                    self.password_input.setText(config.get('password', ''))
                    self.api_url = config.get('backend_url', self.api_url)
                    self.selected_model = config.get('selected_model', self.selected_model)
                    # Mettre à jour le ComboBox avec le modèle chargé
                    model_name = self.get_model_name(self.selected_model)
                    index = self.model_combo.findText(model_name)
                    if index != -1:
                        self.model_combo.setCurrentIndex(index)
            except Exception as e:
                print(f"Erreur lors du chargement des paramètres : {e}")

    def save_config(self):
        try:
            config = {
                'username': self.username_input.text(),
                'password': self.password_input.text(),
                'backend_url': self.api_url,
                'selected_model': self.selected_model
            }
            config_path = get_config_path('config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des paramètres : {e}")

    def load_autoshare_configs(self):
        autoshare_path = get_config_path('autoshare_config.json')
        if os.path.exists(autoshare_path):
            try:
                with open(autoshare_path, 'r') as f:
                    self.autoshare_configs = json.load(f)
            except Exception as e:
                print(f"Erreur lors du chargement des configurations AutoShare : {e}")

    def save_autoshare_configs(self):
        try:
            autoshare_path = get_config_path('autoshare_config.json')
            with open(autoshare_path, 'w') as f:
                json.dump(self.autoshare_configs, f, indent=4)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des configurations AutoShare : {e}")

    def share_with_user(self, user_id, access_type):
        if not hasattr(self, 'selectedUploadId') or not self.selectedUploadId:
            self.show_error_message("Aucune transcription à partager")
            return

        try:
            headers = {'Authorization': f'Bearer {self.token}'}
            payload = {
                'user_id': user_id,
                'access_type': access_type
            }
            response = requests.post(
                f"{self.api_url}/share/{self.selectedUploadId}/user/",
                json=payload,
                headers=headers
            )
            
            if response.status_code not in [200, 201]:
                self.show_error_message(f"Échec du partage avec l'utilisateur ID {user_id}")
                
        except Exception as e:
            print(f"Erreur lors du partage avec l'utilisateur ID {user_id} : {e}")

    def cleanup_temp_files(self):
        temp_dir = tempfile.gettempdir()
        for filename in os.listdir(temp_dir):
            if filename.endswith('.wav'):
                try:
                    file_path = os.path.join(temp_dir, filename)
                    os.unlink(file_path)
                except Exception as e:
                    print(f"Impossible de supprimer le fichier temporaire {file_path} : {e}")

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        self.show_notification(
            "Application Minimisée",
            "L'application continue de fonctionner en arrière-plan"
        )

def main():
    app = QApplication(sys.argv)
    
    # Définir la police globale
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Utiliser le style Fusion
    app.setStyle("Fusion")
    
    # Créer et afficher la fenêtre principale
    main_window = MainWindow()
    main_window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
