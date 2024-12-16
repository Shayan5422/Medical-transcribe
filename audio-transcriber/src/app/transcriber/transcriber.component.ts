// transcriber.component.ts
import { Component, OnInit } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { jsPDF } from 'jspdf';

interface UploadHistory {
  upload_id: number;
  filename: string;
  transcription_filename: string;
  upload_time: string;
}

@Component({
  selector: 'app-transcriber',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './transcriber.component.html',
  styleUrls: ['./transcriber.component.css']
})
export class TranscriberComponent implements OnInit {
  transcription: string | null = null;
  transcriptionFile: string | null = null;
  isRecording: boolean = false;
  isTranscribing: boolean = false;
  mediaRecorder: MediaRecorder | null = null;
  audioChunks: Blob[] = [];
  selectedFile: File | null = null; // Fichier sélectionné
  history: UploadHistory[] = [];
  userId: number = 1; // Identifiant de l'utilisateur, à ajuster selon le contexte
  selectedUploadId: number | null = null;
  selectedTranscription: string | null = null;

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.fetchHistory();
  }

  // Gestion de la sélection de fichier
  onFileSelected(event: any): void {
    const file: File = event.target.files[0];
    if (file) {
      this.selectedFile = file;
      this.transcription = null; // Réinitialiser la transcription précédente
      this.transcriptionFile = null; // Réinitialiser le fichier de transcription précédent
      console.log('Fichier sélectionné:', file);
    }
  }

  // Upload du fichier sélectionné
  uploadSelectedFile(): void {
    if (this.selectedFile) {
      this.uploadAudio(this.selectedFile);
    } else {
      alert('Veuillez sélectionner un fichier audio avant de l\'uploader.');
    }
  }

  // Upload du fichier vers le serveur
  uploadAudio(file: File): void {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', this.userId.toString());

    console.log('Uploading file:', file);
    console.log('File type:', file.type);

    this.isTranscribing = true; // Démarrer la transcription

    this.http.post<any>('http://127.0.0.1:8000/upload-audio/', formData).subscribe(
      response => {
        console.log('Transcription response:', response);
        this.transcription = response.transcription;
        this.transcriptionFile = response.transcription_file;
        this.isTranscribing = false; // Fin de la transcription
        alert('Transcription réussie !');
        this.fetchHistory(); // Actualiser l'historique après un nouvel upload
      },
      (error: HttpErrorResponse) => {
        console.error('Erreur lors de l\'upload du fichier audio :', error);
        if (error.error && error.error.detail) {
          alert(`Erreur : ${error.error.detail}`);
        } else {
          alert('Erreur lors de l\'upload du fichier audio.');
        }
        this.isTranscribing = false; // Fin de la transcription en cas d'erreur
      }
    );
  }

  // Commencer l'enregistrement audio
  startRecording(): void {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
          let mimeType = 'audio/webm';
          if (!MediaRecorder.isTypeSupported(mimeType)) {
            mimeType = 'audio/ogg';
            if (!MediaRecorder.isTypeSupported(mimeType)) {
              mimeType = '';
            }
          }

          const options: MediaRecorderOptions = {};
          if (mimeType) {
            options.mimeType = mimeType;
          }

          this.mediaRecorder = new MediaRecorder(stream, options);
          this.audioChunks = [];
          this.mediaRecorder.start();
          this.isRecording = true;

          this.mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
              this.audioChunks.push(event.data);
            }
          };

          this.mediaRecorder.onstop = () => {
            const mimeType = this.mediaRecorder?.mimeType || 'audio/webm';
            console.log('MediaRecorder MIME type:', mimeType);
            const audioBlob = new Blob(this.audioChunks, { type: mimeType });
            const audioFile = new File([audioBlob], 'recording.webm', { type: mimeType });
            console.log('Recorded audio file:', audioFile);
            this.uploadAudio(audioFile);
          };
        })
        .catch(err => {
          console.error('Erreur lors de l\'accès au microphone :', err);
          alert('Impossible d\'accéder au microphone.');
        });
    } else {
      alert('Votre navigateur ne supporte pas l\'enregistrement audio.');
    }
  }

  // Arrêter l'enregistrement audio
  stopRecording(): void {
    if (this.mediaRecorder) {
      this.mediaRecorder.stop();
      this.isRecording = false;
    }
  }

  // Télécharger la transcription en tant que fichier texte
  downloadTranscription(): void {
    if (this.transcription) {
      const blob = new Blob([this.transcription], { type: 'text/plain' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'transcription.txt';
      a.click();
      window.URL.revokeObjectURL(url);
    }
  }

  // Télécharger la transcription depuis le serveur
  downloadTranscriptionFile(upload_id: number): void {
    const url = `http://127.0.0.1:8000/download-transcription/${upload_id}`;
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcription_${upload_id}.txt`;
    a.click();
  }

  // Télécharger le fichier audio depuis le serveur
  downloadAudioFile(upload_id: number): void {
    const url = `http://127.0.0.1:8000/download-audio/${upload_id}`;
    const a = document.createElement('a');
    a.href = url;
    a.download = `audio_${upload_id}.wav`;
    a.click();
  }

  // Télécharger la transcription en PDF (Optionnel)
  downloadTranscriptionAsPDF(upload_id: number): void {
    if (this.selectedTranscription) {
      const doc = new jsPDF();
      const lines = doc.splitTextToSize(this.selectedTranscription, 180);
      doc.text(lines, 10, 10);
      doc.save(`transcription_${upload_id}.pdf`);
    } else {
      alert('Aucune transcription sélectionnée pour le téléchargement.');
    }
  }

  // Sélectionner une entrée de l'historique et afficher la transcription dans la barre latérale
  selectUpload(upload_id: number): void {
    this.selectedUploadId = upload_id;
    this.http.get<any>(`http://127.0.0.1:8000/get-transcription/${upload_id}`).subscribe(
      response => {
        console.log('Transcription récupérée:', response.transcription);
        this.selectedTranscription = response.transcription;
      },
      (error: HttpErrorResponse) => {
        console.error('Erreur lors de la récupération de la transcription :', error);
        alert('Erreur lors de la récupération de la transcription.');
      }
    );
  }

  // Récupérer l'historique des uploads de l'utilisateur
  fetchHistory(): void {
    this.http.get<any>(`http://127.0.0.1:8000/history/${this.userId}`).subscribe(
      response => {
        console.log('Réponse de l\'historique:', response);
        this.history = response.history;
      },
      (error: HttpErrorResponse) => {
        console.error('Erreur lors de la récupération de l\'historique :', error);
        alert('Erreur lors de la récupération de l\'historique.');
      }
    );
  }
}
