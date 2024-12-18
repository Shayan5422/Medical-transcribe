import { Component, OnInit, NgZone } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms'; // Importer FormsModule pour ngModel
import { HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
import { Router } from '@angular/router';
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
  imports: [CommonModule, FormsModule],
  templateUrl: './transcriber.component.html',
  styleUrls: ['./transcriber.component.css']
})
export class TranscriberComponent implements OnInit {
  transcription: string | null = null;
  transcriptionFile: string | null = null;
  isRecording: boolean = false;
  isTranscribing: boolean = false;
  mediaRecorder: MediaRecorder | null = null;
  mediaStream: MediaStream | null = null; // Ajouté pour gérer le flux média
  audioChunks: Blob[] = [];
  selectedFile: File | null = null; // Fichier sélectionné
  history: UploadHistory[] = [];
  token: string | null = null;
  selectedUploadId: number | null = null;
  selectedTranscription: string | null = null;
  currentTheme: string = 'light';
  isEditing: boolean = false;
  editedTranscription: string = '';
  showCopySuccess: boolean = false;

  
  constructor(
    private http: HttpClient,
    private router: Router,
    private ngZone: NgZone // Injecter NgZone
  ) {}

  ngOnInit(): void {
    this.token = localStorage.getItem('token');
    console.log('Token récupéré :', this.token);
    if (this.token) {
      this.fetchHistory();
    } else {
      console.log('Aucun token trouvé, redirection vers la page de connexion.');
      this.router.navigate(['/login']);
    }

    
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      this.currentTheme = savedTheme;
    }
    document.documentElement.setAttribute('data-theme', this.currentTheme);
  }

  toggleTheme(): void {
    this.currentTheme = this.currentTheme === 'light' ? 'dark' : 'light';
    // Mettre à jour l'attribut data-theme
    document.documentElement.setAttribute('data-theme', this.currentTheme);
    // Sauvegarde du thème dans le localStorage (optionnel)
    localStorage.setItem('theme', this.currentTheme);
  }

  // Gérer la sélection de fichier
  onFileSelected(event: any): void {
    const file: File = event.target.files[0];
    if (file) {
      this.selectedFile = file;
      this.transcription = null; // Réinitialiser la transcription précédente
      this.transcriptionFile = null; // Réinitialiser le fichier de transcription précédent
      console.log('Fichier sélectionné :', file);
    }
  }

  // Télécharger le fichier sélectionné
  uploadSelectedFile(): void {
    if (this.selectedFile) {
      this.uploadAudio(this.selectedFile);
    } else {
      alert('Veuillez sélectionner un fichier audio avant de le télécharger.');
    }
  }

  // Télécharger le fichier audio vers le serveur
  // Dans la méthode selectUpload
selectUpload(upload_id: number): void {
  this.selectedUploadId = upload_id;
  this.transcription = null; // Réinitialiser la transcription courante
  console.log(`Sélection de l'upload ID : ${upload_id}`);

  const headers = new HttpHeaders({
    'Authorization': `Bearer ${this.token}`
  });

  this.http.get<any>(`/api/get-transcription/${upload_id}`, { headers }).subscribe(
    response => {
      console.log('Transcription récupérée :', response.transcription);
      this.ngZone.run(() => {
        this.selectedTranscription = response.transcription;
      });
      console.log('selectedTranscription définie sur :', this.selectedTranscription);
    },
    (error: HttpErrorResponse) => {
      console.error('Erreur lors de la récupération de la transcription :', error);
      alert('Erreur lors de la récupération de la transcription.');
    }
  );
}

// Dans la méthode uploadAudio
uploadAudio(file: File): void {
  const formData = new FormData();
  formData.append('file', file);

  const headers = new HttpHeaders({
    'Authorization': `Bearer ${this.token}`
  });

  console.log('Téléchargement du fichier :', file);
  console.log('Type de fichier :', file.type);

  this.ngZone.run(() => {
    this.isTranscribing = true;
    this.transcription = null;
    this.selectedTranscription = null; // Réinitialiser la transcription sélectionnée
    this.selectedUploadId = null; // Réinitialiser l'ID sélectionné
  });

  this.http.post<any>('/api/upload-audio/', formData, { headers }).subscribe(
    response => {
      console.log('Réponse de transcription :', response);
      this.ngZone.run(() => {
        this.transcription = response.transcription;
        this.transcriptionFile = response.transcription_file;
        this.isTranscribing = false;
      });
      
      console.log('Transcription définie :', this.transcription);
      this.fetchHistory();
    },
    (error: HttpErrorResponse) => {
      console.error('Erreur lors du téléchargement du fichier audio :', error);
      this.ngZone.run(() => {
        if (error.error && error.error.detail) {
          alert(`Erreur : ${error.error.detail}`);
        } else {
          alert('Erreur lors du téléchargement du fichier audio.');
        }
        this.isTranscribing = false;
      });
    }
  );
}

  // Démarrer l'enregistrement audio
  startRecording(): void {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
          this.mediaStream = stream; // Stocker le flux média
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
          console.log('Enregistrement démarré.');

          this.mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
              this.audioChunks.push(event.data);
              console.log('Données disponibles :', event.data);
            }
          };

          this.mediaRecorder.onstop = () => {
            const mimeType = this.mediaRecorder?.mimeType || 'audio/webm';
            console.log('Type MIME du MediaRecorder :', mimeType);
            const audioBlob = new Blob(this.audioChunks, { type: mimeType });
            const audioFile = new File([audioBlob], 'enregistrement.webm', { type: mimeType });
            console.log('Fichier audio enregistré :', audioFile);

            // Utiliser NgZone pour s'assurer que les changements sont détectés
            this.ngZone.run(() => {
              this.uploadAudio(audioFile);
            });
          };
        })
        .catch(err => {
          console.error('Erreur lors de l\'accès au microphone :', err);
          alert('Impossible d\'accéder au microphone.');
        });
    } else {
      alert('Votre navigateur ne prend pas en charge l\'enregistrement audio.');
    }
  }

  // Arrêter l'enregistrement audio
  stopRecording(): void {
    if (this.mediaRecorder) {
      this.mediaRecorder.stop();
      this.isRecording = false;
      console.log('Enregistrement arrêté.');
    }

    // Arrêter tous les tracks du flux média pour libérer le microphone
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
      console.log('Flux média arrêté pour libérer le microphone.');
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
      console.log('Transcription téléchargée en tant que fichier texte.');
    }
  }

  // Télécharger le fichier de transcription depuis le serveur
  downloadTranscriptionFile(upload_id: number): void {
    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });

    const url = `/api/download-transcription/${upload_id}`;
    this.http.get(url, { headers, responseType: 'blob' }).subscribe(blob => {
      const a = document.createElement('a');
      const objectUrl = URL.createObjectURL(blob);
      a.href = objectUrl;
      a.download = `transcription_${upload_id}.txt`;
      a.click();
      URL.revokeObjectURL(objectUrl);
      console.log(`Fichier de transcription transcription_${upload_id}.txt téléchargé.`);
    }, error => {
      console.error('Erreur lors du téléchargement du fichier de transcription :', error);
      alert('Erreur lors du téléchargement du fichier de transcription.');
    });
  }

  // Télécharger le fichier audio depuis le serveur
  downloadAudioFile(upload_id: number): void {
    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });

    const url = `/api/download-audio/${upload_id}`;
    this.http.get(url, { headers, responseType: 'blob' }).subscribe(blob => {
      const a = document.createElement('a');
      const objectUrl = URL.createObjectURL(blob);
      a.href = objectUrl;
      a.download = `audio_${upload_id}.wav`;
      a.click();
      URL.revokeObjectURL(objectUrl);
      console.log(`Fichier audio audio_${upload_id}.wav téléchargé.`);
    }, error => {
      console.error('Erreur lors du téléchargement du fichier audio :', error);
      alert('Erreur lors du téléchargement du fichier audio.');
    });
  }

  // Télécharger la transcription en tant que PDF (Optionnel)
  downloadTranscriptionAsPDF(upload_id: number): void {
    if (this.selectedTranscription) {
      const doc = new jsPDF();
      const lines = doc.splitTextToSize(this.selectedTranscription, 180);
      doc.text(lines, 10, 10);
      doc.save(`transcription_${upload_id}.pdf`);
      console.log(`Transcription téléchargée en tant que PDF : transcription_${upload_id}.pdf`);
    } else {
      alert('Aucune transcription sélectionnée pour le téléchargement.');
    }
  }

  

  // Récupérer l'historique des uploads spécifiques à l'utilisateur
  fetchHistory(): void {
    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });
  
    this.http.get<any>('/api/history/', { headers }).subscribe(
      response => {
        console.log('Réponse de l\'historique :', response);
        this.ngZone.run(() => {
          this.history = response.history; // Assurez-vous que 'history' est une propriété définie dans le composant
        });
        console.log('Historique mis à jour :', this.history);
      },
      (error: HttpErrorResponse) => {
        console.error('Erreur lors de la récupération de l\'historique :', error);
        alert('Erreur lors de la récupération de l\'historique.');
      }
    );
  }
  deleteUpload(upload_id: number): void {
    if (confirm('Êtes-vous sûr de vouloir supprimer cet enregistrement ?')) {
      const headers = new HttpHeaders({
        'Authorization': `Bearer ${this.token}`
      });

      this.http.delete(`/api/history/${upload_id}`, { headers }).subscribe(
        () => {
          this.fetchHistory();
          if (this.selectedUploadId === upload_id) {
            this.selectedUploadId = null;
            this.selectedTranscription = null;
          }
          
        },
        error => {
          console.error('Erreur lors de la suppression:', error);
          alert('Erreur lors de la suppression');
        }
      );
    }
  }

  startEditing(): void {
    this.isEditing = true;
    this.editedTranscription = this.selectedTranscription || '';
  }

  saveTranscription(): void {
    if (!this.selectedUploadId) return;

    const formData = new FormData();
    formData.append('transcription', this.editedTranscription);

    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });

    this.http.put(`/api/history/${this.selectedUploadId}`, formData, { headers }).subscribe(
      () => {
        this.selectedTranscription = this.editedTranscription;
        this.isEditing = false;
        
      },
      error => {
        console.error('Erreur lors de la mise à jour:', error);
        alert('Erreur lors de la mise à jour');
      }
    );
  }

  copyTranscription(): void {
    const text = this.transcription || this.selectedTranscription;
    if (text) {
      navigator.clipboard.writeText(text).then(
        () => {
          // Show success icon
          this.showCopySuccess = true;
          
          // Hide success icon after 2 seconds
          setTimeout(() => {
            this.ngZone.run(() => {
              this.showCopySuccess = false;
            });
          }, 2000);
        },
        () => alert('Erreur lors de la copie')
      );
    }
  }

  cancelEditing(): void {
    this.isEditing = false;
    this.editedTranscription = '';
  }
  closeTranscription(): void {
    this.transcription = null;
    this.selectedTranscription = null;
    this.selectedUploadId = null;
    this.isEditing = false;
    this.editedTranscription = '';
  }
  getCurrentFileName(): string | null {
    if (this.selectedUploadId) {
      const record = this.history.find(h => h.upload_id === this.selectedUploadId);
      return record ? record.filename : null;
    }
    return this.selectedFile ? this.selectedFile.name : null;
  }
}
