import { Component, OnInit, NgZone } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms'; // Importer FormsModule pour ngModel
import { HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
import { Router } from '@angular/router';
import { jsPDF } from 'jspdf';
import { AudioPlayerComponent } from './audio-player.component';
import { UserFilterPipe } from './user-filter.pipe';


interface User {
  id: number;
  username: string;
}

interface UploadHistory {
  upload_id: number;
  filename: string;
  transcription_filename: string;
  upload_time: string;
  is_archived: boolean;
  shared_with: number[];
}

@Component({
  selector: 'app-transcriber',
  standalone: true,
  imports: [CommonModule, FormsModule,AudioPlayerComponent,UserFilterPipe],
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
  isSidebarOpen: boolean = false;
  audioUrl: string | null = null;
  audioStreamUrl: string | null = null;
  showArchived: boolean = false;
  users: User[] = [];
  showShareModal: boolean = false;
  selectedUploadForShare: number | null = null;
  searchQuery: string = '';
  showSharedRecords: boolean = false;


  
  constructor(
    private http: HttpClient,
    private router: Router,
    private ngZone: NgZone // Injecter NgZone
  ) {}

  openShareModal(upload_id: number): void {
    this.selectedUploadForShare = upload_id;
    this.showShareModal = true;
    this.fetchUsers();
  }

  closeShareModal(): void {
    this.showShareModal = false;
    this.selectedUploadForShare = null;
    this.searchQuery = '';
  }
  getSharedWith(record: UploadHistory): number[] {
    return record.shared_with || [];
  }
  
  ngOnInit(): void {
    this.token = localStorage.getItem('token');
    console.log('Token récupéré :', this.token);
    if (this.token) {
      this.fetchHistory();
    } else {
      console.log('Aucun token trouvé, redirection vers la page de connexion.');
      this.router.navigate(['/login']);
    }
    window.addEventListener('resize', () => {
      if (window.innerWidth >= 768) { // md breakpoint
        this.isSidebarOpen = false;
      }
    });
    
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      this.currentTheme = savedTheme;
    }
    document.documentElement.setAttribute('data-theme', this.currentTheme);
  }
  toggleSidebar(): void {
    this.isSidebarOpen = !this.isSidebarOpen;
  }
  fetchUsers(): void {
    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });

    this.http.get<{users: User[]}>('/api/users/', { headers }).subscribe(
      response => {
        this.users = response.users;
      },
      error => {
        console.error('Error fetching users:', error);
      }
    );
  }

  shareWithUser(userId: number): void {
    if (!this.selectedUploadForShare) return;

    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });

    this.http.post(`/api/share/${this.selectedUploadForShare}/user/${userId}`, {}, { headers }).subscribe(
      () => {
        this.fetchHistory();
        alert('Transcription partagée avec succès');
      },
      error => {
        console.error('Error sharing:', error);
        alert('Erreur lors du partage');
      }
    );
  }

  removeShare(uploadId: number, userId: number): void {
    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });

    this.http.delete(`/api/share/${uploadId}/user/${userId}`, { headers }).subscribe(
      () => {
        this.fetchHistory();
      },
      error => {
        console.error('Error removing share:', error);
        alert('Erreur lors de la suppression du partage');
      }
    );
  }

  toggleTheme(): void {
    this.currentTheme = this.currentTheme === 'light' ? 'dark' : 'light';
    // Mettre à jour l'attribut data-theme
    document.documentElement.setAttribute('data-theme', this.currentTheme);
    // Sauvegarde du thème dans le localStorage (optionnel)
    localStorage.setItem('theme', this.currentTheme);
  }
  private getAudioStreamUrl(uploadId: number): string {
    return `/api/stream-audio/${uploadId}`;
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
  getPatientName(upload_id: number): string {
    return `Patient_${upload_id}`;
  }

  // تابع برای دریافت نام فایل فعلی
  getCurrentFileName(): string | null {
    if (this.selectedUploadId) {
      return this.getPatientName(this.selectedUploadId);
    }
    return this.selectedFile ? this.selectedFile.name : null;
  }
  // Télécharger le fichier audio vers le serveur
  // Dans la méthode selectUpload
  selectUpload(upload_id: number): void {
    // Reset audio stream URL first
    this.audioStreamUrl = null;
    this.selectedUploadId = upload_id;
    this.transcription = null;

    // Set new audio stream URL
    this.audioStreamUrl = `/api/stream-audio/${upload_id}`;

    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });

    this.http.get<any>(`/api/get-transcription/${upload_id}`, { headers }).subscribe(
      response => {
        this.ngZone.run(() => {
          this.selectedTranscription = response.transcription;
        });
      },
      error => {
        console.error('Error fetching transcription:', error);
        alert('Error fetching transcription');
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
    this.selectedTranscription = null;
    this.selectedUploadId = null;
    this.audioStreamUrl = null; // Reset audio URL
  });

  this.http.post<any>('/api/upload-audio/', formData, { headers }).subscribe(
    response => {
      console.log('Réponse de transcription :', response);
      this.ngZone.run(() => {
        this.transcription = response.transcription;
        this.transcriptionFile = response.transcription_file;
        this.isTranscribing = false;
        
        this.selectedUploadId = response.upload_id; 
        this.audioStreamUrl = `/api/stream-audio/${response.upload_id}`; 
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
  
  

  // Récupérer l'historique des uploads spécifiques à l'utilisateur
  
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
    this.audioStreamUrl = null; // Reset audio URL
  }
  

  downloadTranscriptionFile(upload_id: number): void {
    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });

    const url = `/api/download-transcription/${upload_id}`;
    this.http.get(url, { headers, responseType: 'blob' }).subscribe(blob => {
      const a = document.createElement('a');
      const objectUrl = URL.createObjectURL(blob);
      a.href = objectUrl;
      a.download = `Patient_${upload_id}.txt`; // تغییر نام فایل
      a.click();
      URL.revokeObjectURL(objectUrl);
      console.log(`Fichier de transcription Patient_${upload_id}.txt téléchargé.`);
    }, error => {
      console.error('Erreur lors du téléchargement du fichier de transcription :', error);
      alert('Erreur lors du téléchargement du fichier de transcription.');
    });
  }

  downloadAudioFile(upload_id: number): void {
    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });

    const url = `/api/download-audio/${upload_id}`;
    this.http.get(url, { headers, responseType: 'blob' }).subscribe(blob => {
      const a = document.createElement('a');
      const objectUrl = URL.createObjectURL(blob);
      a.href = objectUrl;
      a.download = `Patient_${upload_id}.wav`; // تغییر نام فایل
      a.click();
      URL.revokeObjectURL(objectUrl);
      console.log(`Fichier audio Patient_${upload_id}.wav téléchargé.`);
    }, error => {
      console.error('Erreur lors du téléchargement du fichier audio :', error);
      alert('Erreur lors du téléchargement du fichier audio.');
    });
  }

  // دانلود ترنسکریپشن به صورت PDF با نام بیمار
  downloadTranscriptionAsPDF(upload_id: number): void {
    // Utiliser la transcription sélectionnée ou la transcription immédiate
    const transcriptionText = this.selectedTranscription || this.transcription;
  
    console.log('downloadTranscriptionAsPDF appelé avec upload_id:', upload_id);
    console.log('Texte de transcription:', transcriptionText);
  
    if (transcriptionText) {
      try {
        const doc = new jsPDF();
  
        // Définir la police et la taille
        doc.setFont('Helvetica');
        doc.setFontSize(12);
  
        // Diviser le texte en lignes adaptées à la page
        const lines = doc.splitTextToSize(transcriptionText, 180); // Ajustez la largeur si nécessaire
  
        // Ajouter le texte au PDF
        doc.text(lines, 10, 10);
  
        // Sauvegarder le PDF avec le nom du patient
        doc.save(`Patient_${upload_id}.pdf`);
        console.log(`Transcription téléchargée en tant que PDF : Patient_${upload_id}.pdf`);
      } catch (error) {
        console.error('Erreur lors de la création du PDF:', error);
        alert('Erreur lors de la création du fichier PDF.');
      }
    } else {
      alert('Aucune transcription disponible pour le téléchargement en PDF.');
    }
  }
  
  toggleArchive(upload_id: number): void {
    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });

    this.http.post(`/api/toggle-archive/${upload_id}`, {}, { headers }).subscribe(
      () => {
        this.fetchHistory();
      },
      error => {
        console.error('Error toggling archive status:', error);
        alert('Erreur lors de la mise à jour du statut d\'archive');
      }
    );
  }

  fetchHistory(): void {
    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });
  
    let url = '/api/history/';
    if (this.showArchived) {
      url += '?include_archived=true';
    }
    if (this.showSharedRecords) {
      url += (url.includes('?') ? '&' : '?') + 'include_shared=true';
    }
    
    this.http.get<any>(url, { headers }).subscribe(
      response => {
        this.ngZone.run(() => {
          this.history = response.history;
        });
      },
      error => {
        console.error('Error fetching history:', error);
      }
    );
  }

  toggleSharedRecords(): void {
    this.showSharedRecords = !this.showSharedRecords;
    this.fetchHistory();
  }

  getUserName(userId: number): string {
    const user = this.users.find(u => u.id === userId);
    return user ? user.username : `User ${userId}`;
  }

  toggleShowArchived(): void {
    this.showArchived = !this.showArchived;
    this.fetchHistory();
  }

  
}


