import { Component, OnInit, NgZone, Directive, Output, EventEmitter, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms'; // Importer FormsModule pour ngModel
import { HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
import { Router } from '@angular/router';
import { jsPDF } from 'jspdf';
import { AudioPlayerComponent } from './audio-player.component';
import { UserFilterPipe } from './user-filter.pipe';
import { ClickOutsideDirective } from './click-outside.directive';
import { PricingComponent } from './pricing.component';
import { catchError, EMPTY } from 'rxjs';
import { User } from './user.model';
import { environment } from '../../environments/environment';


type AccessType = 'viewer' | 'editor';

interface AutoShareConfig {
  userId: number | null;
  accessType: AccessType;
  username?: string;
}
interface ShareInfo {
  user_id: number;
  access_type: string;
  username?: string; // Add username if available from backend
}

interface ShareCreate {
  user_id: number;
  access_type: string;
}

interface ShareResponse {
  id: number;
  upload_id: number;
  user_id: number;
  access_type: string;
}

interface ShareInfo {
  user_id: number;
  access_type: string;
}

interface UploadHistory {
  upload_id: number;
  filename: string;
  transcription_filename: string;
  upload_time: string;
  is_archived: boolean;
  shared_with: ShareInfo[];
  owner_id: number;
  showMenu?: boolean;
}

@Component({
  selector: 'app-transcriber',
  standalone: true,
  imports: [CommonModule, FormsModule,AudioPlayerComponent,UserFilterPipe, ClickOutsideDirective,PricingComponent],
  templateUrl: './transcriber.component.html',
  styleUrls: ['./transcriber.component.css']
})

export class TranscriberComponent implements OnInit {
  transcription: string | null = null;
  transcriptionFile: string | null = null;
  isRecording: boolean = false;
  isTranscribing: boolean = false;
  mediaRecorder: MediaRecorder | null = null;
  mediaStream: MediaStream | null = null;
  audioChunks: Blob[] = [];
  selectedFile: File | null = null;
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
  currentUserId: number | null = null;
  isEditor: boolean = false; // Variable pour gérer le rôle d'éditeur
  showModelMenu = false;
  selectedModel: string = 'fast'; // Default to fast model
  models = [
    { id: 'fast', name: 'Rapide', value: 'openai/whisper-large-v3-turbo' },
    { id: 'accurate', name: 'Précis', value: 'openai/whisper-large-v3' }
  ];

  // Add this helper method
  getSelectedModelValue(): string {
    return this.models.find(m => m.id === this.selectedModel)?.value || 'openai/whisper-large-v3-turbo';
  }

  autoShareConfig: AutoShareConfig = {
    userId: null,
    accessType: 'viewer',
    username: ''
  };
  showAutoShareModal = false;
  currentAccessType: AccessType = 'viewer';

  onAccessTypeChange(event: Event): void {
    const select = event.target as HTMLSelectElement;
    this.currentAccessType = select.value as AccessType;
  }

  validateAccessType(type: string): AccessType {
    return type === 'editor' ? 'editor' : 'viewer';
  }


  constructor(
    private http: HttpClient,
    private router: Router,
    private ngZone: NgZone // Injecter NgZone
  ) {}

  

  
  getSharedWith(record: UploadHistory): ShareInfo[] {
    return record.shared_with || [];
  }
  
  ngOnInit(): void {
    this.token = localStorage.getItem('token');
    console.log('Token récupéré :', this.token);
    if (this.token) {
      this.fetchHistory();
      this.fetchUsers();
    } else {
      console.log('Aucun token trouvé, redirection vers la page de connexion.');
      this.router.navigate(['/login']);
    }

    window.addEventListener('resize', () => {
      if (window.innerWidth >= 768) { // breakpoint md
        this.isSidebarOpen = false;
      }
    });
    const savedConfig = localStorage.getItem('autoShareConfig');
    if (savedConfig) {
      this.autoShareConfig = JSON.parse(savedConfig);
      // If there's a saved config, fetch the username
      if (this.autoShareConfig.userId) {
        this.fetchUserName(this.autoShareConfig.userId);
      }
    }
  
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

    this.http.get<{users: User[]}>(`${environment.apiUrl}/users/`, { headers }).subscribe(
      response => {
        // Initialiser accessType pour chaque utilisateur
        this.users = response.users.map(user => ({
          ...user,
          accessType: user.accessType || 'viewer' // Valeur par défaut si non défini
        }));
      },
      error => {
        console.error('Erreur lors de la récupération des utilisateurs:', error);
      }
    );
  }

  shareWithUser(userId: number, accessType: AccessType): void {
    if (!this.selectedUploadForShare) {
      console.error('Aucun enregistrement sélectionné pour le partage');
      return;
    }

    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });

    const payload: ShareCreate = {
      user_id: userId,
      access_type: accessType
    };

    this.http.post<ShareResponse>(
      `${environment.apiUrl}/share/${this.selectedUploadForShare}/user/`,
      payload,
      { headers }
    ).pipe(
      catchError(error => {
        console.error('Erreur lors du partage:', error);
        alert('Erreur lors du partage');
        return EMPTY;
      })
    ).subscribe(() => {
      this.fetchHistory();
      this.fetchUsers();
      
      this.closeShareModal();
    });
  }


  closeShareModal(): void {
    this.showShareModal = false;
    this.selectedUploadForShare = null;
    this.searchQuery = '';
  }

  openShareModal(upload_id: number): void {
    this.selectedUploadForShare = upload_id;
    this.showShareModal = true;
    this.fetchUsers();
  }

  toggleTheme(): void {
    this.currentTheme = this.currentTheme === 'light' ? 'dark' : 'light';
    // Mettre à jour l'attribut data-theme
    document.documentElement.setAttribute('data-theme', this.currentTheme);
    // Sauvegarde du thème dans le localStorage (optionnel)
    localStorage.setItem('theme', this.currentTheme);
  }
  private getAudioStreamUrl(uploadId: number): string {
    return `${environment.apiUrl}/stream-audio/${uploadId}`;
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
    this.audioStreamUrl = `${environment.apiUrl}/stream-audio/${upload_id}`;

    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });

    this.http.get<any>(`${environment.apiUrl}/get-transcription/${upload_id}`, { headers }).subscribe(
      response => {
        this.ngZone.run(() => {
          this.selectedTranscription = response.transcription;
          this.isEditor = response.is_editor; // Mettre à jour le rôle
        });
      },
      error => {
        console.error('Error fetching transcription:', error);
        alert('Error fetching transcription');
      }
    );
  }


  // Inside TranscriberComponent class
  uploadAudio(file: File): void {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', this.getSelectedModelValue()); // Add model to form data

    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });

    this.ngZone.run(() => {
      this.isTranscribing = true;
      this.transcription = null;
      this.selectedTranscription = null;
      this.selectedUploadId = null;
      this.audioStreamUrl = null;
    });

    this.http.post<any>(`${environment.apiUrl}/upload-audio/`, formData, { headers }).subscribe(
    response => {
      console.log('Réponse de transcription :', response);
      this.ngZone.run(() => {
        this.transcription = response.transcription;
        this.transcriptionFile = response.transcription_file;
        this.isTranscribing = false;
        this.selectedUploadId = response.upload_id;
        this.selectedUploadForShare = response.upload_id;  // Set this before auto-sharing
        this.audioStreamUrl = `${environment.apiUrl}/stream-audio/${response.upload_id}`;
        
        // Auto-share if configured
        if (this.autoShareConfig.userId) {
          this.shareWithUser(
            this.autoShareConfig.userId,
            this.autoShareConfig.accessType
          );
        }
      });
      
      this.fetchHistory();
    },
    error => {
      console.error('Erreur lors du téléchargement:', error);
      this.ngZone.run(() => {
        if (error.error && error.error.detail) {
          alert(`Erreur : ${error.error.detail}`);
        } else {
          alert('Erreur lors du téléchargement.');
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

      this.http.delete(`${environment.apiUrl}/history/${upload_id}`, { headers }).subscribe(
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
    this.editedTranscription = this.transcription || this.selectedTranscription || '';
  }

  
  saveTranscription(): void {
    if (!this.selectedUploadId) return;

    const formData = new FormData();
    formData.append('transcription', this.editedTranscription);

    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });

    this.http.put(`${environment.apiUrl}/history/${this.selectedUploadId}`, formData, { headers }).subscribe(
      () => {
        // Update both transcription sources
        if (this.transcription) {
          this.transcription = this.editedTranscription;
        }
        this.selectedTranscription = this.editedTranscription;
        this.isEditing = false;
        this.isEditor = false;
        this.fetchHistory();
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
    this.audioStreamUrl = null;
  }
  

  downloadTranscriptionFile(upload_id: number): void {
    const headers = new HttpHeaders({
        'Authorization': `Bearer ${this.token}`
    });

    const url = `${environment.apiUrl}/download-transcription/${upload_id}`;
    this.http.get(url, { 
        headers, 
        responseType: 'blob',
        observe: 'response' 
    }).subscribe(
        response => {
            const blob = new Blob([response.body!], { type: 'text/plain' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `Patient_${upload_id}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        },
        error => {
            console.error('Error downloading transcription file:', error);
            if (error.status === 404) {
                alert('Fichier de transcription non trouvé ou accès refusé');
            } else {
                alert('Erreur lors du téléchargement du fichier de transcription');
            }
        }
    );
}


downloadAudioFile(upload_id: number): void {
  const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
  });

  const url = `${environment.apiUrl}/download-audio/${upload_id}`;
  this.http.get(url, { 
      headers, 
      responseType: 'blob',
      observe: 'response'
  }).subscribe(
      response => {
          const blob = new Blob([response.body!], { type: 'audio/wav' });
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `Patient_${upload_id}.wav`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          window.URL.revokeObjectURL(url);
      },
      error => {
          console.error('Error downloading audio file:', error);
          if (error.status === 404) {
              alert('Fichier audio non trouvé ou accès refusé');
          } else {
              alert('Erreur lors du téléchargement du fichier audio');
          }
      }
  );
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
  
    this.http.post(`${environment.apiUrl}/toggle-archive/${upload_id}`, {}, { headers })
      .pipe(
        catchError(error => {
          console.error('Error toggling archive status:', error);
          if (error.status === 404) {
            alert('Enregistrement non trouvé ou permission refusée');
          } else {
            alert('Erreur lors de la mise à jour du statut d\'archive');
          }
          return EMPTY;
        })
      )
      .subscribe(
        (response: any) => {
          this.fetchHistory();
         
          const status = response.is_archived ? 'archivé' : 'désarchivé';
          
        }
      );
  }

  fetchHistory(): void {
    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });

    let url = `${environment.apiUrl}/history/`;
    if (this.showArchived) {
      url += '?include_archived=true';
    }
    if (this.showSharedRecords) {
      url += (url.includes('?') ? '&' : '?') + 'include_shared=true';
    }
    
    this.http.get<any>(url, { headers }).subscribe(
      response => {
        this.ngZone.run(() => {
          this.history = response.history.map((record: any) => ({
            ...record,
            shared_with: record.shared_with || [],
            showMenu: false
          }));
          this.currentUserId = response.current_user_id;  // Obtenir l'ID de l'utilisateur actuel
          // Pas besoin de fetchUsers ici car fetchUsers est déjà appelé dans ngOnInit
        });
      },
      error => {
        console.error('Erreur lors de la récupération de l\'historique:', error);
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

  removeShare(uploadId: number, userId: number): void {
    const headers = new HttpHeaders({
        'Authorization': `Bearer ${this.token}`
    });

    this.http.delete(`${environment.apiUrl}/share/${uploadId}/user/${userId}`, { headers })
        .pipe(
            catchError(error => {
                console.error('Error removing share:', error);
                alert('Erreur lors de la suppression du partage');
                return EMPTY;
            })
        )
        .subscribe(() => {
            this.fetchHistory();
            this.fetchUsers();  // Refresh users list after removing share
        });
    }
    fetchCurrentUser(): void {
      const headers = new HttpHeaders({
        'Authorization': `Bearer ${this.token}`
      });
  
      this.http.get<any>(`${environment.apiUrl}/current-user/`, { headers }).subscribe(
        response => {
          this.currentUserId = response.id;
        },
        error => {
          console.error('Error fetching current user:', error);
        }
      );
    }
  
    getCurrentUserId(): number | null {
      return this.currentUserId;
    }
  
    isOwner(record: UploadHistory): boolean {
      return record.owner_id === this.currentUserId;
    }
  
    // Method to determine if the current user should see the remove button
    canRemoveShare(record: UploadHistory): boolean {
      return record.owner_id === this.currentUserId;
    }


    openAutoShareModal(): void {
      this.showAutoShareModal = true;
      this.fetchUsers();
    }
  
    closeAutoShareModal(): void {
      this.showAutoShareModal = false;
    }
  
    toggleAutoShare(userId: number): void {
    if (this.autoShareConfig.userId === userId) {
      // Disable auto-share
      this.autoShareConfig = {
        userId: null,
        accessType: 'viewer',
        username: ''
      };
    } else {
      // Enable auto-share
      const user = this.users.find(u => u.id === userId);
      this.autoShareConfig = {
        userId,
        accessType: this.currentAccessType,
        username: user?.username
      };
    }
    
    localStorage.setItem('autoShareConfig', JSON.stringify(this.autoShareConfig));
    this.closeAutoShareModal();
  }
  
    private fetchUserName(userId: number): void {
      const foundUser = this.users.find(u => u.id === userId);
      if (foundUser) {
        this.autoShareConfig.username = foundUser.username;
      }
    }

    getUserAccessType(record: UploadHistory): string | null {
      if (!this.currentUserId) return null;
      
      const shareInfo = record.shared_with.find(share => share.user_id === this.currentUserId);
      return shareInfo ? this.formatAccessType(shareInfo.access_type) : null;
    }
  
    formatAccessType(accessType: string): string {
      switch(accessType.toLowerCase()) {
        case 'editor':
          return 'Éditeur';
        case 'viewer':
          return 'Lecteur';
        default:
          return accessType;
      }
    }
  }
  



