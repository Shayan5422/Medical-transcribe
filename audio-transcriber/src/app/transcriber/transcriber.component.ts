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
  userId: number;
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
  selectedModel: string = 'fast'; 
  autoShareConfigs: AutoShareConfig[] = [];
  showAutoShareModal = false;
  currentAccessType: AccessType = 'viewer';
  models = [
    { id: 'fast', name: 'Rapide', value: 'openai/whisper-large-v3-turbo' },
    { id: 'accurate', name: 'Précis', value: 'openai/whisper-large-v3' }
  ];
  private recordingStartTime: number = 0;
  private chunkInterval: any;
  private currentChunkNumber: number = 0;
  private sessionId: string = '';
  private tempTranscriptions: string[] = [];
  

  // Add this helper method
  getSelectedModelValue(): string {
    return this.models.find(m => m.id === this.selectedModel)?.value || 'openai/whisper-large-v3-turbo';
  }

  autoShareConfig: AutoShareConfig = {
    userId: 0,
    accessType: 'viewer',
    username: ''
  };
 

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
  
    const savedConfigs = localStorage.getItem('autoShareConfigs');
    if (savedConfigs) {
      this.autoShareConfigs = JSON.parse(savedConfigs);
      // Fetch usernames for all configs
      this.autoShareConfigs.forEach(config => {
        if (config.userId) {
          this.fetchUserName(config.userId);
        }
      });
    }
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
    formData.append('model', this.getSelectedModelValue());
  
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
          this.selectedUploadForShare = response.upload_id;
          this.audioStreamUrl = `${environment.apiUrl}/stream-audio/${response.upload_id}`;
          
          // Auto-share with all configured users
          this.autoShareConfigs.forEach(config => {
            this.shareWithUser(config.userId, config.accessType);
          });
        });
        
        // درخواست اضافی برای دریافت اطلاعات transcription و نقش کاربر
        this.fetchTranscriptionInfo(response.upload_id);
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

  fetchTranscriptionInfo(upload_id: number): void {
    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });
  
    this.http.get<any>(`${environment.apiUrl}/get-transcription/${upload_id}`, { headers }).subscribe(
      response => {
        this.ngZone.run(() => {
          this.selectedTranscription = response.transcription;
          this.isEditor = response.is_editor; // تنظیم متغیر isEditor
        });
      },
      error => {
        console.error('Error fetching transcription:', error);
        alert('Error fetching transcription');
      }
    );
  }

  // Start recording with chunk processing
  startRecording(): void {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ 
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          sampleSize: 16,
          echoCancellation: true,
          noiseSuppression: true
        }
      })
        .then(stream => {
          this.mediaStream = stream;
          
          // Find supported MIME type
          const mimeTypes = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/ogg;codecs=opus',
            'audio/ogg'
          ];
          
          let selectedMimeType = '';
          for (const type of mimeTypes) {
            if (MediaRecorder.isTypeSupported(type)) {
              selectedMimeType = type;
              break;
            }
          }
          
          if (!selectedMimeType) {
            throw new Error('No supported MIME type found for audio recording');
          }

          console.log('Using MIME type:', selectedMimeType);
          const options: MediaRecorderOptions = {
            mimeType: selectedMimeType,
            audioBitsPerSecond: 256000
          };

          this.mediaRecorder = new MediaRecorder(stream, options);
          this.audioChunks = [];
          this.tempTranscriptions = [];
          this.currentChunkNumber = 0;
          this.sessionId = Date.now().toString();
          this.recordingStartTime = Date.now();
          
          // Configure data collection
          this.mediaRecorder.ondataavailable = async event => {
            if (event.data && event.data.size > 0) {
              console.log(`Received audio data: ${event.data.size} bytes, type: ${event.data.type}`);
              this.audioChunks.push(event.data);
            }
          };

          this.mediaRecorder.onstop = async () => {
            console.log('MediaRecorder stopped, processing chunk...');
            if (this.audioChunks.length > 0) {
              try {
                // Convert to WAV
                const audioContext = new AudioContext({
                  sampleRate: 16000,
                });
                
                const blob = new Blob(this.audioChunks, { type: selectedMimeType });
                const arrayBuffer = await blob.arrayBuffer();
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                
                // Create WAV file
                const wavBlob = await this.audioBufferToWav(audioBuffer);
                
                // Process the WAV chunk
                await this.processCurrentChunk(false, wavBlob);
                
                audioContext.close();
                
                if (this.isRecording) {
                  console.log('Starting next chunk...');
                  this.audioChunks = [];
                  this.mediaRecorder?.start(1000);
                }
              } catch (error) {
                console.error('Error processing audio chunk:', error);
                if (this.isRecording) {
                  this.audioChunks = [];
                  this.mediaRecorder?.start(1000);
                }
              }
            } else {
              console.log('No audio data to process');
              if (this.isRecording) {
                this.mediaRecorder?.start(1000);
              }
            }
          };

          // Start recording with timeslice to get data frequently
          this.mediaRecorder.start(1000);
          this.isRecording = true;
          console.log('Recording started with timeslice of 1 second');

          // Process chunks every 30 seconds
          this.chunkInterval = setInterval(() => {
            if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
              console.log('Stopping recorder for chunk processing...');
              this.mediaRecorder.stop();
            }
          }, 30000);
        })
        .catch(err => {
          console.error('Error accessing microphone:', err);
          alert('Unable to access microphone.');
        });
    } else {
      alert('Your browser does not support audio recording.');
    }
  }

  // Convert AudioBuffer to WAV Blob
  private audioBufferToWav(buffer: AudioBuffer): Promise<Blob> {
    const numberOfChannels = 1;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;
    
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numberOfChannels * bytesPerSample;
    
    const wav = new ArrayBuffer(44 + buffer.length * bytesPerSample);
    const view = new DataView(wav);
    
    // Write WAV header
    const writeString = (view: DataView, offset: number, string: string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    
    writeString(view, 0, 'RIFF');  // ChunkID
    view.setUint32(4, 36 + buffer.length * bytesPerSample, true);  // ChunkSize
    writeString(view, 8, 'WAVE');  // Format
    writeString(view, 12, 'fmt ');  // Subchunk1ID
    view.setUint32(16, 16, true);  // Subchunk1Size
    view.setUint16(20, format, true);  // AudioFormat
    view.setUint16(22, numberOfChannels, true);  // NumChannels
    view.setUint32(24, sampleRate, true);  // SampleRate
    view.setUint32(28, sampleRate * blockAlign, true);  // ByteRate
    view.setUint16(32, blockAlign, true);  // BlockAlign
    view.setUint16(34, bitDepth, true);  // BitsPerSample
    writeString(view, 36, 'data');  // Subchunk2ID
    view.setUint32(40, buffer.length * bytesPerSample, true);  // Subchunk2Size
    
    // Write audio data
    const offset = 44;
    const data = buffer.getChannelData(0);
    for (let i = 0; i < data.length; i++) {
      const sample = Math.max(-1, Math.min(1, data[i]));
      view.setInt16(offset + i * bytesPerSample, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
    }
    
    return Promise.resolve(new Blob([wav], { type: 'audio/wav' }));
  }

  // Process current audio chunk
  private async processCurrentChunk(isFinal: boolean, wavBlob?: Blob): Promise<void> {
    console.log(`Processing chunk ${this.currentChunkNumber}, isFinal: ${isFinal}`);
    
    if (!wavBlob && this.audioChunks.length === 0) {
      console.log('No audio chunks to process');
      return;
    }

    const audioBlob = wavBlob || await this.convertToWav();
    console.log(`Created WAV blob of size: ${audioBlob.size} bytes`);

    const formData = new FormData();
    formData.append('file', audioBlob, `chunk_${this.currentChunkNumber}.wav`);
    formData.append('chunk_number', this.currentChunkNumber.toString());
    formData.append('session_id', this.sessionId);
    formData.append('is_final', isFinal.toString());
    formData.append('model', this.getSelectedModelValue());

    try {
      console.log(`Sending chunk ${this.currentChunkNumber} to server...`);
      const headers = new HttpHeaders({
        'Authorization': `Bearer ${this.token}`
      });

      const response = await this.http.post<any>(
        `${environment.apiUrl}/process-chunk/`,
        formData,
        { headers }
      ).toPromise();

      console.log(`Server response for chunk ${this.currentChunkNumber}:`, response);

      if (isFinal) {
        this.ngZone.run(() => {
          this.transcription = response.transcription;
          this.isTranscribing = false;
          if (response.upload_id) {
            this.selectedUploadId = response.upload_id;
            this.selectedUploadForShare = response.upload_id;
            this.audioStreamUrl = `${environment.apiUrl}/stream-audio/${response.upload_id}`;
            
            // Auto-share with configured users
            this.autoShareConfigs.forEach(config => {
              this.shareWithUser(config.userId, config.accessType);
            });
          }
          this.fetchHistory();
        });
      } else {
        this.tempTranscriptions[this.currentChunkNumber] = response.chunk_transcription;
        this.currentChunkNumber++;
        
        // Update the displayed transcription with all chunks so far
        this.ngZone.run(() => {
          this.transcription = this.tempTranscriptions.join(' ');
        });
      }
    } catch (error: any) {
      console.error('Error processing chunk:', error);
      this.ngZone.run(() => {
        if (error.error?.detail) {
          alert(`Error: ${error.error.detail}`);
        } else {
          alert('Error processing audio chunk.');
        }
        this.isTranscribing = false;
      });
    }
  }

  private async convertToWav(): Promise<Blob> {
    const audioContext = new AudioContext({
      sampleRate: 16000,
    });
    
    const blob = new Blob(this.audioChunks, { type: this.mediaRecorder?.mimeType });
    const arrayBuffer = await blob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    
    const wavBlob = await this.audioBufferToWav(audioBuffer);
    audioContext.close();
    
    return wavBlob;
  }

  stopRecording(): void {
    console.log('Stopping recording...');
    if (this.mediaRecorder) {
      clearInterval(this.chunkInterval);
      
      if (this.mediaRecorder.state === 'recording') {
        this.mediaRecorder.stop();
        console.log('Processing final chunk...');
        this.processCurrentChunk(true);
      }
      
      this.isRecording = false;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
      console.log('Media stream stopped to release microphone.');
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
      const existingConfigIndex = this.autoShareConfigs.findIndex(
        config => config.userId === userId
      );
  
      if (existingConfigIndex !== -1) {
        // Remove existing config
        this.autoShareConfigs.splice(existingConfigIndex, 1);
      } else {
        // Add new config
        const user = this.users.find(u => u.id === userId);
        const newConfig: AutoShareConfig = {
          userId,
          accessType: this.currentAccessType,
          username: user?.username
        };
        this.autoShareConfigs.push(newConfig);
      }
  
      // Save to localStorage
      localStorage.setItem('autoShareConfigs', JSON.stringify(this.autoShareConfigs));
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
    isAutoShareEnabled(userId: number): boolean {
      return this.autoShareConfigs.some(config => config.userId === userId);
    }
  
    // Helper method to get access type for auto-shared user
    getAutoShareAccessType(userId: number): AccessType | null {
      const config = this.autoShareConfigs.find(c => c.userId === userId);
      return config ? config.accessType : null;
    }
  }
  
  




