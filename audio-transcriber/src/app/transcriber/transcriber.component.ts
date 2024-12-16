// src/app/transcriber/transcriber.component.ts
import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms'; // Import FormsModule for ngModel
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
  audioChunks: Blob[] = [];
  selectedFile: File | null = null; // Selected file
  history: UploadHistory[] = [];
  token: string | null = null;
  selectedUploadId: number | null = null;
  selectedTranscription: string | null = null;

  constructor(private http: HttpClient, private router: Router) {}

  ngOnInit(): void {
    this.token = localStorage.getItem('token');
    console.log('Token retrieved:', this.token);
    if (this.token) {
      this.fetchHistory();
    } else {
      // If not logged in, redirect to login
      console.log('No token found, redirecting to login.');
      this.router.navigate(['/login']);
    }
  }

  // Handle file selection
  onFileSelected(event: any): void {
    const file: File = event.target.files[0];
    if (file) {
      this.selectedFile = file;
      this.transcription = null; // Reset previous transcription
      this.transcriptionFile = null; // Reset previous transcription file
      console.log('Selected file:', file);
    }
  }

  // Upload the selected file
  uploadSelectedFile(): void {
    if (this.selectedFile) {
      this.uploadAudio(this.selectedFile);
    } else {
      alert('Please select an audio file before uploading.');
    }
  }

  // Upload audio file to the server
  uploadAudio(file: File): void {
    const formData = new FormData();
    formData.append('file', file);

    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });

    console.log('Uploading file:', file);
    console.log('File type:', file.type);

    this.isTranscribing = true; // Start transcription

    this.http.post<any>('http://127.0.0.1:8000/upload-audio/', formData, { headers }).subscribe(
      response => {
        console.log('Transcription response:', response);
        this.transcription = response.transcription;
        this.transcriptionFile = response.transcription_file;
        this.isTranscribing = false; // End transcription
        alert('Transcription successful!');
        console.log('Transcription set:', this.transcription);
        this.fetchHistory(); // Refresh history after new upload
      },
      (error: HttpErrorResponse) => {
        console.error('Error uploading audio file:', error);
        if (error.error && error.error.detail) {
          alert(`Error: ${error.error.detail}`);
        } else {
          alert('Error uploading audio file.');
        }
        this.isTranscribing = false; // End transcription on error
      }
    );
  }

  // Start audio recording
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
          console.log('Recording started.');

          this.mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
              this.audioChunks.push(event.data);
              console.log('Data available:', event.data);
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
          console.error('Error accessing microphone:', err);
          alert('Unable to access the microphone.');
        });
    } else {
      alert('Your browser does not support audio recording.');
    }
  }

  // Stop audio recording
  stopRecording(): void {
    if (this.mediaRecorder) {
      this.mediaRecorder.stop();
      this.isRecording = false;
      console.log('Recording stopped.');
    }
  }

  // Download transcription as text file
  downloadTranscription(): void {
    if (this.transcription) {
      const blob = new Blob([this.transcription], { type: 'text/plain' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'transcription.txt';
      a.click();
      window.URL.revokeObjectURL(url);
      console.log('Transcription downloaded as text file.');
    }
  }

  // Download transcription file from server
  downloadTranscriptionFile(upload_id: number): void {
    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });

    const url = `http://127.0.0.1:8000/download-transcription/${upload_id}`;
    this.http.get(url, { headers, responseType: 'blob' }).subscribe(blob => {
      const a = document.createElement('a');
      const objectUrl = URL.createObjectURL(blob);
      a.href = objectUrl;
      a.download = `transcription_${upload_id}.txt`;
      a.click();
      URL.revokeObjectURL(objectUrl);
      console.log(`Transcription file transcription_${upload_id}.txt téléchargé.`);
    }, error => {
      console.error('Error downloading transcription file:', error);
      alert('Error downloading transcription file.');
    });
  }

  // Download audio file from server
  downloadAudioFile(upload_id: number): void {
    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });

    const url = `http://127.0.0.1:8000/download-audio/${upload_id}`;
    this.http.get(url, { headers, responseType: 'blob' }).subscribe(blob => {
      const a = document.createElement('a');
      const objectUrl = URL.createObjectURL(blob);
      a.href = objectUrl;
      a.download = `audio_${upload_id}.wav`;
      a.click();
      URL.revokeObjectURL(objectUrl);
      console.log(`Fichier audio audio_${upload_id}.wav téléchargé.`);
    }, error => {
      console.error('Error downloading audio file:', error);
      alert('Error downloading audio file.');
    });
  }

  // Download transcription as PDF (Optional)
  downloadTranscriptionAsPDF(upload_id: number): void {
    if (this.selectedTranscription) {
      const doc = new jsPDF();
      const lines = doc.splitTextToSize(this.selectedTranscription, 180);
      doc.text(lines, 10, 10);
      doc.save(`transcription_${upload_id}.pdf`);
      console.log(`Transcription téléchargée en tant que PDF: transcription_${upload_id}.pdf`);
    } else {
      alert('No transcription selected for download.');
    }
  }

  // Select an upload entry and display its transcription
  selectUpload(upload_id: number): void {
    this.selectedUploadId = upload_id;
    console.log(`Sélection de l'upload ID: ${upload_id}`);

    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });

    this.http.get<any>(`http://127.0.0.1:8000/get-transcription/${upload_id}`, { headers }).subscribe(
      response => {
        console.log('Retrieved transcription:', response.transcription);
        this.selectedTranscription = response.transcription;
        console.log('selectedTranscription set to:', this.selectedTranscription);
      },
      (error: HttpErrorResponse) => {
        console.error('Error retrieving transcription:', error);
        alert('Error retrieving transcription.');
      }
    );
  }

  // Fetch user-specific upload history
  fetchHistory(): void {
    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });
  
    this.http.get<any>('http://127.0.0.1:8000/history/', { headers }).subscribe(
      response => {
        console.log('History response:', response);
        this.history = response.history; // Assurez-vous que 'history' est une propriété définie dans le composant
        console.log('Historique mis à jour:', this.history);
      },
      (error: HttpErrorResponse) => {
        console.error('Error fetching history:', error);
        alert('Erreur lors de la récupération de l\'historique.');
      }
    );
  }
}
