// audio-player.component.ts
import { Component, Input, OnInit, OnDestroy, ViewChild, ElementRef, ChangeDetectorRef, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpHeaders } from '@angular/common/http';

@Component({
  selector: 'app-audio-player',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="w-full bg-base-200 rounded-lg p-4 shadow-md">
      <audio #audioElement preload="metadata" 
             (loadedmetadata)="onLoadedMetadata()" 
             (timeupdate)="onTimeUpdate()"
             (ended)="onEnded()">
      </audio>

      <!-- Timeline and controls -->
      <div class="flex justify-between text-sm mb-2">
        <span>{{formatTime(currentTime)}}</span>
        <span>{{formatTime(duration)}}</span>
      </div>

      <!-- Progress bar -->
      <div class="relative w-full h-2 bg-base-300 rounded-full mb-4 cursor-pointer"
           (click)="seek($event)">
        <div 
          class="absolute top-0 left-0 h-full bg-primary rounded-full transition-all"
          [style.width.%]="(currentTime / duration) * 100">
        </div>
      </div>

      <!-- Controls -->
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-4">
          <!-- Skip Backward -->
          <button 
            (click)="skipBackward()"
            class="p-2 hover:bg-base-300 rounded-full transition-colors"
            [class.opacity-50]="!isLoaded"
            [disabled]="!isLoaded">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" 
                 stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="m12 19-7-7 7-7"></path>
              <path d="M19 19l-7-7 7-7"></path>
            </svg>
          </button>
          
          <!-- Play/Pause -->
          <button 
            (click)="togglePlay()"
            class="p-3 bg-primary hover:bg-primary-focus text-primary-content rounded-full transition-colors"
            [class.opacity-50]="!isLoaded"
            [disabled]="!isLoaded">
            <ng-container *ngIf="!isPlaying">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" 
                   stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
              </svg>
            </ng-container>
            <ng-container *ngIf="isPlaying">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" 
                   stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="6" y="4" width="4" height="16"></rect>
                <rect x="14" y="4" width="4" height="16"></rect>
              </svg>
            </ng-container>
          </button>
          
          <!-- Skip Forward -->
          <button 
            (click)="skipForward()"
            class="p-2 hover:bg-base-300 rounded-full transition-colors"
            [class.opacity-50]="!isLoaded"
            [disabled]="!isLoaded">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" 
                 stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="m5 19 7-7-7-7"></path>
              <path d="m12 19 7-7-7-7"></path>
            </svg>
          </button>
        </div>

        <!-- Volume Controls -->
        <div class="flex items-center space-x-2">
          <button 
            (click)="toggleMute()"
            class="p-2 hover:bg-base-300 rounded-full transition-colors"
            [class.opacity-50]="!isLoaded"
            [disabled]="!isLoaded">
            <ng-container *ngIf="!isMuted">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" 
                   stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
              </svg>
            </ng-container>
            <ng-container *ngIf="isMuted">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" 
                   stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                <line x1="22" y1="9" x2="16" y2="15"></line>
                <line x1="16" y1="9" x2="22" y2="15"></line>
              </svg>
            </ng-container>
          </button>
          
          <input
            type="range"
            class="w-24 h-2 bg-base-300 rounded-lg appearance-none cursor-pointer"
            [value]="volume"
            min="0"
            max="1"
            step="0.1"
            (input)="onVolumeChange($event)"
            [disabled]="!isLoaded"
          />
        </div>
      </div>

      <!-- Loading indicator -->
      <div *ngIf="isLoading" class="text-center text-sm mt-2">
        Chargement de l'audio...
      </div>
    </div>
  `,
  styles: [`
    input[type="range"] {
      -webkit-appearance: none;
      background: transparent;
    }

    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      height: 16px;
      width: 16px;
      border-radius: 50%;
      background: hsl(var(--p));
      margin-top: -6px;
      cursor: pointer;
    }

    input[type="range"]::-webkit-slider-runnable-track {
      width: 100%;
      height: 4px;
      background: hsl(var(--b3));
      border-radius: 2px;
    }

    input[type="range"]:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
  `]
})
export class AudioPlayerComponent implements OnInit, OnDestroy {
  @Input() audioUrl!: string;
  @Input() token: string | null = null;
  @ViewChild('audioElement') audioElement!: ElementRef<HTMLAudioElement>;

  isPlaying: boolean = false;
  isLoaded: boolean = false;
  isLoading: boolean = false;
  isMuted: boolean = false;
  duration: number = 0;
  currentTime: number = 0;
  volume: number = 1;

  constructor(
    private http: HttpClient,
    private cdr: ChangeDetectorRef
  ) {}

  ngOnInit() {
    this.loadAudio();
  }
  private resetPlayer() {
    const audio = this.audioElement?.nativeElement;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
      audio.src = '';
    }
    this.isPlaying = false;
    this.isLoaded = false;
    this.currentTime = 0;
    this.duration = 0;
    
    this.cdr.detectChanges();
  }

  
  ngOnDestroy() {
    const audio = this.audioElement?.nativeElement;
    if (audio) {
      audio.pause();
      audio.src = '';
      this.resetPlayer();
    }
  }
  ngOnChanges(changes: SimpleChanges) {
    // Check if audioUrl has changed
    if (changes['audioUrl'] && !changes['audioUrl'].firstChange) {
      // Reset player state
      this.resetPlayer();
      // Load new audio
      this.loadAudio();
    }
  }
  private loadAudio() {
    if (!this.token || !this.audioUrl) {
      console.error('Missing token or URL');
      return;
    }

    this.isLoading = true;

    const headers = new HttpHeaders({
      'Authorization': `Bearer ${this.token}`
    });

    this.http.get(this.audioUrl, { 
      responseType: 'blob',
      headers: headers 
    }).subscribe({
      next: (blob) => {
        const url = URL.createObjectURL(blob);
        const audio = this.audioElement.nativeElement;
        audio.src = url;
        
        // Important: Explicitly load the audio
        audio.load();
        
        this.isLoading = false;
        this.cdr.detectChanges();
      },
      error: (error) => {
        console.error('Error loading audio:', error);
        this.isLoading = false;
        this.cdr.detectChanges();
      }
    });
  }

  onLoadedMetadata() {
    const audio = this.audioElement.nativeElement;
    this.duration = audio.duration;
    this.isLoaded = true;
    this.cdr.detectChanges();
  }

  onTimeUpdate() {
    const audio = this.audioElement.nativeElement;
    this.currentTime = audio.currentTime;
    this.cdr.detectChanges();
  }

  onEnded() {
    this.isPlaying = false;
    this.cdr.detectChanges();
  }

  togglePlay() {
    if (!this.isLoaded) return;
    
    const audio = this.audioElement.nativeElement;
    if (this.isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
    this.isPlaying = !this.isPlaying;
  }

  skipForward() {
    if (!this.isLoaded) return;
    const audio = this.audioElement.nativeElement;
    audio.currentTime = Math.min(audio.currentTime + 10, audio.duration);
  }

  skipBackward() {
    if (!this.isLoaded) return;
    const audio = this.audioElement.nativeElement;
    audio.currentTime = Math.max(audio.currentTime - 10, 0);
  }

  toggleMute() {
    if (!this.isLoaded) return;
    const audio = this.audioElement.nativeElement;
    audio.muted = !audio.muted;
    this.isMuted = audio.muted;
  }

  onVolumeChange(event: Event) {
    if (!this.isLoaded) return;
    const value = (event.target as HTMLInputElement).value;
    this.volume = parseFloat(value);
    const audio = this.audioElement.nativeElement;
    audio.volume = this.volume;
    this.isMuted = this.volume === 0;
  }

  seek(event: MouseEvent) {
    if (!this.isLoaded) return;
    const audio = this.audioElement.nativeElement;
    const element = event.currentTarget as HTMLElement;
    const rect = element.getBoundingClientRect();
    const percentage = (event.clientX - rect.left) / rect.width;
    audio.currentTime = percentage * audio.duration;
  }

  formatTime(time: number): string {
    if (isNaN(time)) return '0:00';
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }
}