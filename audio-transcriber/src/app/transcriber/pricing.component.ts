// pricing.component.ts
import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-pricing',
  standalone: true,
  imports: [CommonModule],
  template: `
    <!-- Version Banner -->
    <div class="bg-base-100 p-4 rounded-lg shadow-lg mb-6 relative overflow-hidden">
      <div class="flex justify-between items-center">
        <!-- Current Plan -->
        <div class="flex items-center gap-2">
          <div class="flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 2H2v10l9.29 9.29c.94.94 2.48.94 3.42 0l6.58-6.58c.94-.94.94-2.48 0-3.42L12 2Z"/>
              <path d="M7 7h.01"/>
            </svg>
            <span class="ml-2 font-bold">Version Gratuite</span>
          </div>
          <button 
            class="btn btn-ghost btn-sm" 
            (click)="togglePlans()"
            [class.rotate-180]="showPlans"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="6 9 12 15 18 9"></polyline>
            </svg>
          </button>
        </div>

        <!-- Upgrade Button -->
        <button class="btn btn-primary btn-sm gap-2" (click)="togglePlans()">
          <span>Plans</span>
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="5" y1="12" x2="19" y2="12"></line>
            <polyline points="12 5 19 12 12 19"></polyline>
          </svg>
        </button>
      </div>

      <!-- Plans Section -->
      <div *ngIf="showPlans" class="mt-4 grid md:grid-cols-2 gap-6">
        <!-- Standard Plan -->
        <div class="bg-base-200 p-6 rounded-xl">
          <div class="flex justify-between items-center mb-4">
            <h3 class="text-lg font-bold">Plan Standard</h3>
            <span class="badge badge-success">Gratuit</span>
          </div>
          <ul class="space-y-3">
            <li class="flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22a10 10 0 1 1 0-20 10 10 0 0 1 0 20z"/>
                <path d="M12 6v6l4 2"/>
              </svg>
              <span>Temps de traitement jusqu'à 5 minutes</span>
            </li>
            <li class="flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="20 6 9 17 4 12"></polyline>
              </svg>
              <span>Générations illimitées</span>
            </li>
            <li class="flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                <polyline points="22 4 12 14.01 9 11.01"></polyline>
              </svg>
              <span>Précision haute qualité</span>
            </li>
            <li class="flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                <line x1="12" y1="8" x2="12" y2="16"></line>
                <line x1="8" y1="12" x2="16" y2="12"></line>
              </svg>
              <span>Une tâche à la fois</span>
            </li>
          </ul>
        </div>

        <!-- Premium Plan -->
        <div class="bg-base-200 p-6 rounded-xl relative">
          <div class="absolute top-0 right-0 bg-warning text-warning-content px-3 py-1 rounded-bl-lg rounded-tr-lg text-sm">
            Premium ✨
          </div>
          <div class="flex justify-between items-center mb-4">
            <h3 class="text-lg font-bold">Plan Premium</h3>
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
              <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
            </svg>
          </div>
          <ul class="space-y-3">
            <li class="flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
              </svg>
              <span>Traitement instantané</span>
            </li>
            <li class="flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M2 4l3 12h14l3-12-6 7-4-7-4 7-6-7z"></path>
                <path d="M3 20h18"></path>
              </svg>
              <span>Précision haute qualité optimisée</span>
            </li>
            <li class="flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                <circle cx="9" cy="7" r="4"></circle>
                <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
                <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
              </svg>
              <span>Support prioritaire</span>
            </li>
            <li class="flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
              </svg>
              <span>Formats d'export personnalisés</span>
            </li>
          </ul>
          <button 
            (click)="showContactEmail = true"
            class="btn btn-warning w-full mt-4 gap-2"
          >
            <span>Upgrade</span>
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M2 4l3 12h14l3-12-6 7-4-7-4 7-6-7z"></path>
              <path d="M3 20h18"></path>
            </svg>
          </button>
        </div>

        <!-- Contact Email -->
        <div *ngIf="showContactEmail" class="md:col-span-2 alert alert-info">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="2" y="4" width="20" height="16" rx="2"></rect>
            <path d="m22 4-10 8L2 4"></path>
          </svg>
          <span>Contactez-nous: contact&#64;alasuite.com</span>
        </div>

        <!-- Voice clarity note -->
        
      </div>
    </div>
  `
})
export class PricingComponent {
  showPlans = false;
  showContactEmail = false;

  togglePlans(): void {
    this.showPlans = !this.showPlans;
    if (!this.showPlans) {
      this.showContactEmail = false;
    }
  }
}