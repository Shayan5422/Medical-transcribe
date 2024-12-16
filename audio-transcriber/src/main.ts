// src/main.ts
import { enableProdMode, importProvidersFrom } from '@angular/core';
import { bootstrapApplication } from '@angular/platform-browser';
import { AppComponent } from './app/app.component';
import { provideRouter } from '@angular/router';
import { appRoutes } from './app/app.routes';
import { HTTP_INTERCEPTORS, HttpClientModule } from '@angular/common/http';
import { AuthInterceptor } from './app/auth.interceptor'; // If you have an interceptor



bootstrapApplication(AppComponent, {
  providers: [
    provideRouter(appRoutes),
    importProvidersFrom(HttpClientModule), // Provide HttpClientModule globally
    {
      provide: HTTP_INTERCEPTORS,
      useClass: AuthInterceptor, // If you have an interceptor
      multi: true
    },
    // Add other global providers here if necessary
  ]
})
.catch(err => console.error(err));
