// src/app/app.routes.ts
import { Routes } from '@angular/router';
import { TranscriberComponent } from './transcriber/transcriber.component';
import { LoginComponent } from './login/login.component';
import { RegisterComponent } from './register/register.component';
import { AuthGuard } from './auth.guard';

export const appRoutes: Routes = [
  { path: '', redirectTo: '/transcriber', pathMatch: 'full' },
  { path: 'transcriber', component: TranscriberComponent, canActivate: [AuthGuard] }, // Protected Route
  { path: 'login', component: LoginComponent },
  { path: 'register', component: RegisterComponent },
  // Add other routes here as needed
];
