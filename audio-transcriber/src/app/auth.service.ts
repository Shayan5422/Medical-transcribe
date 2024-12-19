import { Injectable } from '@angular/core';
import { Router } from '@angular/router';
import { JwtHelperService } from '@auth0/angular-jwt';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private tokenKey = 'token';
  private jwtHelper = new JwtHelperService();

  constructor(private router: Router) {
    // Check token validity periodically
    setInterval(() => {
      if (this.isTokenExpired()) {
        this.logout();
      }
    }, 60000); // Check every minute
  }

  setToken(token: string): void {
    localStorage.setItem(this.tokenKey, token);
  }

  getToken(): string | null {
    return localStorage.getItem(this.tokenKey);
  }

  isTokenExpired(): boolean {
    const token = this.getToken();
    if (!token) return true;
    
    try {
      return this.jwtHelper.isTokenExpired(token);
    } catch (error) {
      return true;
    }
  }

  isLoggedIn(): boolean {
    return this.getToken() !== null && !this.isTokenExpired();
  }

  logout(): void {
    localStorage.removeItem(this.tokenKey);
    this.router.navigate(['/login']);
  }
}