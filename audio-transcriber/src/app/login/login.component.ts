// src/app/login/login.component.ts
import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms'; // Import FormsModule for ngModel
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Router } from '@angular/router'; // For navigation

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit {
  username: string = '';
  password: string = '';
  errorMessage: string = '';

  constructor(private http: HttpClient, private router: Router) {}

  ngOnInit(): void {}

  onLogin(): void {
    // Construct form data
    const body = new HttpParams()
      .set('username', this.username)
      .set('password', this.password);
  
    // Set headers
    const headers = new HttpHeaders({
      'Content-Type': 'application/x-www-form-urlencoded'
    });
  
    this.http.post<any>('https://backend.shaz.ai/token/', body.toString(), { headers }).subscribe(
      (response) => {
        // Assuming response contains the token
        localStorage.setItem('token', response.access_token);
        localStorage.setItem('username', this.username);  // Store the username
        // Redirect to transcriber or home page
        this.router.navigate(['/transcriber']).then(() => {
          // Refresh the page after navigation
          window.location.reload();
        });
        
      },
      (error) => {
        if (error.status === 422) {
          this.errorMessage = 'Invalid username or password.';
        } else {
          this.errorMessage = 'Login failed. Please try again later.';
        }
        console.error('Login failed', error);
      }
    );
  }
  
}