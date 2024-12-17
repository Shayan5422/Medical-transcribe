// src/app/register/register.component.ts
import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms'; // Import FormsModule for ngModel
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router'; // For navigation

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.css']
})
export class RegisterComponent implements OnInit {
  username: string = '';
  password: string = '';
  confirmPassword: string = '';
  errorMessage: string = '';
  successMessage: string = '';

  constructor(private http: HttpClient, private router: Router) {}

  ngOnInit(): void {}

  onRegister(): void {
    if (this.password !== this.confirmPassword) {
      this.errorMessage = 'Passwords do not match.';
      this.successMessage = '';
      return;
    }
  
    const registerData = { username: this.username, password: this.password };
    this.http.post<any>('http://51.15.224.218:8000/register', registerData).subscribe(
      (response) => {
        this.successMessage = 'Registration successful. You can now log in.';
        this.errorMessage = '';
        this.router.navigate(['/login']);
      },
      (error) => {
        this.errorMessage = 'Registration failed. Username might already be taken.';
        this.successMessage = '';
        console.error('Registration failed', error);
      }
    );
  }
}
