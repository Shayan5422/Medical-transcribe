import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { Router } from '@angular/router';
import { HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  isAuthenticated: boolean = false;
  username: string = '';
  isMenuOpen = false;

  constructor(private router: Router) {}

  ngOnInit(): void {
    const token = localStorage.getItem('token');
    if (token) {
      this.isAuthenticated = true;
      this.username = localStorage.getItem('username') || 'User';
    }

    // Close mobile menu on window resize
    window.addEventListener('resize', () => {
      if (window.innerWidth >= 768) {
        this.isMenuOpen = false;
      }
    });
  }

  logout(): void {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    this.isAuthenticated = false;
    this.router.navigate(['/login']);
  }

  ngOnDestroy(): void {
    window.removeEventListener('resize', () => {
      if (window.innerWidth >= 768) {
        this.isMenuOpen = false;
      }
    });
  }
}