// src/app/app.component.ts
import { Component ,OnInit} from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { Router } from '@angular/router'; 
import { HttpClientModule } from '@angular/common/http'; // Already provided globally

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

  constructor(private router: Router) {}

  ngOnInit(): void {
    // Vérifiez si l'utilisateur est authentifié en cherchant le token dans le localStorage
    const token = localStorage.getItem('token');
    if (token) {
      this.isAuthenticated = true;
      // Vous pouvez ici également récupérer le nom d'utilisateur à partir du token si nécessaire
      this.username = localStorage.getItem('username') || 'User';  // Utilisez un nom d'utilisateur stocké dans le localStorage
    }
  }

  logout(): void {
    // Supprimez le token et redirigez l'utilisateur vers la page de connexion
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    this.isAuthenticated = false;
    this.router.navigate(['/login']);
    
  }
}
