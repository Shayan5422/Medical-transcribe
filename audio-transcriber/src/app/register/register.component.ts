// src/app/register/register.component.ts
import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms'; // Importer FormsModule pour ngModel
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router'; // Pour la navigation

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
  referralCode: string = ''; // Ajout du champ Code de parrainage
  errorMessage: string = '';
  successMessage: string = '';

  constructor(private http: HttpClient, private router: Router) {}

  ngOnInit(): void {}

  onRegister(): void {
    // Vérification de la correspondance des mots de passe
    if (this.password !== this.confirmPassword) {
      this.errorMessage = 'Les mots de passe ne correspondent pas.';
      this.successMessage = '';
      return;
    }

    // Vérification du code de parrainage
    if (this.referralCode !== 'neurocorehealth') {
      this.errorMessage = 'Le code de parrainage est invalide.';
      this.successMessage = '';
      return;
    }

    const registerData = { 
      username: this.username, 
      password: this.password,
      referralCode: this.referralCode // Envoi du code de parrainage au serveur (optionnel)
    };

    this.http.post<any>('http://127.0.0.1:8000/register/', registerData).subscribe(
      (response) => {
        this.successMessage = 'Inscription réussie. Vous pouvez maintenant vous connecter.';
        this.errorMessage = '';
        this.router.navigate(['/login']);
      },
      (error) => {
        this.errorMessage = 'L\'inscription a échoué. Le nom d\'utilisateur pourrait être déjà pris.';
        this.successMessage = '';
        console.error('Échec de l\'inscription', error);
      }
    );
  }
}
