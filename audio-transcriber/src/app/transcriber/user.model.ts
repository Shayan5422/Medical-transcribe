// src/app/models/user.model.ts

export interface User {
    id: number;
    username: string;
    accessType: 'viewer' | 'editor'; // Rendre 'accessType' obligatoire
  }
  