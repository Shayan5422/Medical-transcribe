// src/app/transcriber/user-filter.pipe.ts

import { Pipe, PipeTransform } from '@angular/core';
import { User } from './user.model';

@Pipe({
  name: 'userFilter',
  standalone: true
})
export class UserFilterPipe implements PipeTransform {

  transform(users: User[], searchQuery: string): User[] {
    if (!searchQuery) {
      return users;
    }
    return users.filter(user => 
      user.username.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }

}
