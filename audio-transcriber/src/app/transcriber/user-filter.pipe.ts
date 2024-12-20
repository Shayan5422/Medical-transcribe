
import { Pipe, PipeTransform } from '@angular/core';

interface User {
  id: number;
  username: string;
}

@Pipe({
  name: 'userFilter',
  standalone: true
})
export class UserFilterPipe implements PipeTransform {
  transform(users: User[], searchText: string): User[] {
    if (!users) return [];
    if (!searchText) return users;

    searchText = searchText.toLowerCase();
    
    return users.filter(user => {
      return user.username.toLowerCase().includes(searchText);
    });
  }
}