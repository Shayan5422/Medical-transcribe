import { Component } from '@angular/core';
import { TranscriberComponent } from './transcriber/transcriber.component';
import { HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [TranscriberComponent, HttpClientModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'audio-transcriber';
}
