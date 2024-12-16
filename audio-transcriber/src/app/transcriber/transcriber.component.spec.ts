import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TranscriberComponent } from './transcriber.component';

describe('TranscriberComponent', () => {
  let component: TranscriberComponent;
  let fixture: ComponentFixture<TranscriberComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TranscriberComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TranscriberComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
