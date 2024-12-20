// click-outside.directive.ts
import { Directive, ElementRef, EventEmitter, Output } from '@angular/core';

@Directive({
  selector: '[clickOutside]',
  standalone: true // Make the directive standalone
})
export class ClickOutsideDirective {
  @Output() clickOutside = new EventEmitter<void>();

  constructor(private elementRef: ElementRef) {
    document.addEventListener('click', this.handleClick.bind(this));
  }

  handleClick(event: MouseEvent) {
    if (!this.elementRef.nativeElement.contains(event.target)) {
      this.clickOutside.emit();
    }
  }

  ngOnDestroy() {
    document.removeEventListener('click', this.handleClick.bind(this));
  }
}