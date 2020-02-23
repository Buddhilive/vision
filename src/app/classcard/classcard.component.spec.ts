import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { ClasscardComponent } from './classcard.component';

describe('ClasscardComponent', () => {
  let component: ClasscardComponent;
  let fixture: ComponentFixture<ClasscardComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ ClasscardComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(ClasscardComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
