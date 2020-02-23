import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { AutoTrainerComponent } from './auto-trainer.component';

describe('AutoTrainerComponent', () => {
  let component: AutoTrainerComponent;
  let fixture: ComponentFixture<AutoTrainerComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ AutoTrainerComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(AutoTrainerComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
