import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { ControllerDataset } from '../app/controllerdataset.directive';
import { HttpClient } from '@angular/common/http';
import { async } from '@angular/core/testing';

interface LearningRateData {
  value: number;
  viewValue: string;
}

interface BatchSizeData {
  value: number;
  viewValue: string;
}

interface EpochsData {
  value: number;
  viewValue: string;
}

interface DenseData {
  value: number;
  viewValue: string;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {

  @ViewChild('labelName', { static: false }) labelName: ElementRef;
  @ViewChild('trainingDataset', { static: false }) trainingDataset: ElementRef;
  @ViewChild('imageArea', { static: false }) imageArea: ElementRef;
  @ViewChild('blvUpload', { static: false }) blvUpload: ElementRef;

  title = 'BuddhiliveVision';
  IMAGE_SIZE = 224;
  NUM_CLASSES = 2;

  controllerDataset = new ControllerDataset(this.NUM_CLASSES);

  consoleLog: any;

  truncatedMobileNet: any;
  model: any;
  trainImg = [];

  // training configuration
  learningRateValues: LearningRateData[] = [
    { value: 0.00001, viewValue: '0.00001' },
    { value: 0.0001, viewValue: '0.0001' },
    { value: 0.001, viewValue: '0.001' },
    { value: 0.003, viewValue: '0.003' }
  ];

  batchSizeValues: BatchSizeData[] = [
    { value: 0.05, viewValue: '0.05' },
    { value: 0.1, viewValue: '0.1' },
    { value: 0.4, viewValue: '0.4' },
    { value: 1, viewValue: '1' }
  ];

  epochsValues: EpochsData[] = [
    { value: 10, viewValue: '10' },
    { value: 20, viewValue: '20' },
    { value: 40, viewValue: '40' }
  ];

  denseValues: DenseData[] = [
    { value: 10, viewValue: '10' },
    { value: 100, viewValue: '100' },
    { value: 200, viewValue: '200' }
  ];

  denseUnits = 100;
  learningRate = 0.0001;
  batchSizeFraction = 0.4;
  epochsNum = 20;
  labelNames = [];
  label = 0;
  autoMode = false;

  //disable properties
  btnTrain = true;
  btnDownload = true;
  btnNew = true;
  btnLabel = true;
  btnSample = false;
  btnAddExample = false;
  btnUpload = true;
  numClassField = false;
  labelInput = false;

  constructor(private httpClient: HttpClient) { }

  updateLearningRate(getValue) {
    this.learningRate = getValue;
  }

  updateBatchSize(getValue) {
    this.batchSizeFraction = getValue;
  }

  updateEpochs(getValue) {
    this.epochsNum = getValue;
  }

  updateDenseUnits(getValue) {
    this.denseUnits = getValue;
  }

  updateControllerData(getValue) {
    this.NUM_CLASSES = +getValue.target.value;
    this.controllerDataset = new ControllerDataset(this.NUM_CLASSES);
  }

  switchTrainingMode() {
    this.autoMode = !this.autoMode;
    if (this.autoMode) {
      this.numClassField = true;
    } else {
      this.numClassField = false;
    }

  }

  async loadTruncatedMobileNet() {
    const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

    const layer = mobilenet.getLayer('conv_pw_13_relu');
    return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
  }

  async getTrainData() {
    this.numClassField = true;
    const newLabel = this.labelName.nativeElement.value;
    if (newLabel == '') {
      this.consoleLog.innerHTML = `Enter a label name!`;
      //throw new Error('Enter a label name!');
    } else {

      this.imageArea.nativeElement.querySelectorAll('.blv-training-img').forEach((elmnt) => {
        //this.imageArea.nativeElement.appendChild(elmnt);
        const img = tf.browser.fromPixels(elmnt).toFloat();
        const offset = tf.scalar(127.5);
        const normalized = img.sub(offset).div(offset);
        const batched = normalized.reshape([1, this.IMAGE_SIZE, this.IMAGE_SIZE, 3]);
        this.controllerDataset.addExample(this.truncatedMobileNet.predict(batched), this.label);

        img.dispose();

      });
      console.log(this.labelNames[this.label], this.label);
      if (this.label < (this.NUM_CLASSES - 1)) {
        this.label++;
        this.newBatch();
        this.btnLabel = true;
      } else {
        this.labelName.nativeElement.value = ``;
        this.btnTrain = false;
        this.btnUpload = true;
        this.labelInput = true;
        this.btnLabel = true;
      }

    }
  }

  async train() {
    if (this.controllerDataset.xs == null) {
      this.consoleLog.innerHTML = `Add some examples before training!`;
      //throw new Error('Add some examples before training!');
    } else {
      this.model = tf.sequential({
        layers: [
          tf.layers.flatten({ inputShape: this.truncatedMobileNet.outputs[0].shape.slice(1) }),
          // Layer 1.
          tf.layers.dense({
            units: this.denseUnits,
            activation: 'relu',
            kernelInitializer: 'varianceScaling',
            useBias: true
          }),

          tf.layers.dense({
            units: this.NUM_CLASSES,
            kernelInitializer: 'varianceScaling',
            useBias: false,
            activation: 'softmax'
          })
        ]
      });

      const optimizer = tf.train.adam(this.learningRate);
      this.model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });
      const batchSize =
        Math.floor(this.controllerDataset.xs.shape[0] * this.batchSizeFraction);
      if (!(batchSize > 0)) {
        throw new Error(
          `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
      }
      let epochIndex = 0;
      // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
      this.model.fit(this.controllerDataset.xs, this.controllerDataset.ys, {
        batchSize,
        epochs: this.epochsNum,
        callbacks: {
          onBatchEnd: async (batch, logs) => {
            console.log('Loss: ' + logs.loss.toFixed(5), batch);
            this.consoleLog.innerHTML = 'Epochs: ' + epochIndex + ' | Loss: ' + logs.loss.toFixed(5);
            epochIndex++;
          }
        }
      }).then(info => {
        console.log('Final accuracy', info);
        this.btnDownload = false;
        this.btnNew = false;
        this.btnTrain = true;
        this.consoleLog.innerHTML = `Training Completed!`;
      });
    }
  }

  downloadModel() {
    this.model.save('downloads://my-model');
    const jsonObj = JSON.stringify({ labels: this.labelNames });
    const blob = new Blob([jsonObj], { type: 'text/plain' });
    const aLink = document.createElement("a");
    const url = window.URL.createObjectURL(blob);
    aLink.href = url;
    aLink.download = 'buddhivision-labels';
    aLink.click();
  }

  getFileUpload(evt) {
    const newLabel = this.labelName.nativeElement.value;
    if (newLabel !== '') {
      this.labelNames.push(newLabel.toLowerCase());
      const csv = evt.target.files;
      for (let i = 0, imgEl; imgEl = csv[i]; i++) {
        if (imgEl.type.match('image.*')) {
          const reader = new FileReader();
          reader.readAsDataURL(imgEl);
          reader.onload = (eData) => {
            const textLinks = reader.result as string;
            //console.log(eData, reader);
            this.consoleLog.innerHTML = 'Uploaded: ' + imgEl.name;
            const elmnt = document.createElement('img');
            elmnt.src = textLinks;
            elmnt.classList.add('blv-training-img');
            elmnt.width = this.IMAGE_SIZE;
            elmnt.height = this.IMAGE_SIZE;
            this.imageArea.nativeElement.appendChild(elmnt);
          };
          this.btnLabel = false;
        } else {
          this.consoleLog.innerHTML = 'Upload only image files';
        }
      }
    } else {
      this.consoleLog.innerHTML = 'Enter a label name!';
    }
  }

  resetApp() {
    /* const classTemplate = "";
    const tempDiv = this.renderer.createElement('div');
    this.renderer.addClass(tempDiv, 'blv-training--class');
    this.renderer.setProperty(tempDiv, 'innerHTML', classTemplate);
    this.renderer.appendChild(this.trainingDataset.nativeElement, tempDiv); */
    this.imageArea.nativeElement.innerHTML = ``;
    this.blvUpload.nativeElement.value = ``;
    this.numClassField = false;
    this.btnTrain = true;
    this.btnDownload = true;
    this.btnNew = true;
    this.btnLabel = true;
    this.labelInput = false;
  }

  enableUpload() {
    const newLabel = this.labelName.nativeElement.value;
    if (newLabel !== '') {
      this.btnUpload = false;
    }
  }

  newBatch() {
    this.imageArea.nativeElement.innerHTML = ``;
    this.labelName.nativeElement.value = '';
    this.btnTrain = true;
    this.btnDownload = true;
    this.btnNew = true;
    this.btnUpload = true;
    this.labelInput = false;
    this.consoleLog.innerHTML = ``;
    this.blvUpload.nativeElement.value = ``;
  }

  async autoTrainModel(fileData) {
    const theFile = fileData.target.files[0];
    const reader = new FileReader();
    let training_Data = [];
    let jsonData;
    reader.readAsText(theFile);
    reader.onload = (eData) => {
      if (theFile.type.match('application/json')) {
        this.consoleLog.innerHTML = 'Uploaded: ' + theFile.name;

        jsonData = JSON.parse(reader.result as string);
        this.NUM_CLASSES = jsonData.num_classes;
        training_Data = jsonData.training_data;
        this.labelNames = jsonData.class_names;

        training_Data.map((linkData, labelIndex) => {
          linkData.map((imgLink, linkIndex) => {
            //this.imageArea.nativeElement.innerHTML = '';
            const elmnt = document.createElement('img');
            elmnt.src = imgLink;
            elmnt.classList.add('blv-training-img');
            elmnt.width = this.IMAGE_SIZE;
            elmnt.height = this.IMAGE_SIZE;
            //this.imageArea.nativeElement.appendChild(elmnt);
            //set training data
            elmnt.onload = async () => {
              const img = tf.browser.fromPixels(elmnt).toFloat();
              const offset = tf.scalar(127.5);
              const normalized = img.sub(offset).div(offset);
              const batched = normalized.reshape([1, this.IMAGE_SIZE, this.IMAGE_SIZE, 3]);
              await this.controllerDataset.addExample(this.truncatedMobileNet.predict(batched), labelIndex);

              img.dispose();
              console.log(labelIndex, linkIndex, linkData.length);
              if (labelIndex === (this.NUM_CLASSES - 1) && linkIndex === (linkData.length - 1)) {
                this.btnTrain = false;
                this.consoleLog.innerHTML = 'Training data is ready for train';
              }
            };
          });
        });
      } else {
        this.consoleLog.innerHTML = 'Upload only .json files.';
      }
      console.log(training_Data);
    };
  }

  async initializeVision() {
    this.truncatedMobileNet = await this.loadTruncatedMobileNet();
    this.consoleLog = document.querySelector('.console');
    this.btnTrain = true;
    this.btnDownload = true;
    this.btnNew = true;
    this.btnLabel = true;
    this.consoleLog.innerHTML = 'Initialized!';
  }

  ngOnInit() {
    this.initializeVision();
  }
}
