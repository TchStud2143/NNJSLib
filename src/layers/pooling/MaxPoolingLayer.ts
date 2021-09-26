import GradientHolder from '../../GradientHolder';
import Layer from '../Layer';
import Tensor from '../../Tensor';
import Utils from '../../utils/Utils';
import OptimizableLayer from '../OptimizableLayer';

export default class MaxPoolingLayer extends OptimizableLayer {
  inputs: GradientHolder;
  inputShape: number[];
  outputShape: number[];
  filterShape: number[];
  padding: number;
  stride: number;
  switchX: number[];
  switchY: number[];

  constructor(
    w: number,
    h: number,
    d: number,
    filterShape: number[],
    padding: number,
    stride: number = null
  ) {
    const outputW = Math.floor((w + padding * 2 - filterShape[0]) / stride + 1);
    const outputH = Math.floor((h + padding * 2 - filterShape[1]) / stride + 1);
    const outputD = d;

    super(filterShape[0], filterShape[1], filterShape[2]);

    this.inputShape = [w, h, d];
    this.outputShape = [outputW, outputH, outputD];
    this.filterShape = filterShape;
    this.switchX = [];
    this.switchY = [];
  }

  feedForward(inputs: GradientHolder): Layer {
    this.inputs = inputs;

    const output: GradientHolder = new Tensor(
      this.outputShape[0],
      this.outputShape[1],
      this.outputShape[2]
    );

    let switchCounter = 0;
    for (
      let currentDepth = 0;
      currentDepth < this.outputShape[2];
      currentDepth++
    ) {
      let x = -this.padding;
      let y = -this.padding;
      for (
        let outputX = 0;
        outputX < this.outputShape[0];
        x += this.stride, outputX++
      ) {
        y = -this.padding;
        for (
          let outputY = 0;
          outputY < this.outputShape[1];
          y += this.stride, outputY++
        ) {
          let max = Number.MIN_VALUE;
          let maxX = -1;
          let maxY = -1;
          for (let filterX = 0; filterX < this.filterShape[0]; filterX++) {
            for (let filterY = 0; filterY < this.filterShape[1]; filterY++) {
              const originalY = y + filterY;
              const originalX = x + filterX;
              if (
                originalY >= 0 &&
                originalY < inputs.shape[1] &&
                originalX >= 0 &&
                originalX < inputs.shape[0]
              ) {
                const currentValue = inputs.get(
                  originalX,
                  originalY,
                  currentDepth
                );
                if (currentValue > max) {
                  max = currentValue;
                  maxX = originalX;
                  maxY = originalY;
                }
              }
            }
          }
          this.switchX[switchCounter] = maxX;
          this.switchY[switchCounter] = maxY;
          switchCounter++;
          output.set(outputX, outputY, currentDepth, max);
        }
      }
    }

    this.out = output.out;
    return this;
  }

  propagateBackwards(): void {
    let inputs = this.inputs;
    inputs.gradv = Utils.buildOneDimensionalArray(inputs.out.length, () => 0);
    let output = this.items;

    let switchIndex = 0;
    for (
      let currentDepth = 0;
      currentDepth < this.outputShape[2];
      currentDepth++
    ) {
      let x = -this.padding;
      let y = -this.padding;
      for (
        let outputX = 0;
        outputX < this.outputShape[0];
        x += this.stride, outputX++
      ) {
        y = -this.padding;
        for (
          let outputY = 0;
          outputY < this.outputShape[1];
          y += this.stride, outputY++
        ) {
          let chain_grad = output.getGrad(outputX, outputY, currentDepth);
          inputs.gradv[this.switchX[switchIndex]][this.switchY[switchIndex]][
            currentDepth
          ] = chain_grad;
          switchIndex++;
        }
      }
    }
  }
}
