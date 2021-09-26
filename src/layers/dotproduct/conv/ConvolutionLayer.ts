import Tensor from '../../../Tensor';
import Assertion from '../../../utils/Assertion';
import Utils from '../../../utils/Utils';
import KernelPaddingType from '../../../padding/KernelPaddingType';
import Layer from '../../Layer';
import GradientHolder from '../../../GradientHolder';
import OptimizableLayer from '../../OptimizableLayer';

function mult(array: number[]) {
  let acc = 1;

  for (const n of array) {
    acc *= n;
  }

  return acc;
}

export default class ConvolutionLayer extends OptimizableLayer {
  inputs: GradientHolder;
  inputShape: number[];
  stride: number;
  padding: number;
  filterCount: number;
  filters: Tensor[];
  biases: Tensor;

  constructor(
    w: number, // Input Width
    h: number, // Input Height
    d: number, // Input Depth
    padding: number,
    filterShape: number[],
    filterCount: number,
    stride: number = null,
    bias: number = 0
  ) {
    const outputW = Math.floor((w + padding * 2 - filterShape[0]) / stride + 1);
    const outputH = Math.floor((h + padding * 2 - filterShape[1]) / stride + 1);
    const outputD = filterCount;

    super(outputW, outputH, outputD);

    this.inputShape = [w, h, d];

    this.filterCount = filterCount;
    this.filters = [];

    for (let filterIndex = 0; filterIndex < filterCount; ++filterIndex) {
      this.filters.push(
        new Tensor(this.shape[0], this.shape[1], this.inputShape[2])
      );
    }

    this.padding = padding;
    this.stride = stride || 1;
    this.biases = new Tensor(1, 1, filterCount);
    this.biases.setOut(Utils.buildOneDimensionalArray(filterCount, () => bias));
  }

  feedForward(inputs: GradientHolder): Layer {
    Assertion.assert(
      Utils.shapeEquals(this.inputShape, inputs.shape),
      'Input Shape must match the convolution input shape!'
    );

    this.inputs = inputs;

    const stride = this.stride;

    for (let filterIndex = 0; filterIndex < this.filterCount; ++filterIndex) {
      const filter: Tensor = this.filters[filterIndex];
      let x: number = this.padding;
      let y: number = this.padding;

      for (let outputY = 0; outputY < this.shape[1]; y += stride, ++outputY) {
        for (let outputX = 0; outputX < this.shape[0]; x += stride, ++outputX) {
          let accumulator = 0;

          for (let filterY = 0; filterY < filter.shape[1]; ++filterY) {
            const inputY = y + filterY;

            for (let filterX = 0; filterX < filter.shape[0]; ++filterX) {
              const inputX = x + filterX;

              // Bounds Check
              if (
                inputY >= 0 &&
                inputY < inputs.shape[1] &&
                inputX >= 0 &&
                inputX < inputs.shape[0]
              ) {
                /*console.log(
                  (inputs.shape[0] * inputY + inputX) * inputs.shape[2] +
                    filterIndex
                );*/
                accumulator +=
                  filter.out[
                    (filter.shape[0] * filterY + filterX) * filter.shape[2] +
                      filterIndex
                  ] *
                  inputs.out[
                    (inputs.shape[0] * inputY + inputX) * inputs.shape[2] +
                      filterIndex
                  ];
              }
            }
          }

          accumulator += this.biases.out[filterIndex];
          this.set(outputX, outputY, filterIndex, accumulator);
        }
      }
    }

    return this;
  }

  propagateBackwards() {
    const inputs = this.inputs;
    inputs.gradv = Utils.buildOneDimensionalArray(inputs.out.length, () => 0);

    const inputsW = inputs.shape[0];
    const inputsH = inputs.shape[1];
    const stride = this.stride;

    for (let filterIndex = 0; filterIndex < this.shape[2]; filterIndex++) {
      const filter = this.filters[filterIndex];
      let x: number;
      let y: number;
      for (let outputY = 0; outputY < this.shape[1]; y += stride, outputY++) {
        x = -this.padding;
        for (let outputX = 0; outputX < this.shape[0]; x += stride, outputX++) {
          const chain_grad = this.getGrad(outputX, outputY, filterIndex);
          for (let filterY = 0; filterY < filter.shape[1]; filterY++) {
            let inputY = y + filterY;
            for (var filterX = 0; filterX < filter.shape[0]; filterX++) {
              let inputX = x + filterX;
              if (
                inputY >= 0 &&
                inputY < inputsH &&
                inputX >= 0 &&
                inputX < inputsW
              ) {
                for (
                  let filterDepth = 0;
                  filterDepth < filter.shape[2];
                  filterDepth++
                ) {
                  let inputIndex =
                    (inputsW * inputY + inputX) * inputs.shape[2] + filterDepth;
                  let filterIndex =
                    (filter.shape[0] * filterY + filterX) * filter.shape[2] +
                    filterDepth;
                  filter.gradv[filterIndex] +=
                    inputs.out[inputIndex] * chain_grad;
                  inputs.gradv[inputIndex] +=
                    filter.out[filterIndex] * chain_grad;
                }
              }
            }
          }
          this.biases.gradv[filterIndex] += chain_grad;
        }
      }
    }
  }
}
