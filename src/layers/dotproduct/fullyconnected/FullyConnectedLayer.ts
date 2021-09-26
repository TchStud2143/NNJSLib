import Utils from '../../../utils/Utils';
import OptimizableLayer from '../../OptimizableLayer';
import Tensor from '../../../Tensor';
import GradientHolder from '../../../GradientHolder';
import Layer from '../../Layer';
export default class FullyConnectedLayer extends OptimizableLayer {
  filters: Tensor[];
  inputShape: number[];
  biases: Tensor;

  constructor(w, h, d, neuronCount, bias: number = 0) {
    super(1, 1, neuronCount);

    this.inputShape = [w, h, d];

    this.filters = [];
    for (var filterIndex = 0; filterIndex < d; filterIndex++) {
      this.filters.push(new Tensor(1, 1, w));
    }
    this.biases = new Tensor(1, 1, neuronCount);
    this.biases.out = Utils.buildOneDimensionalArray(neuronCount, () => bias);
  }

  feedForward(inputs: GradientHolder): Layer {
    const output = new Tensor(1, 1, this.shape[2]);
    const weights = inputs.out;
    for (let i = 0; i < this.shape[2]; i++) {
      let weight = 0.0;
      const filterWeights = this.filters[i].out;
      for (
        let weightIndex = 0;
        weightIndex < this.inputShape[0];
        weightIndex++
      ) {
        weight += weights[weightIndex] * filterWeights[weightIndex];
      }
      weight += this.biases.out[i];
      output.out[i] = weight;
    }

    return this;
  }

  propagateBackwards() {
    const inputs: GradientHolder = this.items;
    inputs.gradv = Utils.buildOneDimensionalArray(inputs.out.length);

    for (
      let currentDepth: number = 0;
      currentDepth < this.shape[2];
      currentDepth++
    ) {
      const filter: Tensor = this.filters[currentDepth];
      const chainedGradient = this.items.gradv[currentDepth];

      for (var index = 0; index < this.inputShape[0]; index++) {
        inputs.gradv[index] += filter.out[index] * chainedGradient;
        filter.gradv[index] += inputs.out[index] * chainedGradient;
      }

      this.biases.gradv[currentDepth] += chainedGradient;
    }
  }
}
