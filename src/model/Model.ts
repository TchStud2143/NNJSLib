import Layer from '../layers/Layer';
import ModelOptions from './ModelOptions';
import Optimizer from '../optimizer/Optimizer';
import GradientHolder from '../GradientHolder';
import LossType from '../loss/LossType';
import OptimizerType from '../optimizer/OptimizerType';
import SGD from '../optimizer/SGD';
import SoftmaxLayer from '../layers/activation/SoftmaxLayer';

export default abstract class Model {
  layers: Layer[];
  optimizer: Optimizer;
  loss: string;
  out: Layer;

  constructor(layers: Layer[], modelOptions: ModelOptions = null) {
    this.layers = layers;

    if (modelOptions !== null) {
      this.compile(modelOptions);
    }
  }

  compile(options: ModelOptions) {
    switch (options.optimizer) {
      case OptimizerType.SGD:
        this.optimizer = new SGD(this, options.learningRate);
        break;
      default:
        this.optimizer = null;
        break;
    }
    this.loss = options.loss;
  }

  abstract forward(input: GradientHolder): void;

  backward(target: number): number {
    let loss: number = 0;
    const layerCount = this.layers.length;
    const lastLayer = this.layers[layerCount - 1];

    if (lastLayer instanceof SoftmaxLayer) {
      switch (this.loss) {
        case LossType.CrossEntropy:
          loss = lastLayer.propagateBackwards(target);
          for (let layerIndex = layerCount - 2; layerIndex >= 0; layerIndex--) {
            this.layers[layerIndex].propagateBackwards();
          }
          break;
        default:
          loss = null;
          break;
      }
    }

    return loss;
  }

  optimize() {
    this.optimizer.optimize();
  }

  train(inputs: GradientHolder, target: number): number[] {
    const output = this.guess(inputs);
    this.backward(target);
    this.optimize();

    return output;
  }

  guess(inputs: GradientHolder) {
    this.forward(inputs);

    return this.out.out;
  }
}
