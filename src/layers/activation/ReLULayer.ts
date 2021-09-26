import GradientHolder from '../../GradientHolder';
import Layer from '../Layer';
import ActivationLayer from './ActivationLayer';
import Utils from '../../utils/Utils';
import Tensor from '../../Tensor';

export default class ReLULayer extends ActivationLayer {
  inputs: GradientHolder;

  feedForward(inputs: GradientHolder): Layer {
    var outputLength = inputs.out.length;
    var weights = [...inputs.out];
    for (var i = 0; i < outputLength; i++) {
      if (weights[i] < 0) weights[i] = 0;
    }
    this.items = new Tensor(this.shape[0], this.shape[1], this.shape[2]);
    this.items.out = weights;

    return this;
  }

  propagateBackwards(): void {
    const inputs: GradientHolder = this.inputs;
    const output = this.items;
    const outputLength = inputs.out.length;
    inputs.gradv = Utils.buildOneDimensionalArray(outputLength, () => 0);
    for (var i = 0; i < outputLength; i++) {
      if (output.out[i] <= 0) inputs.gradv[i] = 0;
      else inputs.gradv[i] = output.gradv[i];
    }
  }
}
