import GradientHolder from '../../GradientHolder';
import Layer from '../Layer';
import ActivationLayer from './ActivationLayer';
import Tensor from '../../Tensor';
import Utils from '../../utils/Utils';

export default class SoftmaxLayer extends ActivationLayer {
  exponents: number[];

  feedForward(inputs: GradientHolder): Layer {
    const length = this.shape[2];
    const tensor: Tensor = new Tensor(1, 1, length, true);
    const activation = inputs.out;
    const activationMax = Math.max(...inputs.out);

    const expArray = Utils.buildOneDimensionalArray(length, () => 0);

    let expSum = 0.0;
    for (let expIndex = 0; expIndex < length; ++expIndex) {
      const exp = Math.exp(activation[expIndex] - activationMax);
      expSum += exp;
      expArray[expIndex] = exp;
    }

    for (let index = 0; index < length; index++) {
      expArray[index] /= expSum;
      tensor.out[index] = expArray[index];
    }

    this.exponents = expArray;
    this.items = tensor;
    return this;
  }

  propagateBackwards(correctLabel: number = 0): number {
    const inputs: GradientHolder = this.items;
    inputs.gradv = Utils.buildOneDimensionalArray(inputs.out.length, () => 0);

    for (var index = 0; index < this.shape[2]; index++) {
      var indicator = index === correctLabel ? 1.0 : 0.0;
      var mul = this.exponents[index] - indicator;
      inputs.gradv[index] = mul;
    }

    return -Math.log(this.exponents[correctLabel]);
  }
}
