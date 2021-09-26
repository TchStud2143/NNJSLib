import GradientHolder from '../../GradientHolder';
import Layer from '../Layer';

export default class InputLayer extends Layer {
  constructor(w: number, h: number, d: number) {
    super(w, h, d);
  }

  feedForward(inputs: GradientHolder): Layer {
    this.out = inputs.out;
    return this;
  }
}
