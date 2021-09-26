import GradientHolder from '../../GradientHolder';
import Layer from '../Layer';

export default abstract class ActivationLayer extends Layer {
  constructor(w, h, d) {
    super(w, h, d);
  }

  abstract feedForward(inputs: GradientHolder): Layer;
}
