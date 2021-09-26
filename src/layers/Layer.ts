import Tensor from '../Tensor';
import GradientHolder from '../GradientHolder';
import BackPropagationNode from '../backpropagation/BackPropagationNode';
import PropagationOperation from '../backpropagation/PropagationOperation';
import Utils from '../utils/Utils';

export default abstract class Layer
  extends GradientHolder
  implements BackPropagationNode
{
  items: GradientHolder;
  W: Tensor;
  b: Tensor;
  feedCache: number;

  constructor(w: number, h: number, d: number) {
    super();

    this.shape = [w, h, d];
    this.out = Utils.buildOneDimensionalArray(w * h * d, () => 0);
    this.W = new Tensor(w, h, d).fillGaussianRandom(0, 0.88);
    this.b = new Tensor(1, 1, d).fillGaussianRandom(0, 0.88);
  }

  forwardPass(): GradientHolder {
    return this.items;
  }

  propagateBackwards(target?: number): void {
    this.items.grad(this.gradv);

    if (this.items instanceof PropagationOperation) {
      this.items.propagateBackwards(target);
    }
  }

  abstract feedForward(inputs: GradientHolder): Layer;
}
