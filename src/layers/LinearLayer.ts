import Assertion from '../utils/Assertion';
import GradientHolder from '../GradientHolder';
import AddPropagation from '../backpropagation/operations/AddPropagation';
import MatMulPropagation from '../backpropagation/operations/MatMulPropagation';
import Layer from './Layer';
import OptimizableLayer from './OptimizableLayer';

export default class LinearLayer extends OptimizableLayer {
  feedForward(inputs: GradientHolder): Layer {
    Assertion.assert(
      inputs.out.length === this.shape[0],
      'Feed Forward was given an invalid input count.'
    );

    const matmulOperation = new MatMulPropagation(inputs, this.W);
    this.items = new AddPropagation(matmulOperation, this.b);
    this.shape = matmulOperation.shape;
    this.out = this.items.out;
    this.grad(this.items.gradv);

    return this;
  }

  /*
   * TODO
   *  Implement different optimizers
   *  The only optimizer supported is
   *  Stochastic gradient descent.
   */
  optimize(learningRate: number): void {
    for (let index = 0; index < this.W.out.length; index++) {
      this.W.out[index] -= learningRate * this.W.gradv[index];
    }

    for (let index = 0; index < this.b.out.length; index++) {
      this.b.out[index] -= learningRate * this.b.gradv[index];
    }
  }
}
