import Layer from './Layer';

export default abstract class OptimizableLayer extends Layer {
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
