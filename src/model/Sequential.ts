import Model from './Model';
import GradientHolder from '../GradientHolder';

export default class Sequential extends Model {
  forward(inputs: GradientHolder): void {
    this.out = this.layers[0].feedForward(inputs);

    for (let index = 1; index < this.layers.length; index++) {
      this.out = this.layers[index].feedForward(this.out);
    }
  }
}
