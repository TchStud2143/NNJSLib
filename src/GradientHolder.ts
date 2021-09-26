export default abstract class GradientHolder {
  out: number[];
  shape: number[];
  gradv: number[];
  grad_required: boolean;

  get(row: number, col: number, depth: number) {
    return this.out[(this.shape[1] * row + col) * this.shape[2] + depth];
  }

  set(row: number, col: number, depth: number, value: number) {
    this.out[(this.shape[1] * row + col) * this.shape[2] + depth] = value;
  }

  getGrad(row: number, col: number, depth: number) {
    return this.gradv[(this.shape[1] * row + col) * this.shape[2] + depth];
  }

  setGrad(row: number, col: number, depth: number, value: number) {
    this.gradv[(this.shape[1] * row + col) * this.shape[2] + depth] = value;
  }

  setOut(out: number[]) {
    for (let i = 0, n = out.length; i < n; i++) {
      this.out[i] = out[i];
    }
  }

  grad(gradv: number[]): void {
    this.gradv = gradv;
  }
}
