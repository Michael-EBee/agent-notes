// Each hidden unit owns two input weights and one bias. The output unit
// owns the last two weights and the last bias, in the same order.
const MlpModel = (() => {
  const samples = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
  ];

  function initial() {
    return [2, -2, -1, -2, 2, -1, 3, 3, -2.5];
  }

  function sigmoid(logit) {
    return 1 / (1 + Math.exp(-logit));
  }

  function activations(params, x1, x2) {
    const h1 = sigmoid(params[0] * x1 + params[1] * x2 + params[2]);
    const h2 = sigmoid(params[3] * x1 + params[4] * x2 + params[5]);
    const logit = params[6] * h1 + params[7] * h2 + params[8];
    const probability = sigmoid(logit);
    return { h1, h2, logit, probability };
  }

  function forward(params, x1, x2) {
    return activations(params, x1, x2).probability;
  }

  function evaluate(params) {
    let loss = 0;
    let correct = 0;
    const probabilities = [];
    for (const [x1, x2, label] of samples) {
      const { logit, probability } = activations(params, x1, x2);
      // Binary cross-entropy in logit space stays finite for saturated outputs.
      loss += Math.max(logit, 0) - logit * label + Math.log1p(Math.exp(-Math.abs(logit)));
      correct += Number((probability >= 0.5 ? 1 : 0) === label);
      probabilities.push(probability);
    }
    return { loss: loss / samples.length, correct, probabilities };
  }

  function gradients(params) {
    const sums = Array(9).fill(0);
    for (const [x1, x2, label] of samples) {
      const { h1, h2, probability } = activations(params, x1, x2);
      const outputDelta = probability - label;
      const h1Delta = outputDelta * params[6] * h1 * (1 - h1);
      const h2Delta = outputDelta * params[7] * h2 * (1 - h2);
      sums[0] += h1Delta * x1;
      sums[1] += h1Delta * x2;
      sums[2] += h1Delta;
      sums[3] += h2Delta * x1;
      sums[4] += h2Delta * x2;
      sums[5] += h2Delta;
      sums[6] += outputDelta * h1;
      sums[7] += outputDelta * h2;
      sums[8] += outputDelta;
    }
    return sums.map(sum => sum / samples.length);
  }

  function train(rate = 0.03, epochs = 300) {
    const history = [initial()];
    const firstMoment = Array(9).fill(0);
    const secondMoment = Array(9).fill(0);
    const beta1 = 0.9;
    const beta2 = 0.999;
    const epsilon = 1e-8;
    for (let epoch = 1; epoch <= epochs; epoch++) {
      const params = history[epoch - 1];
      const gradient = gradients(params);
      const firstCorrection = 1 - beta1 ** epoch;
      const secondCorrection = 1 - beta2 ** epoch;
      const updated = params.map((value, index) => {
        const g = gradient[index];
        firstMoment[index] = beta1 * firstMoment[index] + (1 - beta1) * g;
        secondMoment[index] = beta2 * secondMoment[index] + (1 - beta2) * g * g;
        const correctedFirst = firstMoment[index] / firstCorrection;
        const correctedSecond = secondMoment[index] / secondCorrection;
        return value - rate * correctedFirst / (Math.sqrt(correctedSecond) + epsilon);
      });
      history.push(updated);
    }
    return history;
  }

  return { initial, activations, forward, evaluate, gradients, train };
})();
