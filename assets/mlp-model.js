// Each hidden unit owns two input weights and one bias. Output weights
// follow the hidden triples in unit order, with the output bias last.
const MlpModel = (() => {
  const samples = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
  ];

  function initial(hiddenCount = 2) {
    if (!Number.isInteger(hiddenCount) || hiddenCount < 1 || hiddenCount > 4) {
      throw new RangeError('Hidden unit count must be an integer from 1 to 4.');
    }
    const units = [[2, -2, -1], [-2, 2, -1], [1.2, -0.5, 0.2], [-0.65, 0.95, -0.35]];
    const outputWeights = [3, 3, 0.4, -0.6];
    return [...units.slice(0, hiddenCount).flat(), ...outputWeights.slice(0, hiddenCount), -2.5];
  }

  function sigmoid(logit) {
    return 1 / (1 + Math.exp(-logit));
  }

  function activations(params, x1, x2) {
    const hiddenCount = (params.length - 1) / 4;
    const outputStart = 3 * hiddenCount;
    const hidden = [];
    let logit = 0;
    for (let i = 0; i < hiddenCount; i++) {
      const start = 3 * i;
      const value = sigmoid(params[start] * x1 + params[start + 1] * x2 + params[start + 2]);
      hidden.push(value);
      logit += params[outputStart + i] * value;
    }
    logit += params[params.length - 1];
    const probability = sigmoid(logit);
    return { hidden, h1: hidden[0], h2: hidden[1], logit, probability };
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
    const sums = Array(params.length).fill(0);
    for (const [x1, x2, label] of samples) {
      const { hidden, probability } = activations(params, x1, x2);
      const outputDelta = probability - label;
      const outputStart = 3 * hidden.length;
      for (let i = 0; i < hidden.length; i++) {
        const value = hidden[i];
        const delta = outputDelta * params[outputStart + i] * value * (1 - value);
        sums[3 * i] += delta * x1;
        sums[3 * i + 1] += delta * x2;
        sums[3 * i + 2] += delta;
        sums[outputStart + i] += outputDelta * value;
      }
      sums[sums.length - 1] += outputDelta;
    }
    return sums.map(sum => sum / samples.length);
  }

  function train(rate = 0.03, epochs = 300, hiddenCount = 2) {
    const history = [initial(hiddenCount)];
    const firstMoment = Array(history[0].length).fill(0);
    const secondMoment = Array(history[0].length).fill(0);
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
