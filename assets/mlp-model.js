// Hidden parameters are stored unit by unit (all input weights, then bias).
// The output weights and the output bias occupy the final H+1 positions.
const MlpModel = (() => {
  function initialParameters(inputCount, hiddenCount) {
    // A fixed seed gives each topology reproducible weights, independent of labels.
    let state = 42;
    return Array.from({ length: (inputCount + 2) * hiddenCount + 1 }, () => {
      state = (Math.imul(1664525, state) + 1013904223) >>> 0;
      return 3 * (state / 4294967296 - 0.5);
    });
  }

  function sigmoid(logit) {
    return 1 / (1 + Math.exp(-logit));
  }

  function create(inputCount, hiddenCount) {
    if (!Number.isInteger(inputCount) || inputCount < 2 || inputCount > 6) {
      throw new RangeError('Input count must be an integer from 2 to 6.');
    }
    if (!Number.isInteger(hiddenCount) || hiddenCount < 1 || hiddenCount > 12) {
      throw new RangeError('Hidden unit count must be an integer from 1 to 12.');
    }

    const stride = inputCount + 1;
    const outputStart = stride * hiddenCount;
    const samples = Array.from({ length: 2 ** inputCount }, (_, index) => {
      const input = Array.from({ length: inputCount }, (_, j) => (index >> (inputCount - 1 - j)) & 1);
      return { input, target: input.reduce((parity, bit) => parity ^ bit, 0) };
    });

    function initial() {
      return initialParameters(inputCount, hiddenCount);
    }

    function activations(params, input) {
      const hidden = [];
      let logit = 0;
      for (let i = 0; i < hiddenCount; i++) {
        const start = i * stride;
        let sum = 0;
        for (let j = 0; j < inputCount; j++) {
          sum += params[start + j] * input[j];
        }
        const value = sigmoid(sum + params[start + inputCount]);
        hidden.push(value);
        logit += params[outputStart + i] * value;
      }
      logit += params[params.length - 1];
      return { hidden, logit, probability: sigmoid(logit) };
    }

    function evaluate(params) {
      let loss = 0;
      let correct = 0;
      const probabilities = [];
      for (const { input, target } of samples) {
        const { logit, probability } = activations(params, input);
        loss += Math.max(logit, 0) - logit * target + Math.log1p(Math.exp(-Math.abs(logit)));
        correct += Number((probability >= 0.5 ? 1 : 0) === target);
        probabilities.push(probability);
      }
      return { loss: loss / samples.length, correct, probabilities };
    }

    function gradients(params) {
      const sums = Array(params.length).fill(0);
      for (const { input, target } of samples) {
        const { hidden, probability } = activations(params, input);
        const outputDelta = probability - target;
        for (let i = 0; i < hiddenCount; i++) {
          const value = hidden[i];
          const delta = outputDelta * params[outputStart + i] * value * (1 - value);
          const start = i * stride;
          for (let j = 0; j < inputCount; j++) {
            sums[start + j] += delta * input[j];
          }
          sums[start + inputCount] += delta;
          sums[outputStart + i] += outputDelta * value;
        }
        sums[sums.length - 1] += outputDelta;
      }
      return sums.map(sum => sum / samples.length);
    }

    function train(rate = 0.03, epochs = 300) {
      const history = [initial()];
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

    return { samples, initial, activations, evaluate, gradients, train };
  }

  return { create };
})();
