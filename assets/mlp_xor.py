"""本站的四位奇偶校验实验。运行 python mlp_xor.py 可核对网页默认设置的训练结果。"""

from itertools import product
from math import exp, log1p, sqrt


INPUTS, HIDDEN = 4, 8
SAMPLES = [(bits, sum(bits) % 2) for bits in product((0, 1), repeat=INPUTS)]
# 每个隐藏单元依次存放输入权重和偏置，最后是全部输出权重和输出偏置。
STRIDE = INPUTS + 1
OUTPUT_START = STRIDE * HIDDEN


def initial_parameters():
    state = 42
    params = []
    for _ in range(OUTPUT_START + HIDDEN + 1):
        state = (1664525 * state + 1013904223) % (2 ** 32)
        params.append(3 * (state / (2 ** 32) - 0.5))
    return params


def sigmoid(z):
    return 1 / (1 + exp(-z))


def forward(params, bits):
    hidden = []
    logit = 0.0
    for i in range(HIDDEN):
        start = STRIDE * i
        weighted_input = sum(params[start + j] * bit for j, bit in enumerate(bits))
        value = sigmoid(weighted_input + params[start + INPUTS])
        hidden.append(value)
        logit += params[OUTPUT_START + i] * value
    return hidden, logit + params[-1]


def loss_and_gradient(params):
    loss = 0.0
    gradient = [0.0] * len(params)
    for bits, target in SAMPLES:
        hidden, logit = forward(params, bits)
        loss += max(logit, 0) - target * logit + log1p(exp(-abs(logit)))
        output_delta = sigmoid(logit) - target
        for i, value in enumerate(hidden):
            start = STRIDE * i
            delta = output_delta * params[OUTPUT_START + i] * value * (1 - value)
            for j, bit in enumerate(bits):
                gradient[start + j] += delta * bit
            gradient[start + INPUTS] += delta
            gradient[OUTPUT_START + i] += output_delta * value
        gradient[-1] += output_delta
    return loss / len(SAMPLES), [g / len(SAMPLES) for g in gradient]


def train(params, epochs=300, learning_rate=0.03):
    params = params.copy()
    first_moment = [0.0] * len(params)
    second_moment = [0.0] * len(params)
    beta1, beta2 = 0.9, 0.999
    for epoch in range(1, epochs + 1):
        _, gradient = loss_and_gradient(params)
        for i, g in enumerate(gradient):
            first_moment[i] = beta1 * first_moment[i] + (1 - beta1) * g
            second_moment[i] = beta2 * second_moment[i] + (1 - beta2) * g * g
            m_hat = first_moment[i] / (1 - beta1 ** epoch)
            v_hat = second_moment[i] / (1 - beta2 ** epoch)
            params[i] -= learning_rate * m_hat / (sqrt(v_hat) + 1e-8)
    return params


def report(epoch, params):
    loss, _ = loss_and_gradient(params)
    correct = 0
    print(f"epoch={epoch}  BCE={loss:.8f}")
    for bits, target in SAMPLES:
        _, logit = forward(params, bits)
        probability = sigmoid(logit)
        predicted = int(probability >= 0.5)
        correct += int(predicted == target)
        print(f"  {''.join(map(str, bits))}  target={target}  p(1)={probability:.8f}  class={predicted}")
    print(f"correct={correct}/{len(SAMPLES)}")


if __name__ == "__main__":
    initial = initial_parameters()
    report(0, initial)
    report(300, train(initial))
