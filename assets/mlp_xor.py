"""第二篇第 15 节：用标准库训练 XOR。运行 python mlp_xor.py 即可复现。"""

from math import exp, log1p, sqrt


SAMPLES = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
# 顺序为 h1 的两个输入权重和偏置、h2 的两个输入权重和偏置、
# 两个隐藏单元到输出的权重，以及输出偏置。
INITIAL = [2.0, -2.0, -1.0, -2.0, 2.0, -1.0, 3.0, 3.0, -2.5]


def sigmoid(z):
    return 1 / (1 + exp(-z)) if z >= 0 else exp(z) / (1 + exp(z))


def forward(params, x1, x2):
    w11, w21, b1, w12, w22, b2, v1, v2, bias = params
    h1 = sigmoid(w11 * x1 + w21 * x2 + b1)
    h2 = sigmoid(w12 * x1 + w22 * x2 + b2)
    logit = v1 * h1 + v2 * h2 + bias
    return h1, h2, logit


def loss_and_gradient(params):
    total_loss = 0.0
    gradient = [0.0] * len(params)
    for x1, x2, target in SAMPLES:
        h1, h2, logit = forward(params, x1, x2)
        # 由 logit 直接计算二元交叉熵，避免对接近零的概率取对数。
        total_loss += max(logit, 0) - target * logit + log1p(exp(-abs(logit)))
        d_out = sigmoid(logit) - target
        d_h1 = d_out * params[6] * h1 * (1 - h1)
        d_h2 = d_out * params[7] * h2 * (1 - h2)
        sample_gradient = (d_h1 * x1, d_h1 * x2, d_h1,
                           d_h2 * x1, d_h2 * x2, d_h2,
                           d_out * h1, d_out * h2, d_out)
        for i, derivative in enumerate(sample_gradient):
            gradient[i] += derivative
    return total_loss / len(SAMPLES), [g / len(SAMPLES) for g in gradient]


def report(epoch, params):
    loss, _ = loss_and_gradient(params)
    print(f"epoch={epoch}  BCE={loss:.8f}")
    for x1, x2, target in SAMPLES:
        _, _, logit = forward(params, x1, x2)
        prob = sigmoid(logit)
        print(f"  ({x1}, {x2})  target={target}  p(1)={prob:.8f}  class={int(prob >= 0.5)}")


def main():
    params = INITIAL.copy()
    first_moment = [0.0] * len(params)
    second_moment = [0.0] * len(params)
    beta1, beta2 = 0.9, 0.999
    learning_rate = 0.03
    report(0, params)
    for epoch in range(1, 301):
        _, gradient = loss_and_gradient(params)
        for i, g in enumerate(gradient):
            first_moment[i] = beta1 * first_moment[i] + (1 - beta1) * g
            second_moment[i] = beta2 * second_moment[i] + (1 - beta2) * g * g
            m_hat = first_moment[i] / (1 - beta1 ** epoch)
            v_hat = second_moment[i] / (1 - beta2 ** epoch)
            params[i] -= learning_rate * m_hat / (sqrt(v_hat) + 1e-8)
    report(300, params)


if __name__ == "__main__":
    main()
