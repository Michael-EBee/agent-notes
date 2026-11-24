(() => {
  const byId = id => document.getElementById(id);
  const totalEpochs = 300;
  const inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const targets = [0, 1, 1, 0];
  const labels = [
    'x₁ → h₁', 'x₂ → h₁', 'h₁ 偏置', 'x₁ → h₂', 'x₂ → h₂',
    'h₂ 偏置', 'h₁ → 输出', 'h₂ → 输出', '输出偏置',
  ];
  const nodes = [
    { x: 58, y: 77, label: 'x₁' }, { x: 58, y: 190, label: 'x₂' },
    { x: 277, y: 77, label: 'h₁' }, { x: 277, y: 190, label: 'h₂' },
    { x: 502, y: 133.5, label: 'p' },
  ];
  // from node, to node, parameter index, label position along the connection.
  const edges = [[0, 2, 0, 0.45], [1, 2, 1, 0.29], [0, 3, 3, 0.29],
    [1, 3, 4, 0.45], [2, 4, 6, 0.5], [3, 4, 7, 0.5]];
  let epoch = 0;
  let history;
  let reports;
  let curvePoints;
  let timer = null;

  function signed(value, digits = 3) {
    return (value < 0 ? '−' : '+') + Math.abs(value).toFixed(digits);
  }

  function drawNetwork(params) {
    const index = Number(byId('sample').value);
    const [x1, x2] = inputs[index];
    const { h1, h2, probability } = MlpModel.activations(params, x1, x2);
    const values = [x1, x2, h1.toFixed(3), h2.toFixed(3), probability.toFixed(3)];
    const layers = ['输入层', '隐藏层', '输出概率'].map((name, i) =>
      `<text class="net-layer" x="${nodes[i * 2].x}" y="22" text-anchor="middle">${name}</text>`).join('');
    const lines = [];
    const weights = [];
    for (const [from, to, param, position] of edges) {
      const start = nodes[from];
      const end = nodes[to];
      const value = params[param];
      const color = value < 0 ? '#f0a373' : '#a9d6a5';
      const width = 1 + Math.abs(value) * 0.75;
      lines.push(`<line x1="${start.x}" y1="${start.y}" x2="${end.x}" y2="${end.y}" stroke="${color}" stroke-width="${width}" stroke-opacity=".88"/>`);
      const x = start.x + (end.x - start.x) * position;
      const y = start.y + (end.y - start.y) * position;
      weights.push(`<text class="edge-value" x="${x}" y="${y + 4}" text-anchor="middle" fill="${color}">${signed(value, 2)}</text>`);
    }
    const circles = nodes.map((node, i) => `<g class="net-node${i === 4 ? ' output' : ''}" transform="translate(${node.x} ${node.y})">
      <circle r="28"/><text class="node-symbol" text-anchor="middle" y="-2">${node.label}</text>
      <text class="node-value" text-anchor="middle" y="15">${values[i]}</text></g>`).join('');
    byId('network').innerHTML = layers + lines.join('') + weights.join('') + circles;
    byId('signal').textContent = `输入 (${x1}, ${x2}) 时，隐藏层给出 ${values[2]} 和 ${values[3]}，输出概率为 ${values[4]}。`;
  }

  function render() {
    const params = history[epoch];
    const { loss, correct, probabilities } = reports[epoch];
    byId('epoch').value = epoch;
    byId('epoch-number').textContent = epoch;
    byId('loss').textContent = loss.toFixed(4);
    byId('correct').textContent = correct + ' / 4';
    byId('curve-past').setAttribute('d', 'M' + curvePoints.slice(0, epoch + 1).join(' L'));
    const [dotX, dotY] = curvePoints[epoch].split(',');
    byId('curve-dot').setAttribute('cx', dotX);
    byId('curve-dot').setAttribute('cy', dotY);
    byId('step').disabled = epoch === totalEpochs;
    byId('reset').disabled = epoch === 0;
    byId('play').textContent = timer !== null ? '暂停训练' : epoch === totalEpochs ? '重播训练' : '播放训练';
    drawNetwork(params);
    byId('predictions').innerHTML = probabilities.map((p, i) => {
      const prediction = Number(p >= 0.5);
      const [x1, x2] = inputs[i];
      return `<div class="sample-card"><div class="pair"><small>输入</small> <strong>${x1} ⊕ ${x2}</strong></div>
        <div class="prob-bar" aria-hidden="true"><b style="width:${100 * p}%"></b></div>
        <div>输出概率 <b>${p.toFixed(4)}</b></div>
        <div class="sample-result">目标 ${targets[i]} · 预测 ${prediction} <strong>${prediction === targets[i] ? '✓ 正确' : '× 错误'}</strong></div></div>`;
    }).join('');
    const gradient = MlpModel.gradients(params);
    byId('parameters').innerHTML = params.map((value, i) => `<tr>
      <th scope="row">${labels[i]}</th><td>${signed(history[0][i])}</td><td>${signed(value)}</td>
      <td>${epoch === 0 ? '—' : signed(value - history[epoch - 1][i], 5)}</td><td>${signed(gradient[i], 5)}</td>
      </tr>`).join('');
  }

  function pause() {
    window.clearInterval(timer);
    timer = null;
  }

  function calculate() {
    pause();
    epoch = 0;
    history = MlpModel.train(Number(byId('rate').value), totalEpochs);
    reports = history.map(MlpModel.evaluate);
    curvePoints = reports.map(({ loss }, i) =>
      (32 + 296 * i / totalEpochs).toFixed(2) + ',' + (78 - 66 * loss / 0.6).toFixed(2));
    byId('curve-full').setAttribute('d', 'M' + curvePoints.join(' L'));
    render();
  }

  byId('rate').addEventListener('change', calculate);
  byId('sample').addEventListener('change', () => drawNetwork(history[epoch]));
  byId('epoch').addEventListener('input', event => {
    pause();
    epoch = Number(event.target.value);
    render();
  });
  byId('reset').addEventListener('click', () => {
    pause();
    epoch = 0;
    render();
  });
  byId('step').addEventListener('click', () => {
    pause();
    epoch = Math.min(epoch + 1, totalEpochs);
    render();
  });
  byId('play').addEventListener('click', () => {
    if (timer !== null) {
      pause();
    } else {
      if (epoch === totalEpochs) epoch = 0;
      timer = window.setInterval(() => {
        epoch++;
        if (epoch >= totalEpochs) pause();
        render();
      }, 80);
    }
    render();
  });
  calculate();
  for (const id of ['bench', 'samples-panel', 'parameters-panel']) byId(id).hidden = false;
})();
