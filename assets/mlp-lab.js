(() => {
  const byId = id => document.getElementById(id);
  const totalEpochs = 300;
  const inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
  const targets = [0, 1, 1, 0];
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let nodes, edges, labels, signals;
  let flowFrame = null;
  let epoch = 0;
  let history;
  let reports;
  let curvePoints;
  let timer = null;

  function signed(value, digits = 3) {
    return (value < 0 ? '−' : '+') + Math.abs(value).toFixed(digits);
  }

  function buildNetwork(size) {
    const height = 250 + Math.max(0, size - 2) * 88;
    const mid = (height + 18) / 2;
    const spread = 56 + Math.max(0, size - 2) * 30;
    const hidden = Array.from({ length: size }, (_, i) => ({
      x: 277, y: size === 1 ? mid : 77 + i * (height - 136) / (size - 1),
      label: 'h' + '₁₂₃₄'[i],
    }));
    nodes = [{ x: 58, y: mid - spread, label: 'x₁' }, { x: 58, y: mid + spread, label: 'x₂' },
      ...hidden, { x: 502, y: mid, label: 'p' }];
    edges = [];
    labels = [];
    hidden.forEach((node, i) => {
      for (let input = 0; input < 2; input++) {
        edges.push({ from: input, to: i + 2, param: 3 * i + input, position: input === 0 ? 0.43 : 0.7 });
        labels.push(nodes[input].label + ' → ' + node.label);
      }
      labels.push(node.label + ' 偏置');
    });
    hidden.forEach((node, i) => {
      edges.push({ from: i + 2, to: nodes.length - 1, param: 3 * size + i, position: 0.44 });
      labels.push(node.label + ' → 输出');
    });
    labels.push('输出偏置');
    byId('network').setAttribute('viewBox', '0 0 560 ' + height);
    byId('parameters-title').textContent = labels.length + ' 个参数怎样变化？';
    byId('parameter-description').textContent = `当前有 ${size} 个隐藏单元。每次更新调整 ${edges.length} 个连接权重和 ${size + 1} 个偏置。点击「单步 +1」可对照当前值和上轮的差。`;
  }

  function drawNetwork(params) {
    const index = Number(byId('sample').value);
    const [x1, x2] = inputs[index];
    const { hidden, probability } = MlpModel.activations(params, x1, x2);
    const activations = [x1, x2, ...hidden, probability];
    const values = activations.map((value, i) => i < 2 ? value : value.toFixed(3));
    signals = edges.map(({ from, param }) => activations[from] * params[param]);
    const layers = ['输入层', '隐藏层', '输出概率'].map((name, i) =>
      `<text class="net-layer" x="${[58, 277, 502][i]}" y="22" text-anchor="middle">${name}</text>`).join('');
    const lines = [];
    const weights = [];
    for (const { from, to, param, position } of edges) {
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
    const circles = nodes.map((node, i) => `<g class="net-node${i === nodes.length - 1 ? ' output' : ''}" transform="translate(${node.x} ${node.y})">
      <circle r="28"/><text class="node-symbol" text-anchor="middle" y="-2">${node.label}</text>
      <text class="node-value" text-anchor="middle" y="15">${values[i]}</text></g>`).join('');
    byId('network').innerHTML = layers + lines.join('') + weights.join('') + '<g id="flow-dots" aria-hidden="true"></g>' + circles;
    byId('signal').textContent = `输入 (${x1}, ${x2}) 时，隐藏层给出 ${values.slice(2, -1).join('、')}，输出概率为 ${values.at(-1)}。`;
    const nonzero = signals.filter(value => value !== 0).length;
    byId('flow-summary').textContent = `当前样本前向传递 ${edges.length} 次，其中 ${nonzero} 次的加权值非零。每轮四组样本共完成 ${inputs.length * edges.length} 次前向传递。`;
  }

  function flowStatus(stage, arrived) {
    byId('flow-status').textContent = `${stage} · 已送达 ${arrived} / ${edges.length}`;
  }

  function stopFlow() {
    window.cancelAnimationFrame(flowFrame);
    flowFrame = null;
    byId('flow-dots')?.replaceChildren();
    byId('show-flow').textContent = '播放信号';
    flowStatus('准备就绪', 0);
  }

  function playFlow() {
    stopFlow();
    const finish = () => {
      stopFlow();
      flowStatus('前向传播完成', edges.length);
    };
    if (reducedMotion.matches) {
      finish();
      return;
    }
    byId('show-flow').textContent = '停止演示';
    const firstStageCount = edges.filter(edge => edge.from < 2).length;
    const started = performance.now();
    let lastStage = -1;
    function tick(now) {
      const elapsed = Math.max(0, (now - started) / 850);
      const stage = Math.floor(elapsed);
      if (stage >= 2 || reducedMotion.matches) {
        finish();
        return;
      }
      const start = stage === 0 ? 0 : firstStageCount;
      const end = stage === 0 ? firstStageCount : edges.length;
      const group = byId('flow-dots');
      if (stage !== lastStage) {
        flowStatus(stage === 0 ? '输入层 → 隐藏层' : '隐藏层 → 输出层', start);
        group.innerHTML = signals.slice(start, end).map(value => {
          const magnitude = Math.abs(value);
          const radius = 5 + 6 * magnitude / (1 + magnitude);
          return `<circle class="signal-dot${value === 0 ? ' zero' : ''}" r="${radius}"/>`;
        }).join('');
        lastStage = stage;
      }
      const t = elapsed - stage;
      for (let i = start; i < end; i++) {
        const from = nodes[edges[i].from], to = nodes[edges[i].to];
        const dot = group.children[i - start];
        dot.setAttribute('cx', from.x + (to.x - from.x) * t);
        dot.setAttribute('cy', from.y + (to.y - from.y) * t);
      }
      flowFrame = window.requestAnimationFrame(tick);
    }
    tick(started);
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
    const size = Number(byId('hidden-size').value);
    buildNetwork(size);
    stopFlow();
    history = MlpModel.train(Number(byId('rate').value), totalEpochs, size);
    reports = history.map(MlpModel.evaluate);
    const ceiling = Math.ceil(Math.max(...reports.map(report => report.loss)) * 10) / 10;
    byId('loss-max').textContent = ceiling.toFixed(1);
    byId('loss-mid').textContent = Number((ceiling / 2).toFixed(2));
    curvePoints = reports.map(({ loss }, i) =>
      (32 + 296 * i / totalEpochs).toFixed(2) + ',' + (78 - 66 * loss / ceiling).toFixed(2));
    byId('curve-full').setAttribute('d', 'M' + curvePoints.join(' L'));
    render();
  }

  byId('rate').addEventListener('change', calculate);
  byId('hidden-size').addEventListener('change', () => {
    calculate();
    playFlow();
  });
  byId('sample').addEventListener('change', () => {
    drawNetwork(history[epoch]);
    if (timer === null) playFlow();
  });
  byId('show-flow').addEventListener('click', () => {
    if (flowFrame !== null) {
      stopFlow();
      return;
    }
    pause();
    render();
    playFlow();
  });
  byId('epoch').addEventListener('input', event => {
    pause();
    stopFlow();
    epoch = Number(event.target.value);
    render();
  });
  byId('reset').addEventListener('click', () => {
    pause();
    stopFlow();
    epoch = 0;
    render();
  });
  byId('step').addEventListener('click', () => {
    pause();
    epoch = Math.min(epoch + 1, totalEpochs);
    render();
    playFlow();
  });
  byId('play').addEventListener('click', () => {
    stopFlow();
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
