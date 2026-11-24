(() => {
  const byId = id => document.getElementById(id);
  const totalEpochs = 300;
  const pageSize = 8;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let model, inputCount, hiddenCount, nodes, edges, labels;
  let history, reports, curvePoints, signals;
  let epoch = 0, samplePage = 0;
  let trainingTimer = null;
  let flowFrame = null, flowTimer = null, pass = 0;
  let movingDots = [];

  function signed(value, digits = 3) {
    return (value < 0 ? '−' : '+') + Math.abs(value).toFixed(digits);
  }

  function buildNetwork() {
    const width = Math.max(inputCount, hiddenCount);
    const height = Math.min(472, 222 + width * 20);
    const subscript = n => String(n).replace(/\d/g, digit => '₀₁₂₃₄₅₆₇₈₉'[digit]);
    function layer(count, x, prefix) {
      const gap = Math.min(97, (height - 90) / Math.max(1, count - 1));
      return Array.from({ length: count }, (_, i) => ({
        x, y: (height + 18) / 2 + (i - (count - 1) / 2) * gap,
        label: prefix === 'p' ? 'p' : prefix + subscript(i + 1),
        radius: Math.min(28, gap * 0.38),
      }));
    }
    const hidden = layer(hiddenCount, 277, 'h');
    nodes = [...layer(inputCount, 58, 'x'), ...hidden, ...layer(1, 502, 'p')];
    edges = [];
    labels = [];
    function connect(from, to, fraction) {
      edges.push({ from, to, param: labels.length, fraction });
      labels.push(nodes[from].label + ' → ' + nodes[to].label);
    }
    hidden.forEach((node, i) => {
      for (let j = 0; j < inputCount; j++) connect(j, inputCount + i, j % 2 === 0 ? 0.43 : 0.70);
      labels.push(node.label + ' 偏置');
    });
    hidden.forEach((node, i) => connect(inputCount + i, nodes.length - 1, 0.44));
    labels.push('输出偏置');

    const dense = edges.length > 12;
    const svg = byId('network');
    svg.setAttribute('viewBox', '0 0 560 ' + height);
    svg.classList.toggle('dense-network', dense);
    const headings = ['输入层', '隐藏层', '输出概率'].map((name, i) =>
      `<text class="net-layer" x="${[58, 277, 502][i]}" y="20">${name}</text>`);
    const lines = edges.map(({ from, to }) =>
      `<line x1="${nodes[from].x}" y1="${nodes[from].y}" x2="${nodes[to].x}" y2="${nodes[to].y}"/>`);
    const weights = dense ? [] : edges.map(({ from, to, fraction }) => {
      const x = nodes[from].x + (nodes[to].x - nodes[from].x) * fraction;
      const y = nodes[from].y + (nodes[to].y - nodes[from].y) * fraction;
      return `<text class="edge-value" x="${x}" y="${y + 4}"></text>`;
    });
    const circles = nodes.map((node, i) => {
      const output = i === nodes.length - 1;
      const compact = node.radius < 23;
      const inside = compact ? '<text class="node-symbol" y="5"></text>' :
        `<text class="node-symbol" y="-2">${node.label}</text><text class="node-value" y="15"></text>`;
      const role = output ? 'output' : i < inputCount ? 'input-unit' : 'hidden-unit';
      return `<g class="net-node ${role}${compact ? ' compact' : ''}" transform="translate(${node.x} ${node.y})"><circle r="${node.radius}"/><circle class="activation-ring" r="${node.radius + 3}" pathLength="1" transform="rotate(-90)"/>${inside}</g>`;
    });
    svg.innerHTML = headings.join('') + lines.join('') + weights.join('') + '<g id="flow-dots" aria-hidden="true"></g>' + circles.join('');
    const lineElements = svg.querySelectorAll('line');
    const weightElements = svg.querySelectorAll('.edge-value');
    edges.forEach((edge, i) => {
      edge.line = lineElements[i];
      edge.weightLabel = weightElements[i];
    });
    svg.querySelectorAll('.net-node').forEach((element, i) => {
      nodes[i].readout = element.lastElementChild;
      nodes[i].ring = element.querySelector('.activation-ring');
    });
    byId('topology').textContent = [inputCount, hiddenCount, 1].join(' → ');
    byId('samples-title').textContent = `${model.samples.length} 组输入的预测`;
    byId('parameters-title').textContent = `${labels.length} 个参数怎样变化？`;
    byId('parameter-description').textContent = `网络包含 ${edges.length} 个连接权重和 ${hiddenCount + 1} 个偏置。单步更新可对照每项变化。`;
    byId('parameters').innerHTML = labels.map(label => `<tr><th scope="row">${label}</th><td></td><td></td><td></td><td></td></tr>`).join('');
  }

  function updateDotSizes() {
    for (const { dot, index } of movingDots) {
      const magnitude = Math.abs(signals[index]);
      const scale = edges.length > 30 ? 0.7 : 1;
      dot.setAttribute('r', scale * (4 + 5 * magnitude / (1 + magnitude)));
      dot.classList.toggle('zero', magnitude === 0);
    }
  }

  function drawNetwork() {
    const params = history[epoch];
    const { input } = model.samples[Number(byId('sample').value)];
    const { hidden, probability } = model.activations(params, input);
    const values = [...input, ...hidden, probability];
    signals = edges.map(({ from, param }) => values[from] * params[param]);
    const dense = edges.length > 12;
    for (const edge of edges) {
      const value = params[edge.param];
      const color = value < 0 ? 'var(--edge-negative, #a990ff)' : 'var(--edge-positive, #4fe6d1)';
      edge.line.setAttribute('stroke', color);
      edge.line.setAttribute('stroke-width', dense ? 0.8 + Math.abs(value) * 0.23 : 1 + Math.abs(value) * 0.6);
      edge.line.setAttribute('stroke-opacity', dense ? 0.58 : 0.87);
      if (edge.weightLabel) {
        edge.weightLabel.textContent = signed(value, 2);
        edge.weightLabel.setAttribute('fill', color);
      }
    }
    nodes.forEach((node, i) => {
      const compact = node.radius < 23;
      const value = i < inputCount ? input[i] : values[i].toFixed(3);
      node.readout.textContent = compact && i >= inputCount ? node.label : value;
      node.ring.setAttribute('stroke-dasharray', values[i] + ' 1');
    });
    byId('signal').textContent = `输入 ${input.join('')}，隐藏层激活值 ${hidden.map(v => v.toFixed(3)).join(' / ')}，输出概率 ${probability.toFixed(3)}。`;
    const nonzero = signals.filter(value => value !== 0).length;
    byId('flow-summary').textContent = `每组输入传递 ${edges.length} 次，当前样本中有 ${nonzero} 个非零信号。每轮 ${model.samples.length} 组样本共传递 ${model.samples.length * edges.length} 次。`;
    updateDotSizes();
  }

  function clearFlow() {
    cancelAnimationFrame(flowFrame);
    clearTimeout(flowTimer);
    flowFrame = flowTimer = null;
    movingDots = [];
    byId('flow-dots')?.replaceChildren();
    delete byId('network').dataset.receiving;
  }

  // The signal clock only advances the display. Training has its own clock.
  function flowStage(stage) {
    clearFlow();
    const split = inputCount * hiddenCount;
    const start = [0, split, edges.length][stage];
    if (stage === 0) pass++;
    const name = ['输入 → 隐藏', '隐藏 → 输出', '传播完成'][stage];
    byId('network').dataset.receiving = ['hidden', 'output', ''][stage];
    byId('flow-status').textContent = `第 ${pass} 次 · ${name} · ${start} / ${edges.length}`;
    const delay = stage === 2 ? 430 : 850;
    const advance = () => flowStage((stage + 1) % 3);
    if (stage === 2 || reducedMotion.matches) {
      flowTimer = setTimeout(advance, delay);
      return;
    }
    const end = stage === 0 ? split : edges.length;
    const group = byId('flow-dots');
    group.innerHTML = '<path class="signal-trails"/>' + '<circle class="signal-dot"/>'.repeat(end - start);
    const trails = group.firstElementChild;
    movingDots = Array.from(group.querySelectorAll('circle'), (dot, i) => ({ dot, index: start + i }));
    updateDotSizes();
    const began = performance.now();
    function advanceDots(now) {
      const t = Math.max(0, (now - began) / delay);
      if (t >= 1) {
        advance();
        return;
      }
      const tail = Math.max(0, t - 0.09);
      const segments = [];
      for (const { dot, index } of movingDots) {
        const { from, to } = edges[index];
        const start = nodes[from];
        const dx = nodes[to].x - start.x, dy = nodes[to].y - start.y;
        const x = start.x + dx * t, y = start.y + dy * t;
        dot.setAttribute('cx', x);
        dot.setAttribute('cy', y);
        if (signals[index] !== 0) segments.push(`M${start.x + dx * tail},${start.y + dy * tail}L${x},${y}`);
      }
      trails.setAttribute('d', segments.join(''));
      flowFrame = requestAnimationFrame(advanceDots);
    }
    advanceDots(began);
  }

  function restartFlow() {
    pass = 0;
    flowStage(0);
  }

  function renderSamples() {
    const probabilities = reports[epoch].probabilities;
    const pages = Math.ceil(model.samples.length / pageSize);
    const first = samplePage * pageSize;
    byId('sample-pager').hidden = pages <= 1;
    byId('prev-samples').disabled = samplePage === 0;
    byId('next-samples').disabled = samplePage + 1 >= pages;
    byId('sample-page').textContent = `${samplePage + 1} / ${pages}`;
    byId('predictions').innerHTML = model.samples.slice(first, first + pageSize).map(({ input, target }, i) => {
      const p = probabilities[first + i];
      const prediction = Number(p >= 0.5);
      return `<div class="sample-card"><div class="pair"><small>输入</small> <strong>${input.join('')}</strong></div>
        <div class="prob-bar" aria-hidden="true"><b style="width:${100 * p}%"></b></div>
        <div>输出概率 <b>${p.toFixed(4)}</b></div>
        <div class="sample-result">目标 ${target} · 预测 ${prediction} <strong>${prediction === target ? '✓ 正确' : '× 错误'}</strong></div></div>`;
    }).join('');
  }

  function render() {
    const params = history[epoch];
    const { loss, correct } = reports[epoch];
    byId('epoch').value = epoch;
    byId('epoch-number').textContent = epoch;
    byId('loss').textContent = loss.toFixed(4);
    byId('correct').textContent = correct + ' / ' + model.samples.length;
    byId('curve-past').setAttribute('d', 'M' + curvePoints.slice(0, epoch + 1).join(' L'));
    const [dotX, dotY] = curvePoints[epoch].split(',');
    byId('curve-dot').setAttribute('cx', dotX);
    byId('curve-dot').setAttribute('cy', dotY);
    byId('step').disabled = epoch === totalEpochs;
    byId('reset').disabled = epoch === 0;
    byId('play').textContent = trainingTimer !== null ? '暂停训练' : epoch === totalEpochs ? '重播训练' : '播放训练';
    drawNetwork();
    renderSamples();
    const gradient = model.gradients(params);
    const rows = byId('parameters').rows;
    params.forEach((value, i) => {
      const cells = rows[i].cells;
      if (epoch === 0) cells[1].textContent = signed(history[0][i]);
      cells[2].textContent = signed(value);
      cells[3].textContent = epoch === 0 ? '—' : signed(value - history[epoch - 1][i], 5);
      cells[4].textContent = signed(gradient[i], 5);
    });
  }

  function pauseTraining() {
    clearInterval(trainingTimer);
    trainingTimer = null;
  }

  function calculate() {
    pauseTraining();
    clearFlow();
    epoch = 0;
    inputCount = Number(byId('input-size').value);
    hiddenCount = Number(byId('hidden-size').value);
    model = MlpModel.create(inputCount, hiddenCount);
    const selector = byId('sample');
    const sample = Math.min(Number(selector.value || 5), model.samples.length - 1);
    selector.innerHTML = model.samples.map(({ input }, i) => `<option value="${i}">${input.join('')}</option>`).join('');
    selector.value = sample;
    samplePage = Math.floor(sample / pageSize);
    buildNetwork();
    history = model.train(Number(byId('rate').value), totalEpochs);
    reports = history.map(model.evaluate);
    const ceiling = Math.ceil(Math.max(...reports.map(report => report.loss)) * 10) / 10;
    byId('loss-max').textContent = ceiling.toFixed(1);
    byId('loss-mid').textContent = Number((ceiling / 2).toFixed(2));
    curvePoints = reports.map(({ loss }, i) =>
      (32 + 296 * i / totalEpochs).toFixed(2) + ',' + (78 - 66 * loss / ceiling).toFixed(2));
    byId('curve-full').setAttribute('d', 'M' + curvePoints.join(' L'));
    render();
    restartFlow();
  }

  for (const id of ['rate', 'hidden-size', 'input-size']) byId(id).addEventListener('change', calculate);
  byId('sample').addEventListener('change', () => {
    samplePage = Math.floor(Number(byId('sample').value) / pageSize);
    drawNetwork();
    renderSamples();
    restartFlow();
  });
  for (const [id, direction] of [['prev-samples', -1], ['next-samples', 1]]) {
    byId(id).addEventListener('click', () => {
      samplePage += direction;
      renderSamples();
    });
  }
  reducedMotion.addEventListener('change', restartFlow);
  byId('epoch').addEventListener('input', event => {
    pauseTraining();
    epoch = Number(event.target.value);
    render();
  });
  byId('reset').addEventListener('click', () => {
    pauseTraining();
    epoch = 0;
    render();
  });
  byId('step').addEventListener('click', () => {
    pauseTraining();
    epoch = Math.min(epoch + 1, totalEpochs);
    render();
  });
  byId('play').addEventListener('click', () => {
    if (trainingTimer !== null) pauseTraining();
    else {
      if (epoch === totalEpochs) epoch = 0;
      trainingTimer = setInterval(() => {
        epoch++;
        if (epoch === totalEpochs) pauseTraining();
        render();
      }, 80);
    }
    render();
  });
  calculate();
  for (const id of ['bench', 'samples-panel', 'parameters-panel']) byId(id).hidden = false;
})();
