/** Measure a built or hosted documentation site with Chrome's DevTools protocol. */
import {spawn} from 'node:child_process';
import {mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {parseArgs} from 'node:util';

const {values} = parseArgs({options: {
  'base-url': {type: 'string'},
  trials: {type: 'string', default: '3'},
  chrome: {type: 'string', default: 'google-chrome'},
  output: {type: 'string'},
  page: {type: 'string', multiple: true},
  check: {type: 'boolean', default: false},
  'expect-revision': {type: 'string'},
}});
if (!values['base-url']) throw new Error('--base-url is required');
const trials = Number(values.trials);
if (!Number.isInteger(trials) || trials < 1) throw new Error('--trials must be a positive integer');
const base = new URL(values['base-url']);
if (!['http:', 'https:'].includes(base.protocol)) throw new Error('Use an HTTP(S) documentation URL');

const searchCases = [
  ['deadtime', 'reference/detector_grouping.html'],
  ['grouping', 'reference/detector_grouping.html'],
  ['alpha calibration', 'reference/data_reduction/alpha_calibration.html'],
  ['asymmetry.core.transform.deadtime', 'api/transforms.html'],
];
const apiMathCases = {
  'api/fitting/models.html': [
    {anchor: 'asymmetry.core.fitting.models.abragam'},
    {anchor: 'asymmetry.core.fitting.models.risch_kehr'},
  ],
  'api/fitting/sc.html': [
    {anchor: 'asymmetry.core.fitting.sc.bcs.delta_generalized', precedingInlineMath: 2},
    {
      anchor: 'asymmetry.core.fitting.sc.models.sc_two_gap_ss',
      followingParagraph: 'MgB₂-style alpha-model decomposition',
    },
  ],
};
const pages = values.page || [
  'index.html', 'getting_started/quickstart.html', 'explanation/reduction_pipeline.html',
  'reference/fourier_analysis.html', 'reference/parameter_trending.html', 'explanation/comparison.html',
  ...searchCases.map(([query]) => `search.html?q=${encodeURIComponent(query)}`),
];
const rows = [];
let failed = false;
let environment;
for (const page of pages) {
  const url = new URL(page, base).href;
  const apiMath = Object.entries(apiMathCases).find(([path]) =>
    new URL(url).pathname === new URL(path, base).pathname)?.[1] ?? [];
  const profile = await mkdtemp(join(tmpdir(), 'asymmetry-docs-browser-'));
  const chrome = spawn(values.chrome, [
    '--headless', '--no-sandbox', '--disable-gpu', '--remote-debugging-port=0',
    `--user-data-dir=${profile}`, 'about:blank',
  ], {stdio: 'ignore'});
  let launchError;
  chrome.on('error', error => { launchError = error; });
  const exited = new Promise(resolve => { chrome.once('exit', resolve); chrome.once('error', resolve); });
  let socket;
  const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
  try {
    let port;
    for (let attempt = 0; attempt < 100; attempt++) {
      if (launchError) throw launchError;
      try {
        port = Number((await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]);
        break;
      } catch (error) {
        if (error.code !== 'ENOENT') throw error;
        await pause(50);
      }
    }
    if (!port) throw new Error('Chrome did not expose its DevTools port');
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`, {signal: AbortSignal.timeout(5000)})).json();
    socket = new WebSocket(targets.find(target => target.type === 'page').webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('DevTools connection deadline')), 5000);
      socket.addEventListener('open', () => {clearTimeout(timer); resolve();}, {once: true});
      for (const event of ['error', 'close']) socket.addEventListener(event, () => {
        clearTimeout(timer); reject(new Error(`DevTools connection ${event}`));
      }, {once: true});
    });
    let serial = 0;
    const pending = new Map();
    let requests = [];
    let httpStatus;
    socket.addEventListener('message', event => {
      const message = JSON.parse(event.data);
      if (!message.id) {
        if (message.method === 'Network.requestWillBeSent') requests.push(message.params.request.url);
        if (message.method === 'Network.responseReceived' && message.params.type === 'Document')
          httpStatus = message.params.response.status;
        return;
      }
      const {resolve, reject, timer} = pending.get(message.id);
      pending.delete(message.id);
      clearTimeout(timer);
      if (message.error) reject(new Error(JSON.stringify(message.error)));
      else resolve(message.result);
    });
    socket.addEventListener('close', () => {
      for (const {reject, timer} of pending.values()) {clearTimeout(timer); reject(new Error('DevTools closed'));}
      pending.clear();
    });
    const call = (method, params = {}) => new Promise((resolve, reject) => {
      const id = ++serial;
      const timer = setTimeout(() => reject(new Error(`${method} exceeded 30 seconds`)), 30000);
      pending.set(id, {resolve, reject, timer});
      socket.send(JSON.stringify({id, method, params}));
    });
    const evaluate = async expression => {
      const result = await call('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
      if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
      return result.result.value;
    };
    const browser = await call('Browser.getVersion');
    environment = {
      browser: browser.product, userAgent: browser.userAgent, date: new Date().toISOString(),
      base: base.href, trials, network: 'unthrottled', cpu: 'unthrottled',
      browserIsolation: 'fresh profile per page; repeated cold/warm pairs share that profile',
      cache: 'cold clears HTTP cache; warm immediately repeats the URL in the same browser',
    };
    console.log(JSON.stringify({environment}));
    await call('Page.enable');
    await call('Network.enable');
    await call('Page.addScriptToEvaluateOnNewDocument', {source: `
      window.docsMetrics = {shifts: [], lcp: null, firstResult: null, searchFinished: null};
      new PerformanceObserver(list => {for (const entry of list.getEntries()) {
        if (!entry.hadRecentInput) docsMetrics.shifts.push({value: entry.value, time: entry.startTime});
      }}).observe({type: 'layout-shift', buffered: true});
      new PerformanceObserver(list => {docsMetrics.lcp = list.getEntries().at(-1).startTime;})
        .observe({type: 'largest-contentful-paint', buffered: true});
      if (location.pathname.endsWith('/search.html')) new MutationObserver(() => {
        if (docsMetrics.firstResult === null && document.querySelector('#search-results ul.search>li'))
          docsMetrics.firstResult = performance.now();
        if (docsMetrics.searchFinished === null && document.querySelector('.search-summary')?.textContent.includes('Search finished'))
          docsMetrics.searchFinished = performance.now();
      }).observe(document, {subtree: true, childList: true, characterData: true});
    `});
    for (const viewport of [{width: 1440, height: 1000}, {width: 390, height: 844}]) {
      await call('Emulation.setDeviceMetricsOverride', {...viewport, deviceScaleFactor: 1, mobile: false});
      for (let trial = 0; trial < trials; trial++) for (const cache of ['cold', 'warm']) {
        if (cache === 'cold') await call('Network.clearBrowserCache');
        await call('Network.setCacheDisabled', {cacheDisabled: false});
        await call('Page.stopLoading');
        await call('Page.navigate', {url: 'about:blank'});
        requests = [];
        httpStatus = null;
        const navigation = await call('Page.navigate', {url});
        if (navigation.errorText) throw new Error(navigation.errorText);
        const deadline = Date.now() + 15000;
        let domReady = false;
        while (Date.now() < deadline) {
          domReady = await evaluate(`location.href === ${JSON.stringify(url)} && performance.getEntriesByType('navigation')[0]?.domContentLoadedEventEnd > 0`);
          if (domReady) break;
          await pause(50);
        }
        if (!domReady) throw new Error(`DOM readiness deadline: ${url}; no browser timing inferred`);
        const row = await evaluate(`(async () => {
          const bounded = promise => Promise.race([promise.then(() => true), new Promise(resolve => setTimeout(() => resolve(false), 8000))]);
          const hasMath = !!document.querySelector('.math');
          const mathReady = !hasMath || (!!window.MathJax?.startup && await bounded(MathJax.startup.promise));
          const fontsReady = await bounded(document.fonts.ready);
          const searchPage = location.pathname.endsWith('/search.html');
          if (searchPage) {
            const deadline = performance.now() + 10000;
            while (docsMetrics.searchFinished === null && performance.now() < deadline)
              await new Promise(resolve => setTimeout(resolve, 20));
          }
          const contentReadyMs = performance.now();
          await new Promise(resolve => setTimeout(resolve, 250));
          await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
          const nav = performance.getEntriesByType('navigation')[0];
          const article = document.querySelector('[itemprop="articleBody"]');
          const apiMathFailures = ${JSON.stringify(apiMath)}.flatMap(({anchor, precedingInlineMath, followingParagraph}) => {
            const entry = document.getElementById(anchor)?.nextElementSibling;
            if (!entry?.matches('dd')) return [{anchor, problems: ['Missing API description']}];
            const problems = [];
            const display = entry.querySelector(':scope > div.math');
            if (!display?.querySelector(':scope > mjx-container[display="true"] > mjx-math'))
              problems.push('Missing typeset display equation');
            if (precedingInlineMath) {
              const before = display?.previousElementSibling;
              const rendered = before?.querySelectorAll('span.math > mjx-container:not([display="true"]) > mjx-math');
              if (!before?.matches('p') || rendered.length !== precedingInlineMath)
                problems.push('Missing typeset inline equations before display equation');
            }
            if (followingParagraph) {
              const after = display?.nextElementSibling;
              if (!after?.matches('p') || !after.textContent.includes(followingParagraph))
                problems.push('Missing prose paragraph after display equation');
            }
            if (entry.querySelector('mjx-merror')) problems.push('MathJax error');
            return problems.length ? [{anchor, problems}] : [];
          });
          let cls = 0, session = 0, start = 0, previous = -Infinity;
          for (const shift of docsMetrics.shifts) {
            if (shift.time - previous > 1000 || shift.time - start > 5000) {session = 0; start = shift.time;}
            session += shift.value; previous = shift.time; cls = Math.max(cls, session);
          }
          return {
            revision: document.querySelector('meta[name="asymmetry-source-revision"]')?.content || null,
            networkResponseMs: nav.responseEnd, transferBytes: nav.transferSize, encodedBytes: nav.encodedBodySize,
            domContentLoadedMs: nav.domContentLoadedEventEnd || null,
            resourceTransferBytes: performance.getEntriesByType('resource').reduce((sum, entry) => sum + entry.transferSize, 0), loadMs: nav.loadEventEnd || null,
            fcpMs: performance.getEntriesByName('first-contentful-paint')[0]?.startTime || null,
            lcpMs: docsMetrics.lcp, contentReadyMs, observationEndMs: performance.now(), cls,
            fontsReady, mathReady, overflow: document.documentElement.scrollWidth > ${viewport.width},
            literalMath: article?.textContent.includes(':math:') || false,
            mathErrors: document.querySelectorAll('mjx-merror').length,
            ...(${apiMath.length} ? {apiMathFailures} : {}),
            search: searchPage ? {
              firstResultMs: docsMetrics.firstResult, finishedMs: docsMetrics.searchFinished,
              results: document.querySelectorAll('#search-results ul.search>li').length,
              top: [...document.querySelectorAll('#search-results ul.search>li>a')].slice(0, 5)
                .map(link => ({title: link.textContent, href: new URL(link.href).href})),
            } : null,
          };
        })()`);
        const record = {page, viewport, trial, cache, requestCount: requests.length, httpStatus,
          summaryRequests: requests.filter(url => new URL(url).pathname.endsWith('.html') && url !== new URL(page, base).href).length, ...row};
        rows.push(record);
        console.log(JSON.stringify(record));
        if (values.check) {
          const expected = searchCases.find(([query]) => new URL(url).searchParams.get('q') === query)?.[1];
          if (httpStatus === null || httpStatus < 200 || httpStatus >= 400 || row.overflow || row.literalMath ||
              row.mathErrors || row.apiMathFailures?.length || !row.fontsReady || !row.mathReady ||
              (row.search && (!row.search.finishedMs || (expected &&
                new URL(row.search.top[0]?.href || 'about:blank').pathname !== new URL(expected, base).pathname))) ||
              (values['expect-revision'] && row.revision !== values['expect-revision'])) failed = true;
        }
      }
    }
   } finally {
    try {
      if (values.output) await writeFile(values.output, JSON.stringify({environment, measurements: rows,
        complete: rows.length === pages.length * 2 * trials * 2}, null, 2) + '\n');
    } finally {
      if (socket?.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({id: 0, method: 'Browser.close'}));
        await Promise.race([exited, pause(2000)]);
      }
      socket?.close();
      chrome.kill('SIGTERM');
      await Promise.race([exited, pause(2000)]);
      if (chrome.exitCode === null && chrome.signalCode === null) {
        chrome.kill('SIGKILL');
        await Promise.race([exited, pause(2000)]);
      }
      await rm(profile, {recursive: true, force: true, maxRetries: 5, retryDelay: 100});
    }
  }
}
if (failed) process.exitCode = 1;
