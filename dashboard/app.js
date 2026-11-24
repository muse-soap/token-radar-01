// Launch Radar page: renders window.RADAR (written by main.ts beside this page) with no network.
// Hash routes: #/ the board, #/p/<slug> one protocol's detail, #/?cat=..&chain=.. a filtered board.
'use strict';

const R = window.RADAR;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const money = (n) => (n >= 1e9 ? `$${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `$${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `$${(n / 1e3).toFixed(0)}K` : `$${n}`);
const pct = (n) => (n === null ? '—' : `${n > 0 ? '+' : ''}${n.toFixed(1)}%`);
const ageText = (d) => (d === null ? 'unknown' : d >= 730 ? `${(d / 365).toFixed(1)}y` : `${d}d`);
const bySlug = new Map(R.rows.map((r) => [r.slug, r]));

// Log scale for the interval bars, spanning the whole board's ranges.
const LOG_MIN = Math.log(Math.min(...R.rows.map((r) => r.implied_low_usd)));
const LOG_MAX = Math.log(Math.max(...R.rows.map((r) => r.implied_high_usd)));
const pos = (v) => `${(((Math.log(Math.max(1, v)) - LOG_MIN) / (LOG_MAX - LOG_MIN)) * 100).toFixed(1)}%`;

function verdictClass(v) {
  return { 'Farm first': 'v-farm', 'Worth farming': 'v-worth', 'Small test position': 'v-small', 'Watch only': 'v-watch' }[v] ?? '';
}

// ---- header chips and the model strip ------------------------------------------------------------
$('chips').innerHTML = [
  ['Snapshot', R.as_of],
  ['Protocols scanned', R.protocols_total],
  ['Candidates ranked', R.board_size],
  ['Watchlist covered', `${R.watchlist_size} / ${R.watchlist_size}`],
].map(([k, v]) => `<span class="chip">${k}: <b>${esc(v)}</b></span>`).join('');

const M = R.model;
$('model-strip').innerHTML = `
  <h2>How a launch is priced</h2>
  <p>Fitted on the snapshot's own token-bearing protocols, not asserted: <b>log value = ${M.coefficients.intercept}
  + ${M.coefficients.log_tvl} × log TVL ${M.coefficients.log_age < 0 ? '−' : '+'} ${Math.abs(M.coefficients.log_age)} × log age
  + ${M.coefficients.growth} × 7d growth</b>, a ridge fit (λ ${M.ridge_lambda}) on ${M.training_size} protocols.
  Leave-one-out says the typical miss is <b>×${M.loo.typical_miss_factor}</b>, so every row shows its 80% interval.
  Category medians were tested and dropped: held out, they made the error worse.</p>
  <div class="model-stats">
    <div class="mstat"><div class="v">${M.training_size}</div><div class="k">training protocols</div></div>
    <div class="mstat"><div class="v">×${M.loo.typical_miss_factor}</div><div class="k">typical LOO miss</div></div>
    <div class="mstat"><div class="v">${M.loo.within_2x_pct}%</div><div class="k">held out within ×2</div></div>
    <div class="mstat"><div class="v">${M.loo.within_5x_pct}%</div><div class="k">held out within ×5</div></div>
  </div>`;

// ---- controls ------------------------------------------------------------------------------------
const cats = [...new Set(R.rows.map((r) => r.category ?? 'Uncategorised'))].sort();
$('cat').innerHTML = '<option value="">All categories</option>' + cats.map((c) => `<option>${esc(c)}</option>`).join('');
const chains = [...new Set(R.rows.flatMap((r) => r.chains))].sort();
$('chain').innerHTML = '<option value="">All chains</option>' + chains.map((c) => `<option>${esc(c)}</option>`).join('');

const state = { q: '', cat: '', chain: '', verdict: '', watch: false, sort: 'rank' };
function readControls() {
  state.q = $('q').value.trim().toLowerCase();
  state.cat = $('cat').value;
  state.chain = $('chain').value;
  state.verdict = $('verdict').value;
  state.watch = $('watch').checked;
  state.sort = $('sort').value;
}
function visibleRows() {
  let rows = R.rows.filter((r) =>
    (state.q === '' || r.name.toLowerCase().includes(state.q) || r.slug.includes(state.q)) &&
    (state.cat === '' || (r.category ?? 'Uncategorised') === state.cat) &&
    (state.chain === '' || r.chains.includes(state.chain)) &&
    (state.verdict === '' || r.verdict === state.verdict) &&
    (!state.watch || r.watchlist));
  const key = {
    rank: (r) => r.rank,
    tvl: (r) => -r.tvl_usd,
    growth: (r) => -(r.change_7d ?? -1e9),
    age: (r) => -(r.age_days ?? -1),
    peer: (r) => -r.peer_token_rate,
  }[state.sort];
  return rows.sort((a, b) => key(a) - key(b));
}
function renderBoard() {
  const rows = visibleRows();
  $('count').textContent = `${rows.length} of ${R.rows.length} protocols`;
  $('empty').hidden = rows.length > 0;
  $('rows').innerHTML = rows.map((r) => `
    <tr data-slug="${esc(r.slug)}">
      <td class="num">${r.rank}</td>
      <td><span class="pname">${esc(r.name)}</span>${r.watchlist ? '<span class="badge">watchlist</span>' : ''}<br>
          <span class="pslug">${esc(r.slug)}</span></td>
      <td><span class="cat">${esc(r.category ?? '—')}</span><span class="chains">${esc(r.chains.slice(0, 3).join(', '))}${r.chains.length > 3 ? ` +${r.chains.length - 3}` : ''}</span></td>
      <td class="num">${money(r.tvl_usd)}</td>
      <td class="num ${r.change_7d === null ? '' : r.change_7d >= 0 ? 'up' : 'down'}">${pct(r.change_7d)}</td>
      <td class="num">${ageText(r.age_days)}</td>
      <td><span class="value">${money(r.implied_value_usd)}</span>
          <span class="range-cap">${money(r.implied_low_usd)} – ${money(r.implied_high_usd)}</span>
          <div class="range"><div class="bar" style="left:${pos(r.implied_low_usd)};width:calc(${pos(r.implied_high_usd)} - ${pos(r.implied_low_usd)})"></div><div class="dot" style="left:${pos(r.implied_value_usd)}"></div></div></td>
      <td class="num">${Math.round(r.peer_token_rate * 100)}% <span class="pslug">of ${r.peer_count}</span></td>
      <td><span class="verdict ${verdictClass(r.verdict)}">${esc(r.verdict)}</span></td>
    </tr>`).join('');
  for (const tr of document.querySelectorAll('#rows tr')) {
    tr.addEventListener('click', () => { location.hash = `#/p/${tr.dataset.slug}`; });
  }
}
for (const id of ['q', 'cat', 'chain', 'verdict', 'watch', 'sort']) {
  $(id).addEventListener('input', () => { readControls(); renderBoard(); });
}

// ---- detail view ---------------------------------------------------------------------------------
function renderDetail(r) {
  const growthTerm = M.coefficients.growth * (Math.min(120, Math.max(-60, r.change_7d ?? 0)) / 100);
  const ageTerm = M.coefficients.log_age * Math.log((r.age_days ?? 365) + 30);
  const tvlTerm = M.coefficients.log_tvl * Math.log(Math.max(1, r.tvl_usd));
  $('detail-view').innerHTML = `
    <button class="back" id="back">← Back to the board</button>
    <div class="detail-head">
      <h2>${esc(r.name)}</h2><span class="slug">${esc(r.slug)}</span>
      ${r.watchlist ? '<span class="badge">on the watchlist</span>' : '<span class="badge" style="background:#22314a;color:#a9c3e8">not on the house watchlist</span>'}
      <span class="verdict ${verdictClass(r.verdict)}" style="font-size:15px">${esc(r.verdict)}</span>
    </div>
    <div class="detail-grid">
      <div class="card">
        <h3>The record</h3>
        <div class="stat-grid">
          <div class="stat"><div class="v">${money(r.tvl_usd)}</div><div class="k">TVL</div></div>
          <div class="stat"><div class="v ${r.change_7d === null ? '' : r.change_7d >= 0 ? 'up' : 'down'}">${pct(r.change_7d)}</div><div class="k">TVL, 7 days</div></div>
          <div class="stat"><div class="v">${ageText(r.age_days)}</div><div class="k">tracked, no token</div></div>
          <div class="stat"><div class="v">${money(r.implied_value_usd)}</div><div class="k">implied launch value</div></div>
          <div class="stat"><div class="v">${money(r.implied_low_usd)}–${money(r.implied_high_usd)}</div><div class="k">80% interval</div></div>
          <div class="stat"><div class="v">${Math.round(r.peer_token_rate * 100)}%</div><div class="k">peers tokenized</div></div>
        </div>
        <h3 style="margin-top:18px">How this value is derived</h3>
        <table class="derive">
          <tr><td>Fitted constant</td><td>${M.coefficients.intercept}</td></tr>
          <tr><td>TVL term: ${M.coefficients.log_tvl} × log(${money(r.tvl_usd)})</td><td>+${tvlTerm.toFixed(2)}</td></tr>
          <tr><td>Age term: ${M.coefficients.log_age} × log(age + 30d)</td><td>${ageTerm.toFixed(2)}</td></tr>
          <tr><td>Growth term: ${M.coefficients.growth} × clipped 7d change</td><td>${growthTerm >= 0 ? '+' : ''}${growthTerm.toFixed(2)}</td></tr>
          <tr class="total"><td>Implied launch value</td><td>${money(r.implied_value_usd)}</td></tr>
        </table>
        <p style="color:var(--muted);font-size:13px;margin:12px 0 0">
          Category ${esc(r.category ?? '—')} · chains ${esc(r.chains.join(', ') || '—')} · first tracked ${esc(r.listed_at ?? 'unknown')}.
          Of its ${r.peer_count} category peers within a ×4 TVL band, ${r.peer_tokenized} already record a token —
          a precedent count from this snapshot, shown for the farm decision; it does not move the rank.
          ${esc(r.name)} records no token in this snapshot (no market cap, no CoinGecko id, no symbol); that is
          DefiLlama's record, not proof none exists anywhere.</p>
      </div>
      <div class="card">
        <h3>Farm plan — $${r.farm.cash_cost_usd} cash, about ${r.farm.hours} hours, $${r.farm.capital_usd} of capital</h3>
        <ol class="plan">
          ${r.farm.steps.map((s, i) => `<li><span class="n">${i + 1}</span><div class="body">${esc(s.text)}<div class="meta"><b>Cost:</b> ${esc(s.cost)} · <b>Time:</b> ${esc(s.time)}</div></div></li>`).join('')}
        </ol>
        <p class="plan-total">Total cash cost <b>$${r.farm.cash_cost_usd}</b> over <b>${r.farm.weeks} weeks</b>;
        the $${r.farm.capital_usd} of capital stays in the position and is not spent. No step is sponsored and no
        step asks for a payment to claim.</p>
      </div>
    </div>
    <div class="verdict-box">
      <span class="v">${esc(r.verdict)}</span>
      <p>${esc(r.verdict_reason)}</p>
    </div>`;
  $('back').addEventListener('click', () => { location.hash = '#/'; });
}

// ---- live strip and method footer ----------------------------------------------------------------
$('live').innerHTML = R.live.map((p) => `
  <div class="live-item"><div class="n">${esc(p.name)}</div>
  <div class="m">market cap ${money(p.mcap_usd)} · TVL ${money(p.tvl_usd)} · ${esc(p.category ?? '—')}</div></div>`).join('');

$('method').innerHTML = `
  <h2>Method, stated plainly</h2>
  <h3>The board</h3>
  <p>Every protocol in the snapshot (${R.protocols_total} at $5M TVL and up) was screened the way the house's watchlist
  rule reads: it must record no token (no market cap, no CoinGecko id, no symbol) and must not be a wrapped, staked or
  bridged product of a token that trades, a versioned deployment, or an exchange's own product. ${R.candidates_total}
  protocols pass; all ${R.watchlist_size} watchlist protocols are among them. Of the rest, ${R.off_list_above_floor} sit above
  the floor but off the list — the house's rule also reads a parent-protocol record this snapshot does not carry — and
  ${R.below_floor} are early finds below the $50M floor. Ranks are the fitted values alone, highest first.</p>
  <h3>The value model (${esc(M.version)})</h3>
  <p>A ridge regression of log market cap on log TVL, log age and clipped seven-day growth, fitted at run time on the
  ${M.training_size} token-bearing protocols whose market-cap-to-TVL ratio sits inside [${M.ratio_band[0]}, ${M.ratio_band[1]}]
  (${M.dropped_outliers} records outside the band are dropped by that rule). Age and growth are penalised (λ ${M.ridge_lambda})
  because leave-one-out validation supports only a small weight for them; category medians were fitted, tested and dropped
  for making held-out error worse. Leave-one-out over the training set: median error ×${M.loo.typical_miss_factor},
  ${M.loo.within_2x_pct}% within ×2 and ${M.loo.within_5x_pct}% within ×5 — wide, and the intervals on every row say so.
  An implied value is this entry's own claim about a launch that has not happened, never a prediction that it will.</p>
  <h3>Farm plans and verdicts</h3>
  <p>Each plan is generated from the protocol's own record — its category sets the core action, its chains set where,
  its TVL sets the capital tier, its age sets the number of weekly actions — and every step names its cost and its time.
  The verdict weighs the peer token rate (the share of same-category, similar-size protocols that already record a token)
  against size and the week's direction. Data: ${esc(R.source)}; snapshot as of ${esc(R.as_of)}. “Records no token” is
  DefiLlama's record, not proof: a protocol may have a token its record does not carry.</p>`;

// ---- routing --------------------------------------------------------------------------------------
function route() {
  const hash = location.hash || '#/';
  const detail = hash.match(/^#\/p\/(.+)$/);
  if (detail && bySlug.has(detail[1])) {
    $('board-view').hidden = true;
    $('controls').hidden = true;
    $('detail-view').hidden = false;
    renderDetail(bySlug.get(detail[1]));
    window.scrollTo(0, 0);
    return;
  }
  $('detail-view').hidden = true;
  $('board-view').hidden = false;
  $('controls').hidden = false;
  const q = new URLSearchParams(hash.split('?')[1] ?? '');
  if (q.has('cat')) $('cat').value = q.get('cat');
  if (q.has('chain')) $('chain').value = q.get('chain');
  if (q.has('q')) $('q').value = q.get('q');
  if (q.has('verdict')) $('verdict').value = q.get('verdict');
  if (q.has('sort')) $('sort').value = q.get('sort');
  $('watch').checked = q.get('watch') === '1';
  readControls();
  renderBoard();
}
window.addEventListener('hashchange', route);
route();
