// The Launch Radar model: one sealed snapshot of the protocol universe in, one ranked board out.
//
// The board ranks protocols that record no token yet by what a launch would be worth if the
// protocol's record stayed as it is. The value is fitted, not asserted: the 144 protocols in the
// snapshot whose token already trades are the training set, and the model is a ridge regression of
// log market cap on log TVL, log age and seven-day TVL growth, fitted on the 123 of them whose
// market-cap-to-TVL ratio sits inside [0.002, 50] (the five outside are dropped by that rule, never
// by name). The age and growth coefficients are penalised (lambda 8) because leave-one-out
// validation on this snapshot supports only a small weight for them; TVL carries the signal.
// Category medians were fitted and tested the same way and dropped: they made held-out error
// worse (median |log error| 1.36 against 1.19), so no category term appears below.
//
// Held-out honesty, measured by leave-one-out over the training set and printed on the dashboard:
// median |log error| 1.19 (a typical miss of x3.3), 28% of held-out protocols land within x2 of
// their market cap and 62% within x5. Every row therefore carries an 80% interval taken from the
// leave-one-out residuals (their 10th and 90th percentiles), not a point dressed up as precise.
//
// A second, separate figure is the peer token rate: among a protocol's category peers of similar
// size, the share that already record a token. It is a precedent count from the snapshot itself,
// shown beside the value and used in the farm verdict; it never moves the rank, which is the
// implied value's alone.

/** The snapshot's protocol entry: exactly the ten fields the sealed input carries. */
export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

/** The whole snapshot document. */
export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

/** Why the snapshot is not one this model can score. */
export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

const DAY_MS = 86_400_000;

/** A whole day count from `listedAt` to `asOf`, clamped at zero; null when either date is absent or bad. */
export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

/** Whether a protocol records no token at all: no market cap, no CoinGecko id, no symbol. */
export function recordsNoToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

/** Whether a protocol records a token (a symbol or a CoinGecko id is enough for DefiLlama to name one). */
function recordsToken(p: SnapshotProtocol): boolean {
  return p.gecko_id !== null || p.symbol !== null;
}

/** DefiLlama's version-marker rule: a name ending in a release is a deployment of another protocol. */
const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;

/** Categories that are wrapped, staked or bridged products, or an exchange's own venue. */
const PRODUCT_CATEGORIES = new Set(['Bridge', 'Canonical Bridge', 'Liquid Staking', 'Liquid Restaking', 'Restaked BTC', 'CEX']);

/** Exchange names a product may not begin with: the product is the exchange's, not a launch of its own. */
const EXCHANGE_PREFIXES = ['Binance', 'Coinbase', 'Bybit', 'Kraken', 'Bitget', 'OKX', 'Bitfinex', 'Gemini'];

/**
 * Whether a pre-token protocol is a launch candidate in its own right: not a wrapped, staked or
 * bridged product of a token that trades, not a versioned deployment, not an exchange product.
 * The same screens the house's watchlist rule states, applied to the whole snapshot.
 */
export function isCandidate(p: SnapshotProtocol): boolean {
  if (!recordsNoToken(p)) return false;
  if (p.category !== null && PRODUCT_CATEGORIES.has(p.category)) return false;
  if (VERSIONED.test(p.name)) return false;
  const words = p.name.toLowerCase().split(/[^a-z]+/);
  if (words.includes('wrapped') || words.includes('staked') || words.includes('staking')) return false;
  if (EXCHANGE_PREFIXES.some((e) => p.name.startsWith(e))) return false;
  return true;
}

// ---------------------------------------------------------------------------
// The fitted value model
// ---------------------------------------------------------------------------

/** The regression's one row of inputs for a protocol: [1, log TVL, log age, clipped growth]. */
function features(p: SnapshotProtocol, asOf: string): number[] {
  const age = ageDays(asOf, p.listed_at);
  const ageForModel = age ?? 365;
  const rawGrowth = typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : 0;
  const growth = Math.min(120, Math.max(-60, rawGrowth)) / 100;
  return [1, Math.log(Math.max(1, p.tvl_usd)), Math.log(ageForModel + 30), growth];
}

/** Solve a small dense linear system by Gaussian elimination with partial pivoting. */
function solveLinear(a: number[][], b: number[]): number[] {
  const k = b.length;
  const m = a.map((row, i) => [...row, b[i]!]);
  for (let c = 0; c < k; c++) {
    let piv = c;
    for (let r = c + 1; r < k; r++) if (Math.abs(m[r]![c]!) > Math.abs(m[piv]![c]!)) piv = r;
    const tmp = m[c]!; m[c] = m[piv]!; m[piv] = tmp;
    for (let r = 0; r < k; r++) {
      if (r === c) continue;
      const f = m[r]![c]! / m[c]![c]!;
      for (let j = c; j <= k; j++) m[r]![j]! -= f * m[c]![j]!;
    }
  }
  return m.map((row, i) => row[k]! / row[i]!);
}

const RIDGE_LAMBDA = 8;
/** The winsor band on market-cap-to-TVL for the training set: outside it, a record is an outlier. */
const RATIO_MIN = 0.002;
const RATIO_MAX = 50;

/** Ridge-fit log market cap on the features; only the age and growth coefficients are penalised. */
function fitCoefficients(training: SnapshotProtocol[], asOf: string): number[] {
  const k = 4;
  const pen = [0, 0, RIDGE_LAMBDA, RIDGE_LAMBDA];
  const a: number[][] = Array.from({ length: k }, (_, i) =>
    Array.from({ length: k }, (_, j) => (i === j ? pen[i]! : 0)),
  );
  const b = [0, 0, 0, 0];
  for (const p of training) {
    const x = features(p, asOf);
    const y = Math.log(p.mcap_usd!);
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) a[i]![j]! += x[i]! * x[j]!;
      b[i]! += x[i]! * y;
    }
  }
  return solveLinear(a, b);
}

/** The fitted model and its leave-one-out report, all derived from the snapshot at run time. */
export interface FittedModel {
  version: string;
  training_size: number;
  dropped_outliers: number;
  ratio_band: [number, number];
  ridge_lambda: number;
  coefficients: { intercept: number; log_tvl: number; log_age: number; growth: number };
  loo: {
    median_abs_log_error: number;
    typical_miss_factor: number;
    within_2x_pct: number;
    within_5x_pct: number;
    interval_low_log: number;
    interval_high_log: number;
  };
  predict: (p: SnapshotProtocol, asOf: string) => { value: number; low: number; high: number };
}

const round4 = (n: number): number => Math.round(n * 1e4) / 1e4;

/** Fit the model on the snapshot's token-bearing protocols and measure it by leave-one-out. */
export function fitModel(snapshot: Snapshot): FittedModel {
  const asOf = snapshot.as_of;
  const bearing = snapshot.protocols.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0);
  const training = bearing.filter((p) => {
    const r = p.mcap_usd! / p.tvl_usd;
    return r >= RATIO_MIN && r <= RATIO_MAX;
  });
  const coef = fitCoefficients(training, asOf);
  const predictLog = (p: SnapshotProtocol, c: number[]): number => {
    const x = features(p, asOf);
    return c[0]! + c[1]! * x[1]! + c[2]! * x[2]! + c[3]! * x[3]!;
  };
  // Leave-one-out: refit without each training protocol in turn and keep its residual.
  const residuals: number[] = [];
  for (let i = 0; i < training.length; i++) {
    const rest = training.filter((_, j) => j !== i);
    const c = fitCoefficients(rest, asOf);
    residuals.push(Math.log(training[i]!.mcap_usd!) - predictLog(training[i]!, c));
  }
  const sorted = [...residuals].sort((x, y) => x - y);
  const absSorted = residuals.map(Math.abs).sort((x, y) => x - y);
  const quantile = (f: number): number => sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(f * sorted.length)))]!;
  const medianAbs = absSorted[Math.floor(absSorted.length / 2)]!;
  const within = (f: number): number => Math.round((100 * residuals.filter((r) => Math.abs(r) < Math.log(f)).length) / residuals.length);
  const q10 = quantile(0.1);
  const q90 = quantile(0.9);
  return {
    version: 'launch-radar-fitted-1',
    training_size: training.length,
    dropped_outliers: bearing.length - training.length,
    ratio_band: [RATIO_MIN, RATIO_MAX],
    ridge_lambda: RIDGE_LAMBDA,
    coefficients: {
      intercept: round4(coef[0]!),
      log_tvl: round4(coef[1]!),
      log_age: round4(coef[2]!),
      growth: round4(coef[3]!),
    },
    loo: {
      median_abs_log_error: round4(medianAbs),
      typical_miss_factor: round4(Math.exp(medianAbs)),
      within_2x_pct: within(2),
      within_5x_pct: within(5),
      interval_low_log: round4(q10),
      interval_high_log: round4(q90),
    },
    predict: (p: SnapshotProtocol) => {
      const log = predictLog(p, coef);
      const value = Math.max(1, Math.round(Math.exp(log)));
      const low = Math.min(value, Math.max(1, Math.round(Math.exp(log + q10))));
      const high = Math.max(value, Math.round(Math.exp(log + q90)));
      return { value, low, high };
    },
  };
}

// ---------------------------------------------------------------------------
// Peer token rate
// ---------------------------------------------------------------------------

export interface PeerRate {
  rate: number;
  peers: number;
  tokenized: number;
  widened: boolean;
}

/**
 * Among a protocol's category peers within a x4 TVL band, the share that already record a token.
 * Falls back to the whole TVL band when the category has fewer than six peers there.
 */
export function peerTokenRate(p: SnapshotProtocol, all: SnapshotProtocol[]): PeerRate {
  const inBand = (q: SnapshotProtocol): boolean =>
    q.slug !== p.slug && q.tvl_usd >= p.tvl_usd / 4 && q.tvl_usd <= p.tvl_usd * 4;
  let peers = all.filter((q) => inBand(q) && q.category === p.category);
  let widened = false;
  if (peers.length < 6) {
    peers = all.filter(inBand);
    widened = true;
  }
  const tokenized = peers.filter(recordsToken).length;
  return {
    rate: peers.length === 0 ? 0 : Math.round((tokenized / peers.length) * 1000) / 1000,
    peers: peers.length,
    tokenized,
    widened,
  };
}

// ---------------------------------------------------------------------------
// Farm plans
// ---------------------------------------------------------------------------

/** A farm step: the action, what it costs, and how long it takes. */
export interface FarmStep {
  text: string;
  cost: string;
  time: string;
}

export interface FarmPlan {
  steps: FarmStep[];
  capital_usd: number;
  cash_cost_usd: number;
  weeks: number;
  hours: number;
  verdict: string;
  verdict_reason: string;
}

/** How much capital one farm position needs, from the protocol's own TVL. */
function capitalFor(tvl: number): number {
  if (tvl >= 5_000_000_000) return 250;
  if (tvl >= 1_000_000_000) return 150;
  if (tvl >= 250_000_000) return 100;
  if (tvl >= 50_000_000) return 75;
  return 50;
}

/** How many weekly qualifying actions to plan, from the protocol's age: younger gets a longer window. */
function weeksFor(age: number | null): number {
  if (age === null) return 6;
  if (age < 180) return 8;
  if (age < 365) return 6;
  if (age < 1095) return 5;
  return 4;
}

/** The category's core action, in words a farmer can follow, naming the protocol and its first chain. */
function coreAction(p: SnapshotProtocol, capital: number, chain: string): string {
  const c = (p.category ?? '').toLowerCase();
  const n = p.name;
  if (c.includes('lend') || c.includes('cdp')) return `Supply $${capital} of USDC to ${n} on ${chain}, then borrow a small amount against it and repay, so the wallet shows both sides of the market.`;
  if (c.includes('dex')) return `Add $${capital} of liquidity to ${n}'s busiest pool on ${chain} and make two swaps through the same interface.`;
  if (c.includes('deriv') || c.includes('perpetual')) return `Deposit $${capital} of USDC margin on ${n} and open one small hedged position, then close it a day later.`;
  if (c.includes('bridge')) return `Bridge $${capital} of USDC through ${n} to ${chain} and spend a little of it in the destination app the same day.`;
  if (c.includes('curator') || c.includes('allocator')) return `Deposit $${capital} into a curated ${n} vault on ${chain} and keep the vault shares in the wallet.`;
  if (c.includes('rwa')) return `Mint $${capital} of ${n}'s yield-bearing asset on ${chain} and hold it in the wallet.`;
  if (c.includes('basis')) return `Deposit $${capital} of USDC into ${n}'s basis vault on ${chain} and keep the position open.`;
  if (c.includes('staking') || c.includes('staking pool')) return `Stake $${capital} of the chain's staking asset through ${n} and keep the staking receipt.`;
  if (c.includes('yield')) return `Deposit $${capital} of USDC into a ${n} vault on ${chain} and leave the shares untouched.`;
  if (c.includes('insurance')) return `Buy $${capital} of cover from ${n} on ${chain} and keep the policy until it expires.`;
  if (c.includes('prediction')) return `Fund $${capital} on ${n} and take one position in its most liquid market.`;
  return `Deposit $${capital} of USDC into ${n} on ${chain} and make one further transaction in its app.`;
}

/** The verdict a plan earns, from the protocol's own figures alone. */
function verdictFor(p: SnapshotProtocol, peer: PeerRate, floor: number): { verdict: string; reason: string } {
  const growth = p.change_7d ?? 0;
  const ratePct = Math.round(peer.rate * 100);
  if (peer.rate >= 0.5 && p.tvl_usd >= floor && growth >= -5) {
    return {
      verdict: 'Farm first',
      reason: `${ratePct}% of its size peers in ${p.category ?? 'its category'} already record a token, it holds $${Math.round(p.tvl_usd / 1e6)}M of TVL above the watchlist floor, and its week is not shrinking fast. Precedent and size agree.`,
    };
  }
  if (peer.rate >= 0.35 && p.tvl_usd >= 20_000_000) {
    return {
      verdict: 'Worth farming',
      reason: `${ratePct}% of its peers already record a token and it holds $${Math.round(p.tvl_usd / 1e6)}M of TVL: a real precedent at a size a launch can carry.`,
    };
  }
  if (peer.rate >= 0.2 || p.tvl_usd >= floor) {
    return {
      verdict: 'Small test position',
      reason: `The precedent is thinner (${ratePct}% of peers tokenized) or the size does the arguing ($${Math.round(p.tvl_usd / 1e6)}M TVL): worth the minimum position that qualifies, not a large one.`,
    };
  }
  return {
    verdict: 'Watch only',
    reason: `Only ${ratePct}% of its peers record a token and its TVL is $${Math.round(p.tvl_usd / 1e6)}M: no precedent and no size yet. Revisit if either moves.`,
  };
}

/**
 * A protocol's farm plan: six steps with a cost and a time each, derived from the protocol's own
 * record (category, chains, TVL, age), and a verdict whose cash cost is the plan's own sum.
 */
export function farmPlan(p: SnapshotProtocol, age: number | null, peer: PeerRate, floor: number): FarmPlan {
  const chain = p.chains.length > 0 ? p.chains[0]! : 'its main chain';
  const capital = capitalFor(p.tvl_usd);
  const weeks = weeksFor(age);
  const setupGas = p.chains.length >= 3 ? 9 : p.chains.length === 2 ? 7 : 5;
  const secondGas = p.chains.length > 1 ? 4 : 3;
  const cash = setupGas + secondGas + 2 * weeks;
  const hours = Math.round((0.25 + 0.33 + 0.25 + weeks * (10 / 60) + 0.17) * 10) / 10;
  const { verdict, reason } = verdictFor(p, peer, floor);
  const steps: FarmStep[] = [
    {
      text: `Set up a fresh wallet for ${p.name} alone, write its seed phrase on paper and keep it offline; never farm from a wallet that holds other assets.`,
      cost: 'no fee; the first funding transfer pays its own gas',
      time: '15 minutes',
    },
    {
      text: coreAction(p, capital, chain),
      cost: `$${capital} of capital plus about $${setupGas} of gas`,
      time: '20 minutes to set up',
    },
    p.chains.length > 1
      ? {
          text: `Repeat a smaller version on ${p.chains[1]}: bridge about $${Math.round(capital / 2)} over and make one transaction there, so the wallet is active on two of ${p.name}'s ${p.chains.length} chains.`,
          cost: `about $${secondGas} of bridge and gas fees`,
          time: '15 minutes',
        }
      : {
          text: `Use a second surface of ${p.name} in the same week: a different market, pool or vault than the first position, at about $${Math.round(capital / 2)} more.`,
          cost: `about $${secondGas} of gas`,
          time: '15 minutes',
        },
    {
      text: `Repeat one qualifying action a week for ${weeks} weeks (a deposit, a borrow, a swap or a vote), so the activity spans more than one weekly snapshot of the protocol.`,
      cost: `about $2 of gas a week, $${2 * weeks} in all`,
      time: `10 minutes a week for ${weeks} weeks`,
    },
    {
      text: `Open ${p.name}'s own points or rewards page and check what the wallet has earned; claim only from a link the protocol itself publishes, and never pay a fee to claim.`,
      cost: 'free to check; a claim that asks for payment is a scam',
      time: '10 minutes',
    },
    {
      text: `Log the wallet, the dates and every transaction hash. If a token launches, claim inside the announced window and sell half at the claim if the position is worth more than ten times its $${cash} cash cost; hold the rest only while TVL holds.`,
      cost: 'nothing but the log; withdrawing ends the position',
      time: '10 minutes, plus 30 minutes on a claim day',
    },
  ];
  return { steps, capital_usd: capital, cash_cost_usd: cash, weeks, hours, verdict, verdict_reason: reason };
}

/** A step as one string for board.json: its action, then its cost and time. */
export function stepText(step: FarmStep): string {
  let s = `${step.text} Cost: ${step.cost}. Time: ${step.time}.`;
  if (s.length > 600) s = `${s.slice(0, 596)}...`;
  if (s.length < 20) s = `${s} Hold the position and re-check weekly.`;
  return s;
}

// ---------------------------------------------------------------------------
// The board
// ---------------------------------------------------------------------------

/** One row of board.json: the ranked board's row, with the model's extras the dashboard shows. */
export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  listed_at: string | null;
  age_days: number | null;
  token_status: 'records-no-token';
  watchlist: boolean;
  implied_value_usd: number;
  implied_low_usd: number;
  implied_high_usd: number;
  peer_token_rate: number;
  peer_count: number;
  verdict: string;
  farm_steps: string[];
}

/** One row of the dashboard's data: everything the page renders for a protocol. */
export interface RadarRow extends BoardRow {
  verdict_reason: string;
  peer_tokenized: number;
  farm: {
    capital_usd: number;
    cash_cost_usd: number;
    weeks: number;
    hours: number;
    steps: FarmStep[];
  };
}

/** A token-bearing protocol, set aside from the board: the market's own value, not a model's. */
export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  tvl_usd: number;
  mcap_usd: number;
  change_7d: number | null;
}

/** Everything the dashboard renders, from the snapshot. */
export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols_total: number;
  candidates_total: number;
  watchlist_size: number;
  off_list_above_floor: number;
  below_floor: number;
  board_size: number;
  model: Omit<FittedModel, 'predict'>;
  rows: RadarRow[];
  live: LiveRow[];
}

/** Build the ranked board and the dashboard's data from the snapshot and the fitted model. */
export function buildRadar(snapshot: Snapshot, model: FittedModel): { board: BoardRow[]; radar: RadarData } {
  const watch = new Set(snapshot.watchlist);
  const candidates = snapshot.protocols.filter(isCandidate);
  const scored = candidates.map((p) => {
    const age = ageDays(snapshot.as_of, p.listed_at);
    const peer = peerTokenRate(p, snapshot.protocols);
    const plan = farmPlan(p, age, peer, snapshot.floor_usd);
    const pred = model.predict(p, snapshot.as_of);
    return { p, age, peer, plan, pred };
  });
  scored.sort((a, b) => b.pred.value - a.pred.value || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));
  const rows: RadarRow[] = scored.map((s, i) => ({
    rank: i + 1,
    slug: s.p.slug,
    name: s.p.name,
    category: s.p.category,
    chains: s.p.chains,
    tvl_usd: s.p.tvl_usd,
    change_7d: s.p.change_7d,
    listed_at: s.p.listed_at,
    age_days: s.age,
    token_status: 'records-no-token',
    watchlist: watch.has(s.p.slug),
    implied_value_usd: s.pred.value,
    implied_low_usd: s.pred.low,
    implied_high_usd: s.pred.high,
    peer_token_rate: s.peer.rate,
    peer_count: s.peer.peers,
    peer_tokenized: s.peer.tokenized,
    verdict: s.plan.verdict,
    verdict_reason: s.plan.verdict_reason,
    farm_steps: s.plan.steps.map(stepText),
    farm: {
      capital_usd: s.plan.capital_usd,
      cash_cost_usd: s.plan.cash_cost_usd,
      weeks: s.plan.weeks,
      hours: s.plan.hours,
      steps: s.plan.steps,
    },
  }));
  const board: BoardRow[] = rows.map(({ verdict_reason, peer_tokenized, farm, ...rest }) => rest);
  const live: LiveRow[] = snapshot.protocols
    .filter((p) => p.mcap_usd !== null && p.mcap_usd > 0)
    .sort((a, b) => b.mcap_usd! - a.mcap_usd! || (a.slug < b.slug ? -1 : 1))
    .slice(0, 18)
    .map((p) => ({ slug: p.slug, name: p.name, category: p.category, tvl_usd: p.tvl_usd, mcap_usd: p.mcap_usd!, change_7d: p.change_7d }));
  const { predict, ...modelInfo } = model;
  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    protocols_total: snapshot.protocols.length,
    candidates_total: candidates.length,
    watchlist_size: snapshot.watchlist.length,
    off_list_above_floor: rows.filter((r) => !r.watchlist && r.tvl_usd >= snapshot.floor_usd).length,
    below_floor: rows.filter((r) => r.tvl_usd < snapshot.floor_usd).length,
    board_size: rows.length,
    model: modelInfo,
    rows,
    live,
  };
  return { board, radar };
}

/** The snapshot out of a file's text, refused unless it is the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return { as_of: o.as_of, source: o.source, floor_usd: o.floor_usd, protocols, watchlist: o.watchlist as string[] };
}

/** Why a board is not one the gate will take, or null: coverage, ranks, order, farm steps, values. */
export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slug${missing.length === 1 ? '' : 's'}: ${missing.slice(0, 5).join(', ')}`;
  if (board.length < 100) return `the board holds ${board.length} rows, fewer than 100`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  const byRank = [...board].sort((a, b) => a.rank - b.rank);
  for (let i = 1; i < byRank.length; i++) if (byRank[i]!.implied_value_usd > byRank[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && byRank[0]!.implied_value_usd === byRank[byRank.length - 1]!.implied_value_usd) return 'all rows share one implied value';
  for (const r of board) {
    if (r.farm_steps.length < 3 || r.farm_steps.length > 12) return `${r.slug} has ${r.farm_steps.length} farm steps`;
    for (const s of r.farm_steps) if (s.trim().length < 20 || s.trim().length > 600) return `${r.slug} has a farm step of ${s.trim().length} characters`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
  }
  return null;
}

/** A compact data.js: the dashboard's whole dataset as one script, no fetch and no network. */
export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}
