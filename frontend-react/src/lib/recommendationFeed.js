/**
 * The recommendation browsing sequence, independent of React.
 *
 * One "sequence" is everything shown for one profile: the first batch, every
 * Next Batch after it, and each in-place replacement. The feed owns the three
 * guarantees the page makes about it:
 *
 *   No repeats.       A title surfaced once in a sequence is never surfaced
 *                     again in it, whatever the server returns.
 *   No stale results. A response that belongs to an older profile is dropped.
 *                     Every profile change starts a new epoch, and any request
 *                     that finishes in a later epoch than it began is ignored.
 *   Skip is not dislike. Replace marks a title skipped for this page session
 *                     only. It is sent as an exclusion and never written to the
 *                     profile.
 *
 * Where the titles come from is a *source*. Next Batch and Replace both read
 * the same source, in order, so they cannot disagree about what comes next.
 *
 *   Cursor source     The server ranks one sequence for a frozen profile
 *                     snapshot and the client holds a cursor into it. Requests
 *                     carry the snapshot and the cursor, never the titles
 *                     already shown, so they do not grow with paging.
 *   Exclusion source  For a backend that does not offer cursor paging: every
 *                     request excludes everything already shown.
 */
import { blockedIds, blockedTitleKeys, titleKey } from './session.js';
import { BATCH_SIZE, POOL_SIZE, SEQUENCE_PAGE_SIZE, uniqueEligible } from './recommendationPool.js';

const idOf = (item) => Number(item?.id);
const isAbort = (error) => error?.name === 'AbortError';
const isExpired = (error) => error?.code === 'sequence_expired' || error?.status === 409;
// One click may need several pages when the client filter removes most of one
// (after a sequence restart, for example). Bounded so a click always returns.
const MAX_REQUESTS_PER_ACTION = 8;

/**
 * Cursor paging against `/api/recommend/page`.
 *
 * `buildRequest(session, {skippedIds})` returns `{body, accounting}`; the body is
 * frozen for the life of the sequence. `requestPage(body, {cursor, limit,
 * signal})` returns the API payload.
 */
export function createCursorSource({ buildRequest, requestPage, pageSize = SEQUENCE_PAGE_SIZE }) {
  const read = (payload, state) => {
    const sequence = payload?.sequence || {};
    return {
      results: payload?.results || [],
      hasMore: Boolean(sequence.has_more && sequence.next_cursor),
      accounting: state.accounting,
      payload,
      state: { ...state, cursor: sequence.next_cursor || null },
    };
  };
  return {
    kind: 'cursor',
    async open(session, { skippedIds = [], signal } = {}) {
      const { body, accounting } = buildRequest(session, { skippedIds });
      const state = { kind: 'cursor', body, accounting, cursor: null };
      return read(await requestPage(body, { cursor: null, limit: pageSize, signal }), state);
    },
    async more(state, { signal } = {}) {
      if (!state.cursor) return { results: [], hasMore: false, accounting: state.accounting, state };
      return read(await requestPage(state.body, { cursor: state.cursor, limit: pageSize, signal }), state);
    },
  };
}

/**
 * Paging by exclusion. `fetchPage(session, {shownIds, skippedIds, limit,
 * signal})` returns `{results, accounting}`. The server gives no "has more"
 * signal here, so the feed treats a page with nothing new as the end.
 */
export function createExclusionSource({ fetchPage, poolSize = POOL_SIZE }) {
  const read = (payload) => ({
    results: payload?.results || [],
    hasMore: true,
    accounting: payload?.accounting || null,
    payload: payload || {},
    state: { kind: 'exclusion' },
  });
  return {
    kind: 'exclusion',
    async open(session, { skippedIds = [], signal } = {}) {
      return read(await fetchPage(session, { shownIds: [], skippedIds, limit: poolSize, signal }));
    },
    async more(_state, { session, shownIds = [], skippedIds = [], signal } = {}) {
      return read(await fetchPage(session, { shownIds, skippedIds, limit: poolSize, signal }));
    },
  };
}

/** Use `cursor` when `available()` says the backend supports it, else `exclusion`. */
export function createAutoSource({ cursor, exclusion, available }) {
  const byKind = { cursor, exclusion };
  return {
    kind: 'auto',
    open(session, options) {
      return (available() ? cursor : exclusion).open(session, options);
    },
    more(state, options) {
      return byKind[state?.kind || 'exclusion'].more(state, options);
    },
  };
}

export function createRecommendationFeed({ source, fetchPage, batchSize = BATCH_SIZE, poolSize = POOL_SIZE } = {}) {
  const pages = source || createExclusionSource({ fetchPage, poolSize });
  let epoch = 0;
  let resetting = false;
  let visible = [];
  let pool = [];
  let state = null;
  let hasMore = false;
  let exhausted = false;
  let restarts = 0;
  let accounting = null;
  let queue = Promise.resolve();
  const shown = new Set();
  // Survives profile changes on purpose: liking another title must not bring a
  // just-skipped one straight back. It is in memory only, so a reload clears it.
  const skipped = new Set();

  function eligible(items, session, already = new Set()) {
    const forbidden = new Set([...blockedIds(session), ...skipped]);
    const titles = blockedTitleKeys(session);
    return uniqueEligible(items, forbidden, already).filter(item => !titles.has(titleKey(item.title)));
  }
  const snapshot = (extra = {}) => ({
    applied: true, epoch, visible: [...visible], exhausted, accounting, ...extra,
  });
  const stale = () => ({ applied: false, epoch });

  /**
   * Read the source until `need` unseen titles are in hand or it runs out.
   * What is left of the reserve comes first: it is earlier in the sequence.
   * Returns null when the profile changed while a request was in flight.
   */
  async function gather(session, mine, need) {
    let available = eligible(pool, session, shown);
    let payload = null;
    for (let request = 0; request < MAX_REQUESTS_PER_ACTION && available.length < need && hasMore; request++) {
      let page;
      try {
        page = await pages.more(state, { session, shownIds: [...shown], skippedIds: [...skipped] });
      } catch (error) {
        if (!isExpired(error)) throw error;
        // The server can no longer continue this sequence: a deployment changed
        // what ranks it. Start a new one. `shown` still prevents any repeat.
        page = await pages.open(session, { skippedIds: [...skipped] });
        restarts += 1;
      }
      if (mine !== epoch) return null;
      state = page.state;
      hasMore = page.hasMore;
      accounting = page.accounting || accounting;
      payload = page.payload || payload;
      const fresh = eligible(page.results, session, shown);
      // An exclusion source has no end marker; a page with nothing new is one.
      if (!fresh.length && page.state?.kind !== 'cursor') hasMore = false;
      available = uniqueEligible([...available, ...fresh]);
    }
    if (!hasMore && available.length < need) exhausted = true;
    return { payload, available };
  }
  /** Run one operation at a time, in click order, and never across a reset. */
  function enqueue(operation) {
    const run = queue.then(() => (resetting ? { applied: false, epoch, ignored: true } : operation(epoch)));
    queue = run.catch(() => {});
    return run;
  }

  async function fill(session, item, mine) {
    const id = idOf(item);
    if (!visible.some(entry => idOf(entry) === id)) return snapshot({ changed: false });
    let available = eligible(pool, session, shown);
    let error = null;
    let payload = null;
    if (!available.length && hasMore) {
      try {
        const more = await gather(session, mine, 1);
        if (!more) return stale();
        ({ available, payload } = more);
      } catch (failure) { error = failure; }
    }
    if (mine !== epoch) return stale();
    const replacement = available[0] || null;
    if (replacement) {
      pool = available.slice(1);
      shown.add(idOf(replacement));
      visible = visible.map(entry => (idOf(entry) === id ? replacement : entry));
    } else {
      if (!hasMore) exhausted = true;
      visible = visible.filter(entry => idOf(entry) !== id);
    }
    return snapshot({ changed: true, replacement, error, payload });
  }

  return {
    /** Begin a new sequence for a (possibly changed) profile. */
    async start(session, { signal } = {}) {
      const mine = ++epoch;
      resetting = true;
      try {
        // Nothing already shown is sent: a new profile ranks from the top.
        const page = await pages.open(session, { skippedIds: [...skipped], signal });
        if (mine !== epoch || signal?.aborted) return stale();
        shown.clear();
        state = page.state;
        hasMore = page.hasMore;
        exhausted = false;
        accounting = page.accounting || null;
        pool = eligible(page.results, session);
        let payload = page.payload;
        if (pool.length < batchSize && hasMore && page.state?.kind === 'cursor') {
          // The client filter thinned the first page; read on to fill the batch.
          const more = await gather(session, mine, batchSize);
          if (!more) return stale();
          pool = more.available;
          payload = payload || more.payload;
        }
        if (page.state?.kind !== 'cursor' && !pool.length) hasMore = false;
        visible = pool.slice(0, batchSize);
        pool = pool.slice(batchSize);
        for (const item of visible) shown.add(idOf(item));
        if (!hasMore && !pool.length) exhausted = true;
        return snapshot({ payload, partial: visible.length < batchSize });
      } catch (error) {
        if (mine !== epoch || isAbort(error)) return stale();
        return snapshot({ error });
      } finally {
        // Only the newest reset may reopen the feed; a superseded one must not.
        if (mine === epoch) resetting = false;
      }
    },

    /** Advance to the next batch. Never returns a title already shown. */
    nextBatch(session) {
      return enqueue(async (mine) => {
        let available = eligible(pool, session, shown);
        let payload = null;
        if (available.length < batchSize && hasMore) {
          try {
            const more = await gather(session, mine, batchSize);
            if (!more) return stale();
            ({ available, payload } = more);
          } catch (error) {
            return mine === epoch ? snapshot({ changed: false, error }) : stale();
          }
        }
        if (mine !== epoch) return stale();
        const next = available.slice(0, batchSize);
        if (!next.length) {
          exhausted = true;
          return snapshot({ changed: false });
        }
        pool = available.slice(next.length);
        visible = next;
        for (const item of next) shown.add(idOf(item));
        if (!hasMore && !pool.length) exhausted = true;
        return snapshot({ changed: true, partial: next.length < batchSize, payload });
      });
    },

    /** Replace / Skip: hide for this page session and show another title. */
    skip(session, item) {
      return enqueue((mine) => {
        skipped.add(idOf(item));
        return fill(session, item, mine);
      });
    },

    /** Fill the slot of a title the profile itself now blocks (saved or rejected). */
    drop(session, item) {
      return enqueue((mine) => fill(session, item, mine));
    },

    /** What may be rendered from `items`: nothing saved, rejected, skipped, or repeated. */
    eligible(items, session) {
      return eligible(items, session);
    },
    unskip(item) { skipped.delete(idOf(item)); },
    forgetSkips() { skipped.clear(); },
    state() {
      return {
        epoch, resetting, exhausted, hasMore, restarts,
        source: state?.kind || null, cursor: state?.cursor || null,
        visibleIds: visible.map(idOf), poolIds: pool.map(idOf),
        shownIds: [...shown], skippedIds: [...skipped],
      };
    },
  };
}
