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
 * This is the single seam for the paginated recommendation API. Paging is
 * expressed today as "exclude what was shown"; `fetchPage` receives the shown
 * and skipped IDs and can translate them to a cursor or offset without any
 * change to the callers.
 */
import { blockedIds, blockedTitleKeys, titleKey } from './session.js';
import { BATCH_SIZE, POOL_SIZE, uniqueEligible } from './recommendationPool.js';

const idOf = (item) => Number(item?.id);
const isAbort = (error) => error?.name === 'AbortError';

export function createRecommendationFeed({ fetchPage, batchSize = BATCH_SIZE, poolSize = POOL_SIZE } = {}) {
  let epoch = 0;
  let resetting = false;
  let visible = [];
  let pool = [];
  let exhausted = false;
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

  async function request(session, signal) {
    const payload = await fetchPage(session, {
      shownIds: [...shown], skippedIds: [...skipped], limit: poolSize, signal,
    });
    return payload || {};
  }
  /** Ask for more candidates, keeping whatever is left of the current pool. */
  async function refill(session, mine) {
    const payload = await request(session);
    if (mine !== epoch) return null;
    accounting = payload.accounting || accounting;
    const fresh = eligible(payload.results, session, shown);
    if (!fresh.length) exhausted = true;
    return { payload, available: uniqueEligible([...fresh, ...eligible(pool, session, shown)]) };
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
    if (!available.length && !exhausted) {
      try {
        const more = await refill(session, mine);
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
        // `shown` is deliberately not sent: a new profile ranks from the top.
        const payload = await fetchPage(session, {
          shownIds: [], skippedIds: [...skipped], limit: poolSize, signal,
        }) || {};
        if (mine !== epoch || signal?.aborted) return stale();
        const list = eligible(payload.results, session);
        shown.clear();
        visible = list.slice(0, batchSize);
        pool = list.slice(batchSize);
        exhausted = false;
        accounting = payload.accounting || null;
        for (const item of visible) shown.add(idOf(item));
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
        if (available.length < batchSize && !exhausted) {
          try {
            const more = await refill(session, mine);
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
        epoch, resetting, exhausted,
        visibleIds: visible.map(idOf), poolIds: pool.map(idOf),
        shownIds: [...shown], skippedIds: [...skipped],
      };
    },
  };
}
