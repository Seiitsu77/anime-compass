/** Immutable helpers: never display saved or previously surfaced titles twice. */
export const BATCH_SIZE = 18;
export const POOL_SIZE = 50;
// One cursor request: the batch on screen plus one batch held in reserve, so a
// Replace or the next Next Batch is usually answered without a round trip.
export const SEQUENCE_PAGE_SIZE = 2 * BATCH_SIZE;
export function uniqueEligible(items, forbidden = new Set(), already = new Set()) {
  const seen = new Set(already);
  return (Array.isArray(items) ? items : []).filter(item => {
    const id = Number(item?.id);
    if (!Number.isInteger(id) || id<=0 || forbidden.has(id) || seen.has(id)) return false;
    seen.add(id);
    return true;
  });
}
export function takeFromPool(pool, visible, shown, forbidden, count) {
  const occupied=new Set([...visible.map(x=>Number(x.id)), ...shown]);
  const next=uniqueEligible(pool,forbidden,occupied).slice(0,Math.max(0,count));
  const chosen=new Set(next.map(item=>Number(item.id)));
  return { chosen: next, rest: pool.filter(item=>!chosen.has(Number(item.id))) };
}
