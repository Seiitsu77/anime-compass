import React, { useState } from 'react';
import { ChevronDown, SlidersHorizontal, X } from 'lucide-react';
import { S } from '../lib/strings.js';

export const EMPTY_FILTERS = { genre:'', year:'', format:'', minScore:'', maxEpisodes:'', studio:'', sort:'relevance' };
const SORT_OPTIONS = [
  ['relevance',S.filters.sortBestMatch], ['popularity',S.filters.sortPopularity], ['score',S.filters.sortScore],
  ['start_year',S.filters.sortNewest], ['title',S.filters.sortTitle],
];
export function FilterBar({ filters, setFilters, genres=[], formats=[] }) {
  const [open, setOpen] = useState(false);
  const update=(field, value)=>setFilters(old=>({ ...old, [field]: value }));
  const hasActive=Object.entries(filters).some(([k,v])=>v && !(k==='sort' && v==='relevance'));
  return <section className="filter-shell" aria-label={S.filters.regionLabel}>
    <div className="quick-filters">
      <div className="filter-heading"><SlidersHorizontal size={16}/> <span>{S.filters.heading}</span></div>
      <label className="filter-control"><span>{S.filters.genre}</span><select value={filters.genre} onChange={e=>update('genre',e.target.value)}><option value="">{S.filters.allGenres}</option>{genres.map(g=><option value={g} key={g}>{g}</option>)}</select><ChevronDown size={14}/></label>
      <label className="filter-control"><span>{S.filters.year}</span><select value={filters.year} onChange={e=>update('year',e.target.value)}><option value="">{S.filters.anyYear}</option>{Array.from({length:27},(_,i)=>2026-i).map(y=><option key={y} value={y}>{y}</option>)}</select><ChevronDown size={14}/></label>
      <label className="filter-control"><span>{S.filters.format}</span><select value={filters.format} onChange={e=>update('format',e.target.value)}><option value="">{S.filters.allFormats}</option>{formats.map(f=><option key={f} value={f}>{f}</option>)}</select><ChevronDown size={14}/></label>
      <button className="more-filters" onClick={()=>setOpen(v=>!v)} aria-expanded={open}>{open?S.filters.fewerFilters:S.filters.moreFilters} <ChevronDown size={15} className={open?'rotated':''}/></button>
      {hasActive && <button className="clear-filters" onClick={()=>setFilters({...EMPTY_FILTERS})} aria-label={S.filters.clearLabel}><X size={15}/> {S.filters.clear}</button>}
    </div>
    {open && <div className="advanced-filters">
      <label>{S.filters.minimumScore} <input type="number" min="0" max="10" step="0.1" placeholder={S.filters.any} value={filters.minScore} onChange={e=>update('minScore',e.target.value)}/></label>
      <label>{S.filters.maximumEpisodes} <input type="number" min="1" placeholder={S.filters.any} value={filters.maxEpisodes} onChange={e=>update('maxEpisodes',e.target.value)}/></label>
      <label>{S.filters.studio} <input value={filters.studio} maxLength={100} onChange={e=>update('studio',e.target.value)} placeholder={S.filters.studioPlaceholder}/></label>
      <label>{S.filters.sortBy} <select value={filters.sort} onChange={e=>update('sort',e.target.value)}>{SORT_OPTIONS.map(([value,label])=><option key={value} value={value}>{label}</option>)}</select></label>
    </div>}
  </section>;
}
