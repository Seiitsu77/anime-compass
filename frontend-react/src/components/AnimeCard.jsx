import React from 'react';
import { Bookmark, Heart, RefreshCw, Star, ThumbsDown, Undo2 } from 'lucide-react';
import { S } from '../lib/strings.js';

export function AnimeCard({ anime, liked, watchlisted=false, onToggleLike, onToggleWatchlist, onReplace, onNotInterested, onRestore, onDetails, compact=false, actionsDisabled=false }) {
  if (!anime) return null;
  const image = typeof anime.image_url === 'string' ? anime.image_url : '';
  const score = Number(anime.score);
  return <article className={`anime-card ${compact?'anime-card-compact':''}`} data-anime-id={anime.id}>
    <button className="poster-button" onClick={() => onDetails(anime)} aria-label={S.actions.viewDetailsLabel(anime.title)}>
      <span className="poster">
        {image ? <img src={image} loading="lazy" alt={S.actions.posterAlt(anime.title)} /> : <span className="poster-fallback">{S.brand.posterFallback}</span>}
      </span>
    </button>
    <div className="card-top-actions">
      <button onClick={() => onToggleLike(anime)} className={`heart-btn ${liked?'liked':''}`} aria-pressed={liked} aria-label={S.actions.likeLabel(anime.title)} title={liked?S.actions.liked:S.actions.like}><Heart size={17} fill={liked?'currentColor':'none'}/></button>
      {onToggleWatchlist&&<button onClick={() => onToggleWatchlist(anime)} className={`bookmark-btn ${watchlisted?'saved':''}`} aria-pressed={watchlisted} aria-label={S.actions.wantToWatchLabel(anime.title)} title={watchlisted?S.actions.wantToWatchRemove:S.actions.wantToWatch}><Bookmark size={16} fill={watchlisted?'currentColor':'none'}/></button>}
    </div>
    <div className="anime-info">
      <button className="anime-title" onClick={() => onDetails(anime)} title={anime.title}>{anime.title}</button>
      <div className="anime-submeta">
        <span>{[anime.type,anime.start_year].filter(Boolean).join(' · ')}</span>
        {Number.isFinite(score) && score > 0 ? <span className="score"><Star size={13} fill="currentColor"/> {score.toFixed(1)}</span> : null}
      </div>
      {!compact && Array.isArray(anime.genres) && anime.genres.length>0 && <div className="genre-tags">{anime.genres.slice(0,2).map(g=><span key={g}>{g}</span>)}</div>}
      {(onReplace||onNotInterested)&&<div className="card-actions">
        {onReplace&&<button type="button" className="card-action card-replace" disabled={actionsDisabled} onClick={()=>onReplace(anime)} title={S.actions.replaceHint} aria-label={S.actions.replaceLabel(anime.title)}><RefreshCw size={13}/> {S.actions.replace}</button>}
        {onNotInterested&&<button type="button" className="card-action card-not-interested" disabled={actionsDisabled} onClick={()=>onNotInterested(anime)} title={S.actions.notInterestedHint} aria-label={S.actions.notInterestedLabel(anime.title)}><ThumbsDown size={13}/> {S.actions.notInterested}</button>}
      </div>}
      {onRestore&&<div className="card-actions"><button type="button" className="card-action card-replace" onClick={()=>onRestore(anime)} aria-label={S.actions.notInterestedRemoveLabel(anime.title)}><Undo2 size={13}/> {S.actions.notInterestedRemove}</button></div>}
    </div>
  </article>;
}
