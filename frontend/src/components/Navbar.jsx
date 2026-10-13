import React from 'react';
import { Compass, Heart } from 'lucide-react';
import { PillNav } from './animations/PillNav.jsx';
import { S } from '../lib/strings.js';

const TABS = [
  ['home',S.nav.home], ['discover',S.nav.discover], ['for-you',S.nav.forYou], ['agent',S.nav.agent],
];
export function Navbar({ page, navigate, likeCount, health }) {
  const online = Boolean(health?.agent?.available);
  const count = health?.catalog?.count;
  return <header className="topbar">
    <button className="brand" onClick={() => navigate('home')} aria-label={S.brand.homeLabel}>
      <span className="brand-symbol"><Compass size={27} strokeWidth={1.7}/></span>
      <span><strong>{S.brand.name}</strong><small>{S.brand.tagline}</small></span>
    </button>
    <PillNav items={TABS} value={page} onChange={navigate} label={S.nav.label}/>
    <div className="nav-meta">
      {count > 0 && <span className="catalog-status">{S.status.catalogCount(count)}</span>}
      {health && <span className={`service-dot ${online?'ready':'offline'}`} title={online?S.status.agentOnlineHint:S.status.agentOfflineHint}><i/>{online?S.status.agentOnline:S.status.catalogReady}</span>}
      <button className="likes-shortcut" onClick={() => navigate('for-you')} title={S.status.likedShortcut} aria-label={S.status.likedShortcutLabel(likeCount)}><Heart size={18}/><span>{likeCount}</span></button>
    </div>
  </header>;
}
