import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowRight, ArrowUpRight, Bookmark, Check, ChevronDown, Compass, Heart, RefreshCw, Search, Sparkles, ThumbsDown, X } from 'lucide-react';
import { Navbar } from './components/Navbar.jsx';
import { AnimeCard } from './components/AnimeCard.jsx';
import { AgentPage } from './components/AgentPage.jsx';
import { EMPTY_FILTERS, FilterBar } from './components/FilterBar.jsx';
import { api, buildSequenceRequest, createPreferenceSync, recommend, recommendPage, resolveLimits, resolvePagination, searchCatalog } from './lib/api.js';
import { clearHiddenTitle, clearNotInterested, legacyHiddenTitles, loadSession, markNotInterested, normalizeSession, persistSession, toggleLiked, toggleWatchlist } from './lib/session.js';
import { BATCH_SIZE } from './lib/recommendationPool.js';
import { createAutoSource, createCursorSource, createExclusionSource, createRecommendationFeed } from './lib/recommendationFeed.js';
import { createDiagnosticsLog } from './lib/diagnostics.js';
import { S } from './lib/strings.js';
import { BlurText } from './components/animations/BlurText.jsx';

const validPages=new Set(['home','discover','for-you','agent']);
const COLLECTION_PREVIEW=12;
// A burst of Likes becomes one recompute instead of one request per click.
const PROFILE_DEBOUNCE_MS=200;
function hashPage() {const page=location.hash.replace(/^#\/?/,'').split('?')[0];return validPages.has(page)?page:'home';}
const getItems=(payload)=>Array.isArray(payload?.results)?payload.results:[];
const sameAnime=(a,b)=>Number(a?.id)===Number(b?.id);
// Development only. Vite compiles `import.meta.env.DEV` to `false` for a
// production build, which removes the log and its console global from the bundle.
const diagnostics=import.meta.env.DEV?createDiagnosticsLog(true):null;

function SectionTitle({eyebrow, title, caption, onAction, action=S.home.seeMore}) {
  return <div className="section-heading"><div><div className="section-kicker">{eyebrow}</div><h2>{title}</h2>{caption && <p>{caption}</p>}</div>{onAction&&<button className="section-link" onClick={onAction}>{action} <ArrowRight size={15}/></button>}</div>;
}

export function App() {
  const [page,setPage]=useState(hashPage);
  const [session,setSession]=useState(loadSession);
  const [health,setHealth]=useState(null);
  const [ready,setReady]=useState(false);
  const [healthError,setHealthError]=useState('');
  const [popular,setPopular]=useState([]);
  const [recommendations,setRecommendations]=useState([]);
  const [loadingPopular,setLoadingPopular]=useState(true);
  const [loadingRec,setLoadingRec]=useState(true);
  const [refreshing,setRefreshing]=useState(false);
  const [recBusy,setRecBusy]=useState(false);
  const [recNotice,setRecNotice]=useState(null);
  const [profileNotice,setProfileNotice]=useState('');
  const [expandCollection,setExpandCollection]=useState(false);
  const [collectionTab,setCollectionTab]=useState('liked');
  const [recommendError,setRecommendError]=useState('');
  const [filters,setFilters]=useState({...EMPTY_FILTERS});
  const [query,setQuery]=useState('');
  const [heroQuery,setHeroQuery]=useState('');
  const [discoverResults,setDiscoverResults]=useState([]);
  const [discoverBusy,setDiscoverBusy]=useState(false);
  const [discoverError,setDiscoverError]=useState('');
  const [more,setMore]=useState(false);
  const [total,setTotal]=useState(0);
  const [detail,setDetail]=useState(null);
  const [detailBusy,setDetailBusy]=useState(false);
  const [messages,setMessages]=useState([]);
  const [notice,setNotice]=useState('');
  const fileInput=useRef(null);
  const sessionRef=useRef(session);
  sessionRef.current=session;
  const limits=useMemo(()=>resolveLimits(health),[health]);
  const limitsRef=useRef(limits);
  limitsRef.current=limits;
  const pagination=useMemo(()=>resolvePagination(health),[health]);
  const paginationRef=useRef(pagination);
  paginationRef.current=pagination;
  const feedRef=useRef(null);
  // Cursor paging when the backend advertises it; otherwise page by exclusion.
  if(!feedRef.current)feedRef.current=createRecommendationFeed({source:createAutoSource({
    available:()=>Boolean(paginationRef.current),
    cursor:createCursorSource({
      buildRequest:(current,{skippedIds})=>buildSequenceRequest(current,{skippedIds,limits:limitsRef.current}),
      requestPage:(body,options)=>recommendPage(body,{...options,endpoint:paginationRef.current?.endpoint}),
    }),
    exclusion:createExclusionSource({
      fetchPage:(current,{shownIds,skippedIds,limit,signal})=>recommend(current,signal,{shownIds,skippedIds,limit,limits:limitsRef.current}),
    }),
  })});
  const feed=feedRef.current;
  const syncRef=useRef(null);
  if(!syncRef.current)syncRef.current=createPreferenceSync();
  const firstLoad=useRef(true);
  const busyCount=useRef(0);

  function navigate(target) {location.hash=target==='home'?'#home':`#${target}`;setPage(target);window.scrollTo({top:0,behavior:'smooth'});}
  useEffect(()=>{const onHash=()=>setPage(hashPage());window.addEventListener('hashchange',onHash);return()=>window.removeEventListener('hashchange',onHash);},[]);
  useEffect(()=>{if(!persistSession(session))setNotice(S.status.storageUnavailable);},[session]);
  useEffect(()=>{
    let active=true;let timer;
    const started=Date.now();
    async function probe() {
      try {
        await api('/api/ready');
        // Health carries the request limits, so it is read before the first recommendation.
        const report=await api('/api/health').catch(()=>null);
        if(!active)return;
        setHealth(report);setHealthError('');setReady(true);
      } catch {
        if(!active)return;
        setHealthError(Date.now()-started>120000?S.status.backendSlow:S.status.backendStarting);
        timer=setTimeout(probe,3000);
      }
    }
    probe();
    return()=>{active=false;clearTimeout(timer);};
  },[]);
  // After a cold start the server-side profile is empty; restore it once.
  useEffect(()=>{if(ready)syncRef.current.push(sessionRef.current);},[ready]);
  useEffect(()=>{
    if(!ready)return;
    const c=new AbortController();setLoadingPopular(true);
    searchCatalog({sort:'popularity',top_k:12},c.signal)
      .then(x=>setPopular(getItems(x))).catch(e=>{if(e.name!=='AbortError')setNotice(S.status.catalogError(e.message));})
      .finally(()=>{if(!c.signal.aborted)setLoadingPopular(false);});
    return()=>c.abort();
  },[ready]);
  const likedIds=new Set(session.liked.map(x=>Number(x.id)));
  const watchIds=new Set((session.watchlist||[]).map(x=>Number(x.id)));
  const likedKey=session.liked.map(x=>Number(x.id)).join(',');

  // Only Likes are profile inputs, so only a change to them recomputes. Want to
  // Watch, Not Interested and Skip remove or replace cards in place instead:
  // none of them moves the model, and recomputing would restart the sequence.
  useEffect(()=>{
    if(!ready)return;
    const c=new AbortController();
    const current=session;
    setRecommendError('');
    if(firstLoad.current)setLoadingRec(true);else setRefreshing(true);
    const timer=setTimeout(async()=>{
      const result=await feed.start(current,{signal:c.signal});
      // Superseded by a newer profile: that request owns the screen now.
      if(!result.applied)return;
      firstLoad.current=false;
      setLoadingRec(false);setRefreshing(false);
      if(result.error){setRecommendError(result.error.message);return;}
      setRecommendations(result.visible);
      setRecNotice(null);
      const dropped=result.accounting?.likedDropped||0;
      setProfileNotice(dropped>0
        ? S.recommendations.likesTruncated(result.accounting.likedSent,result.accounting.likedTotal)
        : result.visible.length&&result.visible.length<BATCH_SIZE?S.recommendations.fewerThanBatch(result.visible.length):'');
      diagnostics?.record('profile',result.payload,{likedCount:current.liked.length});
    },firstLoad.current?0:PROFILE_DEBOUNCE_MS);
    return()=>{clearTimeout(timer);c.abort();};
    // eslint-disable-next-line react-hooks/exhaustive-deps
  },[ready,session.sessionId,likedKey,limits.liked_ids]);

  function commit(next) {
    sessionRef.current=next;
    setSession(next);
    // Local storage is the source of truth; the server copy feeds the agent.
    syncRef.current.push(next);
  }
  /** Apply one feed operation. Results from an older profile are ignored. */
  async function settle(operation,onApplied) {
    busyCount.current+=1;setRecBusy(true);
    try {
      const result=await operation;
      if(result?.applied) {
        setRecommendations(result.visible);
        if(result.payload)diagnostics?.record('refill',result.payload);
        onApplied?.(result);
      }
    } finally {
      busyCount.current-=1;
      if(!busyCount.current)setRecBusy(false);
    }
  }
  function nextBatch() {
    if(recBusy||loadingRec||refreshing)return;
    settle(feed.nextBatch(sessionRef.current),result=>{
      if(result.error)setRecNotice({text:S.recommendations.nextBatchFailed(result.error.message)});
      else if(!result.changed)setRecNotice({text:S.recommendations.exhausted});
      else if(result.accounting?.excludedDropped>0)setRecNotice({text:S.recommendations.exclusionsTruncated(result.accounting.excludedDropped)});
      else setRecNotice(result.partial?{text:S.recommendations.partialBatch(result.visible.length)}:null);
    });
  }
  function replaceRecommendation(item) {
    // Skip is not a dislike: nothing is written to the profile or sent as feedback.
    settle(feed.skip(sessionRef.current,item),result=>{
      if(!result.changed)return;
      if(result.error)setRecNotice({text:S.recommendations.replaceFailed(item.title,result.error.message)});
      else setRecNotice({text:result.replacement?S.recommendations.skipped(item.title):S.recommendations.skippedNoReplacement(item.title)});
    });
  }
  function restoreInterest(item) {
    commit(clearNotInterested(sessionRef.current,item));
    setRecNotice({text:S.recommendations.notInterestedRemoved(item.title)});
  }
  function rejectRecommendation(item) {
    const next=markNotInterested(sessionRef.current,item);
    commit(next);
    settle(feed.drop(next,item),()=>setRecNotice({
      text:S.recommendations.markedNotInterested(item.title),
      undo:()=>restoreInterest(item),
    }));
  }
  const filterKey=JSON.stringify(filters);
  useEffect(()=>{
    if(page!=='discover' || !ready)return;
    const c=new AbortController();setDiscoverBusy(true);setDiscoverError('');setMore(false);
    searchCatalog({...filters,query,top_k:24},c.signal)
      .then(r=>{setDiscoverResults(getItems(r));setTotal(Number(r.total||0));setMore(Boolean(r.has_more));})
      .catch(e=>{if(e.name!=='AbortError')setDiscoverError(e.message);})
      .finally(()=>{if(!c.signal.aborted)setDiscoverBusy(false);});
    return()=>c.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  },[ready,page,query,filterKey]);
  async function loadMore() {
    if(discoverBusy)return;
    setDiscoverBusy(true);
    try {const r=await searchCatalog({...filters,query,offset:discoverResults.length,top_k:24});setDiscoverResults(old=>[...old,...getItems(r)]);setMore(Boolean(r.has_more));}catch(e){setDiscoverError(e.message);}finally{setDiscoverBusy(false);}
  }
  function likeAnime(item) {
    // Changing the Liked set changes `likedKey`, which recomputes recommendations.
    commit(toggleLiked(sessionRef.current,item));
  }
  function saveToWatchlist(item) {
    const next=toggleWatchlist(sessionRef.current,item);
    commit(next);
    // Want to Watch is future intent: the card leaves the recommendations, and
    // the title is never sent as a Like or as watched.
    if((next.watchlist||[]).some(x=>sameAnime(x,item)))settle(feed.drop(next,item));
  }
  function searchFromHero(event) {event.preventDefault();setQuery(heroQuery.trim());navigate('discover');}
  function changeFilters(next) {setFilters(next);if(page==='home')navigate('discover');}
  async function openDetails(item) {
    setDetail(item);setDetailBusy(true);
    try{const r=await api(`/api/anime/${Number(item.id)}`);setDetail(current=>current?.id===item.id?(r.result||item):current);}catch{/* show data from known record */}
    finally{setDetailBusy(false);}
  }
  async function importBackup(file) {
    if(!file)return;
    try {
      const parsed=JSON.parse(await file.text());
      if(!Array.isArray(parsed?.liked))throw new Error(S.backup.invalid);
      if(!window.confirm(S.backup.confirmImport(parsed.liked.length)))return;
      const restored=normalizeSession(parsed);
      feed.forgetSkips();
      commit(restored);setNotice(S.backup.imported(restored.liked.length));
    }catch(e){setNotice(S.backup.failed(e.message));}
    finally{if(fileInput.current)fileInput.current.value='';}
  }
  function exportBackup() {
    const blob=new Blob([JSON.stringify(session,null,2)],{type:'application/json'});
    const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=S.backup.fileName;a.click();URL.revokeObjectURL(url);
  }
  const filtersSection=<FilterBar filters={filters} setFilters={changeFilters} genres={health?.catalog?.genres||[]} formats={health?.catalog?.types||[]}/>;
  const recommendationsLocked=loadingRec||refreshing;
  const renderCards=(items,{compact=false,recommendation=false,restorable=false}={})=>items.map(item=><AnimeCard
    key={item.id} anime={item} compact={compact} liked={likedIds.has(Number(item.id))}
    watchlisted={watchIds.has(Number(item.id))} onToggleLike={likeAnime}
    onToggleWatchlist={saveToWatchlist} onDetails={openDetails}
    onReplace={recommendation?replaceRecommendation:undefined}
    onNotInterested={recommendation?rejectRecommendation:undefined}
    onRestore={restorable?restoreInterest:undefined}
    actionsDisabled={recommendation&&recommendationsLocked}
  />);
  // Re-filtered on every render, so a card disappears the moment it is saved or rejected.
  const visibleRecommendations=feed.eligible(recommendations,session);
  const collections={liked:session.liked,want:session.watchlist||[],hidden:session.notInterested||[]};
  const currentCollection=collections[collectionTab];
  const shownCollection=expandCollection?currentCollection:currentCollection.slice(0,COLLECTION_PREVIEW);
  const hiddenTitles=legacyHiddenTitles(session);
  const emptyCollection={
    liked:[S.forYou.emptyLikedTitle,S.forYou.emptyLikedBody],
    want:[S.forYou.emptyWantTitle,S.forYou.emptyWantBody],
    hidden:[S.forYou.emptyNotInterestedTitle,S.forYou.emptyNotInterestedBody],
  }[collectionTab];
  const selectTab=(tab)=>{setCollectionTab(tab);setExpandCollection(false);};
  const recommendationGrid=(items,options,gridClass='anime-grid')=>loadingRec
    ? <p className="loading-grid" role="status">{S.recommendations.preparing}</p>
    : recommendError?<p className="error-message" role="alert">{recommendError}</p>
    : items.length?<div className={`${gridClass} ${refreshing?'is-refreshing':''}`} aria-busy={refreshing}>{renderCards(items,options)}</div>
    : <p className="empty-note">{S.recommendations.empty}</p>;

  return <div className="app-shell">
    <Navbar page={page} navigate={navigate} likeCount={session.liked.length} health={health}/>
    <main>
      {notice&&<div className="global-notice" role="status"><span>{notice}</span><button onClick={()=>setNotice('')} aria-label={S.status.dismissNotice}><X size={16}/></button></div>}
      {healthError&&<div className="backend-warning" role="status">{healthError}</div>}
      {page==='home'&&<>
        <section className="hero">
          <div className="hero-copy"><span className="kicker"><Sparkles size={15}/> {S.home.kicker}</span><BlurText as="h1" text={S.home.headline}/><p>{S.home.intro}</p>
          <form className="hero-search" onSubmit={searchFromHero}><Search size={19}/><input value={heroQuery} onChange={e=>setHeroQuery(e.target.value)} placeholder={S.home.searchPlaceholder} aria-label={S.home.searchLabel}/><button type="submit">{S.home.search} <ArrowRight size={17}/></button></form>
          <div className="hero-actions"><button className="subtle-cta" onClick={()=>navigate('discover')}>{S.home.exploreCatalog} <ArrowUpRight size={15}/></button><button className="subtle-cta" onClick={()=>navigate('agent')}>{S.home.askAgent} <Sparkles size={15}/></button></div>
          </div>
          <div className="hero-art" aria-hidden="true"><span className="deco-circle d1"/><span className="deco-circle d2"/><span className="deco-star">✦</span>{popular.slice(0,3).map((item,i)=>item.image_url&&<img className={`hero-poster hero-poster-${i}`} key={item.id} src={item.image_url} alt=""/>)}</div>
        </section>
        {filtersSection}
        <section className="catalog-section"><SectionTitle eyebrow={S.home.popularEyebrow} title={S.home.popularTitle} caption={S.home.popularCaption} onAction={()=>navigate('discover')} action={S.home.viewAll}/>
          {loadingPopular?<div className="loading-grid" role="status">{S.home.popularLoading}</div>:popular.length?<div className="anime-grid">{renderCards(popular.slice(0,6))}</div>:<p className="empty-note">{S.home.popularEmpty}</p>}
        </section>
        <section className="catalog-section"><SectionTitle eyebrow={S.home.tasteEyebrow} title={session.liked.length?S.home.tasteTitlePersonal:S.home.tasteTitleStarter} caption={session.liked.length?S.home.tasteCaptionPersonal(session.liked.length):S.home.tasteCaptionStarter} onAction={()=>navigate('for-you')} action={S.home.seeMyPicks}/>
          {refreshing&&<p className="batch-note" role="status">{S.recommendations.updating}</p>}
          {recommendationGrid(visibleRecommendations.slice(0,6),{compact:true,recommendation:true},'anime-grid small-grid')}
        </section>
        <section className="agent-cta"><div className="cta-star"><Sparkles size={24}/></div><div><h2>{S.home.agentCtaTitle}</h2><p>{S.home.agentCtaBody}</p></div><button onClick={()=>navigate('agent')}>{S.home.agentCtaButton} <ArrowRight size={17}/></button></section>
      </>}
      {page==='discover'&&<section className="page-panel discover-page"><div className="page-intro"><div className="kicker">{S.discover.kicker}</div><h1>{S.discover.headline}</h1><p>{S.discover.intro}</p></div>
        <form className="page-search" onSubmit={e=>e.preventDefault()}><Search size={20}/><input value={query} onChange={e=>setQuery(e.target.value)} placeholder={S.discover.searchPlaceholder} aria-label={S.discover.searchLabel}/><button type="button" onClick={()=>setQuery('')} aria-label={S.discover.clearSearch}><X size={17}/></button></form>
        {filtersSection}
        <SectionTitle eyebrow={S.discover.resultsEyebrow} title={S.discover.resultsTitle} caption={discoverBusy?S.discover.searching:S.discover.matchCount(total)}/>
        {discoverError&&<p className="error-message" role="alert">{discoverError}</p>}
        <div className="anime-grid">{renderCards(discoverResults)}</div>
        {discoverBusy&&<p className="loading-grid" role="status">{S.discover.loadingResults}</p>}
        {!discoverBusy&&!discoverResults.length&&!discoverError&&<p className="empty-note">{S.discover.empty}</p>}
        {more&&<button className="load-more" disabled={discoverBusy} onClick={loadMore}>{discoverBusy?S.discover.loading:S.discover.showMore} <ArrowRight size={15}/></button>}
      </section>}
      {page==='for-you'&&<section className="page-panel profile-page"><div className="page-intro"><div className="kicker"><Heart size={14}/> {S.forYou.kicker}</div><h1>{S.forYou.headline}</h1><p>{S.forYou.intro}</p></div>
        <h2 className="collection-title">{S.forYou.collectionTitle}</h2>
        <div className="profile-heading">
          <div className="collection-tabs" role="tablist" aria-label={S.forYou.collectionTabsLabel}>
            <button role="tab" aria-selected={collectionTab==='liked'} className={collectionTab==='liked'?'selected':''} onClick={()=>selectTab('liked')}><Heart size={15}/> {S.forYou.tabLiked(session.liked.length)}</button>
            <button role="tab" aria-selected={collectionTab==='want'} className={collectionTab==='want'?'selected':''} onClick={()=>selectTab('want')}><Bookmark size={15}/> {S.forYou.tabWantToWatch(collections.want.length)}</button>
            <button role="tab" aria-selected={collectionTab==='hidden'} className={collectionTab==='hidden'?'selected':''} onClick={()=>selectTab('hidden')}><ThumbsDown size={15}/> {S.forYou.tabNotInterested(collections.hidden.length+hiddenTitles.length)}</button>
          </div>
          <div className="backup-actions"><button onClick={exportBackup}>{S.forYou.exportBackup}</button><button onClick={()=>fileInput.current?.click()}>{S.forYou.importBackup}</button><input ref={fileInput} type="file" accept="application/json,.json" hidden onChange={e=>importBackup(e.target.files?.[0])}/></div>
        </div>
        {currentCollection.length?<>
          <div className="anime-grid liked-grid">{renderCards(shownCollection,{restorable:collectionTab==='hidden'})}</div>
          {currentCollection.length>COLLECTION_PREVIEW&&<div className="collection-expand"><button onClick={()=>setExpandCollection(v=>!v)} aria-expanded={expandCollection}>{expandCollection?S.forYou.showLess:S.forYou.showMore(currentCollection.length-COLLECTION_PREVIEW)} <ChevronDown className={expandCollection?'rotated':''} size={16}/></button></div>}
        </>:!(collectionTab==='hidden'&&hiddenTitles.length)&&<div className="empty-profile"><Compass size={32}/><h2>{emptyCollection[0]}</h2><p>{emptyCollection[1]}</p><button className="primary-button" onClick={()=>navigate('discover')}>{S.forYou.exploreAnime} <ArrowRight size={16}/></button></div>}
        {collectionTab==='hidden'&&hiddenTitles.length>0&&<div className="hidden-titles"><h3>{S.forYou.hiddenByTitleHeading}</h3><p>{S.forYou.hiddenByTitleBody}</p><ul>{hiddenTitles.map(title=><li key={title}><span>{title}</span><button onClick={()=>commit(clearHiddenTitle(sessionRef.current,title))} aria-label={S.actions.notInterestedRemoveLabel(title)} title={S.actions.notInterestedRemove}><X size={13}/></button></li>)}</ul></div>}
        <section className="catalog-section">
          <div className="recommend-heading"><SectionTitle eyebrow={S.forYou.recommendationsEyebrow} title={session.liked.length?S.forYou.recommendationsTitlePersonal:S.forYou.recommendationsTitleStarter} caption={session.liked.length?S.forYou.recommendationsCaptionPersonal(BATCH_SIZE):S.forYou.recommendationsCaptionStarter}/>
          <button className="refresh-batch" disabled={recBusy||recommendationsLocked||!!recommendError} onClick={nextBatch} title={S.forYou.nextBatchHint}><RefreshCw className={recBusy?'spinning':''} size={17}/> {recBusy?S.forYou.loading:S.forYou.nextBatch}</button></div>
          {profileNotice&&<p className="batch-note" role="status">{profileNotice}</p>}
          {refreshing&&<p className="batch-note" role="status">{S.recommendations.updating}</p>}
          {recNotice&&<p className="batch-note" role="status"><span>{recNotice.text}</span>{recNotice.undo&&<button onClick={recNotice.undo}>{S.actions.undo}</button>}</p>}
          {recommendationGrid(visibleRecommendations,{recommendation:true})}
        </section>
      </section>}
      {page==='agent'&&<AgentPage session={session} messages={messages} setMessages={setMessages} syncProfile={syncRef.current.push}/>}
    </main>
    <footer className="site-footer"><span><Compass size={15}/> {S.footer.brand}</span><span>{S.footer.note}</span></footer>
    {detail&&<div className="modal-layer" role="presentation" onMouseDown={e=>{if(e.target===e.currentTarget)setDetail(null);}}><div className="modal-card" role="dialog" aria-modal="true" aria-label={S.details.dialogLabel(detail.title)}><button className="modal-close" onClick={()=>setDetail(null)} aria-label={S.details.close}><X size={19}/></button><div className="modal-content"><div className="modal-poster">{detail.image_url&&<img src={detail.image_url} alt={S.actions.posterAlt(detail.title)}/>}</div><div className="modal-description"><span className="kicker">{S.details.kicker}</span><h2>{detail.title}</h2><p className="modal-meta">{[detail.type,detail.start_year,detail.episodes?S.details.episodes(detail.episodes):null].filter(Boolean).join(' · ')}</p>{detail.score&&<p className="modal-score">{S.details.score(Number(detail.score).toFixed(2))}</p>}{detail.genres?.length>0&&<div className="genre-tags">{detail.genres.slice(0,6).map(g=><span key={g}>{g}</span>)}</div>}<p className="synopsis">{detail.synopsis||detail.description||S.details.noSynopsis}</p>{detailBusy&&<small>{S.details.loadingMore}</small>}<div className="modal-actions"><button className="primary-button" aria-pressed={likedIds.has(Number(detail.id))} onClick={()=>likeAnime(detail)}>{likedIds.has(Number(detail.id))?<Check size={17}/>:<Heart size={17}/>} {likedIds.has(Number(detail.id))?S.actions.liked:S.actions.like}</button><button className="secondary-button" aria-pressed={watchIds.has(Number(detail.id))} onClick={()=>saveToWatchlist(detail)}><Bookmark size={17} fill={watchIds.has(Number(detail.id))?'currentColor':'none'}/> {watchIds.has(Number(detail.id))?S.actions.wantToWatchAdded:S.actions.wantToWatch}</button></div></div></div></div></div>}
  </div>;
}
