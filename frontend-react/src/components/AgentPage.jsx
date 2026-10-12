import React, { useState, useEffect, useRef } from 'react';
import { Send, Sparkles, CornerDownLeft } from 'lucide-react';
import { api } from '../lib/api.js';
import { S } from '../lib/strings.js';

export function AgentPage({ session, messages, setMessages, syncProfile }) {
  const [input,setInput]=useState('');
  const [pending,setPending]=useState(false);
  const [error,setError]=useState('');
  const bottom=useRef(null);
  // Braces matter: scrollIntoView() returns a Promise in current Chromium, and an
  // effect that returns one makes React call it as a clean-up and crash the app.
  useEffect(()=>{bottom.current?.scrollIntoView({block:'nearest',behavior:'smooth'});},[messages,pending]);
  async function send(text) {
    const message=text.trim();
    if (!message || pending) return;
    const history=messages.slice(-8).map(({role,content})=>({role,content}));
    setMessages(prev=>[...prev,{role:'user',content:message}]);
    setInput('');setError('');setPending(true);
    try {
      // Ensure the stateless browser's profile is available after Cloud Run scale-to-zero.
      await syncProfile(session);
      const response=await api('/api/agent', {
        method:'POST',
        body:JSON.stringify({message,history,session_id:session.sessionId}),
      });
      setMessages(prev=>[...prev,{role:'assistant',content:response.answer || S.agent.noResponse}]);
    } catch(e) {setError(e.message || S.agent.failed);}
    finally {setPending(false);}
  }
  return <section className="agent-page page-panel">
    <div className="page-intro"><div className="kicker"><Sparkles size={15}/> {S.agent.kicker}</div><h1>{S.agent.headline}</h1><p>{S.agent.intro}</p></div>
    <div className="chat-area" aria-live="polite">
      {messages.length===0 && <div className="agent-welcome"><div className="sparkle-orb"><Sparkles size={24}/></div><h2>{S.agent.welcomeTitle}</h2><p>{S.agent.welcomeBody}</p><div className="suggestions">{S.agent.examples.map(x=><button key={x} onClick={()=>send(x)} disabled={pending}>{x}</button>)}</div></div>}
      {messages.map((m,i)=><div key={i} className={`chat-bubble ${m.role==='user'?'user':'assistant'}`}><span className="chat-speaker">{m.role==='user'?S.agent.speakerUser:S.agent.speakerAgent}</span><p>{m.content}</p></div>)}
      {pending && <div className="chat-bubble assistant busy"><span className="chat-speaker">{S.agent.speakerAgent}</span><p>{S.agent.thinking}</p></div>}
      <div ref={bottom}/>
    </div>
    {error && <p role="alert" className="error-message">{error} <button onClick={()=>setError('')}>{S.agent.dismiss}</button></p>}
    <form className="chat-form" onSubmit={e=>{e.preventDefault();send(input);}}><input value={input} onChange={e=>setInput(e.target.value)} placeholder={S.agent.inputPlaceholder} maxLength={2000} aria-label={S.agent.inputLabel}/><button type="submit" disabled={pending||!input.trim()} aria-label={S.agent.send}><Send size={18}/></button></form>
    <p className="chat-footnote"><CornerDownLeft size={12}/> {S.agent.footnote}</p>
  </section>;
}
