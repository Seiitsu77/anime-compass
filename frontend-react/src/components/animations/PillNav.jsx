import React, {useEffect,useRef,useState} from 'react';
/** React Bits-inspired sliding pill indicator with keyboard-friendly real buttons. */
export function PillNav({items,value,onChange,label}) {
  const nav=useRef(null);
  const nodes=useRef(new Map());
  const [bounds,setBounds]=useState(null);
  useEffect(()=>{
    function move() {
      const root=nav.current, target=nodes.current.get(value);
      if(!root || !target)return;
      setBounds({left:target.offsetLeft,top:target.offsetTop,width:target.offsetWidth,height:target.offsetHeight});
    }
    move();
    const observer=typeof ResizeObserver==='function'?new ResizeObserver(move):null;
    if(nav.current) observer?.observe(nav.current);
    window.addEventListener('resize',move);
    return()=>{observer?.disconnect();window.removeEventListener('resize',move);};
  },[value,items]);
  return <nav className="main-nav pill-nav" ref={nav} aria-label={label}>
    {bounds&&<span className="pill-nav-active-track" aria-hidden="true" style={{left:bounds.left,top:bounds.top,width:bounds.width,height:bounds.height}}/>}
    {items.map(([id,label])=><button ref={node=>{if(node)nodes.current.set(id,node);else nodes.current.delete(id);}} key={id} onClick={()=>onChange(id)} className={value===id?'active':''} aria-current={value===id?'page':undefined}>{label}</button>)}
  </nav>;
}
