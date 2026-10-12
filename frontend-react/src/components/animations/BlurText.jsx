import React from 'react';
/** React Bits-inspired blur-word reveal. Lightweight CSS variant, no animation runtime. */
export function BlurText({text, className='', as:Tag='span', delay=65}) {
  const words=String(text).split(/\s+/).filter(Boolean);
  return <Tag className={`blur-text ${className}`} aria-label={text}>
    {words.map((word,i)=><React.Fragment key={`${i}-${word}`}><span aria-hidden="true" className="blur-word" style={{'--blur-delay':`${i*delay}ms`}}>{word}</span>{i<words.length-1?' ':null}</React.Fragment>)}
  </Tag>;
}
