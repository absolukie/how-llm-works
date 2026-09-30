/* How an LLM Works — all client-side. No network, no keys. */
(function(){
"use strict";

/* ---------- accordions: tap toggles, never scrolls ---------- */
document.querySelectorAll(".node-head, .sub-head, .deeper-toggle").forEach(function(btn){
  btn.addEventListener("click", function(){
    var target = btn.getAttribute("aria-controls")
      ? document.getElementById(btn.getAttribute("aria-controls"))
      : btn.nextElementSibling;
    if(!target) return;
    var open = target.hasAttribute("hidden");
    if(open){ target.removeAttribute("hidden"); }
    else{ target.setAttribute("hidden", ""); }
    btn.setAttribute("aria-expanded", open ? "true" : "false");
  });
});

/* ============================================================
   DEMO 1 · tiny real BPE tokenizer (trained in-browser)
   ============================================================ */
var BPE_CORPUS = [
  "the cat sat on the mat","the dog ran in the park","a bird sang in the tree",
  "the fish swam in the pond","she read a book","he kicked the ball",
  "the sun is hot","the moon is bright","we like to play","they went to school",
  "i love my family","the baby laughed","birds can fly","fish can swim",
  "dogs bark loudly","cats sleep all day","the car is red","a big green tree",
  "she sings a song","he draws a picture","we eat lunch","they drink water",
  "the stars shine","rain falls down","snow is white","fire is hot",
  "ice is cold","the wind blows","leaves fall down","flowers bloom","bees make honey"
];
var SEP = "\u0001";

function trainBPE(lines, numMerges){
  var freq = new Map();
  lines.forEach(function(line){
    line.split(/\s+/).forEach(function(w){
      if(w) freq.set(w, (freq.get(w)||0)+1);
    });
  });
  var splits = new Map();
  freq.forEach(function(f, w){ splits.set(w, w.split("").concat(["</w>"])); });
  var vocab = new Map(), id = 0;
  var chars = new Set(["</w>"]);
  freq.forEach(function(f, w){ w.split("").forEach(function(ch){ chars.add(ch); }); });
  Array.from(chars).sort().forEach(function(ch){ vocab.set(ch, id++); });
  var ranks = new Map();
  for(var m=0; m<numMerges; m++){
    var pairCounts = new Map();
    freq.forEach(function(f, w){
      var syms = splits.get(w);
      for(var i=0;i<syms.length-1;i++){
        var p = syms[i]+SEP+syms[i+1];
        pairCounts.set(p, (pairCounts.get(p)||0)+f);
      }
    });
    var best=null, bestC=0;
    pairCounts.forEach(function(c,p){ if(c>bestC){bestC=c;best=p;} });
    if(!best) break;
    ranks.set(best, ranks.size);
    var ab = best.split(SEP), a=ab[0], b=ab[1], merged=a+b;
    vocab.set(merged, id++);
    freq.forEach(function(f, w){
      var syms=splits.get(w), out=[];
      for(var i=0;i<syms.length;){
        if(i<syms.length-1 && syms[i]===a && syms[i+1]===b){ out.push(merged); i+=2; }
        else{ out.push(syms[i]); i+=1; }
      }
      splits.set(w,out);
    });
  }
  return {vocab:vocab, ranks:ranks};
}

function bpeEncodeWord(word, ranks){
  var syms = word.split("").concat(["</w>"]);
  for(;;){
    var bi=-1, br=Infinity;
    for(var i=0;i<syms.length-1;i++){
      var r = ranks.get(syms[i]+SEP+syms[i+1]);
      if(r!==undefined && r<br){ br=r; bi=i; }
    }
    if(bi<0) break;
    syms.splice(bi, 2, syms[bi]+syms[bi+1]);
  }
  return syms;
}

var bpeModel = trainBPE(BPE_CORPUS, 48);

function esc(s){ return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

var tokInput=document.getElementById("tok-input"),
    tokOut=document.getElementById("tok-out"),
    tokStats=document.getElementById("tok-stats");

function renderTokens(){
  var text = tokInput.value;
  var toks=[];
  text.toLowerCase().split(/\s+/).forEach(function(w){
    var clean=w.replace(/[^a-z]/g,"");
    if(!clean) return;
    bpeEncodeWord(clean, bpeModel.ranks).forEach(function(s){
      toks.push({t:s, id:bpeModel.vocab.get(s)});
    });
  });
  if(!toks.length){
    tokOut.innerHTML='<span style="color:#5f6b88;font-size:13px">type a–z letters above…</span>';
    tokStats.textContent="";
    return;
  }
  tokOut.innerHTML = toks.map(function(tk){
    return '<span class="tok">'+esc(tk.t.replace("</w>",""))+'<i>'+tk.id+'</i></span>';
  }).join("");
  var chars=text.replace(/\s/g,"").replace(/[^a-zA-Z]/g,"").length;
  tokStats.innerHTML = "<b>"+toks.length+"</b> tokens · "+chars+" letters · vocab <b>"+bpeModel.vocab.size+"</b> pieces · "+
    "decodes back to: “"+esc(toks.map(function(tk){return tk.t;}).join("").replace(/<\/w>/g," ").trim())+"”";
}
var tokT=null;
tokInput.addEventListener("input", function(){ clearTimeout(tokT); tokT=setTimeout(renderTokens,120); });
renderTokens();

/* ============================================================
   DEMO 2 · real scaled dot-product attention, toy embeddings
   ============================================================ */
function mulberry32(a){
  return function(){
    a|=0; a=a+0x6D2B79F5|0;
    var t=Math.imul(a^a>>>15,1|a);
    t=t+Math.imul(t^t>>>7,61|t)^t;
    return ((t^t>>>14)>>>0)/4294967296;
  };
}
var ATTN_SENTS=[
  "the cat sat on the mat".split(" "),
  "the dog chased its tail".split(" "),
  "she gave him her umbrella".split(" ")
];
var ATTN_D=8, attnSeed=7, attnIdx=0, attnW=null, attnWords=ATTN_SENTS[0];

function matVec(M,v){
  return M.map(function(row){
    var s=0; for(var i=0;i<v.length;i++) s+=row[i]*v[i]; return s;
  });
}
function computeAttention(words, seed){
  var rnd=mulberry32(seed*2654435761>>>0 || 1);
  var n=words.length, d=ATTN_D;
  function rmat(){ var M=[]; for(var i=0;i<d;i++){var r=[];for(var j=0;j<d;j++)r.push(rnd()*2-1);M.push(r);} return M; }
  function rvec(){ var v=[]; for(var j=0;j<d;j++)v.push(rnd()*2-1); return v; }
  var Wq=rmat(), Wk=rmat();
  var X=words.map(rvec);
  var Q=X.map(function(x){return matVec(Wq,x);});
  var K=X.map(function(x){return matVec(Wk,x);});
  var W=[];
  for(var i=0;i<n;i++){
    var scores=[];
    for(var j=0;j<n;j++){
      if(j>i){ scores.push(-Infinity); continue; } /* causal mask */
      var s=0; for(var k=0;k<d;k++) s+=Q[i][k]*K[j][k];
      scores.push(s/Math.sqrt(d));
    }
    var m=Math.max.apply(null,scores.filter(isFinite));
    var ex=scores.map(function(s){return s===-Infinity?0:Math.exp(s-m);});
    var sum=ex.reduce(function(a,b){return a+b;},0);
    W.push(ex.map(function(e){return e/sum;}));
  }
  return W;
}

var heat=document.getElementById("attn-heat"),
    read=document.getElementById("attn-read");

function renderAttention(){
  attnWords=ATTN_SENTS[attnIdx];
  attnW=computeAttention(attnWords, attnSeed);
  var n=attnWords.length;
  heat.style.gridTemplateColumns="54px repeat("+n+", 1fr)";
  var html='<div class="hcell corner"></div>';
  attnWords.forEach(function(w){ html+='<div class="hcell collab">'+esc(w)+'</div>'; });
  for(var i=0;i<n;i++){
    html+='<div class="hcell rowlab">'+esc(attnWords[i])+'</div>';
    for(var j=0;j<n;j++){
      var wgt=attnW[i][j], masked=j>i;
      html+='<button type="button" class="hcell" data-i="'+i+'" data-j="'+j+'" '+
        'style="background:rgba(233,161,59,'+(masked?0.02:(wgt*0.92).toFixed(3))+')" '+
        'aria-label="attention from '+esc(attnWords[i])+' to '+esc(attnWords[j])+'">'+
        (masked?"·":wgt.toFixed(2))+'</button>';
    }
  }
  heat.innerHTML=html;
  read.textContent="Tap any cell to read its weight.";
}
heat.addEventListener("click", function(e){
  var c=e.target.closest(".hcell[data-i]");
  if(!c||!attnW) return;
  var i=+c.getAttribute("data-i"), j=+c.getAttribute("data-j"), w=attnW[i][j];
  if(j>i){ read.textContent="“"+attnWords[i]+"” → “"+attnWords[j]+"”: masked — a token can never see the future."; }
  else{
    var pct=(w*100).toFixed(1);
    read.textContent="“"+attnWords[i]+"” attends to “"+attnWords[j]+"” with weight "+w.toFixed(3)+" ("+pct+"% of its mix).";
  }
});
document.getElementById("attn-sentences").addEventListener("click", function(e){
  var b=e.target.closest("button"); if(!b) return;
  this.querySelectorAll("button").forEach(function(x){x.classList.remove("on");});
  b.classList.add("on");
  attnIdx=+b.getAttribute("data-s");
  renderAttention();
});
document.getElementById("attn-shuffle").addEventListener("click", function(){
  attnSeed=(attnSeed*1103515245+12345)>>>0;
  renderAttention();
});
renderAttention();

/* ============================================================
   DEMO 3 · sampling lab: real softmax + temperature + top-k
   ============================================================ */
var CANDS=[[" the",2.2],[" cat",1.5],[" dog",1.2],[" mat",0.8],
           [" moon",0.1],[" zebra",-0.6],[" quantum",-1.2],[" banana",-2.1]];
var sampTemp=document.getElementById("samp-temp"),
    sampTopk=document.getElementById("samp-topk"),
    tmpVal=document.getElementById("tmp-val"),
    topkVal=document.getElementById("topk-val"),
    bars=document.getElementById("samp-bars"),
    pick=document.getElementById("samp-pick"),
    seq=document.getElementById("samp-seq");

function softmax(logits){
  var m=Math.max.apply(null,logits);
  var ex=logits.map(function(z){return Math.exp(z-m);});
  var s=ex.reduce(function(a,b){return a+b;},0);
  return ex.map(function(e){return e/s;});
}
function currentProbs(){
  var k=+sampTopk.value, T=+sampTemp.value;
  var order=CANDS.map(function(c,i){return i;})
    .sort(function(a,b){return CANDS[b][1]-CANDS[a][1];}).slice(0,k);
  var keep={}; order.forEach(function(i){keep[i]=1;});
  var logits=CANDS.map(function(c,i){return keep[i]?c[1]/T:-Infinity;});
  var finite=softmax(logits.map(function(z){return z===-Infinity?-1e9:z;}));
  /* softmax with -1e9 approximates masking; renormalize over kept only */
  var probs=CANDS.map(function(c,i){
    if(!keep[i]) return 0;
    return finite[i];
  });
  var s=probs.reduce(function(a,b){return a+b;},0);
  return {probs:probs.map(function(p){return p/s;}), keep:keep};
}
function renderBars(){
  var r=currentProbs();
  tmpVal.textContent=(+sampTemp.value).toFixed(2);
  topkVal.textContent=sampTopk.value;
  bars.innerHTML=CANDS.map(function(c,i){
    var p=r.probs[i], cut=!r.keep[i];
    return '<div class="bar-row'+(cut?' cut':'')+'">'+
      '<span class="tok">'+esc(c[0])+'</span>'+
      '<span class="bar-track"><span class="bar-fill" style="width:'+(p*100).toFixed(1)+'%"></span></span>'+
      '<span class="bar-pct">'+(cut?"cut":(p*100).toFixed(1)+"%")+'</span></div>';
  }).join("");
  return r;
}
function drawSample(probs){
  var r=Math.random(), acc=0;
  for(var i=0;i<probs.length;i++){ acc+=probs[i]; if(r<=acc) return i; }
  return probs.length-1;
}
function sampleOnce(){
  var r=renderBars(), i=drawSample(r.probs);
  pick.innerHTML="Drew <b>“"+esc(CANDS[i][0])+"”</b> <span style='color:#98a0b3'>(p="+(r.probs[i]*100).toFixed(1)+"%)</span>";
  var chip=document.createElement("span");
  chip.className="tok"; chip.textContent=CANDS[i][0];
  seq.appendChild(chip);
  while(seq.children.length>48) seq.removeChild(seq.firstChild);
}
sampTemp.addEventListener("input", renderBars);
sampTopk.addEventListener("input", renderBars);
document.getElementById("samp-once").addEventListener("click", sampleOnce);
document.getElementById("samp-20").addEventListener("click", function(){ for(var n=0;n<20;n++) sampleOnce(); });
document.getElementById("samp-clear").addEventListener("click", function(){
  seq.innerHTML=""; pick.textContent="Press “Sample once”.";
});
renderBars();

})();
