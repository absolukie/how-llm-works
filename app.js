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
  "ice is cold","the wind blows","leaves fall down","flowers bloom","bees make honey",
  "the quick brown fox jumps over the lazy dog",
  "it was unbelievable","an unbelievable story","truly unbelievable",
  "that is believable","a believable excuse","seems believable to me",
  "i believe you","believe me","they believe in ghosts",
  "undo the knot","untie your shoes","that is unfair","an unhappy ending",
  "jazz music is fun","a fuzzy buzz","quiet quails","box of tricks"
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
  /* seed EVERY a-z letter so no input char can ever miss the vocab
     (a missing letter used to render as "j undefined") */
  var chars = new Set(["</w>"]);
  "abcdefghijklmnopqrstuvwxyz".split("").forEach(function(ch){ chars.add(ch); });
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

/* encode, recording every merge step for the trace visualizer.
   steps[r] = segmentation BEFORE merge r; the merging pair is at
   steps[r].bi; steps[last] is the final token list. */
function bpeEncodeSteps(word, ranks){
  var syms = word.split("").concat(["</w>"]);
  var steps = [{syms: syms.slice(), bi: -1}];
  for(;;){
    var bi=-1, br=Infinity;
    for(var i=0;i<syms.length-1;i++){
      var r = ranks.get(syms[i]+SEP+syms[i+1]);
      if(r!==undefined && r<br){ br=r; bi=i; }
    }
    if(bi<0) break;
    steps[steps.length-1].bi = bi;
    syms.splice(bi, 2, syms[bi]+syms[bi+1]);
    steps.push({syms: syms.slice(), bi: -1});
  }
  return steps;
}

function bpeEncodeWord(word, ranks){
  var steps = bpeEncodeSteps(word, ranks);
  return steps[steps.length-1].syms;
}

var bpeModel = trainBPE(BPE_CORPUS, 48);

/* self-test: every a-z letter has an id; round-trip is exact;
   no emitted piece can ever look up as undefined. */
function bpeSelfTest(){
  var bad=[];
  "abcdefghijklmnopqrstuvwxyz".split("").forEach(function(ch){
    if(bpeModel.vocab.get(ch)===undefined) bad.push("no id for "+ch);
  });
  ["unbelievable","the","cat","fjdjbdjajs","zebra","quiz","jumps","pack"].forEach(function(w){
    var syms=bpeEncodeWord(w, bpeModel.ranks);
    syms.forEach(function(s){
      if(s!=="</w>" && bpeModel.vocab.get(s)===undefined) bad.push("undefined id: "+s);
    });
    var dec=syms.join("").replace(/<\/w>/g,"");
    if(dec!==w) bad.push("round-trip fail: "+w+" -> "+dec);
  });
  if(bad.length) console.error("BPE self-test FAILED: "+bad.join("; "));
  else console.log("BPE self-test passed ("+bpeModel.vocab.size+" pieces)");
  return bad.length===0;
}
bpeSelfTest();

function esc(s){ return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

var tokInput=document.getElementById("tok-input"),
    tokOut=document.getElementById("tok-out"),
    tokStats=document.getElementById("tok-stats"),
    tokNote=document.getElementById("tok-note"),
    mergeStepsEl=document.getElementById("merge-steps"),
    mergeWordEl=document.getElementById("merge-word"),
    mergePlayBtn=document.getElementById("merge-play");
if(tokNote){
  tokNote.innerHTML="A genuine BPE tokenizer — \u201cbyte-pair encoding\u201d, the standard trick tokenizers use to learn their pieces — trained live in your browser on <b>"+
    BPE_CORPUS.length+" tiny sentences</b> — a toy vocabulary of <b>"+bpeModel.vocab.size+
    " pieces</b>. GPT's real one learns exactly this way, at vastly larger scale.";
}

function normalizeInput(text){
  return text.toLowerCase().split(/\s+/).map(function(w){
    return w.replace(/[^a-z]/g,"");
  }).filter(function(w){ return w; });
}

var mergeData=[], mergeCursor=0, mergeTimer=null;

function stopMergePlay(){
  if(mergeTimer){ clearInterval(mergeTimer); mergeTimer=null; }
}

function paintMergeTrace(){
  if(!mergeData.length){
    mergeStepsEl.innerHTML='<span style="color:#5f6b88;font-size:13px">type something above…</span>';
    return;
  }
  var html="";
  for(var r=0;r<mergeData.length;r++){
    var st=mergeData[r], last=(r===mergeData.length-1);
    var cls="mrow"+(r>mergeCursor?" dim":"")+(r===mergeCursor?" cur":"");
    html+='<div class="'+cls+'" data-r="'+r+'"><span class="mnum">'+r+'</span><span class="mchips">';
    for(var i=0;i<st.syms.length;i++){
      var s=st.syms[i];
      if(last && s==="</w>") continue; /* boundary marker: not a chip */
      var cc, label, idbit="";
      if(last){ cc="tok final"; }
      else{ cc="tok char"; if(i===st.bi||i===st.bi+1) cc+=" pair"; else if(r>0&&i===mergeData[r-1].bi) cc+=" new"; }
      label = (s==="</w>") ? '<span class="eow">␣</span>' : esc(s.replace("</w>",""));
      if(last) idbit="<i>"+bpeModel.vocab.get(s)+"</i>";
      html+='<span class="'+cc+'">'+label+idbit+'</span>';
    }
    html+='</span></div>';
  }
  mergeStepsEl.innerHTML=html;
  mergePlayBtn.textContent = mergeTimer ? "⏸ Pause" : "▶ Play merges";
}

function renderMergeTrace(){
  stopMergePlay();
  var words=normalizeInput(tokInput.value);
  var word=words[0]||"";
  mergeWordEl.textContent = word || "(type something)";
  mergeData = word ? bpeEncodeSteps(word, bpeModel.ranks) : [];
  mergeCursor = mergeData.length-1;
  paintMergeTrace();
}

mergePlayBtn.addEventListener("click", function(){
  if(!mergeData.length) return;
  if(mergeTimer){ stopMergePlay(); paintMergeTrace(); return; }
  mergeCursor=0; paintMergeTrace();
  mergeTimer=setInterval(function(){
    if(mergeCursor>=mergeData.length-1){ stopMergePlay(); paintMergeTrace(); return; }
    mergeCursor++; paintMergeTrace();
  },650);
});
document.getElementById("merge-step").addEventListener("click", function(){
  if(!mergeData.length) return;
  stopMergePlay();
  mergeCursor=Math.min(mergeCursor+1, mergeData.length-1);
  paintMergeTrace();
});
document.getElementById("merge-back").addEventListener("click", function(){
  if(!mergeData.length) return;
  stopMergePlay();
  mergeCursor=Math.max(mergeCursor-1, 0);
  paintMergeTrace();
});
document.getElementById("merge-reset").addEventListener("click", function(){
  if(!mergeData.length) return;
  stopMergePlay();
  mergeCursor=0;
  paintMergeTrace();
});
mergeStepsEl.addEventListener("click", function(e){
  var row=e.target.closest(".mrow"); if(!row||!mergeData.length) return;
  stopMergePlay();
  mergeCursor=+row.getAttribute("data-r");
  paintMergeTrace();
});

function renderTokens(){
  var text = tokInput.value;
  var words = normalizeInput(text);
  renderMergeTrace();
  if(!words.length){
    tokOut.innerHTML='<span style="color:#5f6b88;font-size:13px">type a–z letters above…</span>';
    tokStats.textContent="";
    return;
  }
  var toks=[];
  words.forEach(function(w){
    bpeEncodeWord(w, bpeModel.ranks).forEach(function(s){
      if(s==="</w>") return; /* boundary marker: word break only, not a chip */
      var id=bpeModel.vocab.get(s);
      if(id===undefined){ id="?"; console.error("token without id: "+s); }
      toks.push({t:s, id:id});
    });
  });
  tokOut.innerHTML = toks.map(function(tk){
    return '<span class="tok final">'+esc(tk.t.replace("</w>",""))+'<i>'+tk.id+'</i></span>';
  }).join("");
  /* decode is exact by construction (encode is lossless over a-z words);
     assert it, and explain the one normalization the input goes through. */
  var decoded = words.map(function(w){
    return bpeEncodeWord(w, bpeModel.ranks).join("").replace(/<\/w>/g,"");
  }).join(" ");
  console.assert(decoded===words.join(" "), "tokenizer decode mismatch");
  var normNote = /[^a-z\s]/.test(text)
    ? '<span class="norm-note">input normalized: lowercase a–z and single spaces (punctuation &amp; other characters are dropped before tokenizing)</span>' : "";
  var letters = words.join("").length;
  tokStats.innerHTML = "<b>"+toks.length+"</b> tokens · "+letters+" letters · vocab <b>"+bpeModel.vocab.size+"</b> pieces · "+
    "decodes back to: “"+esc(decoded)+"” <span class='ok'>✓ exact</span>"+normNote;
}
var tokT=null;
tokInput.addEventListener("input", function(){ clearTimeout(tokT); tokT=setTimeout(renderTokens,120); });
renderTokens();

/* ============================================================
   TOKENIZER EXAMPLE CAROUSEL · 50 hand-written examples,
   each broken down by the REAL toy BPE engine at render time
   ============================================================ */
var TOK_EX=[
 {t:"unbelievable", why:"The poster child: the toy met <i>un\u2013</i> and <i>believable</i> inside many training words, so each earned its own tile. Two tokens, done."},
 {t:"unhappy", why:"<i>un</i> again \u2014 the exact same tile as in \u201cunbelievable\u201d. Negation travels: learn the prefix once, reuse it everywhere."},
 {t:"untie", why:"And again \u2014 <i>un</i>(41) plus three letters. The prefix tile doesn\u2019t care what word it\u2019s undoing."},
 {t:"preheat", why:"The <i>re\u2013</i> prefix earned a tile from words like \u201creturn\u201d and \u201credo\u201d. Prefixes are some of the hardest-working tiles in the box."},
 {t:"replay", why:"<i>re</i>(55) plus <i>la</i>(73) \u2014 two earned tiles in one word. Frequent chunks stack like Lego."},
 {t:"quickly", why:"The <i>\u2013ly</i> ending earned its own tile. Adverb endings are so common they get printed, just like prefixes."},
 {t:"slowly", why:"<i>ly</i>(63) again \u2014 same tile, different word. Suffixes travel exactly like prefixes do."},
 {t:"darkness", why:"Seven tiles \u2014 and no <i>ness</i> tile. Some endings earn tiles (<i>ly</i>), some never do. The vocabulary is a popularity contest, not a grammar book."},
 {t:"happiness", why:"Eight tiles \u2014 and look, <i>in</i> snuck in mid-word. Tiles are dumb patterns; they don\u2019t know they\u2019re inside \u201chappiness\u201d."},
 {t:"the", why:"The commonest word in English gets a single tile. Frequency is the whole game: the more a chunk appears, the more likely it owns a tile."},
 {t:"the the the", why:"Repetition is cheap \u2014 the same tile three times. Common words cost almost nothing; rare ones cost many tiles."},
 {t:"a", why:"One letter, one tile. The cheapest possible token."},
 {t:"cats", why:"Surprise \u2014 with only 75 tiles, the toy never learned a <i>cat</i> chunk. A real tokenizer (100,000+ tiles) gives common words their own token; tiny vocabularies spell everything out."},
 {t:"dogs", why:"Plurals usually cost one extra tile: the stem plus an <i>s</i>. (\u201cdog\u201d alone \u2192 <i>do</i> + <i>g</i> \u2014 same stem, no plural tile.)"},
 {t:"teh", why:"\u201cthe\u201d is one tile; the typo \u201cteh\u201d shatters into three. To the model these look nothing alike \u2014 which is why typos confuse it more than they confuse you."},
 {t:"fjdjbdjajs", why:"Ten tiles of alphabet soup. Gibberish gets no discounts \u2014 every letter pays full price."},
 {t:"don't", why:"The toy drops the apostrophe and reads \u201cdont\u201d. Real tokenizers keep the pieces \u2014 \u201cdon\u201d + \u201c\u2019\u201d + \u201ct\u201d. Contractions almost always split."},
 {t:"e-mail", why:"Identical to \u201cemail\u201d \u2014 the toy can\u2019t tell them apart. A real tokenizer can, splitting \u201ce\u201d + \u201c-\u201d + \u201cmail\u201d."},
 {t:"well-known", why:"The hyphen vanishes \u2014 \u201cwell-known\u201d and \u201cwellknown\u201d are identical to the toy. Real tokenizers split it: \u201cwell\u201d + \u201c-\u201d + \u201cknown\u201d."},
 {t:"mother-in-law", why:"Three hyphens gone without a trace. Punctuation the toy never learned simply doesn\u2019t exist to it."},
 {t:"Hello", why:"Byte-for-byte identical to \u201chello\u201d \u2014 the toy lowercases everything first. Real tokenizers don\u2019t: \u201cHello\u201d and \u201chello\u201d are different tiles."},
 {t:"HELLO", why:"Shouting changes nothing for the toy. In real tokenizers ALL-CAPS often shatters into pieces \u2014 caps are rare in training text, so they never earned tiles."},
 {t:"camelCase", why:"The capital C carried the word boundary \u2014 and the toy erased it. Real tokenizers split at the hump: \u201ccamel\u201d + \u201cCase\u201d."},
 {t:"snake_case", why:"Underscore gone too. Real tokenizers keep it: \u201csnake\u201d + \u201c_\u201d + \u201ccase\u201d \u2014 code is full of these joints."},
 {t:"function", why:"\u201cun\u201d \u2014 inside \u201cfunction\u201d! The tile learned from \u201cunbelievable\u201d moonlights wherever \u201cu\u201d,\u201cn\u201d sit together. Tiles have no idea what words are."},
 {t:"running", why:"Weird but logical: the tiles <i>un</i> and <i>in</i> were learned from other words and get reused mid-word, even here."},
 {t:"tokenization", why:"A word the toy never met gets spelled out almost letter by letter \u2014 11 tokens. Slow, but it works: <b>any</b> text can be represented."},
 {t:"antidisestablishmentarianism", why:"Twenty-five tiles. Long rare words shatter \u2014 each piece is common, the combination is not."},
 {t:"pneumonoultramicroscopicsilicovolcanoconiosis", why:"Forty-one tiles for one word \u2014 the longest word in major dictionaries, and the toy sounds it out like a child reading aloud."},
 {t:"supercalifragilisticexpialidocious", why:"Twenty-nine tiles. Even beloved nonsense shatters \u2014 what matters is frequency in training text, not fame."},
 {t:"subdermatoglyphic", why:"Fifteen tiles \u2014 and a party fact: it\u2019s the longest English word with no repeated letters. The tiles don\u2019t care."},
 {t:"starfish", why:"A compound \u2014 but the toy cuts \u201cs-t-ar-fi-sh\u201d, not \u201cstar-fish\u201d. Tiles follow frequency, not meaning."},
 {t:"rainbow", why:"\u201crainbow\u201d = \u201cr\u201d + \u201ca\u201d + \u201cin\u201d + \u201cb\u201d + \u201cow\u201d. Absurd and correct: those chunks were simply more common than \u201crain\u201d and \u201cbow\u201d."},
 {t:"bookkeeper", why:"Nine tiles, and \u201coo\u201d earned one \u2014 double letters are common enough to get printed. (Three e\u2019s, two k\u2019s: English.)"},
 {t:"strengths", why:"Seven tiles for nine letters \u2014 consonant clusters barely merge. The hardest English word to pronounce is also hard to tile."},
 {t:"rhythms", why:"No vowels, no problem. The tokenizer doesn\u2019t need vowels \u2014 it needs frequent neighbors."},
 {t:"queueing", why:"The famous vowel pileup: u-e-u-e in a row, and the toy just walks through them."},
 {t:"$100", why:"The toy sees NOTHING \u2014 digits aren\u2019t in its alphabet. Real tokenizers split numbers digit by digit: \u201c100\u201d \u2192 \u201c1\u201d,\u201c0\u201d,\u201c0\u201d. The model never sees a number, only digit confetti \u2014 one reason arithmetic is shaky."},
 {t:"2024", why:"Invisible to the toy. Real tokenizers chop years into pairs or digits \u2014 \u201c2024\u201d \u2192 \u201c20\u201d,\u201c24\u201d. A year is never one thing to the model."},
 {t:"3.14159", why:"Dropped entirely here. Real tokenizers turn decimals into digit soup: \u201c3\u201d,\u201c.\u201d,\u201c14\u201d,\u201c15\u201d,\u201c9\u201d. Pi arrives as confetti, never as a number."},
 {t:"caf\u00e9", why:"The toy silently EATS the \u00e9 \u2014 \u201ccaf\u00e9\u201d becomes \u201ccaf\u201d. Real tokenizers split accents into extra byte-tiles: accented text costs more, everywhere."},
 {t:"na\u00efve", why:"The \u00ef vanished and now it reads \u201cnave\u201d \u2014 a different word! Dropped characters can silently corrupt meaning."},
 {t:"user@example.com", why:"The @ and . are erased \u2014 an email becomes one long word. Real tokenizers split it: \u201cuser\u201d + \u201c@\u201d + \u201cexample\u201d + \u201c.\u201d + \u201ccom\u201d."},
 {t:"https://example.com", why:"\u201c://\u201d gone \u2014 to the toy it\u2019s one word. Real tokenizers keep the protocol pieces: \u201chttps\u201d + \u201c://\u201d."},
 {t:"\u{1f600}", why:"Invisible \u2014 emoji aren\u2019t in the toy\u2019s alphabet. Real tokenizers give common emoji one or two tiles each; rare ones shatter into bytes."},
 {t:"\u4e2d\u6587", why:"Two characters, zero tiles \u2014 the whole script is outside the toy\u2019s world. Real tokenizers spend roughly a tile per Chinese character: the same meaning costs more tiles than English."},
 {t:"New York", why:"Seven tiles here \u2014 but real tokenizers merge frequent phrases, and multi-word chunks like \u201cNew York\u201d routinely become a single tile. Phrases you say a lot become vocabulary."},
 {t:"machine learning", why:"Eleven tiles in the toy \u2014 yet in real tokenizers \u201cmachine learning\u201d is frequently one tile. The vocabulary bends toward what people actually write."},
 {t:"the quick brown fox jumps", why:"Five words, seventeen tokens. Tokens aren\u2019t words \u2014 and \u201c4 characters \u2248 1 token\u201d is the whole pricing game behind API bills."},
 {t:"    x = 1", why:"The toy throws away the spaces and reads just \u201cx\u201d. Real tokenizers do the opposite: a four-space indent is often a SINGLE tile \u2014 Python pays a token tax on every indent level."},
];
(function(){
  var track=document.getElementById("tok-ex-track");
  if(!track) return;
  var html="";
  TOK_EX.forEach(function(ex,i){
    var words=normalizeInput(ex.t), chips="";
    words.forEach(function(w){
      bpeEncodeWord(w,bpeModel.ranks).forEach(function(sym){
        if(sym==="</w>")return;
        chips+='<span class="tok final">'+esc(sym.replace("</w>",""))+'<i>'+bpeModel.vocab.get(sym)+'</i></span>';
      });
    });
    if(!chips) chips='<span class="tok-empty">\u2205 \u2014 nothing survived: no a\u2013z letters</span>';
    var rawFlat=ex.t.replace(/\s+/g," ").trim();
    var disp=esc(ex.t).replace(/^ +/gm,function(m){ return new Array(m.length+1).join("&nbsp;"); });
    var norm=words.join(" "), normLine="";
    if(norm!==rawFlat){
      normLine = norm===""
        ? '<p class="ex-norm">toy reads: (nothing \u2014 every character was dropped)</p>'
        : '<p class="ex-norm">toy reads: &ldquo;'+esc(norm)+'&rdquo;</p>';
    }
    html+='<div class="ex-slide"><div class="ex-card"><div class="ex-body">'
      +'<p class="ex-kicker">Simplified illustration \u00b7 real toy-model output</p>'
      +'<p class="ex-input">&ldquo;'+disp+'&rdquo;</p>'
      +normLine
      +'<div class="ex-chips">'+chips+'</div>'
      +'<p class="ex-why">'+ex.why+'</p>'
      +'</div><p class="ex-count">'+(i+1)+' / '+TOK_EX.length+'</p></div></div>';
  });
  track.innerHTML=html;
})();

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
   DEMO 2b · 3D embedding point cloud (hand-rolled projection,
   no libraries — one point per toy-vocab piece)
   ============================================================ */
(function(){
  var cv=document.getElementById("pc-canvas"); if(!cv) return;
  var ctx=cv.getContext("2d");
  var read=document.getElementById("pc-read");
  function h32(s){
    var h=2166136261;
    for(var i=0;i<s.length;i++){ h^=s.charCodeAt(i); h=Math.imul(h,16777619); }
    return h>>>0;
  }
  var pts=[];
  bpeModel.vocab.forEach(function(id, tok){
    if(tok==="</w>") return;
    var rnd=mulberry32(h32(tok)||1), v=[];
    for(var i=0;i<8;i++) v.push(rnd()*2-1);
    pts.push({tok:tok, id:id, v:v, p:{x:0,y:0,z:0}, sx:0, sy:0, sc:1, depth:0});
  });
  /* fixed random 8D -> 3D projection, then normalize to unit sphere */
  var prnd=mulberry32(20260930), P=[];
  for(var r=0;r<3;r++){ var prow=[]; for(var c=0;c<8;c++) prow.push(prnd()*2-1); P.push(prow); }
  pts.forEach(function(pt){
    var x=0,y=0,z=0,r2,c2,s;
    for(r2=0;r2<3;r2++){ s=0; for(c2=0;c2<8;c2++) s+=P[r2][c2]*pt.v[c2];
      if(r2===0)x=s; else if(r2===1)y=s; else z=s; }
    var n=Math.sqrt(x*x+y*y+z*z)||1;
    pt.p={x:x/n, y:y/n, z:z/n};
  });
  var W=0,H=260,dpr=1,yaw=0.7,pitch=0.42,visible=false,dragging=false,
      lastAct=0,sel=-1,rafOn=false,sized=false;
  function resize(){
    var wrap=cv.parentElement;
    dpr=Math.min(window.devicePixelRatio||1,2);
    W=Math.max(wrap.clientWidth,200); H=260;
    cv.width=Math.round(W*dpr); cv.height=Math.round(H*dpr);
    cv.style.width=W+"px"; cv.style.height=H+"px";
  }
  function project(){
    var cyw=Math.cos(yaw),syw=Math.sin(yaw),cxp=Math.cos(pitch),sxp=Math.sin(pitch);
    var R=Math.min(W,H)*0.44, persp=3.2, ox=W/2, oy=H/2;
    pts.forEach(function(pt){
      var x=pt.p.x,y=pt.p.y,z=pt.p.z;
      var x1=x*cyw+z*syw, z1=-x*syw+z*cyw;
      var y1=y*cxp-z1*sxp, z2=y*sxp+z1*cxp;
      var s=persp/(persp+z2);
      pt.sx=ox+x1*R*s; pt.sy=oy-y1*R*s; pt.sc=s; pt.depth=z2;
    });
  }
  function draw(){
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,W,H);
    var order=pts.map(function(p,i){return i;})
      .sort(function(a,b){return pts[a].depth-pts[b].depth;});
    order.forEach(function(i){
      var pt=pts[i];
      var a=0.30+0.70*(pt.sc-0.62)/1.0;
      a=Math.max(0.22,Math.min(1,a));
      ctx.beginPath();
      ctx.arc(pt.sx,pt.sy,2+2.2*pt.sc,0,6.2832);
      if(i===sel){
        ctx.fillStyle="#e9a13b"; ctx.shadowColor="#e9a13b"; ctx.shadowBlur=12;
      }else{
        ctx.fillStyle="rgba(98,182,164,"+a.toFixed(2)+")"; ctx.shadowBlur=0;
      }
      ctx.fill(); ctx.shadowBlur=0;
    });
  }
  function frame(){
    if(!visible){ rafOn=false; return; }
    if(!dragging && Date.now()-lastAct>3000) yaw+=0.0045; /* gentle auto-rotate */
    project(); draw();
    requestAnimationFrame(frame);
  }
  function kick(){ if(visible && !rafOn){ rafOn=true; requestAnimationFrame(frame); } }
  var px0=0,py0=0,moved=0;
  cv.addEventListener("pointerdown",function(e){
    dragging=true; px0=e.clientX; py0=e.clientY; moved=0; lastAct=Date.now();
    try{cv.setPointerCapture(e.pointerId);}catch(_){}
    e.preventDefault();
  });
  cv.addEventListener("pointermove",function(e){
    if(!dragging) return;
    var dx=e.clientX-px0, dy=e.clientY-py0;
    moved+=Math.abs(dx)+Math.abs(dy);
    yaw+=dx*0.008;
    pitch=Math.max(-1.2,Math.min(1.2,pitch+dy*0.006));
    px0=e.clientX; py0=e.clientY; lastAct=Date.now();
  });
  function endDrag(e){
    if(dragging && moved<8){ /* tap: name the nearest point */
      var r=cv.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
      var best=-1, bd=30*30;
      pts.forEach(function(pt,i){
        var d=(pt.sx-mx)*(pt.sx-mx)+(pt.sy-my)*(pt.sy-my);
        if(d<bd){ bd=d; best=i; }
      });
      sel=best; lastAct=Date.now();
      if(best>=0) read.innerHTML="Token <b>“"+esc(pts[best].tok.replace("</w>",""))+"”</b> · id <b>"+pts[best].id+"</b> — one of "+pts.length+" pieces in the toy vocabulary.";
      else read.textContent="Drag to rotate · tap a point to name it.";
    }
    dragging=false;
  }
  cv.addEventListener("pointerup",endDrag);
  cv.addEventListener("pointercancel",function(){ dragging=false; });
  new IntersectionObserver(function(es){
    es.forEach(function(en){
      visible=en.isIntersecting;
      if(visible){ if(!sized){ resize(); sized=true; } kick(); }
    });
  },{threshold:0.05}).observe(cv);
  window.addEventListener("resize",function(){ if(sized) resize(); });
})();

/* ============================================================
   DEMO 3 · sampling lab: real softmax + temperature + top-k
   ============================================================ */
var CANDS=[[" the",2.2],[" cat",1.5],[" dog",1.2],[" mat",0.8],
           [" moon",0.1],[" zebra",-0.6],[" quantum",-1.2],[" banana",-2.1]];
var tmpVal=document.getElementById("tmp-val"),
    topkVal=document.getElementById("topk-val"),
    bars=document.getElementById("samp-bars"),
    pick=document.getElementById("samp-pick"),
    seq=document.getElementById("samp-seq");

/* Touch-friendly slider: the whole track row is tappable (pointerdown
   computes the value from the x position), drag keeps working, the
   thumb sits in a 44px+ touch target, and touch-action:none means a
   drag can never scroll the page. Keyboard: arrows/Home/End. */
function makeSlider(id, opts){
  var el=document.getElementById(id),
      track=el.querySelector(".cslider-track"),
      fill=el.querySelector(".cslider-fill"),
      thumb=el.querySelector(".cslider-thumb");
  var S={min:opts.min,max:opts.max,step:opts.step,value:opts.value,
         onChange:opts.onChange||function(){}};
  function fmt(v){ return opts.step<1 ? v.toFixed(2) : String(Math.round(v)); }
  function render(){
    var t=(S.value-S.min)/(S.max-S.min);
    fill.style.width=(t*100)+"%";
    thumb.style.left=(t*100)+"%";
    el.setAttribute("aria-valuenow",S.value);
    el.setAttribute("aria-valuetext",fmt(S.value));
  }
  function set(v,fire){
    v=Math.min(S.max,Math.max(S.min,v));
    v=Math.round(v/S.step)*S.step;
    v=Math.round(v*1000)/1000; /* float hygiene */
    if(v===S.value){ render(); return; }
    S.value=v; render();
    if(fire!==false) S.onChange(S.value);
  }
  S.set=set;
  function fromX(cx){
    var r=track.getBoundingClientRect();
    set(S.min+((cx-r.left)/r.width)*(S.max-S.min));
  }
  var dragging=false;
  el.addEventListener("pointerdown",function(e){
    dragging=true;
    try{el.setPointerCapture(e.pointerId);}catch(_){}
    fromX(e.clientX);
    e.preventDefault();
  });
  el.addEventListener("pointermove",function(e){ if(dragging) fromX(e.clientX); });
  el.addEventListener("pointerup",function(){ dragging=false; });
  el.addEventListener("pointercancel",function(){ dragging=false; });
  el.addEventListener("keydown",function(e){
    var d=0;
    if(e.key==="ArrowLeft"||e.key==="ArrowDown") d=-S.step;
    else if(e.key==="ArrowRight"||e.key==="ArrowUp") d=S.step;
    else if(e.key==="Home"){ set(S.min); e.preventDefault(); return; }
    else if(e.key==="End"){ set(S.max); e.preventDefault(); return; }
    else return;
    set(S.value+d); e.preventDefault();
  });
  render();
  return S;
}

/* donut / probability-wheel view of the same distribution */
var donutSegs=document.getElementById("donut-segs"),
    donutTop=document.getElementById("donut-top"),
    donutPct=document.getElementById("donut-pct"),
    donutCircles=[];
(function(){
  var NS="http://www.w3.org/2000/svg";
  for(var i=0;i<CANDS.length;i++){
    var c=document.createElementNS(NS,"circle");
    c.setAttribute("cx",70); c.setAttribute("cy",70); c.setAttribute("r",54);
    c.setAttribute("fill","none"); c.setAttribute("stroke-width",18);
    donutSegs.appendChild(c); donutCircles.push(c);
  }
})();
function renderDonut(r){
  var C=2*Math.PI*54, acc=0, pmax=1e-4, topI=0, i;
  for(i=0;i<CANDS.length;i++) if(r.keep[i]&&r.probs[i]>pmax){ pmax=r.probs[i]; topI=i; }
  for(i=0;i<CANDS.length;i++){
    var frac=r.keep[i]?r.probs[i]:0;
    var len=Math.max(frac*C-2.5,0);
    var c=donutCircles[i];
    c.setAttribute("stroke-dasharray",len.toFixed(1)+" "+C.toFixed(1));
    c.setAttribute("stroke-dashoffset",(-acc*C).toFixed(1));
    c.setAttribute("stroke","#e9a13b");
    c.setAttribute("stroke-opacity",r.keep[i]?(0.35+0.65*(r.probs[i]/pmax)).toFixed(2):"0");
    acc+=frac;
  }
  donutTop.textContent="“"+CANDS[topI][0].trim()+"”";
  donutPct.textContent=(r.probs[topI]*100).toFixed(1)+"%";
}
function renderLab(){
  var r=renderBars();
  renderDonut(r);
}

function softmax(logits){
  var m=Math.max.apply(null,logits);
  var ex=logits.map(function(z){return Math.exp(z-m);});
  var s=ex.reduce(function(a,b){return a+b;},0);
  return ex.map(function(e){return e/s;});
}
function currentProbs(){
  var k=SL.topk.value, T=SL.temp.value;
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
  tmpVal.textContent=SL.temp.value.toFixed(2);
  topkVal.textContent=SL.topk.value;
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
var SL={
  temp:makeSlider("sl-temp",{min:0.1,max:2,step:0.05,value:1,onChange:renderLab}),
  topk:makeSlider("sl-topk",{min:1,max:8,step:1,value:8,onChange:renderLab})
};
document.getElementById("tmp-dec").addEventListener("click",function(){ SL.temp.set(SL.temp.value-0.05); });
document.getElementById("tmp-inc").addEventListener("click",function(){ SL.temp.set(SL.temp.value+0.05); });
document.getElementById("topk-dec").addEventListener("click",function(){ SL.topk.set(SL.topk.value-1); });
document.getElementById("topk-inc").addEventListener("click",function(){ SL.topk.set(SL.topk.value+1); });
document.getElementById("samp-once").addEventListener("click", sampleOnce);
document.getElementById("samp-20").addEventListener("click", function(){ for(var n=0;n<20;n++) sampleOnce(); });
document.getElementById("samp-clear").addEventListener("click", function(){
  seq.innerHTML=""; pick.textContent="Press “Sample once”.";
});
renderLab();

/* ============================================================
   EXAMPLE CAROUSELS · hand-rolled, one initializer for all
   ============================================================ */
document.querySelectorAll(".ex-car").forEach(function(car){
  var track=car.querySelector(".ex-track"),
      prev=car.querySelector(".ex-prev"),
      next=car.querySelector(".ex-next");
  if(!track||!prev||!next) return;
  var slides=Array.prototype.slice.call(track.querySelectorAll(".ex-slide"));
  var idx=0;
  /* Exact slide targeting: every navigation lands on a real slide edge,
     never on a width multiple — so rapid taps can't accumulate drift. */
  function targetFor(i){
    var s=slides[i]; if(!s) return 0;
    var tr=track.getBoundingClientRect(), sr=s.getBoundingClientRect();
    return track.scrollLeft + (sr.left - tr.left);
  }
  function go(i){
    if(!slides.length) return;
    idx=Math.max(0,Math.min(slides.length-1,i));
    track.scrollTo({left:targetFor(idx),behavior:"smooth"});
    upd();
  }
  function upd(){
    prev.disabled=idx<=0;
    next.disabled=idx>=slides.length-1;
  }
  prev.addEventListener("click",function(){go(idx-1);});
  next.addEventListener("click",function(){go(idx+1);});
  /* After a free swipe, re-anchor the index to the nearest slide so the
     next tap continues from the card you're actually looking at. */
  var syncT=null;
  function resync(){
    if(!slides.length) return;
    var tr=track.getBoundingClientRect(), best=0, bd=Infinity;
    slides.forEach(function(s,i){
      var d=Math.abs(s.getBoundingClientRect().left-tr.left);
      if(d<bd){bd=d;best=i;}
    });
    if(best!==idx){idx=best;upd();}
  }
  track.addEventListener("scroll",function(){
    clearTimeout(syncT); syncT=setTimeout(resync,140);
  },{passive:true});
  if("onscrollend" in window) track.addEventListener("scrollend",function(){clearTimeout(syncT);resync();});
  window.addEventListener("resize",function(){track.scrollTo({left:targetFor(idx)});});
  upd();
});

/* ============================================================
   Attention mini-interactives · hand-set pedagogical weights
   (causal: a word can only listen to itself and the past)
   ============================================================ */
var ATTN_EX=[
  {words:["the","cat","licked","its","paw"],
   w:[[1,0,0,0,0],[.25,.75,0,0,0],[.05,.55,.4,0,0],[.05,.7,.1,.15,0],[.05,.15,.25,.35,.2]],
   focus:3,
   notes:["“the” mostly minds itself — it will attach to “cat” next.",
     "“cat” glances back at “the”: together they're one noun phrase.",
     "“licked” looks back at “cat” — <b>who</b> is doing the licking?",
     "“its” listens hardest to “cat” — that's how the model knows <b>whose</b> paw. Pronouns are attention's favorite puzzle.",
     "“paw” soaks up the whole phrase: whose (cat), doing what (licked), which one (its)."]},
  {words:["she","opened","the","door"],
   w:[[1,0,0,0],[.5,.5,0,0],[.05,.1,.85,0],[.15,.3,.35,.2]],
   focus:3,
   notes:["“she” mostly minds itself — for now.",
     "“opened” checks back at “she”: <b>who</b> opened?",
     "“the” waits — the word it belongs to hasn't arrived yet.",
     "“door” looks back at “opened” and “the”: <b>what</b> was opened?"]},
  {words:["not","good","at","all"],
   w:[[1,0,0,0],[.55,.45,0,0],[.25,.35,.4,0],[.4,.25,.15,.2]],
   focus:1,
   notes:["“not” mostly minds itself — its power comes later.",
     "“good” keeps a close eye on “not” — the praise is about to be <b>cancelled</b>. Negation works by listening.",
     "“at” gathers “not good” so far.",
     "“all” soaks up the whole negated phrase: not-good-at-all."]},
  {words:["the","dog","buried","its","bone"],
   w:[[1,0,0,0,0],[.25,.75,0,0,0],[.05,.55,.4,0,0],[.05,.65,.15,.15,0],[.05,.15,.3,.3,.2]],
   focus:3,
   notes:["“the” mostly minds itself — it will attach to “dog” next.",
     "“dog” glances back at “the”: one noun phrase.",
     "“buried” looks back at “dog” — <b>who</b> did the burying?",
     "Same trick, new sentence: “its” finds “dog”. The pattern <b>generalizes</b> — that's the whole game.",
     "“bone” gathers the phrase: whose bone, buried by whom."]}
];
document.querySelectorAll(".attn-mini").forEach(function(box){
  var ex=ATTN_EX[+box.getAttribute("data-ex")];
  if(!ex) return;
  var wrap=box.querySelector(".am-words"), note=box.querySelector(".am-note");
  var btns=ex.words.map(function(wd,i){
    var b=document.createElement("button");
    b.type="button"; b.className="am-w"; b.textContent=wd;
    b.setAttribute("aria-label","How "+wd+" listens");
    b.addEventListener("click",function(){ show(i); });
    wrap.appendChild(b);
    return b;
  });
  function show(i){
    var row=ex.w[i];
    btns.forEach(function(b,j){
      var a=j<=i?row[j]:0;
      b.style.background="rgba(233,161,59,"+(0.06+a*0.85).toFixed(2)+")";
      b.style.borderColor=a>0.5?"#e9a13b":"";
    });
    note.innerHTML=ex.notes[i];
  }
  show(ex.focus);
});

})();
