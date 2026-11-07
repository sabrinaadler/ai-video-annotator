'use strict';
const $ = id => document.getElementById(id);
let state, duration, fps, sessionName='fieldnote', selected = null, busy = false, undo = [], queue = Promise.resolve();
const video = $('video');
const time = seconds => {
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2,'0')}:${(seconds % 60).toFixed(3).padStart(6,'0')}`;
};
const range = row => `${time(row.start)} – ${time(row.end)}`;
const clone = value => JSON.parse(JSON.stringify(value));
function error(message) { $('error').textContent = message; $('error').hidden = !message; }
async function request(path, body) {
  const response = await fetch(path, body ? {method:'POST', headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Request failed');
  return data;
}
function current() { return state.results.find(row => row.id === selected); }
function boundaries() {
  const start = Number($('start').value), end = Number($('end').value);
  if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start || end > duration) throw new Error('Use an interval with 0 ≤ in < out ≤ recording duration.');
  return {start, end};
}
function seek(position) { video.currentTime = Math.min(duration,Math.max(0,position)); }
function render() {
  $('behavior').replaceChildren(...state.ethogram.map(entry=>{const option=document.createElement('option');option.value=entry.name;option.textContent=entry.name;return option;}));
  $('behavior').value = state.behavior; $('method').value = state.method;
  const activeBehavior=state.ethogram.find(e=>e.name===state.behavior);
  $('behaviorDefinition').textContent=activeBehavior.description || 'Define this behavior in your ethogram before coding.';
  $('behaviorKeys').textContent=state.ethogram.filter(e=>e.key).map(e=>`${e.key.toUpperCase()} · ${e.name}`).join(' / ');
  $('codingPad').replaceChildren();
  state.ethogram.forEach(entry=>{const button=document.createElement('button'),key=document.createElement('kbd');
    button.textContent=(Object.hasOwn(state.open_events,entry.name)?'■ Stop ':'')+entry.name;
    key.textContent=entry.key.toUpperCase()||'—';button.append(key);button.disabled=busy;
    button.classList.toggle('recording',Object.hasOwn(state.open_events,entry.name));
    button.title=entry.type==='point'?'Record a point event':'Start / stop a state event';
    button.onclick=()=>codeBehavior(entry.name);$('codingPad').append(button);
  });
  const open=Object.entries(state.open_events);
  $('codingStatus').textContent=open.length?open.map(([name,start])=>`${name} recording from ${time(start)}`).join(' · '):'Behavior keys: state = start / stop · point = one press.';
  $('editEthogram').disabled=busy; $('saveEthogram').disabled=busy;
  const reviewing = state.results.length > 0;
  $('behavior').disabled = busy; $('method').disabled = reviewing || busy;
  $('search').disabled = busy || !state.examples.length || activeBehavior.type==='point';
  $('search').textContent = reviewing ? 'Update suggestions ↗' : 'Find similar moments ↗';
  $('negativeWeight').value = String(state.negative_weight);
  $('proposalPolicy').value=state.proposal_policy||'fixed';
  $('proposalPolicy').disabled=busy || !(state.identity.window_sizes?.length>1);
  $('negativeWeight').disabled = busy;
  $('addNegative').hidden = reviewing;
  $('negativeCount').textContent = `${state.negative_examples.length} / 10`;
  $('encoderLabel').textContent = state.identity.model.split('/').pop();
  $('durationPolicy').textContent = state.identity.window_sizes ?
    `Proposal durations: ${state.identity.window_sizes.join(' / ')} seconds.` :
    `Proposal duration: ${state.identity.window_seconds || 'cached'} seconds.`;
  $('negatives').replaceChildren();
  state.negative_examples.forEach(row => {
    const li=document.createElement('li'), jump=document.createElement('button'), remove=document.createElement('button');
    jump.textContent=range(row); jump.onclick=()=>seek(row.start);
    remove.textContent='×'; remove.disabled=busy; remove.setAttribute('aria-label','Remove counterexample');
    remove.onclick=()=>mutate(()=>{state.negative_examples=state.negative_examples.filter(r=>r.id!==row.id);});
    li.append(jump,remove); $('negatives').append(li);
  });
  $('add').hidden = reviewing; $('apply').hidden = !reviewing;
  $('markIn').disabled = busy; $('markOut').disabled = busy;
  $('exampleCount').textContent = `${state.examples.length} / 10`;
  $('phaseExamples').classList.toggle('active', !reviewing);
  $('phaseReview').classList.toggle('active', reviewing);
  $('examples').replaceChildren();
  state.examples.forEach(row => {
    const li = document.createElement('li'), button = document.createElement('button');
    button.textContent = range(row); button.onclick = () => { seek(row.start); if(!reviewing){$('start').value=row.start; $('end').value=row.end;} };
    li.append(button);
    if (!reviewing) {const remove = document.createElement('button'); remove.textContent='×'; remove.disabled=busy; remove.setAttribute('aria-label','Remove example'); remove.onclick=()=>mutate(()=>{state.examples=state.examples.filter(r=>r.id!==row.id);}); li.append(remove);}
    $('examples').append(li);
  });
  $('results').replaceChildren();
  state.results.forEach((row,index) => {
    const button=document.createElement('button'); button.className=`candidate ${row.status} ${row.id===selected?'selected':''}`;
    const rank=document.createElement('span'); rank.className='rank'; rank.textContent=`${String(index+1).padStart(2,'0')} / SUGGESTION`;
    const interval=document.createElement('strong'); interval.textContent=range(row);
    const status=document.createElement('span'); status.className='status'; status.textContent=row.status==='later'?'Review later':row.status;
    button.append(rank,interval,status); button.onclick=()=>select(row.id); $('results').append(button);
  });
  const row=current(); $('reviewPanel').hidden=!row;
  ['add','addNegative','apply','useNegative','accept','reject','later','previous','next','inBack','inForward','outBack','outForward'].forEach(id=>{$(id).disabled=busy;});
  $('note').disabled=busy; $('restoreBounds').disabled=busy || !row;
  $('editorTitle').textContent=reviewing?'REFINE THE SELECTED INTERVAL':'MARK A REFERENCE EXAMPLE';
  $('editorHint').textContent=activeBehavior.type==='point'?'Point event: use its assigned key or the coding pad.':reviewing?'Edit boundaries, then apply before making a decision.':'Use I / O to mark the event. Enter adds an example.';
  if(row){$('candidateTitle').textContent=`Suggestion ${state.results.indexOf(row)+1} · ${row.status}`; $('score').textContent=`${row.score_type==='contrastive'?'Contrastive rank':'Cosine'} ${row.score.toFixed(5)}`; if(document.activeElement!==$('note'))$('note').value=row.note;}
  const accepted=state.results.filter(r=>r.status==='accepted'), pending=state.results.filter(r=>r.status==='pending'), later=state.results.filter(r=>r.status==='later');
  $('summary').textContent=`${accepted.length} accepted · ${state.examples.length} examples`;
  $('queueHint').textContent=reviewing?`${pending.length} pending · ${later.length} later`:'Your examples guide the search.';
  $('ledger').replaceChildren();
  [...state.examples.map(r=>({...r,source:'researcher example'})),...accepted.map(r=>({...r,source:'reviewed suggestion'}))].sort((a,b)=>a.start-b.start).forEach(r=>{
    const line=document.createElement('div'); line.className='ledger-row';
    const interval=document.createElement('span'), source=document.createElement('span'); interval.textContent=range(r); source.textContent=r.source; line.append(interval,source); $('ledger').append(line);
  });
  $('undo').disabled=!undo.length || busy;
  state.manual_events.slice().sort((a,b)=>a.start-b.start).forEach(event=>{
    const line=document.createElement('div');line.className='manual-row';
    const jump=document.createElement('button');jump.textContent=`${event.behavior} · ${event.type==='point'?time(event.start):range(event)} · manual`;jump.onclick=()=>seek(event.start);
    const remove=document.createElement('button');remove.className='delete-event';remove.textContent='Remove';remove.disabled=busy;remove.setAttribute('aria-label',`Remove ${event.behavior} event at ${time(event.start)}`);
    remove.onclick=()=>mutate(()=>{state.manual_events=state.manual_events.filter(r=>r.id!==event.id);});line.append(jump,remove);$('ledger').append(line);
  });
  Object.entries(state.open_events).forEach(([name,start])=>{const line=document.createElement('div');line.className='open-state';line.textContent=`● ${name} open from ${time(start)} `;
    const cancel=document.createElement('button');cancel.textContent='Cancel start';cancel.disabled=busy;cancel.onclick=()=>mutate(()=>{delete state.open_events[name];});line.append(cancel);$('ledger').append(line);});
  $('summary').textContent=`${accepted.length} accepted · ${state.examples.length} examples · ${state.manual_events.length} coded across behaviors`;
  ['add','addNegative','markIn','markOut'].forEach(id=>{$(id).disabled=busy || activeBehavior.type==='point';});
  $('footerStatus').textContent=reviewing?'Update suggestions keeps reviewed decisions, notes and boundary edits · scores are not probabilities':'Mark positive examples and optional counterexamples · all edits save locally';
}
function select(id) { selected=id; const row=current(); if(!row)return; $('start').value=row.start; $('end').value=row.end;seek(Math.max(0,row.start-.75)); render(); }
function mutate(change) {
  queue = queue.then(async()=> {
    busy=true; error(''); const previous=clone(state), previousSelected=selected;
    try {
      change(); state.position=video.currentTime;
      $('save').textContent='Saving…';render();
      state=await request('/api/session',state);
      undo.push({state:previous,selected:previousSelected}); if(undo.length>50)undo.shift();
      $('save').textContent='Saved to disk';
    } catch(exc) { state=previous;selected=previousSelected;error(exc.message);$('save').textContent='Edit not saved'; }
    finally {busy=false;render();}
  });
  return queue;
}
function addExample(){mutate(()=>{if(state.ethogram.find(e=>e.name===state.behavior).type==='point')throw new Error('Use the coding pad for point events.');if(state.results.length)throw new Error('Reference examples are locked after searching.');if(state.examples.length>=10)throw new Error('Maximum 10 examples.'); const interval=boundaries();if(state.examples.some(r=>Math.max(r.start,interval.start)<Math.min(r.end,interval.end)))throw new Error('Choose a distinct, nonoverlapping example.');state.examples.push({id:crypto.randomUUID(),...interval,note:''});});}
function addCounterexample(fromReview=false){
  let interval;try{interval=boundaries();}catch(exc){error(exc.message);return;}
  const id=selected, note=fromReview?$('note').value:'';
  mutate(()=>{
    if(state.negative_examples.length>=10)throw new Error('Maximum 10 counterexamples.');
    if([...state.examples,...state.negative_examples].some(r=>Math.max(r.start,interval.start)<Math.min(r.end,interval.end)))throw new Error('Choose a distinct counterexample that does not overlap a reference.');
    if(fromReview){const row=state.results.find(r=>r.id===id);if(!row)throw new Error('Select a suggestion.');Object.assign(row,interval);row.status='rejected';row.note=note;}
    state.negative_examples.push({id:crypto.randomUUID(),...interval,note});
    if(fromReview){const next=state.results.find(r=>r.status==='pending'&&r.id!==id);if(next)select(next.id);}
  });
}
function decide(status){if(!current())return;let interval;try{interval=boundaries();}catch(exc){error(exc.message);return;}const id=selected,note=$('note').value;mutate(()=>{const row=state.results.find(r=>r.id===id);Object.assign(row,interval);row.status=status;row.note=note;const next=state.results.find(r=>r.status==='pending'&&r.id!==selected);if(next){select(next.id);} });}
function navigate(delta){if(!state.results.length)return;const index=state.results.findIndex(r=>r.id===selected);select(state.results[(index+delta+state.results.length)%state.results.length].id);}
$('add').onclick=addExample;
$('addNegative').onclick=()=>addCounterexample(false);
$('useNegative').onclick=()=>addCounterexample(true);
$('proposalPolicy').onchange=()=>{const value=$('proposalPolicy').value;mutate(()=>{state.proposal_policy=value;});};
$('negativeWeight').onchange=()=>{const weight=Number($('negativeWeight').value);mutate(()=>{state.negative_weight=weight;});};
$('apply').onclick=()=>mutate(()=>{const row=current();if(!row)throw new Error('Select a suggestion.');Object.assign(row,boundaries());row.note=$('note').value;});
$('behavior').onchange=()=>switchBehavior($('behavior').value);
$('method').onchange=()=>{const value=$('method').value;mutate(()=>{state.method=value;});};
$('note').onchange=()=>{const value=$('note').value,id=selected;mutate(()=>{state.results.find(r=>r.id===id).note=value;});};
$('search').onclick=()=>{
  if(busy)return;
  queue=queue.then(async()=>{
    busy=true;error('');render();$('save').textContent='Updating suggestions…';
    try{state=await request('/api/search',{revision:state.revision,identity:state.identity});undo=[];
      if(state.results.length)select(state.results.find(r=>r.status==='pending')?.id||state.results[0].id);
      else error('No unseen candidates remain.');$('save').textContent='Search saved';
    }catch(exc){error(exc.message);$('save').textContent='Search failed';}
    finally{busy=false;render();}
  });return queue;
};
$('accept').onclick=()=>decide('accepted');$('reject').onclick=()=>decide('rejected');$('later').onclick=()=>decide('later');
$('next').onclick=()=>navigate(1);$('previous').onclick=()=>navigate(-1);
$('undo').onclick=async()=>{await queue;if(!undo.length||busy)return;const item=undo.pop();const revision=state.revision;busy=true;try{state=await request('/api/session',{...item.state,revision});selected=item.selected;if(current())select(selected);$('save').textContent='Undo saved';}catch(exc){undo.push(item);error(exc.message);}finally{busy=false;render();}};
$('play').onclick=()=>{if(video.paused)video.play().catch(exc=>error(exc.message));else video.pause();};
video.onplay=()=>{$('play').textContent='Ⅱ Pause';};video.onpause=()=>{$('play').textContent='▶ Play';};
video.onerror=()=>error('The browser could not decode this recording. Use a browser with MP4/H.264 support.');
video.ontimeupdate=()=>{$('scrub').value=video.currentTime;$('timecode').textContent=time(video.currentTime);if($('loop').checked){const a=Number($('start').value),b=Number($('end').value);if(b>a&&video.currentTime>=b)seek(a);}};
$('scrub').oninput=()=>seek(Number($('scrub').value));
$('back').onclick=()=>{video.pause();seek(video.currentTime-1/fps);};$('forward').onclick=()=>{video.pause();seek(video.currentTime+1/fps);};
function nudgeBoundary(which,delta){
  try{const interval=boundaries();const key=which==='start'?'start':'end';
    const scaled=interval[key]*fps, nearest=Math.round(scaled);
    const base=Math.abs(scaled-nearest)<1e-4?nearest:(delta>0?Math.floor(scaled):Math.ceil(scaled));
    interval[key]=(base+delta)/fps;
    if(interval.start<0||interval.end>duration+1e-9||interval.end<=interval.start)throw new Error('The interval must stay inside the recording, with in before out.');
    $(which).value=Math.min(duration,interval[key]).toFixed(6);seek(interval[key]);
  }catch(exc){error(exc.message);}
}
$('inBack').onclick=()=>nudgeBoundary('start',-1);$('inForward').onclick=()=>nudgeBoundary('start',1);
$('outBack').onclick=()=>nudgeBoundary('end',-1);$('outForward').onclick=()=>nudgeBoundary('end',1);
$('restoreBounds').onclick=()=>{const row=current();if(row){$('start').value=row.original_start;$('end').value=row.original_end;seek(row.original_start);}};
$('markIn').onclick=()=>{$('start').value=video.currentTime.toFixed(3);};$('markOut').onclick=()=>{$('end').value=video.currentTime.toFixed(3);};
$('speed').onchange=()=>{video.playbackRate=Number($('speed').value);};
$('help').onclick=()=>$('shortcuts').showModal();$('closeHelp').onclick=()=>$('shortcuts').close();
document.addEventListener('keydown',event=>{if(event.target.matches('input,textarea,select')||$('shortcuts').open||$('ethogramEditor').open||!state||event.repeat)return;const key=event.key.toLowerCase();
if((key==='arrowleft'||key==='arrowright')&&(event.altKey||event.shiftKey)&&!event.metaKey&&!event.ctrlKey){event.preventDefault();nudgeBoundary(event.altKey?'start':'end',key==='arrowleft'?-1:1);return;}
if((event.metaKey||event.ctrlKey)&&key==='z'){event.preventDefault();$('undo').click();return;}if(event.metaKey||event.ctrlKey||event.altKey)return;const behavior=state.ethogram.find(e=>e.key&&e.key===key);if(behavior){event.preventDefault();codeBehavior(behavior.name);return;}if(busy)return;const actions={' ':()=>$('play').click(),arrowleft:()=>$('back').click(),arrowright:()=>$('forward').click(),i:()=>$('markIn').click(),o:()=>$('markOut').click(),enter:()=>{if(!state.results.length)addExample();},a:()=>decide('accepted'),r:()=>decide('rejected'),x:()=>addCounterexample(!!state.results.length),d:()=>decide('later'),n:()=>navigate(1),p:()=>navigate(-1),l:()=>{$('loop').checked=!$('loop').checked;},'?':()=>$('help').click()};if(actions[key]){event.preventDefault();actions[key]();}});
document.querySelectorAll('a[download]').forEach(link=>{link.onclick=async event=>{event.preventDefault();await queue;try{const response=await fetch(link.getAttribute('href'));if(!response.ok)throw new Error('Export failed');const blob=await response.blob(),url=URL.createObjectURL(blob),download=document.createElement('a');download.href=url;const stem=sessionName.replace(/\.json$/i,'');download.download=stem+(link.href.endsWith('.csv')?'.annotations.csv':'.session.json');download.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(exc){error(exc.message);}};});
request('/api/session').then(data=>{sessionName=data.session_name||'fieldnote';state=data.state;state.negative_examples??=[];state.negative_weight??=0;state.proposal_policy??='fixed';duration=data.duration;fps=data.fps;$('scrub').max=duration;$('filename').textContent=data.filename;$('meta').textContent=`${time(duration)} · ${fps} fps · ${state.identity.preprocessing} · local`;$('videoId').textContent=state.identity.video_id;$('save').textContent='Session ready';render();if(state.results.length)select(state.results.find(r=>r.status==='pending')?.id||state.results[0].id);else if(video.readyState>=1)seek(state.position);else video.addEventListener('loadedmetadata',()=>seek(state.position),{once:true});}).catch(exc=>error(exc.message));

// Behavior selection changes retrieval context; it never relabels previous annotations.
function switchBehavior(name){
  queue=queue.then(async()=>{busy=true;error('');render();
    try{state=await request('/api/behavior',{identity:state.identity,revision:state.revision,behavior:name});selected=null;undo=[];
      $('start').value=video.currentTime.toFixed(3);$('end').value=Math.min(duration,video.currentTime+1).toFixed(3);
      if(state.results.length)select(state.results.find(r=>r.status==='pending')?.id||state.results[0].id);
      $('save').textContent='Behavior workspace saved';
    }catch(exc){error(exc.message);}finally{busy=false;render();}
  });return queue;
}
function codeBehavior(name){
  const position=video.currentTime;
  mutate(()=>{const entry=state.ethogram.find(e=>e.name===name);if(!entry)throw new Error('Unknown behavior');
    if(entry.type==='state'&&!Object.hasOwn(state.open_events,name)){state.open_events[name]=position;return;}
    const start=entry.type==='point'?position:state.open_events[name];
    if(entry.type==='state'&&position<=start)throw new Error('Stop after the state start, or cancel its start in the ledger.');
    state.manual_events.push({id:crypto.randomUUID(),behavior:name,type:entry.type,start,end:position,note:''});
    delete state.open_events[name];
  });
}
let ethogramDraft=[];
const availableKeys='1234567890bcefghjkmqstuvwyz';
function behaviorUsed(name){const data=name===state.behavior?state:state.behavior_tracks[name];return !!(data&&(data.examples.length||data.results.length||data.negative_examples.length)||Object.hasOwn(state.open_events,name)||state.manual_events.some(e=>e.behavior===name));}
function nextKey(){return [...availableKeys].find(k=>!ethogramDraft.some(e=>e.key===k))||'';}
function renderEthogram(){
  $('ethogramRows').replaceChildren();
  ethogramDraft.forEach((entry,index)=>{
    const row=document.createElement('div');row.className='ethogram-row';
    for(const [field,title] of [['name','Behavior'],['type','Event type'],['key','Key'],['description','Definition']]){
      const label=document.createElement('label');label.textContent=title;
      const input=document.createElement(field==='type'?'select':'input');input.setAttribute('aria-label',`${title} ${index+1}`);
      if(field==='type'){for(const type of ['state','point']){const option=document.createElement('option');option.value=type;option.textContent=type;input.append(option);}}
      input.value=entry[field]||'';if(field==='key')input.maxLength=1;if(field==='name')input.maxLength=80;
      input.disabled=(field==='name'||field==='type')&&behaviorUsed(entry.originalName||entry.name);
      input.oninput=()=>{entry[field]=field==='key'?input.value.toLowerCase():input.value;};label.append(input);row.append(label);
    }
    const remove=document.createElement('button');remove.textContent='×';remove.setAttribute('aria-label',`Remove behavior ${index+1}`);
    remove.disabled=behaviorUsed(entry.originalName||entry.name)||(entry.originalName===state.behavior);
    remove.onclick=()=>{ethogramDraft.splice(index,1);renderEthogram();};row.append(remove);$('ethogramRows').append(row);
  });
}
$('editEthogram').onclick=()=>{ethogramDraft=state.ethogram.map(e=>({...e,originalName:e.name}));$('ethogramError').textContent='';renderEthogram();$('ethogramEditor').showModal();};
$('cancelEthogram').onclick=()=>$('ethogramEditor').close();
$('addBehavior').onclick=()=>{ethogramDraft.push({name:'',type:'state',key:nextKey(),description:''});renderEthogram();};
const presets={fly:[['lunge','A rapid forward thrust directed at another fly.'],['wing extension','One or both wings held away from the body.'],['grooming','Cleaning movements directed at the body.'],['chase','Following another fly in directed pursuit.']],mouse:[['investigation','Directed sniffing or close inspection of another mouse.'],['mount','Forelimbs placed on the other animal with mounting posture.'],['grooming','Cleaning movements directed at the body.']]};
$('addPreset').onclick=()=>{for(const [name,description] of presets[$('preset').value])if(!ethogramDraft.some(e=>e.name===name))ethogramDraft.push({name,type:'state',key:nextKey(),description});renderEthogram();};
function cleanEthogram(){return ethogramDraft.map(({name,type,key,description})=>({name:name.trim(),type,key,description}));}
$('saveEthogram').onclick=async()=>{
  const entries=cleanEthogram(),renamed=ethogramDraft.find(e=>e.originalName===state.behavior);
  await mutate(()=>{state.ethogram=entries;if(renamed)state.behavior=renamed.name.trim();});
  if($('save').textContent==='Saved to disk')$('ethogramEditor').close();else $('ethogramError').textContent=$('error').textContent;
};
$('exportEthogram').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify({version:1,ethogram:cleanEthogram()},null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='fieldnote-ethogram.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('importEthogram').onchange=async event=>{try{
  const file=event.target.files[0];if(!file)return;if(file.size>200000)throw new Error('Ethogram file is too large');
  const data=JSON.parse(await file.text());if(!Array.isArray(data.ethogram)||data.ethogram.length>50)throw new Error('Expected an ethogram JSON with at most 50 behaviors');
  for(const entry of data.ethogram){if(!entry||typeof entry.name!=='string'||!['state','point'].includes(entry.type)||typeof entry.key!=='string'||typeof entry.description!=='string')throw new Error('Invalid behavior definition');}
  // Append only: importing cannot overwrite keys or definitions of existing work.
  for(const entry of data.ethogram)if(!ethogramDraft.some(e=>e.name===entry.name))ethogramDraft.push({...entry});
  renderEthogram();$('ethogramError').textContent='Imported new behaviors. Resolve any duplicate keys, then save.';
}catch(exc){$('ethogramError').textContent=exc.message;}finally{event.target.value='';}};
