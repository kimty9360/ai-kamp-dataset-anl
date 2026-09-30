'use strict';
const DATA = JSON.parse(document.getElementById('embeddedData').textContent);
const $ = id => document.getElementById(id);
const F = DATA.features;
const recordsById = new Map(DATA.records.map(r => [r.id,r]));
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (x,n=3) => Number.isFinite(x) ? x.toLocaleString('ko-KR',{maximumFractionDigits:n}) : '—';
const config = {responsive:true,displaylogo:false,scrollZoom:false,toImageButtonOptions:{format:'png',scale:2},modeBarButtonsToRemove:['select2d','lasso2d']};
let selected=[], tab='timeline', observer=null, generation=0, correlation=null, correlationGeneration=-1, correlationMethod='';
let normalVisible=true;
let featureAudit=[], analysisIndices=[];
function updateFeatureAudit(){
 const scope=candidates();
 featureAudit=F.map((f,index)=>{
  const values=scope.map(r=>r.values[index]).filter(Number.isFinite), unique=new Set(values);
  const reason=!scope.length?'':!values.length?'모두 결측':values.length===scope.length&&unique.size===1&&values[0]===0?'모두 0':scope.length>=2&&values.length===scope.length&&unique.size===1?'상수':'';
  return {index,name:f.name,reason,value:values[0]};
 });
 analysisIndices=featureAudit.filter(f=>!f.reason||$('showExcluded').checked).map(f=>f.index);
 const excluded=featureAudit.filter(f=>f.reason);
 $('preprocessingNote').textContent=`기본 정리: 완전 중복 ${num(DATA.duplicates,0)}행 제외 · 공정값 결측 ${DATA.audit.numeric_missing_cells}개. 선택 차종·좌우의 전체 기간 기준으로 0/상수/전체 결측 ${excluded.length}개 컬럼을 ${$('showExcluded').checked?'포함':'제외'}하여 ${analysisIndices.length}개를 분석합니다. 날짜 범위만 바꿔도 제외 기준은 유지됩니다. 관측 1건만으로는 비영(非零) 상수 여부를 판단하지 않습니다.`;
 $('excludedColumns').innerHTML=excluded.length?'<ul>'+excluded.map(f=>`<li>${esc(f.name)}: ${esc(f.reason)}${Number.isFinite(f.value)?' (값 '+num(f.value,6)+')':''}</li>`).join('')+'</ul>':'<p>제외할 컬럼이 없습니다.</p>';
}

const reasonStyle = {'가스':{color:'#dc2626',symbol:'x'},'미성형':{color:'#9333ea',symbol:'triangle-up'},'초기허용불량':{color:'#ea580c',symbol:'diamond'},'사유 미기재':{color:'#be123c',symbol:'cross'}};
function candidates(){return DATA.records.filter(r => ($('product').value==='ALL'||r.product===$('product').value)&&($('side').value==='ALL'||r.side===$('side').value));}
function resetDates(){const c=candidates(); const dates=c.map(r=>r.time.slice(0,10)).sort();$('start').value=dates[0]||'';$('end').value=dates.at(-1)||'';for(const id of ['start','end']){$(id).min=dates[0]||'';$(id).max=dates.at(-1)||'';}}
function update(){
 const start=$('start').value,end=$('end').value;
 const invalid=start&&end&&start>end;$('filterError').hidden=!invalid;$('filterError').textContent='시작일이 종료일보다 늦습니다. 날짜를 다시 선택해 주세요.';
 const beforeReview=invalid?[]:candidates().filter(r=>(!start||r.time.slice(0,10)>=start)&&(!end||r.time.slice(0,10)<=end));
 const flagged=beforeReview.filter(r=>r.reviewFlag),exclude=$('reviewMode').value==='exclude';
 selected=exclude?beforeReview.filter(r=>!r.reviewFlag):beforeReview;
 $('reviewNote').textContent=`${exclude?'검토 대상 제외 · 비교용':'전체 포함 · 원본 기준'}: 선택 범위 ${num(beforeReview.length,0)}건 중 검토 대상 ${flagged.length}건, 제외 ${exclude?flagged.length:0}건 → 분석 ${num(selected.length,0)}건. 그래프·이동 통계·상관계수·CSV에 같은 기준을 적용합니다. 제외 후 불량 비율은 분석 대상 구성이 달라져 변할 수 있습니다.`;
 $('reviewDetails').hidden=!flagged.length;
 $('rpmNote').hidden=!['CN7','ALL'].includes($('product').value);
 $('reviewTable').innerHTML='<table><thead><tr><th>기록 시각</th><th>좌우</th><th>원본 판정</th><th>사유</th></tr></thead><tbody>'+flagged.map(r=>`<tr class="clickable" tabindex="0" data-id="${esc(r.id)}"><td>${esc(r.time)}</td><td>${esc(r.side)}</td><td>${r.label?'불량':'양품'}</td><td>다변수 값군 검토</td></tr>`).join('')+'</tbody></table>';
 $('reviewTable').querySelectorAll('[data-id]').forEach(el=>{el.onclick=()=>showRecord(el.dataset.id);el.onkeydown=e=>{if(e.key==='Enter')showRecord(el.dataset.id);};});
 generation++;correlation=null;focusTime=null;updateFeatureAudit();
 const defects=selected.filter(r=>r.label===1).length,days=new Set(selected.map(r=>r.time.slice(0,10))),equipment=new Set(selected.map(r=>r.equipment));
 $('recordCount').textContent=num(selected.length,0)+'건';$('defectCount').textContent=num(defects,0)+'건';$('defectRate').textContent=selected.length?num(defects/selected.length*100,2)+'%':'—';$('coverageCount').textContent=days.size+'일 · '+equipment.size+'대';
 const product=$('product').value;
 let note=selected.length?`선택 기간 ${selected[0].time} ~ ${selected.at(-1).time} · ${[...equipment].join(', ')} · 완전 중복 ID는 한 번만 집계합니다. 관측 비율은 공장 전체 불량률이 아닙니다.`:'선택한 조건의 기록이 없습니다. 차종·좌우·날짜를 확인해 주세요.';
 if(product==='ALL')note+=' 서로 다른 부품·설비를 섞은 참고 자료입니다. 제품별 분석을 우선하세요.';
 if(['JX1','SP2'].includes(product))note+=' 이 제품은 고유 양품 1건만 있어 상관계수를 계산할 수 없습니다.';
 $('scopeNote').textContent=note;
 renderActive();
}
function switchTab(next){tab=next;document.querySelectorAll('[data-tab]').forEach(b=>{b.classList.toggle('active',b.dataset.tab===tab);b.setAttribute('aria-pressed',b.dataset.tab===tab?'true':'false');});for(const t of ['timeline','correlation','defects','guide'])$(t+'Panel').hidden=t!==tab;if(observer){observer.disconnect();observer=null;}clearTimePlots();renderActive();}
function renderActive(){if(tab==='timeline'){renderOverview();renderPairSummary();renderTimelines();}if(tab==='correlation')renderCorrelation();if(tab==='defects')renderDefects();}
function renderOverview(){
 const dates=[...new Set(selected.map(r=>r.time.slice(0,10)))].sort();const counts=new Map(dates.map(d=>[d,[0,0]]));selected.forEach(r=>counts.get(r.time.slice(0,10))[r.label]++);
 Plotly.react($('overview'),[{x:dates,y:dates.map(d=>counts.get(d)[0]),type:'bar',name:'양품 기록',marker:{color:'#91b8e9'}},{x:dates,y:dates.map(d=>counts.get(d)[1]),type:'bar',name:'불량 기록',text:dates.map(d=>counts.get(d)[1]?counts.get(d)[1]+' 불량':''),textposition:'outside',textfont:{color:'#b91c1c',size:11},cliponaxis:false,marker:{color:'#dc2626'}}],{barmode:'stack',height:250,margin:{l:65,r:15,t:15,b:50},paper_bgcolor:'white',plot_bgcolor:'#fafcff',xaxis:{type:'category',title:'관측 날짜'},yaxis:{title:'기록 수',rangemode:'tozero'},legend:{orientation:'h',y:1.13},annotations:dates.length?[]:[{text:'표시할 기록이 없습니다',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false}]},config);
 $('overview').removeAllListeners?.('plotly_click');$('overview').on('plotly_click',event=>{const date=event.points[0]?.x;if(date){$('start').value=date;$('end').value=date;update();}});
}
function clearTimePlots(){document.querySelectorAll('.time-plot').forEach(el=>{if(el._fullLayout)Plotly.purge(el);});}
function renderTimelines(){
 if(observer)observer.disconnect();clearTimePlots();$('timelineCharts').replaceChildren();
 const query=$('featureSearch').value.trim().toLowerCase(),category=$('category').value;
 const features=analysisIndices.map(i=>({...F[i],index:i})).filter(f=>(category==='ALL'||f.category===category)&&(!query||(f.name+' '+f.label).toLowerCase().includes(query)));
 $('featureCount').textContent=`${features.length} / 분석 ${analysisIndices.length}개 (원본 ${F.length}개)`;
 if(!selected.length||!features.length){$('timelineCharts').innerHTML='<div class="empty">'+(!selected.length?'선택한 조건의 기록이 없습니다.':'검색 조건에 맞는 컬럼이 없습니다.')+'</div>';return;}
 const version=generation;
 observer=new IntersectionObserver(entries=>entries.forEach(e=>{
  const el=e.target;if(version!==generation||tab!=='timeline')return;
  if(e.isIntersecting){if(!el._fullLayout&&!el.dataset.busy){el.dataset.busy='1';drawTimePlot(el,Number(el.dataset.index),version).finally(()=>{delete el.dataset.busy;});}}
  else if(el._fullLayout){Plotly.purge(el);}
 }),{rootMargin:'300px 0px'});
 for(const f of features){
  const card=document.createElement('article');card.className='chart-card';card.innerHTML=`<div class="card-header"><div><h3>${esc(f.label)}</h3><p class="technical">${esc(f.name)} · ${esc(f.category)}</p></div><span class="unit">${esc(f.unit)}</span></div><p class="note">${esc(f.note)}</p><div class="time-plot" data-index="${f.index}" aria-label="${esc(f.name)} 시간 그래프"></div>`;
  $('timelineCharts').append(card);observer.observe(card.querySelector('.time-plot'));
 }
}
function pairedEvents(rows){
 const groups=new Map();
 for(const r of rows){const key=[r.equipment,r.product,r.time].join('|');if(!groups.has(key))groups.set(key,[]);groups.get(key).push(r);}
 return [...groups.values()].filter(g=>g.length===2&&new Set(g.map(r=>r.side)).size===2).map(g=>({time:g[0].time,product:g[0].product,equipment:g[0].equipment,lh:g.find(r=>r.side==='LH'),rh:g.find(r=>r.side==='RH')}));
}
let focusTime=null;
function renderPairSummary(){
 const pairs=pairedEvents(selected),different=pairs.filter(p=>p.lh.label!==p.rh.label);
 const left=different.filter(p=>p.lh.label===1).length,right=different.length-left;
 $('pairSummary').textContent=`같은 설비·차종·시각의 좌우 쌍 ${pairs.length}개 · LH만 불량 ${left}개 · RH만 불량 ${right}개. 아래 시각을 선택하면 모든 컬럼에서 전후 15분을 확대합니다. 상대쪽 기록이 없으면 양품으로 간주하지 않습니다.`;
 const picker=$('pairEvent');const previous=focusTime?picker.value:'';
 picker.innerHTML='<option value="">한쪽만 불량인 시각 선택</option>'+different.map(p=>`<option value="${esc(p.equipment+'|'+p.product+'|'+p.time)}">${esc(p.time)} · ${esc(p.product)} · ${p.lh.label?'LH 불량 / RH 양품':'LH 양품 / RH 불량'}</option>`).join('');
 if([...picker.options].some(o=>o.value===previous))picker.value=previous;
}
// Trailing windows include the current observation, never future rows.
// Start fresh at the selected period boundary, a >10-minute gap, or a missing value.
function rollingTemperature(group,index,windowSize){
 const x=[],median=[],std=[];let window=[],prev=null;
 for(const r of group){
  if(prev&&(new Date(r.time.replace(' ','T')+'Z')-new Date(prev.time.replace(' ','T')+'Z'))>600000){window=[];x.push(null);median.push(null);std.push(null);}
  const value=r.values[index];
  if(!Number.isFinite(value))window=[];else{window.push(value);if(window.length>windowSize)window.shift();}
  x.push(r.time);
  if(window.length===windowSize){const sorted=[...window].sort((a,b)=>a-b),mean=window.reduce((a,b)=>a+b,0)/window.length;median.push(sorted[Math.floor(window.length/2)]);std.push(Math.sqrt(window.reduce((a,b)=>a+(b-mean)**2,0)/window.length));}
  else{median.push(null);std.push(null);}
  prev=r;
 }
 return {x,median,std};
}
async function drawTimePlot(el,index,version){
 const f=F[index],traces=[],annotations=[],shapes=[];
 const windowSize=Number($('temperatureWindow').value),trend=f.category==='온도'&&windowSize>0;
 el.style.height=trend?'620px':'400px';
 const sides=$('side').value==='ALL'?['LH','RH']:[$('side').value];
 let range=selected.length>1&&selected[0].time!==selected.at(-1).time?[selected[0].time,selected.at(-1).time]:undefined;
 if(focusTime){const t=new Date(focusTime.replace(' ','T')+'Z').getTime();range=[new Date(t-15*60000).toISOString().slice(0,19).replace('T',' '),new Date(t+15*60000).toISOString().slice(0,19).replace('T',' ')];}
 const visibleRows=focusTime?selected.filter(r=>r.time>=range[0]&&r.time<=range[1]):selected;
 const valid=visibleRows.map(r=>r.values[index]).filter(Number.isFinite);
 const min=valid.length?Math.min(...valid):0,max=valid.length?Math.max(...valid):1,pad=(max-min)*.08||Math.max(Math.abs(min)*.02,.1);
 const yrange=[min-pad,max+pad];
 const layout={height:trend?620:400,margin:{l:65,r:25,t:48,b:62},paper_bgcolor:'white',plot_bgcolor:'#fafcff',showlegend:false,hovermode:'closest',annotations,shapes};
 for(let panel=0;panel<sides.length;panel++){
  const side=sides[panel],axis=panel===0?'':String(panel+1),xref='x'+axis,yref='y'+axis;
  const domain=sides.length===1?[0,1]:(panel===0?[0,.455]:[.545,1]);
  layout['xaxis'+axis]={type:'date',domain,range,title:trend?'':'기록 시각',anchor:yref,gridcolor:'#e7edf5',...(panel?{matches:'x'}:{})};
  layout['yaxis'+axis]={domain:trend?[.43,1]:[0,1],range:yrange,title:panel?'':(f.unit.length>22?'원자료 수치':f.unit),anchor:xref,zeroline:false,gridcolor:'#e7edf5',...(panel?{matches:'y'}:{})};
  annotations.push({text:'<b>'+side+' · '+(side==='LH'?'왼쪽':'오른쪽')+'</b>',xref:'paper',yref:'paper',x:(domain[0]+domain[1])/2,y:trend?1.055:1.12,showarrow:false,font:{size:15,color:side==='LH'?'#2563eb':'#0891b2'}});
  const bottomAxis=String(panel+3),bottomX='x'+bottomAxis,bottomY='y'+bottomAxis;
  if(trend){
   layout['xaxis'+bottomAxis]={type:'date',domain,range,anchor:bottomY,matches:'x',title:'기록 시각',gridcolor:'#e7edf5'};
   layout['yaxis'+bottomAxis]={domain:[0,.25],anchor:bottomX,rangemode:'tozero',title:panel?'':'표준편차',gridcolor:'#e7edf5',...(panel?{matches:'y3'}:{})};
   annotations.push({text:'최근 '+windowSize+'개 관측의 변동 폭 · 이동 표준편차',xref:'paper',yref:'paper',x:(domain[0]+domain[1])/2,y:.31,showarrow:false,font:{size:12,color:'#64748b'}});
  }
  const rows=selected.filter(r=>r.side===side),color=side==='LH'?'#2563eb':'#0891b2';
  const groups=new Map();for(const r of rows){const key=r.equipment+'|'+r.product;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(r);}
  function makeTrace(group,name,symbol,size,traceColor){return {type:'scatter',mode:'markers',name,x:group.map(r=>r.time),y:group.map(r=>r.values[index]),xaxis:xref,yaxis:yref,customdata:group.map(r=>[r.id,r.side,r.label?'불량':'양품',r.reason||'없음',r.equipment]),marker:{color:traceColor,opacity:trend&&size<=6?.4:1,symbol,size,line:{width:size>6?1:0,color:traceColor}},hovertemplate:'기록: %{x}<br>값: %{y:.6g}<br>%{customdata[1]} · %{customdata[2]}<br>사유: %{customdata[3]}<extra>'+esc(name)+'</extra>'};}
  for(const group of groups.values()){
   // Connect chronological observations only inside a <=10 minute observed segment.
   const x=[],y=[];let prev=null;
   for(const r of group){if(prev&&(new Date(r.time.replace(' ','T')+'Z')-new Date(prev.time.replace(' ','T')+'Z'))>600000){x.push(null);y.push(null);}x.push(r.time);y.push(r.values[index]);prev=r;}
   traces.push({type:'scatter',mode:'lines',name:side+' 관측 연결선',x,y,xaxis:xref,yaxis:yref,line:{color,width:trend?.8:1.3},opacity:trend?.35:1,hoverinfo:'skip',connectgaps:false});
   if(trend){
    const rolling=rollingTemperature(group,index,windowSize);
    traces.push({type:'scatter',mode:'lines',name:side+' 이동 중앙값',meta:{kind:'median',windowSize},x:rolling.x,y:rolling.median,xaxis:xref,yaxis:yref,line:{color:side==='LH'?'#1e3a8a':'#115e59',width:3},connectgaps:false,hovertemplate:'%{x}<br>최근 '+windowSize+'개 중앙값: %{y:.6g}<extra>'+side+'</extra>'});
    traces.push({type:'scatter',mode:'lines',name:side+' 이동 표준편차',meta:{kind:'std',windowSize},x:rolling.x,y:rolling.std,xaxis:bottomX,yaxis:bottomY,line:{color,width:2},fill:'tozeroy',connectgaps:false,hovertemplate:'%{x}<br>최근 '+windowSize+'개 표준편차: %{y:.6g}<extra>'+side+'</extra>'});
   }
  }
  if(normalVisible){const good=rows.filter(r=>r.label===0&&!r.reviewFlag);if(good.length)traces.push(makeTrace(good,side+' 양품','circle',5,color));}
  const reviewGood=rows.filter(r=>r.label===0&&r.reviewFlag);if(normalVisible&&reviewGood.length)traces.push(makeTrace(reviewGood,side+' 검토 대상 · 원본 양품','square-open',11,'#a16207'));
  for(const [reason,style] of Object.entries(reasonStyle)){
   const bad=rows.filter(r=>r.label===1&&(r.reason||'사유 미기재')===reason);
   if(bad.length)traces.push(makeTrace(bad,side+' '+reason,style.symbol,12,style.color));
  }
  if(!rows.length)annotations.push({text:side+' 관측 없음',xref:'paper',yref:'paper',x:(domain[0]+domain[1])/2,y:.5,showarrow:false,font:{color:'#64748b'}});
  if(focusTime)shapes.push({type:'line',xref,yref:'paper',x0:focusTime,x1:focusTime,y0:0,y1:1,line:{color:'#b91c1c',width:1.5,dash:'dot'}});
 }
 await Plotly.newPlot(el,traces,layout,config);
 if(version!==generation||tab!=='timeline'||!el.isConnected){Plotly.purge(el);return;}
 el.on('plotly_click',event=>{const id=event.points[0]?.customdata?.[0];if(id)showRecord(id);});
}

function pearson(a,b){
 let n=0,ma=0,mb=0;for(let i=0;i<a.length;i++)if(Number.isFinite(a[i])&&Number.isFinite(b[i])){n++;ma+=a[i];mb+=b[i];}if(n<2)return null;ma/=n;mb/=n;
 let xx=0,yy=0,xy=0;let amin=Infinity,amax=-Infinity,bmin=Infinity,bmax=-Infinity;
 for(let i=0;i<a.length;i++)if(Number.isFinite(a[i])&&Number.isFinite(b[i])){const x=a[i]-ma,y=b[i]-mb;xx+=x*x;yy+=y*y;xy+=x*y;amin=Math.min(amin,a[i]);amax=Math.max(amax,a[i]);bmin=Math.min(bmin,b[i]);bmax=Math.max(bmax,b[i]);}
 if(amin===amax||bmin===bmax||xx===0||yy===0)return null;
 return Math.max(-1,Math.min(1,xy/Math.sqrt(xx*yy)));
}
function ranks(a){const sorted=a.map((v,i)=>({v,i})).filter(o=>Number.isFinite(o.v)).sort((a,b)=>a.v-b.v);const result=Array(a.length).fill(null);for(let i=0;i<sorted.length;){let end=i+1;while(end<sorted.length&&sorted[end].v===sorted[i].v)end++;const rank=(i+end-1)/2+1;for(let j=i;j<end;j++)result[sorted[j].i]=rank;i=end;}return result;}
function computeCorrelation(rows,method){
 const vectors=F.map((_,i)=>rows.map(r=>r.values[i]));const complete=vectors.every(a=>a.every(Number.isFinite));const ranked=method==='spearman'?vectors.map(ranks):vectors;
 const matrix=vectors.map(()=>Array(F.length).fill(null));
 for(let i=0;i<F.length;i++)for(let j=i;j<F.length;j++){
  let value;
  if(method==='spearman'&&!complete){const positions=rows.map((_,k)=>k).filter(k=>Number.isFinite(vectors[i][k])&&Number.isFinite(vectors[j][k]));value=pearson(ranks(positions.map(k=>vectors[i][k])),ranks(positions.map(k=>vectors[j][k])));}
  else value=pearson(ranked[i],ranked[j]);matrix[i][j]=matrix[j][i]=value;
 }
 const labels=rows.map(r=>r.label);const target=vectors.map(a=>{const positions=a.map((_,i)=>i).filter(i=>Number.isFinite(a[i]));const x=positions.map(i=>a[i]),y=positions.map(i=>labels[i]);return method==='spearman'?pearson(ranks(x),ranks(y)):pearson(x,y);});
 return {matrix,target};
}
function renderCorrelation(){
 const method=$('method').value;
 if(!correlation||correlationGeneration!==generation||correlationMethod!==method){correlation=computeCorrelation(selected,method);correlationGeneration=generation;correlationMethod=method;}
 const indices=analysisIndices.filter(i=>correlation.matrix[i][i]!==null);const defects=selected.filter(r=>r.label).length;
 $('correlationNote').textContent=`${num(selected.length,0)}개 기록 · 불량 ${defects}개 · ${method==='pearson'?'Pearson':'Spearman'} · 기본 정리 후 ${analysisIndices.length}개 컬럼 중 현재 기간에서 계산 가능한 ${indices.length}개를 그림에 표시합니다. 계산 불가 값은 표에 ‘—’로 표시합니다. 좌우의 공정값 중복과 날짜별 운전 조건이 상관에 영향을 줄 수 있습니다.`;
 if(indices.length&&selected.length>=2){
  $('correlationPlot').style.minWidth=window.innerWidth<800?'1000px':'0';const names=indices.map(i=>F[i].name),z=indices.map(i=>indices.map(j=>correlation.matrix[i][j]));
  Plotly.react($('correlationPlot'),[{type:'heatmap',x:names,y:names,z,zmin:-1,zmax:1,colorscale:'RdBu',reversescale:true,texttemplate:'%{z:.2f}',textfont:{size:9},hovertemplate:'%{y}<br>↔ %{x}<br>상관계수: %{z:.6f}<extra></extra>',colorbar:{title:'상관',thickness:15}}],{height:Math.max(700,indices.length*25+240),margin:{l:205,r:65,t:20,b:220},xaxis:{tickangle:-90,tickfont:{size:10}},yaxis:{autorange:'reversed',tickfont:{size:10}},paper_bgcolor:'white'},config);
 }else{if($('correlationPlot')._fullLayout)Plotly.purge($('correlationPlot'));$('correlationPlot').style.minWidth='0';$('correlationPlot').innerHTML='<div class="empty">변동이 있는 관측이 부족해 상관계수를 계산할 수 없습니다.</div>';}
 const format=x=>Number.isFinite(x)?x.toFixed(3):'—';
 $('correlationTable').innerHTML='<table><thead><tr><th>공정 컬럼</th>'+analysisIndices.map(i=>'<th>'+esc(F[i].name)+'</th>').join('')+'</tr></thead><tbody>'+analysisIndices.map(i=>'<tr><td>'+esc(F[i].name)+'</td>'+analysisIndices.map(j=>'<td>'+format(correlation.matrix[i][j])+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
 const sorted=analysisIndices.map(i=>({...F[i],value:correlation.target[i]})).sort((a,b)=>(b.value===null?-1:Math.abs(b.value))-(a.value===null?-1:Math.abs(a.value)));
 $('targetTable').innerHTML='<table><thead><tr><th>공정 컬럼</th><th>설명</th><th>불량과의 상관</th></tr></thead><tbody>'+sorted.map(f=>'<tr><td>'+esc(f.name)+'</td><td>'+esc(f.label)+'</td><td>'+format(f.value)+'</td></tr>').join('')+'</tbody></table>';
}
function renderDefects(){
 const rows=selected.filter(r=>r.label===1);$('defectTable').innerHTML=rows.length?'<table><thead><tr><th>기록 시각</th><th>차종</th><th>좌우</th><th>설비</th><th>불량 사유</th></tr></thead><tbody>'+rows.map(r=>`<tr class="clickable" tabindex="0" data-id="${esc(r.id)}"><td>${esc(r.time)}</td><td>${esc(r.product)}</td><td>${esc(r.side)}</td><td>${esc(r.equipment)}</td><td>${esc(r.reason||'사유 미기재')}</td></tr>`).join('')+'</tbody></table>':'<div class="empty">선택한 조건에는 불량 기록이 없습니다.</div>';
 $('defectTable').querySelectorAll('[data-id]').forEach(el=>{el.onclick=()=>showRecord(el.dataset.id);el.onkeydown=e=>{if(e.key==='Enter')showRecord(el.dataset.id);};});
}
function showRecord(id){const r=recordsById.get(id);if(!r)return;$('recordDetails').innerHTML=`<p><strong>${esc(r.time)} · ${esc(r.product)} ${esc(r.side)}</strong><br>${esc(r.part)}<br>${esc(r.equipment)} / ${esc(r.equipmentName)} · ${r.label?'불량':'양품'} · 사유 ${esc(r.reason||'없음')}</p><p class="small">ID: ${esc(r.id)} · 데이터 검토: ${r.reviewFlag?'대상 (불량 재판정 아님)':'해당 없음'}</p><div class="table-scroll"><table><thead><tr><th>공정 컬럼</th><th>설명</th><th>원자료 값</th><th>단위</th></tr></thead><tbody>${F.map((f,i)=>`<tr><td>${esc(f.name)}</td><td>${esc(f.label)}</td><td>${num(r.values[i],6)}</td><td>${esc(f.unit)}</td></tr>`).join('')}</tbody></table></div>`;$('recordDialog').showModal();}
function csvCell(v){return '"'+String(v??'').replace(/"/g,'""')+'"';}
function downloadCSV(rows,name){const blob=new Blob(['\ufeff'+rows.map(row=>row.map(csvCell).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8;'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('exportData').onclick=()=>downloadCSV([['TimeStamp','PART_NAME','EQUIP_CD','PassOrFail','Reason','Data_Review_Flag',...analysisIndices.map(i=>F[i].name)],...selected.map(r=>[r.time,r.part,r.equipment,r.label?'N':'Y',r.reason,r.reviewFlag?1:0,...analysisIndices.map(i=>r.values[i])])],`labeled_${$('product').value}_${$('side').value}_${$('reviewMode').value}.csv`);
$('exportCorrelation').onclick=()=>{if(!correlation)renderCorrelation();downloadCSV([['feature',...analysisIndices.map(i=>F[i].name)],...analysisIndices.map(i=>[F[i].name,...analysisIndices.map(j=>correlation.matrix[i][j])])],`correlation_${$('product').value}_${$('side').value}_${$('method').value}_${$('reviewMode').value}.csv`);};
$('reviewMode').onchange=update;
$('showExcluded').onchange=update;
$('closeDialog').onclick=()=>$('recordDialog').close();
$('product').onchange=()=>{resetDates();update();};$('side').onchange=()=>{resetDates();update();};$('start').onchange=update;$('end').onchange=update;$('resetDates').onclick=()=>{resetDates();update();};
$('pairEvent').onchange=()=>{focusTime=$('pairEvent').value?$('pairEvent').value.split('|').at(-1):null;renderTimelines();};
$('clearFocus').onclick=()=>{focusTime=null;$('pairEvent').value='';renderTimelines();};
$('temperatureWindow').onchange=renderTimelines;
$('category').onchange=renderTimelines;let searchTimer;$('featureSearch').oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(renderTimelines,180);};$('showNormal').onchange=()=>{normalVisible=$('showNormal').checked;renderTimelines();};$('method').onchange=renderCorrelation;
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>switchTab(b.dataset.tab));
$('provenance').textContent=`자료: ${DATA.source} · 원본 ${num(DATA.rawRows,0)}행 / 중복 제외 ${num(DATA.uniqueRows,0)}기록 · 공정 컬럼 ${F.length}개 · 생성 ${DATA.builtAt}\n출처 ZIP SHA-256: ${DATA.sourceHash}`;
resetDates();update();
// Expose read-only diagnostics for automated numerical and UI checks.
window.dashboardDiagnostics={rollingTemperature,getSelection:()=>selected.map(r=>r.id),getCorrelation:()=>correlation,pearson,ranks,computeCorrelation,featureCount:F.length,getPairs:()=>pairedEvents(selected),getAnalysisIndices:()=>[...analysisIndices],getFeatureAudit:()=>featureAudit.map(f=>({...f}))};
