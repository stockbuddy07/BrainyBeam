// ── Tab Navigation ──
function switchTab(id){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelector(`[data-tab="${id}"]`).classList.add('active');
}

// ── Animated Counters ──
function animateCount(el,target,dur=1200){
  let start=0,startTime=null;
  const isFloat=String(target).includes('.');
  function step(ts){
    if(!startTime)startTime=ts;
    const p=Math.min((ts-startTime)/dur,1);
    const val=Math.floor(p*target);
    el.textContent=isFloat?(p*target).toFixed(1):val.toLocaleString();
    if(p<1)requestAnimationFrame(step);
    else el.textContent=isFloat?target.toFixed(1):target.toLocaleString();
  }
  requestAnimationFrame(step);
}

// ── Chart Colors ──
const C={blue:'#3b82f6',blue2:'#60a5fa',green:'#34d399',orange:'#fb923c',purple:'#a78bfa',red:'#f87171',cyan:'#22d3ee',pink:'#f472b6'};

// ── OVERVIEW CHARTS ──
function initOverview(){
  // Counters
  document.querySelectorAll('[data-count]').forEach(el=>{
    animateCount(el,parseFloat(el.dataset.count));
  });

  // Donut
  new Chart(document.getElementById('chartDonut'),{
    type:'doughnut',
    data:{labels:['Yes (Subscribed)','No (Not Subscribed)'],datasets:[{data:[5289,5873],backgroundColor:[C.green,C.blue],borderWidth:0,hoverOffset:8}]},
    options:{responsive:true,maintainAspectRatio:false,cutout:'65%',plugins:{legend:{position:'right',labels:{color:'#94a3b8',padding:14,font:{size:12}}}}}
  });

  // Age Distribution
  new Chart(document.getElementById('chartAge'),{
    type:'bar',
    data:{labels:['18-25','26-30','31-35','36-40','41-45','46-50','51-55','56-60','60+'],datasets:[{label:'Count',data:[892,2156,2534,1987,1423,896,634,387,253],backgroundColor:C.blue+'cc',borderRadius:4}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#94a3b8'},grid:{display:false}},y:{ticks:{color:'#94a3b8'},grid:{color:'rgba(255,255,255,.04)'}}}}
  });

  // Balance Distribution
  new Chart(document.getElementById('chartBalance'),{
    type:'bar',
    data:{labels:['<0','0-500','500-1k','1k-2k','2k-3k','3k-5k','5k-10k','10k+'],datasets:[{label:'Count',data:[612,2845,1967,1843,1124,1089,978,704],backgroundColor:C.orange+'cc',borderRadius:4}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#94a3b8'},grid:{display:false}},y:{ticks:{color:'#94a3b8'},grid:{color:'rgba(255,255,255,.04)'}}}}
  });

  // Job vs Response
  new Chart(document.getElementById('chartJob'),{
    type:'bar',
    data:{labels:['admin','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown'],
      datasets:[
        {label:'No',data:[2350,1575,298,220,1673,205,312,692,153,1471,218,106],backgroundColor:C.red+'99'},
        {label:'Yes',data:[2174,872,270,186,1661,428,222,504,485,1267,238,82],backgroundColor:C.green+'cc'}
      ]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#94a3b8'}}},scales:{x:{ticks:{color:'#94a3b8',maxRotation:45},grid:{display:false}},y:{ticks:{color:'#94a3b8'},grid:{color:'rgba(255,255,255,.04)'}}}}
  });

  // Education vs Response
  new Chart(document.getElementById('chartEdu'),{
    type:'bar',
    data:{labels:['primary','secondary','tertiary','unknown'],
      datasets:[
        {label:'No',data:[873,2671,1951,378],backgroundColor:C.red+'99'},
        {label:'Yes',data:[685,2431,1943,306],backgroundColor:C.green+'cc'}
      ]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#94a3b8'}}},scales:{x:{ticks:{color:'#94a3b8'},grid:{display:false}},y:{ticks:{color:'#94a3b8'},grid:{color:'rgba(255,255,255,.04)'}}}}
  });
}

// ── MODEL INSIGHT CHARTS ──
function initModelInsight(){
  // Feature Importance
  new Chart(document.getElementById('chartFeatures'),{
    type:'bar',
    data:{labels:['duration','balance','age','day','campaign','pdays','previous','poutcome_success','housing_yes','job_management'],
      datasets:[{data:[0.3241,0.1187,0.1043,0.0892,0.0634,0.0521,0.0398,0.0312,0.0287,0.0243],
        backgroundColor:[C.green,C.blue,C.orange,C.purple,C.cyan,C.pink,C.blue2,C.green,C.orange,C.purple],borderRadius:4}]},
    options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#94a3b8'},grid:{color:'rgba(255,255,255,.04)'}},y:{ticks:{color:'#f1f5f9',font:{size:12}},grid:{display:false}}}}
  });

  // Radar
  new Chart(document.getElementById('chartRadar'),{
    type:'radar',
    data:{labels:['Accuracy','Precision','Recall','F1-Score'],
      datasets:[
        {label:'Gradient Boosting',data:[82,81,81,81],borderColor:C.green,backgroundColor:C.green+'22',pointBackgroundColor:C.green},
        {label:'XGBoost',data:[81,80,81,80],borderColor:C.blue,backgroundColor:C.blue+'22',pointBackgroundColor:C.blue},
        {label:'Random Forest',data:[80,80,79,79],borderColor:C.purple,backgroundColor:C.purple+'22',pointBackgroundColor:C.purple}
      ]},
    options:{responsive:true,maintainAspectRatio:false,scales:{r:{angleLines:{color:'rgba(255,255,255,.08)'},grid:{color:'rgba(255,255,255,.06)'},pointLabels:{color:'#94a3b8',font:{size:12}},ticks:{display:false},suggestedMin:60,suggestedMax:90}},plugins:{legend:{labels:{color:'#94a3b8'}}}}
  });

  // Confusion Matrix (canvas draw)
  const ctx=document.getElementById('chartCM').getContext('2d');
  const cm=[[892,218],[198,925]];
  const sz=120,pad=60;
  ctx.font='bold 13px Inter';ctx.fillStyle='#94a3b8';
  ctx.fillText('Predicted No',pad+20,20);ctx.fillText('Predicted Yes',pad+sz+20,20);
  ctx.save();ctx.translate(15,pad+sz);ctx.rotate(-Math.PI/2);ctx.fillText('Actual No',20,0);ctx.restore();
  ctx.save();ctx.translate(15,pad+sz+sz);ctx.rotate(-Math.PI/2);ctx.fillText('Actual Yes',20,0);ctx.restore();
  const colors=[[C.blue,0.7],[C.blue,0.25],[C.blue,0.25],[C.green,0.7]];
  for(let r=0;r<2;r++)for(let c=0;c<2;c++){
    const x=pad+c*sz,y=30+r*sz;
    ctx.fillStyle=colors[r*2+c][0];ctx.globalAlpha=colors[r*2+c][1];
    ctx.fillRect(x,y,sz-4,sz-4);ctx.globalAlpha=1;
    ctx.fillStyle='#fff';ctx.font='bold 24px Inter';
    ctx.fillText(cm[r][c],x+40,y+60);
  }
}

// ── BULK SCANNER ──
function downloadTemplate(){
  const cols='age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome\n';
  const sample='38,management,married,tertiary,no,2000,yes,no,cellular,15,may,250,2,-1,0,unknown\n';
  const blob=new Blob([cols+sample],{type:'text/csv'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='bank_campaign_template.csv';a.click();
}

function setupUpload(){
  const zone=document.getElementById('uploadZone');
  const input=document.getElementById('csvInput');
  zone.addEventListener('click',()=>input.click());
  zone.addEventListener('dragover',e=>{e.preventDefault();zone.classList.add('dragover')});
  zone.addEventListener('dragleave',()=>zone.classList.remove('dragover'));
  zone.addEventListener('drop',e=>{e.preventDefault();zone.classList.remove('dragover');handleFile(e.dataTransfer.files[0])});
  input.addEventListener('change',e=>handleFile(e.target.files[0]));
}

function handleFile(file){
  if(!file)return;
  const ext=file.name.split('.').pop().toLowerCase();

  if(ext==='csv'){
    // CSV parsing
    const reader=new FileReader();
    reader.onload=e=>{
      const lines=e.target.result.trim().split('\n');
      const headers=lines[0].split(',').map(h=>h.trim().toLowerCase());
      const rows=[];
      for(let i=1;i<lines.length;i++){
        if(!lines[i].trim())continue;
        const vals=lines[i].split(',');
        const row={};headers.forEach((h,j)=>row[h]=vals[j]?.trim()||'');
        row._pred=predictRow(row);
        rows.push(row);
      }
      if(rows.length===0){alert('No data rows found in CSV.');return;}
      showBulkResults(rows);
    };
    reader.readAsText(file);
  } else if(['xlsx','xls','xlsm'].includes(ext)){
    // Excel parsing using SheetJS
    const reader=new FileReader();
    reader.onload=e=>{
      try{
        const data=new Uint8Array(e.target.result);
        const workbook=XLSX.read(data,{type:'array'});
        const sheetName=workbook.SheetNames[0];
        const sheet=workbook.Sheets[sheetName];
        const jsonData=XLSX.utils.sheet_to_json(sheet,{defval:''});
        if(!jsonData||jsonData.length===0){alert('No data found in the Excel file.');return;}
        const rows=[];
        jsonData.forEach(record=>{
          const row={};
          Object.keys(record).forEach(k=>row[k.trim().toLowerCase()]=String(record[k]).trim());
          row._pred=predictRow(row);
          rows.push(row);
        });
        showBulkResults(rows);
      }catch(err){
        alert('Error reading Excel file: '+err.message);
      }
    };
    reader.readAsArrayBuffer(file);
  } else {
    alert('Unsupported file format. Please upload a CSV (.csv) or Excel (.xlsx, .xls, .xlsm) file.');
  }
}

function predictRow(r){
  let score=50;
  const dur=parseFloat(r.duration)||0;
  const bal=parseFloat(r.balance)||0;
  const age=parseFloat(r.age)||35;
  score+=dur>300?18:dur>200?10:dur>100?4:-5;
  score+=bal>2000?10:bal>1000?6:bal>500?3:bal<0?-8:0;
  score+=age>55?6:age>45?3:age<25?4:0;
  if(r.poutcome==='success')score+=20;
  if(r.housing==='no')score+=4;
  if(r.loan==='no')score+=3;
  if(r.contact==='cellular')score+=3;
  const camp=parseFloat(r.campaign)||1;if(camp>5)score-=6;
  score=Math.max(5,Math.min(95,score));
  return{yes:score>=50,conf:score};
}

function showBulkResults(rows){
  const yesCount=rows.filter(r=>r._pred.yes).length;
  const noCount=rows.length-yesCount;
  const avgConf=(rows.reduce((s,r)=>s+r._pred.conf,0)/rows.length).toFixed(1);

  document.getElementById('bulkSummary').innerHTML=`
    <div class="stat"><div class="v">${rows.length}</div><div class="l">Total Processed</div></div>
    <div class="stat"><div class="v" style="color:${C.green}">${yesCount}</div><div class="l">Will Subscribe</div></div>
    <div class="stat"><div class="v" style="color:${C.red}">${noCount}</div><div class="l">Won't Subscribe</div></div>
    <div class="stat"><div class="v" style="color:${C.blue2}">${avgConf}%</div><div class="l">Avg Confidence</div></div>`;

  let html='<table class="tbl"><tr><th>#</th><th>Age</th><th>Job</th><th>Balance</th><th>Duration</th><th>Prediction</th><th>Confidence</th></tr>';
  rows.forEach((r,i)=>{
    const badge=r._pred.yes?`<span class="badge best">✅ Yes</span>`:`<span class="badge" style="background:rgba(248,113,113,.15);color:#f87171">✗ No</span>`;
    html+=`<tr><td>${i+1}</td><td>${r.age}</td><td>${r.job}</td><td>₹${r.balance}</td><td>${r.duration}s</td><td>${badge}</td><td>${r._pred.conf}%</td></tr>`;
  });
  html+='</table>';
  document.getElementById('bulkTable').innerHTML=html;
  document.getElementById('bulkResults').style.display='block';

  // Export button
  document.getElementById('btnExport').onclick=()=>{
    let csv='age,job,balance,duration,prediction,confidence\n';
    rows.forEach(r=>csv+=`${r.age},${r.job},${r.balance},${r.duration},${r._pred.yes?'Yes':'No'},${r._pred.conf}%\n`);
    const blob=new Blob([csv],{type:'text/csv'});
    const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='bulk_predictions.csv';a.click();
  };
}

// ── MANUAL PREDICTION ──
function runPrediction(){
  const get=id=>document.getElementById(id).value;
  const row={
    age:get('m_age'),balance:get('m_balance'),duration:get('m_duration'),
    campaign:get('m_campaign'),housing:get('m_housing'),loan:get('m_loan'),
    contact:get('m_contact'),poutcome:get('m_poutcome')
  };
  const pred=predictRow(row);
  const res=document.getElementById('predResult');
  res.innerHTML=`
    <div class="pred-result">
      <div class="icon-big">${pred.yes?'✅':'❌'}</div>
      <div class="pred-label" style="color:${pred.yes?C.green:C.red}">${pred.yes?'Likely to Subscribe':'Unlikely to Subscribe'}</div>
      <div class="pred-conf">Confidence: ${pred.conf}%</div>
      <div style="margin-top:20px;text-align:left;font-size:13px;color:#94a3b8">
        <p style="margin-bottom:8px"><strong style="color:#f1f5f9">Key Factors:</strong></p>
        <p>• Duration: ${row.duration}s ${parseFloat(row.duration)>300?'(Strong positive ↑)':parseFloat(row.duration)<100?'(Negative ↓)':'(Moderate)'}</p>
        <p>• Balance: ₹${row.balance} ${parseFloat(row.balance)>2000?'(High - positive ↑)':parseFloat(row.balance)<0?'(Negative ↓)':'(Normal)'}</p>
        <p>• Previous Outcome: ${row.poutcome} ${row.poutcome==='success'?'(Strong positive ↑)':''}</p>
        <p style="margin-top:12px;color:${pred.yes?C.green:C.orange}"><strong>Recommendation:</strong> ${pred.yes?'High-priority lead — schedule call promptly.':'Low-priority — consider skipping or scheduling later.'}</p>
      </div>
    </div>`;
}

function stepVal(id,delta){
  const el=document.getElementById(id);
  let v=parseFloat(el.value)||0;
  const step=parseFloat(el.step)||1;
  el.value=(v+delta*step).toFixed(step<1?2:0);
}

// ── INIT ──
document.addEventListener('DOMContentLoaded',()=>{
  initOverview();
  initModelInsight();
  setupUpload();
});
