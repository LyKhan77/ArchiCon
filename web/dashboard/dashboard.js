'use strict';
const $=id=>document.getElementById(id);
const state={token:null,selected:null,job:null,busy:false,upload:null,submitting:false,file:null,objectURL:null,inputs:[],tab:'input',outputTab:'tour',polling:false};
const execution={queued:'Menunggu',running:'Sedang diproses',completed:'Komputasi selesai',failed:'Proses terhenti',waiting_resources:'Menunggu resource',unknown:'Status belum tersedia'};
const reviews={pending:'Menunggu review',passed:'Review selesai',needs_capture:'Perlu rekaman tambahan',needs_human_review:'Perlu review manual'};
const phases={transfer:'Memverifikasi dan mengirim video',processing:'Merekonstruksi ruang',downloading:'Mengambil hasil dari server',packaging:'Menyiapkan denah dan room tour',browser_qa:'Memeriksa HTML di browser',awaiting_review:'Hasil siap untuk review manual',waiting_resources:'Menunggu resource server'};
const groups=[['audit','Audit video','S02'],['camera','Estimasi kamera','S03'],['depth','Estimasi depth','S05'],['mesh','Fusion dan mesh','S06_'],['render','Render preview','S06R'],['export','Ekspor denah dan viewer','S07']];
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const bytes=n=>n>=1024**3?(n/1024**3).toFixed(2)+' GiB':(n/1024**2).toFixed(1)+' MiB';
const duration=n=>!Number.isFinite(n)?'—':n<60?Math.round(n)+' dtk':Math.floor(n/60)+' m '+Math.round(n%60)+' dtk';
const artifact=(job,path,download=false)=>'/api/jobs/'+encodeURIComponent(job)+'/artifact?path='+encodeURIComponent(path)+(download?'&download=1':'');
function notify(message){$('alert-text').textContent=message;$('alert').hidden=false;}
async function api(path,options={}){
  const response=await fetch(path,{...options,headers:{...(options.body?{'Content-Type':'application/json'}:{}),...(options.method==='POST'?{'X-ArchiCon-Token':state.token}:{}),...options.headers}});
  let data;try{data=await response.json();}catch{throw new Error('Respons server tidak dapat dibaca.');}
  if(!response.ok)throw new Error(data.error||'Permintaan gagal.');return data;
}
function connected(ok){$('connection').textContent=ok?'Backend lokal aktif':'Backend terputus';$('connection').classList.toggle('offline',!ok);}
function tab(name){
  state.tab=name;
  for(const button of document.querySelectorAll('[data-tab]')){const yes=button.dataset.tab===name;button.setAttribute('aria-selected',String(yes));button.tabIndex=yes?0:-1;$('panel-'+button.dataset.tab).hidden=!yes;}
  if(name==='output')renderOutput();
}
function updateRunButton(){
  $('start-job').disabled=state.busy||!!state.upload||state.submitting;
  $('busy-notice').hidden=!state.busy;
  $('busy-notice').textContent='Satu rekonstruksi sedang berjalan. Anda dapat menyiapkan video; mulai setelah job aktif selesai.';
}
function updateScopeFields(){
  const scoped=$('scope-kind').value!=='full_traversal';
  $('scope-fields').hidden=!scoped;
  $('scope-name').required=scoped;
}
function readScope(){
  const kind=$('scope-kind').value;
  if(kind==='full_traversal')return {kind};
  const scope={kind,name:$('scope-name').value.trim()};
  const start=$('scope-start').value.trim(),end=$('scope-end').value.trim();
  if(start)scope.start_time_seconds=Number(start);
  if(end)scope.end_time_seconds=Number(end);
  return scope;
}
async function refreshInputs(){
  state.inputs=await api('/api/inputs');const selected=$('existing-video').value;
  $('existing-video').innerHTML='<option value="">Pilih video yang sudah tersedia</option>'+state.inputs.map(x=>`<option value="${esc(x.path)}">${esc(x.path.replace(/^raw_videos\//,''))} · ${bytes(x.size_bytes)}</option>`).join('');
  $('existing-video').value=selected;
}
function setVideo(url,label){
  const video=$('source-video');video.src=url;video.hidden=false;$('video-empty').hidden=true;
  $('video-meta').textContent=label;
}
function suggestId(name){
  if($('job-id').value)return;
  const prefix=name.replace(/\.[^.]+$/,'').toLowerCase().replace(/[^a-z0-9_-]/g,'_').replace(/^[^a-z0-9]+/,'').slice(0,16)||'ruangan';
  const date=new Date();const stamp=String(date.getMonth()+1).padStart(2,'0')+String(date.getDate()).padStart(2,'0')+'_'+String(date.getHours()).padStart(2,'0')+String(date.getMinutes()).padStart(2,'0');
  $('job-id').value=prefix+'_'+stamp;
}
function chooseFile(file){
  if(!file)return;
  if(file.size>state.maxUpload){notify('Video melebihi batas upload '+bytes(state.maxUpload)+'.');return;}
  if(!/\.(mov|mp4|m4v|webm|avi|mkv)$/i.test(file.name)){notify('Pilih format video yang didukung.');return;}
  state.file=file;$('existing-video').value='';$('file-name').textContent=file.name;$('file-help').textContent=bytes(file.size)+' · dipilih dari perangkat';
  if(state.objectURL)URL.revokeObjectURL(state.objectURL);state.objectURL=URL.createObjectURL(file);
  setVideo(state.objectURL,'Video lokal');suggestId(file.name);
}
function upload(file){
  return new Promise((resolve,reject)=>{
    const xhr=new XMLHttpRequest();state.upload=xhr;updateRunButton();$('upload-state').hidden=false;
    xhr.open('POST','/api/uploads?filename='+encodeURIComponent(file.name));xhr.setRequestHeader('X-ArchiCon-Token',state.token);xhr.setRequestHeader('Content-Type','application/octet-stream');
    xhr.upload.onprogress=event=>{if(event.lengthComputable){const value=Math.round(event.loaded/event.total*100);$('upload-progress').value=value;$('upload-label').textContent='Upload ke laptop · '+value+'%';}};
    xhr.onload=()=>{let data;try{data=JSON.parse(xhr.responseText);}catch{reject(new Error('Respons upload tidak valid.'));return;}if(xhr.status>=200&&xhr.status<300)resolve(data);else reject(new Error(data.error||'Upload gagal.'));};
    xhr.onerror=()=>reject(new Error('Upload terputus. Periksa backend lokal.'));xhr.onabort=()=>reject(new Error('Upload dibatalkan.'));
    xhr.onloadend=()=>{state.upload=null;updateRunButton();$('upload-state').hidden=true;};xhr.send(file);
  });
}
function renderHistory(jobs){
  $('job-count').textContent=jobs.length;
  $('job-list').innerHTML=jobs.length?jobs.map(j=>`<button class="job-item${j.job_id===state.selected?' active':''}" data-job="${esc(j.job_id)}"${j.job_id===state.selected?' aria-current="true"':''}><strong>${esc(j.job_id)}</strong><small><span class="state-dot ${esc(j.execution_status)}"></span>${esc(execution[j.execution_status]||j.execution_status)}</small></button>`).join(''):'<p class="muted sidebar-empty">Belum ada rekonstruksi.</p>';
}
function stageRow(label,status,detail,time,index){return `<li class="${esc(status)}"><span class="stage-marker">${status==='passed'?'✓':status==='running'?'·':status==='failed'?'!':index}</span><div><strong>${esc(label)}</strong><small>${esc(detail)}</small></div><span class="stage-time">${esc(time||'')}</span></li>`;}
function renderProgress(){
  const job=state.job;$('progress-empty').hidden=!!job;$('progress-content').hidden=!job;if(!job)return;
  const status=job.execution_status;const phase=job.phase?.phase;
  const runningStage=job.stages.find(s=>s.status==='running');
  const scopeLabel=job.scope?.kind==='single_room'?'Single room':job.scope?.kind==='connected_area'?'Connected area':'Full traversal';
  $('phase-title').textContent=status==='completed'?'Hasil siap untuk review manual':status==='failed'?'Proses terhenti':status==='waiting_resources'?'Menunggu resource server':phases[phase]||'Merekonstruksi ruang';
  $('telemetry-note').dataset.scope=scopeLabel+(job.scope?.name?' · '+job.scope.name:'');
  if(runningStage&&phase==='processing')$('phase-title').textContent=groups.find(g=>runningStage.stage.startsWith(g[2]))?.[1]||'Merekonstruksi ruang';
  $('execution-badge').textContent=execution[status]||status;$('execution-badge').className='status-label '+status;
  $('resume-job').hidden=job.controller_active||status==='completed';$('resume-job').disabled=state.busy;
  const note=job.telemetry.error||(job.telemetry.stale?'Snapshot progress belum diperbarui. Worker server mungkin masih berjalan; jangan membuat job duplikat.':'')||job.reason;
  $('telemetry-note').hidden=!note;$('telemetry-note').textContent=note;
  const coverage=job.coverage;$('metric-frames').textContent=coverage?`${coverage.processed} / ${coverage.attempted}`:'—';$('metric-gaps').textContent=coverage?String(coverage.gaps?.length??0):'—';
  $('elapsed-label').textContent=job.elapsed_basis==='controller'?'Durasi controller':'Total durasi stage';
  $('metric-elapsed').textContent=duration(job.elapsed_seconds);$('metric-review').textContent=reviews[job.review_status]||'Belum dinilai';
  let rows=[];const transferDone=job.stages.length>0||['processing','downloading','packaging','browser_qa','awaiting_review'].includes(phase)||status==='completed';
  rows.push(stageRow('Verifikasi dan transfer',transferDone?'passed':phase==='transfer'?'running':'pending',transferDone?'Input tersedia untuk proses':phase==='transfer'?'Memeriksa checksum dan mengirim ke server':'Menunggu controller','',1));
  for(const [id,label,prefix] of groups){
    const records=job.stages.filter(x=>x.stage.startsWith(prefix));const passed=records.filter(x=>x.status==='passed').length;
    const s=records.some(x=>x.status==='running')?'running':records.some(x=>x.status==='failed')?'failed':passed?'passed':'pending';
    const detail=records.length?`${passed}/${records.length} stage berhasil`+(records.some(x=>x.attempts>1)?' · beberapa attempt':''):status==='completed'?'Selesai melalui pipeline / hasil terverifikasi':'Menunggu stage sebelumnya';
    rows.push(stageRow(label,s,detail,records.length?duration(records.reduce((a,x)=>a+(x.elapsed_seconds||0),0)):'',rows.length+1));
  }
  const packaged=status==='completed';rows.push(stageRow('HTML dan pemeriksaan browser',packaged?'passed':['packaging','browser_qa','downloading'].includes(phase)?'running':'pending',packaged?'Produk tersedia · review kualitas terpisah':phases[phase]==='Memeriksa HTML di browser'?'Menguji viewer desktop dan mobile':'Menunggu hasil server','',8));
  $('stage-list').innerHTML=rows.join('');
  const key=JSON.stringify(job.previews);
  if($('live-previews').dataset.key!==key){$('live-previews').dataset.key=key;$('live-previews').innerHTML=job.previews.length?job.previews.map(p=>`<figure class="preview-figure"><a href="${artifact(job.job_id,p.path)}" target="_blank" rel="noopener"><img src="${artifact(job.job_id,p.path)}" alt="${esc(p.label)}" loading="lazy"></a><figcaption>${esc(p.label)}</figcaption></figure>`).join(''):'<p class="muted">Contact sheet dan render muncul setelah stage terkait selesai.</p>';}
  $('process-log').textContent=job.log.trim()||'Belum ada log.';
}
function outputMode(mode){state.outputTab=mode;$('show-tour').classList.toggle('selected',mode==='tour');$('show-plan').classList.toggle('selected',mode==='plan');$('tour-frame').hidden=mode!=='tour';$('plan-grid').hidden=mode!=='plan';}
function reviewDecision(){return document.querySelector('input[name="review-decision"]:checked')?.value||'';}
function renderReviewSession(){
  const review=state.job?.review;const panel=$('review-session');
  if(!review?.available){panel.hidden=true;return;}
  panel.hidden=false;
  const gaps=review.coverage?.gaps?.length||0;const applied=!!review.applied;
  $('review-session-status').textContent=applied?'Keputusan tersimpan':'Menunggu keputusan';
  $('review-session-status').className='status-label '+(applied?'completed':'');
  $('review-session-summary').textContent=`${review.component_count||0} komponen · ${gaps} gap kamera · keputusan global`;
  $('review-coverage').textContent=gaps?`${gaps} gap terdeteksi. Accept akan menjadi accepted_partial.`:'Tidak ada gap kamera pada laporan coverage.';
  const links=(review.evidence||[]).slice(0,12);
  $('review-evidence').innerHTML=links.length?links.map(path=>`<a href="${artifact(state.job.job_id,path)}" target="_blank" rel="noopener">${esc(path.split('/').slice(-2).join(' / '))} ↗</a>`).join(''):'<span class="helper">Bukti preview belum tersedia.</span>';
  $('review-form').hidden=applied;$('review-applied').hidden=!applied;$('review-partial-note').hidden=applied||!gaps;
  if(applied){$('review-applied-detail').textContent=`Status kualitas: ${review.quality_status||'unassessed'} · delivery: ${review.delivery||'—'}`;}
}
function renderOutput(){
  const job=state.job;const ready=!!job?.delivery;$('output-empty').hidden=ready;$('output-content').hidden=!ready;if(!ready)return;
  const scopeNames={full_traversal:'Full traversal',single_room:'Single room',connected_area:'Connected area'};
  const scope=job.scope||{kind:'full_traversal'};
  $('scope-label').textContent='Scope: '+(scopeNames[scope.kind]||scope.kind)+(scope.name?' — '+scope.name:'')+' · kualitas tetap menunggu review manual';
  $('download-package').href='/api/jobs/'+encodeURIComponent(job.job_id)+'/package';$('delivery-path').textContent=job.delivery;
  const note=$('review-note');note.querySelector('strong').textContent=reviews[job.review_status]||'Review manual';
  note.querySelector('span').textContent=job.review_status==='needs_capture'?'Ada area yang perlu direkam ulang. Periksa coverage dan catatan review.':job.review_status==='passed'?'Keputusan review tersimpan. Periksa batas kualitas dan kalibrasi ukuran pada laporan.':'Komputasi selesai. Kualitas geometri dan kelengkapan ruang perlu diperiksa.';
  const tour=job.downloads.find(f=>f.path.endsWith('/room_tour/index.html'));
  if(tour){const url=artifact(job.job_id,tour.path);$('open-tour').href=url;if($('tour-frame').getAttribute('src')!==url)$('tour-frame').src=url;}
  const plans=job.downloads.filter(f=>f.path.includes('/denah/')&&f.path.endsWith('.svg'));
  const key=JSON.stringify(job.downloads);
  if($('download-list').dataset.key!==key){
    $('download-list').dataset.key=key;
    $('plan-grid').innerHTML=plans.map(p=>`<figure class="preview-figure"><img src="${artifact(job.job_id,p.path)}" alt="Estimasi denah ${esc(p.name)}"><figcaption>${esc(p.name)} · skala ${esc(job.scale_status)}</figcaption></figure>`).join('')||'<p class="muted">Denah belum tersedia.</p>';
    $('download-list').innerHTML=job.downloads.map(f=>`<a href="${artifact(job.job_id,f.path,true)}"><span>${esc(f.name)}<small>${esc(f.path.includes('/mesh/')?'Mesh detail':f.path.includes('/denah/')?'Estimasi denah':f.kind==='review'?'Input review manual':f.path.includes('/reports/')?'Laporan':'Artefak hasil')}</small></span><span aria-hidden="true">↓</span></a>`).join('');
  }
  renderReviewSession();
  outputMode(state.outputTab);
}
async function selectJob(jid){
  state.selected=jid;state.job=null;renderProgress();renderOutput();
  $('tour-frame').removeAttribute('src');
  await poll();tab(state.job?.execution_status==='completed'?'output':'progress');
}
async function poll(){
  if(state.polling)return;state.polling=true;
  try{
    const result=await api('/api/jobs');state.busy=result.control.busy;updateRunButton();renderHistory(result.jobs);
    if(state.selected){
      const requested=state.selected;const job=await api('/api/jobs/'+encodeURIComponent(requested));if(requested!==state.selected)return;state.job=job;
      $('page-title').textContent=job.job_id;$('page-subtitle').textContent=job.title;$('breadcrumb').textContent='Workspace / Rekonstruksi';renderProgress();if(state.tab==='output')renderOutput();
    }
    connected(true);
  }catch(error){connected(false);if(state.job){$('telemetry-note').hidden=false;$('telemetry-note').textContent='Backend lokal tidak terjangkau. Controller dan worker terpisah mungkin masih berjalan.';}}
  finally{state.polling=false;}
}
$('input-form').addEventListener('submit',async event=>{
  event.preventDefault();if(state.busy||state.upload||state.submitting)return;
  let path=$('existing-video').value;const id=$('job-id').value.trim();
  if(!state.file&&!path){notify('Pilih video terlebih dahulu.');return;}
  state.submitting=true;updateRunButton();$('alert').hidden=true;
  try{
    if(state.file){const result=await upload(state.file);path=result.path;state.file=null;await refreshInputs();$('existing-video').value=path;}
    const result=await api('/api/jobs',{method:'POST',body:JSON.stringify({input_path:path,job_id:id,scope:readScope()})});state.selected=result.job_id;await poll();tab('progress');
  }catch(error){notify(error.message);}finally{state.submitting=false;updateRunButton();}
});
$('video-file').addEventListener('change',()=>chooseFile($('video-file').files[0]));
$('existing-video').addEventListener('change',()=>{const path=$('existing-video').value;if(!path)return;state.file=null;$('video-file').value='';$('file-name').textContent='Pilih atau letakkan video di sini';$('file-help').textContent='Menggunakan video lokal yang tersedia';setVideo('/api/video?path='+encodeURIComponent(path),'Video tersimpan');suggestId(path.split('/').pop());});
$('source-video').addEventListener('loadedmetadata',()=>{$('video-meta').textContent=duration($('source-video').duration);});
$('source-video').addEventListener('error',()=>{$('video-meta').textContent='Codec preview tidak didukung browser';});
for(const name of ['dragenter','dragover'])$('drop-zone').addEventListener(name,event=>{event.preventDefault();$('drop-zone').classList.add('dragging');});
for(const name of ['dragleave','drop'])$('drop-zone').addEventListener(name,event=>{event.preventDefault();$('drop-zone').classList.remove('dragging');if(name==='drop')chooseFile(event.dataTransfer.files[0]);});
$('cancel-upload').addEventListener('click',()=>state.upload?.abort());$('dismiss-alert').addEventListener('click',()=>{$('alert').hidden=true;});
  $('new-job').addEventListener('click',()=>{state.selected=null;state.job=null;$('page-title').textContent='Dari rekaman, menjadi ruang.';$('page-subtitle').textContent='Unggah video ruangan. Ikuti prosesnya. Jelajahi hasilnya.';$('breadcrumb').textContent='Workspace / Rekonstruksi baru';$('job-id').value='';$('scope-kind').value='full_traversal';$('scope-name').value='';$('scope-start').value='';$('scope-end').value='';updateScopeFields();$('tour-frame').removeAttribute('src');renderProgress();tab('input');poll();});
$('job-list').addEventListener('click',event=>{const button=event.target.closest('[data-job]');if(button)selectJob(button.dataset.job).catch(error=>notify(error.message));});
for(const button of document.querySelectorAll('[data-tab]')){button.addEventListener('click',()=>tab(button.dataset.tab));button.addEventListener('keydown',event=>{const items=[...document.querySelectorAll('[data-tab]')];const i=items.indexOf(button);let target;if(event.key==='ArrowRight')target=items[(i+1)%items.length];if(event.key==='ArrowLeft')target=items[(i+items.length-1)%items.length];if(event.key==='Home')target=items[0];if(event.key==='End')target=items.at(-1);if(target){event.preventDefault();tab(target.dataset.tab);target.focus();}});}
$('resume-job').addEventListener('click',async()=>{try{$('resume-job').disabled=true;await api('/api/jobs/'+encodeURIComponent(state.selected)+'/resume',{method:'POST'});await poll();}catch(error){notify(error.message);$('resume-job').disabled=false;}});
$('review-form').addEventListener('submit',async event=>{
  event.preventDefault();const decision=reviewDecision();const reason=$('review-reason').value.trim();
  $('review-validation').hidden=true;
  if(!decision){$('review-validation').textContent='Pilih satu keputusan terlebih dahulu.';$('review-validation').hidden=false;return;}
  if(decision==='reject'&&!reason){$('review-validation').textContent='Alasan wajib untuk Reject / perlu rekaman ulang.';$('review-validation').hidden=false;return;}
  const button=$('submit-review');button.disabled=true;state.submitting=true;
  try{await api('/api/jobs/'+encodeURIComponent(state.selected)+'/review',{method:'POST',body:JSON.stringify({decision,reason})});await poll();renderOutput();}
  catch(error){$('review-validation').textContent=error.message;$('review-validation').hidden=false;}
  finally{state.submitting=false;button.disabled=false;}
});
$('show-tour').addEventListener('click',()=>outputMode('tour'));$('show-plan').addEventListener('click',()=>outputMode('plan'));
$('scope-kind').addEventListener('change',updateScopeFields);
async function init(){
  try{const config=await api('/api/config');state.token=config.token;state.maxUpload=config.max_upload_bytes;updateScopeFields();$('file-help').textContent='MOV, MP4, M4V, WebM, AVI, MKV · maks. '+bytes(state.maxUpload);await refreshInputs();if(config.control.busy)state.selected=config.control.job_id;await poll();if(state.selected)tab('progress');}
  catch(error){connected(false);notify(error.message+' Jalankan backend ArchiCon lalu muat ulang.');}
  setInterval(poll,4000);
}
init();
