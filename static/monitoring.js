/* Independent monitoring connections; one browser connection per farm/tank. */
'use strict';
function streamKey(id) { return `${State.farm}:${id}`; }
function tankStream(id) { return State.streams.get(streamKey(id)); }
function monitorTanks() { return (State.report?.tanks || []).filter(t => t.status !== 'inactive'); }
function sourceKind(tank) { return String(tank?.camera_source || '').match(/^(\d+$|rtsps?:\/\/)/) ? 'camera' : 'video'; }
function isMonitoring(entry) { return entry && ['starting', 'running', 'stopping','reconnecting'].includes(entry.status); }
function monitorConfig(id) { return State.monitoring?.tanks.find(t=>t.tank_id===id); }
function ensureProductionViews() { if (State.production.enabled && State.view==='tanks') monitorTanks().filter(t=>monitorConfig(t.tank_id)?.enabled && ['running','starting','reconnecting'].includes(monitorConfig(t.tank_id)?.status)).forEach(t=>{ if (!isMonitoring(tankStream(t.tank_id))) startTankMonitoring(t.tank_id,true); }); }
function freshMonitoring(entry) { return entry?.status === 'running' && entry.data && Date.now() - entry.receivedAt < 10000; }
const displayDefaults={show_boxes:true,show_labels:true,show_conf:false,show_trails:true,show_dots:false};
let displayPreferences={...displayDefaults};
function displayKey() { return 'aquatrack-display:'+State.user.id; }
function loadDisplayPreferences() { displayPreferences={...displayDefaults}; try { const saved=JSON.parse(localStorage.getItem(displayKey()) || '{}'); for(const key of Object.keys(displayDefaults))if(typeof saved[key]==='boolean')displayPreferences[key]=saved[key]; }catch(_){} }
function displayControls() { return `<details class="display-options"><summary>Display options</summary><fieldset><legend>Camera overlays · only your view</legend>${[['show_boxes','Bounding boxes'],['show_labels','Fish labels / IDs'],['show_conf','Confidence percentages'],['show_trails','Motion trails'],['show_dots','Center dots']].map(([key,label])=>`<label><input type="checkbox" data-display-choice="${key}" ${displayPreferences[key] ? 'checked' : ''}>${label}</label>`).join('')}<div class="display-actions">${action('Clean video','display-clean','','secondary small')}${action('Reset display','display-reset','','secondary small')}</div><small>These choices do not change counting or automatic population updates.</small></fieldset></details>`; }
function setDisplayPreferences(changes) { Object.assign(displayPreferences,changes); try { localStorage.setItem(displayKey(),JSON.stringify(displayPreferences)); }catch(_){} document.querySelectorAll('[data-display-choice]').forEach(input=>input.checked=displayPreferences[input.dataset.displayChoice]); updateMonitoringViews(); }
function drawCameraOverlay(canvas,data,image) {
  if(!canvas || !image?.complete || !image.naturalWidth)return;
  canvas.width=data.frame_width || image.naturalWidth; canvas.height=data.frame_height || image.naturalHeight;
  const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,canvas.width,canvas.height); canvas.hidden=!data.raw_frame;
  if(!data.raw_frame)return;
  for(const fish of data.detections || []) {
    const color=`hsl(${((fish.track_id || 0)*67)%360} 80% 65%)`; ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=2;
    if(displayPreferences.show_trails && fish.trail?.length>1) {ctx.beginPath(); fish.trail.forEach(([x,y],i)=>i ? ctx.lineTo(x,y) : ctx.moveTo(x,y));ctx.stroke();}
    if(displayPreferences.show_boxes)ctx.strokeRect(fish.x1,fish.y1,fish.x2-fish.x1,fish.y2-fish.y1);
    if(displayPreferences.show_dots){ctx.beginPath();ctx.arc((fish.x1+fish.x2)/2,(fish.y1+fish.y2)/2,3,0,Math.PI*2);ctx.fill();}
    const label=[displayPreferences.show_labels ? (fish.track_id ? '#'+fish.track_id+' Fish' : 'Fish') : '',displayPreferences.show_conf ? Math.round(fish.conf*100)+'%' : ''].filter(Boolean).join(' ');
    if(label){ctx.font='13px sans-serif'; const width=ctx.measureText(label).width+8, x=Math.max(0,Math.min(canvas.width-width,fish.x1)), y=Math.max(18,fish.y1);ctx.fillStyle='#123b32';ctx.fillRect(x,y-18,width,18);ctx.fillStyle='white';ctx.fillText(label,x+4,y-4);}
  }
}
function updateCameraPreview(image,canvas,entry) {
  const data=entry?.status==='running' ? entry.data : null, frame=data?.raw_frame;
  image.dataset.previewError=data?.frame && !frame ? 'Preview unavailable. Reload the app and restart this camera to enable display controls.' : '';
  image.hidden=!frame; if(canvas)canvas.hidden=!frame;
  if(!frame){image.onload=null;image.removeAttribute('src');delete image.dataset.frameId;if(canvas){canvas.width=canvas.width;canvas.hidden=true;}return false;}
  const id=entry.key+':'+data.frame_idx;
  image.onload=()=>drawCameraOverlay(canvas,data,image);
  if(image.dataset.frameId!==id){image.src='data:image/jpeg;base64,'+frame;image.dataset.frameId=id;}
  else drawCameraOverlay(canvas,data,image);
  return true;
}
function monitorButton(tank, entry = tankStream(tank.tank_id)) {
  if (entry?.status === 'stopping') return 'Stopping…';
  if (entry?.status === 'starting') return 'Cancel connection';
  const video = sourceKind(tank) === 'video';
  return isMonitoring(entry) ? (video ? 'Stop video' : 'Stop camera') : (video && tank.camera_source ? 'Play video' : 'Start camera');
}
function monitoringTile(tank) {
  const config=monitorConfig(tank.tank_id);
  return `<article class="monitoring-tile" data-monitor-tank="${escapeHTML(tank.tank_id)}" data-open-tank="${escapeHTML(tank.tank_id)}" tabindex="0" role="group" aria-label="Open ${escapeHTML(tank.name)} live view"><div class="monitoring-tile-header"><div><h3><button class="tank-title" data-action="count" data-id="${escapeHTML(tank.tank_id)}">${escapeHTML(tank.name)}</button></h3><small>${escapeHTML(tank.tank_id)} · ${tank.camera_source ? sourceKind(tank) === 'video' ? 'Recorded video' : 'Tank camera' : 'Source not assigned'}</small></div><span class="pill neutral" data-monitor-field="status">Stopped</span></div><div class="monitoring-preview"><img data-monitor-field="image" alt="Fish view for ${escapeHTML(tank.name)}" hidden><canvas data-monitor-field="overlay" aria-hidden="true" hidden></canvas><div class="monitoring-placeholder" data-monitor-field="placeholder">Start monitoring to see this tank.</div></div><div class="monitoring-values"><div><span>Visible fish</span><strong data-monitor-field="count">—</strong></div><div><span>Camera-view occupancy</span><strong data-monitor-field="occupancy">—</strong></div></div><div class="monitoring-values"><div><span>Saved population</span><strong data-monitor-field="saved">${fmt(tank.current_count)}</strong></div><div><span>Estimated daily feed</span><strong>${fmt(tank.daily_feed_kg,3)} kg</strong></div></div><div class="monitoring-detail"><span data-monitor-field="crossings">Crossings unavailable</span><span data-monitor-field="fps">— updates/s</span></div><p class="monitoring-message" data-monitor-field="message" role="status">${tank.camera_source ? 'Ready to start.' : 'Assign a camera in Edit Tank.'}</p><div class="tank-actions">${action('Start camera', 'monitor-toggle', tank.tank_id, '', 'camera')}${action('Edit Tank', 'tank-edit', tank.tank_id, 'secondary small','edit')}</div></article>`;
}
function updateMonitoringViews() {
  if (!State.user) return;
  const tanks = monitorTanks();
  const active = tanks.filter(t => State.production.enabled ? ['running','starting','reconnecting'].includes(monitorConfig(t.tank_id)?.status) : isMonitoring(tankStream(t.tank_id))).length;
  $('monitoringStatus').hidden = active === 0;
  const fresh = tanks.map(t => tankStream(t.tank_id)).filter(freshMonitoring);
  document.querySelectorAll('[data-monitor-summary="active"]').forEach(el => el.textContent = `${active} / ${tanks.length}`);
  document.querySelectorAll('[data-monitor-summary="visible"]').forEach(el => el.textContent = fresh.length ? fmt(fresh.reduce((sum, e) => sum + e.data.live_count, 0)) : '—');
  document.querySelectorAll('[data-action="monitor-all"]').forEach(el => { el.innerHTML = icon(active ? 'stop' : 'play') + (active ? 'Stop all' : 'Start all'); el.disabled = !tanks.some(t => t.camera_source); });
  document.querySelectorAll('[data-monitor-tank]').forEach(root => {
    const tank = tanks.find(t => t.tank_id === root.dataset.monitorTank);
    if (!tank) return;
    const config=monitorConfig(tank.tank_id);
    const entry = tankStream(tank.tank_id) || (State.production.enabled ? config : null), fresh = freshMonitoring(entry);
    const delayed = entry?.status === 'running' && !fresh && entry.data;
    const label = delayed ? 'Delayed' : ({starting:'Connecting',running:entry?.data?.count_reliable===false ? 'Counting unreliable' : entry?.data ? 'Monitoring' : 'Waiting for frames',stopping:'Stopping',error:'Unavailable',reconnecting:'Reconnecting',setup_required:'Setup required',validating:'Checking camera',paused:'Paused',ready:'Ready'})[entry?.status] || 'Stopped';
    const data = fresh ? entry.data : null;
    const value = (name, text) => root.querySelectorAll(`[data-monitor-field="${name}"]`).forEach(el => el.textContent = text);
    value('status', label);
    value('count', data ? fmt(data.live_count) : '—');
    value('saved',fmt(data?.saved_population ?? tank.current_count));
    value('occupancy', data ? `${fmt(data.live_count / Math.max(1, tank.max_capacity) * 100, 1)}%` : '—');
    value('crossings', data ? `${fmt(data.count_in)} in / ${fmt(data.count_out)} out` : 'Crossings unavailable');
    value('fps', data ? `${fmt(data.fps, 1)} updates/s` : '— updates/s');
    value('message', entry?.data?.frame && !entry.data.raw_frame ? 'Preview unavailable. Reload the app and restart this camera to enable display controls.' : delayed ? 'Updates delayed. The last frame is not current.' : entry?.message || (tank.camera_source ? 'Ready to start.' : 'Assign a camera or upload a video.'));
    const badge = root.querySelector('[data-monitor-field="status"]');
    if (badge) badge.className = `pill ${fresh ? '' : delayed || entry?.status === 'starting' ? 'amber' : entry?.status === 'error' ? 'red' : 'neutral'}`;
    const image = root.querySelector('[data-monitor-field="image"]'), placeholder = root.querySelector('[data-monitor-field="placeholder"]');
    if (image) {
      const frame = updateCameraPreview(image,root.querySelector('[data-monitor-field="overlay"]'),entry);
      if (placeholder) { placeholder.hidden = !!frame; placeholder.textContent = image.dataset.previewError || (label === 'Stopped' ? 'Click this tank to open its live view.' : label); }
    }
    root.querySelectorAll('[data-action="monitor-toggle"]').forEach(button => { button.innerHTML = icon(isMonitoring(entry) ? 'stop' : 'play') + escapeHTML(monitorButton(tank, entry)); button.disabled = entry?.status === 'stopping' || !tank.camera_source || (State.production.enabled && (!config?.validated || config?.status==='validating')); button.setAttribute('aria-pressed', String(isMonitoring(entry))); });
    
  });
  if (inspectorOpen() && !State.sampleMode) {
    const tank = tanks.find(t => t.tank_id === State.countTank), entry = tankStream(State.countTank);
    const data = freshMonitoring(entry) ? entry.data : null;
    State.estimate = data?.live_count ?? null;
    $('countValue').textContent = data ? fmt(data.live_count) : '—';
    $('occupancyValue').textContent = data && tank ? `${fmt(data.live_count / Math.max(1,tank.max_capacity) * 100,1)}%` : '—';
    $('crossingValue').textContent = data ? `${fmt(data.count_in)} in / ${fmt(data.count_out)} out` : '—';
    $('countStatus').textContent = entry?.status === 'running' && !data ? 'Waiting for current frames. The preview may be delayed.' : entry?.message || 'Ready to monitor this tank.';
    const frame = updateCameraPreview($('countPreview'),$('countOverlay'),entry);
    $('countPlaceholder').hidden = !!frame; $('countPlaceholder').textContent=$('countPreview').dataset.previewError || 'Start monitoring to see this tank.';
    $('useCount').disabled = !(State.estimate > 0);
  }
  if(inspectorOpen() && State.sampleMode && State.samplePreview)showCountingResult(State.samplePreview);
  updateCameraSetup();
  if (inspectorOpen()) {
    const tank = tanks.find(t => t.tank_id === State.countTank);
    $('toggleCamera').textContent = State.sampleMode ? 'Start monitoring' : tank ? monitorButton(tank) : 'Start camera';
    const config=monitorConfig(State.countTank);
    $('toggleCamera').disabled = !tank?.camera_source || tankStream(State.countTank)?.status === 'stopping' || (State.production.enabled && (!config?.validated || config.status==='validating'));
    if(tank){const population=tankStream(State.countTank)?.data?.saved_population ?? tank.current_count;
      $('inspectorInventory').innerHTML=`<div class="labelled-value"><span>Saved population</span><strong>${fmt(population)} fish</strong></div><div class="labelled-value"><span>Capacity</span><strong>${fmt(tank.max_capacity)} fish</strong></div><div class="labelled-value"><span>Estimated biomass</span><strong>${fmt(population*tank.avg_weight_g/1000,3)} kg</strong></div><div class="labelled-value"><span>Daily feed estimate</span><strong>${fmt(population*tank.avg_weight_g/1000*tank.feed_rate_pct,3)} kg</strong></div>`;
    }
  }
}
function finishMonitoring(entry, status, message) {
  if (State.streams.get(entry.key) !== entry) return;
  clearTimeout(entry.stopTimer); clearTimeout(entry.connectTimer);
  entry.status = status; entry.message = message; entry.data = null;
  entry.ws.onclose = entry.ws.onmessage = entry.ws.onerror = entry.ws.onopen = null;
  entry.ws.close(); entry.stopped?.(); entry.stopped = null;
  updateMonitoringViews();
}
function startTankMonitoring(id, observe=false) {
  const tank = monitorTanks().find(t => t.tank_id === id);
  if (!tank?.camera_source) { toast('Choose a source in Edit Tank first.', true); return; }
  if (isMonitoring(tankStream(id))) return;
  const key = streamKey(id), query = new URLSearchParams({csrf:State.csrf});
  if (State.user.role === 'admin') query.set('farm_id',State.farm);
  const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/live/${encodeURIComponent(id)}?${query}`);
  const entry = {key, farm:State.farm, tankId:id, source:tank.camera_source, ws:socket, status:'starting', message:'Connecting to this tank…', data:null, receivedAt:0};
  State.streams.set(key,entry);
  const valid = () => State.farm === entry.farm && State.streams.get(key) === entry;
  entry.connectTimer = setTimeout(() => { if (valid() && entry.status === 'starting') finishMonitoring(entry,'error','Connection timed out. Check this tank source and restart.'); },30000);
  socket.onopen = () => { if (valid() && entry.status === 'starting') socket.send(JSON.stringify({type:observe ? 'subscribe' : 'start_stream'})); else socket.close(); };
  socket.onmessage = event => {
    if (!valid()) return;
    let data; try { data = JSON.parse(event.data); } catch { return; }
    if (data.type === 'stream_state') {
      if (data.status === 'error') { finishMonitoring(entry,'error',data.message); return; }
      if (data.status === 'stopped') { finishMonitoring(entry,'stopped','Monitoring stopped.'); return; }
      if (entry.status === 'stopping') return;
      entry.status = data.status; entry.message = data.message;
      if (data.status === 'running') clearTimeout(entry.connectTimer);
    } else if (data.type === 'error') { finishMonitoring(entry,'error',data.message); return; }
    else if (data.type === 'telemetry' && data.frame && entry.status !== 'stopping') {
      clearTimeout(entry.connectTimer); entry.status = 'running'; entry.data = data; entry.receivedAt = Date.now();
      entry.message = data.production ? data.message : `${data.source_kind === 'video' ? 'Recorded video · loop ' + (data.playback_cycle + 1) : 'Camera live'} · ${fmt(data.fps,1)} updates/s`;
    }
    updateMonitoringViews();
  };
  socket.onclose = event => { if (valid()) { finishMonitoring(entry,entry.status === 'stopping' ? 'stopped' : 'error',entry.status === 'stopping' ? 'Monitoring stopped.' : 'Connection ended. Restart this tank.'); if ([4401,4403].includes(event.code)) { stopAllMonitoring(true); location.href='/login'; } } };
  socket.onerror = () => { if (valid()) finishMonitoring(entry,'error','Unable to connect. Check this tank source and restart.'); };
  updateMonitoringViews();
}
function stopTankMonitoring(id) {
  const entry = tankStream(id);
  if (!isMonitoring(entry)) return Promise.resolve();
  if (entry.stopPromise) return entry.stopPromise;
  entry.status = 'stopping'; entry.data = null; entry.message = 'Stopping this tank…';
  entry.stopPromise = new Promise(resolve => entry.stopped = resolve);
  if (entry.ws.readyState === WebSocket.OPEN) {
    entry.ws.send(JSON.stringify({type:'stop_stream'}));
    entry.stopTimer = setTimeout(() => finishMonitoring(entry,'stopped','Monitoring stopped.'),20000);
  } else finishMonitoring(entry,'stopped','Monitoring stopped.');
  updateMonitoringViews(); return entry.stopPromise;
}
function stopAllMonitoring(clear = false) {
  for (const entry of State.streams.values()) {
    if (clear) finishMonitoring(entry,'stopped','Monitoring stopped.');
    else if (entry.farm === State.farm) stopTankMonitoring(entry.tankId);
  }
  if (clear) State.streams.clear();
}
async function toggleTankMonitoring(id) { if (State.production.enabled) { const config=monitorConfig(id); await api('/api/tanks/'+encodeURIComponent(id)+'/monitoring',{method:'PUT',body:{enabled:!config?.enabled || !['running','starting','reconnecting'].includes(config?.status)}}); if (!config?.enabled) startTankMonitoring(id,true); await refresh(true); return; } if (isMonitoring(tankStream(id))) return stopTankMonitoring(id); startTankMonitoring(id); }
async function toggleAllMonitoring() { if (State.production.enabled) { const active=State.monitoring.tanks.some(t=>['running','starting','reconnecting'].includes(t.status)); await Promise.all(monitorTanks().filter(t=>monitorConfig(t.tank_id)?.validated).map(t=>api('/api/tanks/'+encodeURIComponent(t.tank_id)+'/monitoring',{method:'PUT',body:{enabled:!active}}))); await refresh(true); return; } if (monitorTanks().some(t => isMonitoring(tankStream(t.tank_id)))) stopAllMonitoring(); else monitorTanks().filter(t => t.camera_source).forEach(t => startTankMonitoring(t.tank_id)); }
