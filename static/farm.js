/* Farm UI: authenticated requests and explicitly enabled automatic production census. */
'use strict';
const $ = id => document.getElementById(id);
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (value, digits = 0) => value == null ? 'N/A' : Number(value).toLocaleString('en-PH', { maximumFractionDigits: digits });
const pct = value => value == null ? 'N/A' : `${Number(value).toFixed(2)}%`;
const money = value => `₱ ${Number(value || 0).toLocaleString('en-PH', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const paths = {
  dashboard: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  food: '<path d="M7 3h10l-1 4 4 13H4L8 7z"/><path d="M8 7h8M9 13c3-3 6 0 6 0s-3 4-6 0z"/>',
  tanks: '<path d="M3 5h18v14H3zM3 11c3-3 6 3 9 0s6 3 9 0"/><path d="M7 19v2m10-2v2"/>',
  dispersals: '<path d="M3 4h12v13H3zM15 8h4l3 5v4h-7M4 17v-5m4-4h4m-2-2v4"/><circle cx="7" cy="18" r="2"/><circle cx="18" cy="18" r="2"/>',
  analytics: '<path d="M4 3h11l5 5v13H4zM14 3v6h6M8 17v-3m4 3v-6m4 6v-4"/>',
  accounts: '<circle cx="9" cy="7" r="3"/><path d="M3 21v-3a6 6 0 0 1 12 0v3M16 4a3 3 0 0 1 0 6m2 4a5 5 0 0 1 3 4v3"/>',
  benchmarks: '<path d="M4 20V4m0 16h17M8 16v-5m5 5V7m5 9v-8"/><path d="M7 5l3-2"/>',
  fish: '<path d="M4 12c4-8 12-8 16 0-4 8-12 8-16 0zM4 12l-3-4v8z"/><circle cx="16" cy="11" r=".6"/>',
  mortality: '<path d="M12 3l10 18H2zM12 9v5m0 3h.01"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  arrow: '<path d="M5 12h14m-5-5l5 5-5 5"/>',
  camera: '<rect x="3" y="6" width="12" height="12" rx="2"/><path d="M15 10l6-3v10l-6-3"/>',
  play: '<path d="M8 4l12 8-12 8z"/>',
  stop: '<rect x="5" y="5" width="14" height="14" rx="2"/>',
  expand: '<path d="M9 3H3v6m12-6h6v6M3 15v6h6m6 0h6v-6"/>',
  trash: '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7"/>',
  edit: '<path d="M16 3l5 5L8 21H3v-5zM13 6l5 5"/>',
  download: '<path d="M12 3v12m-5-5l5 5 5-5M4 16v5h16v-5"/>'
};
const icon = name => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.dashboard}</svg>`;
const State = {
  user: null, csrf: '', farm: '', farms: [], view: 'dashboard', report: null, mortality: null, admin: null, accounts: [], audit: [], engine: null, media: [], generation: 0, refreshing: false, production: {enabled:false}, monitoring: {tanks:[],alerts:[],coverage:[]},
  filters: { tank_id: 'all', days: 14, severity: 'all', metric: 'mortality_rate' }, streams: new Map(), tankView:'grid', countTank: null, estimate: null, sampleMode: false, sampleSequence: 0
};
let toastTimer;
function toast(message, error = false) { $('toast').hidden = false; $('toast').textContent = message; $('toast').className = `toast${error ? ' error' : ''}`; clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 5000); }
function scopedURL(path) { const url = new URL(path, location.origin); if (State.user?.role === 'admin' && State.farm) url.searchParams.set('farm_id', State.farm); return url.pathname + url.search; }
async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (State.csrf) headers.set('X-CSRF-Token', State.csrf);
  if (State.user?.role === 'admin' && State.farm) headers.set('X-Farm-ID', State.farm);
  if (options.body && !(options.body instanceof FormData)) { headers.set('Content-Type', 'application/json'); options.body = JSON.stringify(options.body); }
  const response = await fetch(path, { ...options, headers });
  if (response.status === 401) { location.href = '/login'; throw new Error('Your session ended. Sign in again.'); }
  if (!response.ok) { let error; try { error = await response.json(); } catch { error = {}; } const failure=new Error(typeof error.detail === 'string' ? error.detail : `Request failed (${response.status}).`); failure.status=response.status; throw failure; }
  if (options.raw) return response;
  return response.json();
}
const action = (label, name, id = '', style = 'secondary', symbol = '') => `<button class="button ${style}" data-action="${name}" data-id="${escapeHTML(id)}">${symbol ? icon(symbol) : ''}${escapeHTML(label)}</button>`;
const heading = (title, description, actions = '') => `<div class="page-heading"><div><h1>${escapeHTML(title)}</h1><p class="muted">${escapeHTML(description)}</p></div><div class="heading-actions">${actions}</div></div>`;
const empty = (title, description, extra = '') => `<div class="empty-state"><h3>${escapeHTML(title)}</h3><p>${escapeHTML(description)}</p>${extra}</div>`;
const table = (headers, rows, message = 'No records yet.') => rows.length ? `<div class="table-wrap"><table><thead><tr>${headers.map(h => `<th scope="col">${escapeHTML(h)}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>` : empty(message, 'Saved records will appear here.');
const kpi = (label, value, note, symbol, style = '') => `<article class="kpi ${style}"><div class="kpi-top"><span>${escapeHTML(label)}</span><span class="kpi-icon">${icon(symbol)}</span></div><strong>${escapeHTML(value)}</strong><small>${escapeHTML(note)}</small></article>`;
const tankOptions = (selected = '', includeAll = false) => (includeAll ? '<option value="all">All tanks</option>' : '') + (State.report?.tanks || []).filter(t => t.status !== 'inactive').map(t => `<option value="${escapeHTML(t.tank_id)}" ${t.tank_id === selected ? 'selected' : ''}>${escapeHTML(t.name)} (${escapeHTML(t.tank_id)})</option>`).join('');
function download(label, path) { return `<a class="button secondary" href="${escapeHTML(scopedURL(path))}">${icon('download')}${escapeHTML(label)}</a>`; }

async function start() {
  try {
    const identity = await api('/api/auth/me'); State.user = identity.user; State.csrf = identity.csrf;
    loadDisplayPreferences();
    $('userName').textContent = State.user.display_name; $('avatar').textContent = State.user.display_name.slice(0, 1).toUpperCase();
    $('roleLabel').textContent = State.user.role === 'admin' ? 'Administrator workspace' : 'User · your private farm';
    if (State.user.must_change_password) { renderNavigation(); $('main').innerHTML = empty('Set your own password', 'Change the initial password provided by Admin to open your farm.'); passwordForm(true); return; }
    State.engine = await api('/api/engine');
    if (State.user.role === 'admin') { State.farms = await api('/api/admin/farms'); $('farmSelectorWrap').hidden = false; updateFarmSelector(); }
    else State.farm = State.user.farm_id;
    renderNavigation();
    const requested = location.hash.slice(1); if (['dashboard', 'food', 'tanks', 'dispersals', 'analytics', 'accounts', 'benchmarks'].includes(requested) && !(State.user.role !== 'admin' && ['accounts', 'benchmarks'].includes(requested))) State.view = requested;
    await refresh();
    setInterval(updateMonitoringViews, 1000);
    setInterval(() => { if (!document.hidden) refresh(true); }, 10000);
  } catch (error) { $('main').innerHTML = empty('Unable to open your workspace', error.message); }
}
function updateFarmSelector() { $('farmSelector').innerHTML = '<option value="">All farms overview</option>' + State.farms.map(f => `<option value="${escapeHTML(f.id)}">${escapeHTML(f.name)}</option>`).join(''); $('farmSelector').value = State.farm; }
function renderNavigation() {
  let nav = [['dashboard', 'Dashboard'], ['food', 'Food Management'], ['tanks', 'Tank Management'], ['dispersals', 'Dispersal Management'], ['analytics', 'Analytics & Reports']];
  if (State.user.role === 'admin') { nav.splice(1, 0, ['accounts', 'Farms & Accounts']); if (!State.production.enabled) nav.push(['benchmarks', 'Benchmarks']); }
  $('navigation').innerHTML = nav.map(([name, label]) => `<button class="nav-button${State.view === name ? ' active' : ''}" data-view="${name}" ${State.view === name ? 'aria-current="page"' : ''}>${icon(name)}${label}</button>`).join('');
}
async function navigate(view) { if(view!=='tanks')stopSampleCounting(); State.generation++; State.view = view; location.hash = view; document.body.classList.remove('menu-open'); $('menuButton').setAttribute('aria-expanded', 'false'); renderNavigation(); await refresh(); }
async function refresh(silent = false) {
  if (State.refreshing) { State.refreshQueued = true; return; }
  State.refreshing = true; const generation = State.generation;
  try {
    const production = await api('/api/production');
    if (production.enabled !== State.production.enabled) { stopSampleCounting(); stopAllMonitoring(true); closeInspector(false); $('formDialog').close(); }
    State.production = production;
    if (State.production.enabled && State.view === 'benchmarks') State.view = 'dashboard';
    if (State.user.role === 'admin' && !State.farm && State.view === 'dashboard') State.admin = await api('/api/admin/dashboard');
    else if (State.view === 'accounts') { const [users, farms, audit] = await Promise.all([api('/api/admin/accounts'), api('/api/admin/farms'), api('/api/admin/audit')]); State.accounts = users; State.farms = farms; State.audit = audit; updateFarmSelector(); }
    else if (State.view === 'benchmarks') State.benchmarks = await api('/api/evaluation-benchmarks/grouped');
    else if (State.farm) {
      const report = await api('/api/reports');
      if (generation !== State.generation) return;
      State.report = report;
      State.monitoring = await api('/api/monitoring');
      if (State.production.enabled) for (const tank of State.report.tanks) { tank.demonstration_source=tank.camera_source; tank.camera_source = State.monitoring.tanks.find(t => t.tank_id === tank.tank_id)?.live_source || ''; }
      for (const entry of State.streams.values()) if (entry.farm === State.farm && !report.tanks.some(t => t.tank_id === entry.tankId && t.status !== 'inactive')) stopTankMonitoring(entry.tankId);
      if (State.view === 'analytics' || State.view === 'dashboard' || State.view === 'tanks') {
        const params = new URLSearchParams({ tank_id: State.view === 'analytics' ? State.filters.tank_id : 'all', days: String(State.view === 'analytics' ? State.filters.days : 7), severity: State.view === 'analytics' ? State.filters.severity : 'all' });
        State.mortality = await api('/api/analytics/mortality?' + params);
      }
    }
    if (generation === State.generation) render();
  } catch (error) { if (!silent) $('main').innerHTML = empty('Records could not be loaded', error.message, action('Try again', 'refresh')); else toast(error.message, true); }
  finally { State.refreshing = false; if (State.refreshQueued) { State.refreshQueued = false; refresh(); } }
}
function render() {
  const previouslyFocused=document.activeElement;
  const focusKey=previouslyFocused?.dataset.action ? `[data-action="${CSS.escape(previouslyFocused.dataset.action)}"][data-id="${CSS.escape(previouslyFocused.dataset.id || '')}"]` : previouslyFocused?.dataset.displayChoice ? `[data-display-choice="${CSS.escape(previouslyFocused.dataset.displayChoice)}"]` : previouslyFocused?.dataset.openTank ? `[data-open-tank="${CSS.escape(previouslyFocused.dataset.openTank)}"]` : previouslyFocused?.dataset.view ? `[data-view="${CSS.escape(previouslyFocused.dataset.view)}"]` : previouslyFocused?.id ? '#'+CSS.escape(previouslyFocused.id) : '';
  const displayOpen=document.querySelector('.display-options')?.open;
  renderNavigation(); const names = { dashboard: 'Dashboard', accounts: 'Farms & Accounts', food: 'Food Management', tanks: 'Tank Management', dispersals: 'Dispersal Management', analytics: 'Analytics & Reports', benchmarks: 'Benchmarks' };
  $('pageTitle').textContent = names[State.view];
  $('workspaceLabel').textContent = State.user.role === 'admin' ? (State.farms.find(f => f.id === State.farm)?.name || 'Admin · all farms') : 'User · private farm';
  if (State.user.role === 'admin' && !State.farm && !['dashboard', 'accounts', 'benchmarks'].includes(State.view)) { $('main').innerHTML = heading(names[State.view], 'Choose a farm in the header to view and manage its records.') + empty('Select a farm', 'Admin can manage one selected farm at a time.'); return; }
  const pages = { dashboard: renderDashboard, accounts: renderAccounts, food: renderFood, tanks: renderTanks, dispersals: renderDispersals, analytics: renderAnalytics, benchmarks: renderBenchmarks };
  $('monitoringDock').append($('countDialog'));
  $('main').innerHTML = pages[State.view]();
  mountInspector();
  if(displayOpen && document.querySelector('.display-options'))document.querySelector('.display-options').open=true;
  document.body.classList.toggle('production-mode',State.production.enabled);
  ensureProductionViews();
  updateMonitoringViews();
  if (focusKey && (!previouslyFocused.isConnected || document.activeElement===document.body)) document.querySelector(focusKey)?.focus({preventScroll:true});
}
function renderDashboard() {
  if (State.user.role === 'admin' && !State.farm) return renderAdminDashboard();
  const s = State.report?.summary || {}, trends = State.report?.daily_trends || [];
  const running = State.monitoring.tanks.filter(t => ['running','starting','reconnecting'].includes(t.status)).length;
  const coverage = State.monitoring.coverage.reduce((sum,c) => sum+c.valid_seconds,0), observed = State.monitoring.coverage.reduce((sum,c) => sum+c.observed_seconds,0);
  return heading('Dashboard', 'Understand population changes, losses and feeding needs. Open Tank Management for live cameras.', (s.estimated_mortality_count ? action('Review estimated losses','review-losses','','secondary','mortality') : '')+action('Open tank monitoring','view','tanks','','camera')) + productionPanel() +
    `<div class="kpis">${kpi('Saved population',fmt(s.total_population),State.production.enabled ? 'Automatically reconciled from validated cameras' : 'Saved tank inventory','fish')}${kpi('Estimated mortality today',fmt(s.estimated_mortality_count || 0),'Unexplained population losses; may be corrected','mortality','amber')}${kpi('Confirmed deaths today',fmt(s.confirmed_mortality_count || 0),'Farmer-confirmed records only','mortality','red')}${kpi('Estimated daily feed',fmt(s.total_feed_kg,3)+' kg','Current population x configured feeding inputs','food')}</div>` + engineNotice() +
    `<div class="grid-two"><section class="panel"><div class="panel-header"><h2>Population trend</h2></div>${chart(trends.map(t=>t.label),[{name:'Saved population',color:'#087f8c',values:trends.map(t=>t.population)}],'Saved population over seven days')}</section><section class="panel"><div class="panel-header"><h2>Mortality trend</h2></div>${chart(trends.map(t=>t.label),[{name:'Estimated losses',color:'#a86e0a',values:trends.map(t=>t.estimated_mortality)},{name:'Confirmed deaths',color:'#b44f78',values:trends.map(t=>t.confirmed_mortality)}],'Estimated and confirmed daily losses')}</section></div>` +
    `<section class="panel"><div class="panel-header"><h2>Farm insights & monitoring health</h2></div><div class="labelled-value"><span>Background monitoring</span><strong>${running} / ${monitorTanks().length} tanks</strong></div><div class="labelled-value"><span>Reliable frames during observed time today</span><strong>${observed ? fmt(coverage/observed*100,1)+'%' : 'No observations'}</strong></div><p>${s.estimated_mortality_count ? 'Some losses are estimated. Record any dispersal to explain removals and correct mortality.' : 'No unexplained losses have been recorded today.'}</p><p>${monitorTanks().some(t=>t.capacity_pct>100) ? 'A tank exceeds its configured capacity. Review its population and planned dispersal.' : 'Saved populations are within configured tank capacities.'}</p><p class="muted">Coverage describes recorded observation time, not a guarantee of an entire day. Daily loss rate includes estimates: ${pct(s.mortality_rate_pct)}.</p></section>` + monitoringAlerts();
}
function productionPanel() { if(State.user.role!=='admin')return ''; const on=State.production.enabled; return `<section class="panel production-panel"><div class="panel-header"><div><h2>Production Mode <span class="pill ${on ? '' : 'neutral'}">${on ? 'On' : 'Off'}</span></h2><p>${on ? 'Live cameras monitor independently of the browser. Reliable censuses update saved population and estimated losses.' : 'Demonstration sources are available. Camera estimates do not automatically change inventory.'}</p></div>${action(on ? 'Turn Production Mode off' : 'Turn Production Mode on','production-toggle','','secondary','camera')}</div><p class="muted">Windows startup: ${State.production.startup_configured ? 'configured; starts only when Production Mode is on' : 'not configured; Admin setup required'}.</p></section>`; }
function monitoringAlerts() { return `<section class="panel"><div class="panel-header"><h2>Monitoring alerts</h2></div>${State.monitoring.alerts.length ? State.monitoring.alerts.slice(0,20).map(a=>`<div class="labelled-value"><div><strong>${escapeHTML(a.message)}</strong><small>${escapeHTML(a.created_at)}${a.tank_id ? ' · '+escapeHTML(a.tank_id) : ''}</small></div><span class="pill ${a.resolved_at ? 'neutral' : 'amber'}">${a.resolved_at ? 'Recovered' : 'Needs attention'}</span></div>`).join('') : '<p class="muted">No monitoring alerts recorded.</p>'}</section>`; }
function engineNotice() { return State.engine?.status==='ready' ? '' : `<p class="service-status" role="status">${State.engine?.status==='degraded' ? 'Counting service: reduced coverage. Ask Admin to check it.' : 'Counting service unavailable. Ask Admin to check it.'}</p>`; }
function renderAdminDashboard() {
  const data = State.admin || { summary: {}, farms: [] }, s = data.summary;
  return heading('Farm administration', 'Review all farms, follow recording gaps, and open a farm to manage its records.', action('Create farmer account', 'account-new', '', '', 'plus')) + productionPanel() +
    `<div class="kpis">${kpi('Registered farms', fmt(s.total_farms), 'Includes the preserved existing farm', 'accounts')}${kpi('Saved fish population', fmt(s.total_population), 'Across all farm inventories', 'fish')}${kpi("Today's mortality", pct(s.mortality_rate_pct), 'Includes estimates; weighted across observed farms', 'mortality', 'red')}${kpi('Tanks without observations', fmt(s.missing_checks), 'No recorded mortality observation today', 'analytics', 'amber')}</div>` + engineNotice() +
    `<section class="panel"><div class="panel-header"><div><h2>Farm overview</h2><p>Only Admin can see this combined view.</p></div>${action('Refresh', 'refresh', '', 'secondary small')}</div>` + table(['Farm', 'Saved fish', 'Estimated feed / day', 'Mortality today', 'Observed tanks', 'Manage'], data.farms.map(f => `<tr><td><strong>${escapeHTML(f.name)}</strong></td><td>${fmt(f.summary.total_population)}</td><td>${fmt(f.summary.total_feed_kg, 3)} kg</td><td>${pct(f.summary.mortality_rate_pct)}</td><td>${f.summary.mortality_checked_tanks}/${f.summary.mortality_expected_tanks}</td><td>${action('Open farm', 'farm', f.id, 'secondary small', 'arrow')}</td></tr>`)) + '</section>';
}
function tankCard(t, full = true) {
  const status = t.status === 'inactive' ? 'Archived' : t.capacity_pct > 100 ? 'Over capacity' : t.status === 'quarantine' ? 'Quarantine' : 'Active';
  return `<article class="tank-card" data-monitor-tank="${escapeHTML(t.tank_id)}"><div class="tank-card-top"><div><h3>${escapeHTML(t.name)}</h3><span class="muted">${escapeHTML(t.tank_id)}</span></div><span class="pill ${status === 'Over capacity' ? 'red' : status === 'Quarantine' ? 'amber' : status === 'Archived' ? 'neutral' : ''}">${status}</span></div><div class="tank-count">${fmt(t.current_count)}<small>saved fish</small></div><div class="capacity-bar"><span style="width:${Math.min(100, Number(t.capacity_pct))}%"></span></div><small class="muted">${fmt(t.current_count)} of ${fmt(t.max_capacity)} capacity · ${fmt(t.capacity_pct, 1)}%</small><div class="tank-details"><div><span>Estimated biomass</span><strong>${fmt(t.biomass_kg, 3)} kg</strong></div><div><span>Daily feed estimate</span><strong>${fmt(t.daily_feed_kg, 3)} kg</strong></div><div><span>Camera-view occupancy</span><strong data-monitor-field="occupancy">${freshMonitoring(tankStream(t.tank_id)) ? fmt(tankStream(t.tank_id).data.live_count / Math.max(1,t.max_capacity) * 100,1)+'%' : '\u2014'}</strong></div></div><div class="tank-actions">${action('Monitor tank', 'count', t.tank_id, '', 'camera')}${full ? action('Stock fish', 'stock', t.tank_id, 'secondary small') : ''}${action('Mortality check', 'mortality', t.tank_id, 'secondary small')}${full ? action('Edit tank', 'tank-edit', t.tank_id, 'secondary small') : ''}</div></article>`;
}
function renderTanks() {
  const tanks=monitorTanks();
  if(!tanks.some(t=>t.tank_id===State.countTank)) { State.countTank=tanks[0]?.tank_id; if(!State.countTank)State.tankView='grid'; }
  return heading('Tank Management','Click a tank to open its live view. Other cameras keep running.',action('Start all','monitor-all','','','play')+action('Add tank','tank-new','','secondary','plus'))+productionPanel()+
    `<div class="tank-view-tools"><span class="muted">${State.tankView==='inspect' ? 'Live tank view' : 'Your tank cameras'}</span>${displayControls()}</div>`+
    (State.tankView==='inspect' ? '<div id="tankInspectorHost"></div>' : `<div class="monitoring-grid">${tanks.map(monitoringTile).join('') || empty('Create your first tank','Add its starting population and camera.',action('Add tank','tank-new','','','plus'))}</div>`);
}
function renderFood() {
  const tanks = monitorTanks(), total = tanks.reduce((sum,t) => sum+t.daily_feed_kg,0), biomass = tanks.reduce((sum,t) => sum+t.biomass_kg,0);
  return heading('Food Management', 'Plan feed from saved population and your configured feeding settings.', action('Edit tank settings', 'view', 'tanks', 'secondary', 'tanks')) +
    `<div class="kpis">${kpi('Estimated feed today', fmt(total,3)+' kg', 'Total daily requirements across active tanks', 'food')}${kpi('Seven-day feed estimate', fmt(total*7,3)+' kg', 'Assumes unchanged stock, weight and feeding rate', 'food', 'amber')}${kpi('Estimated biomass', fmt(biomass,3)+' kg', 'Saved fish x configured average weight', 'fish')}${kpi('Tanks in the feed plan', fmt(tanks.length), 'Archived tanks are excluded', 'tanks')}</div>` +
    `<section class="panel"><div class="panel-header"><div><h2>Today's estimated feed requirements</h2><p>Saved fish &times; configured weight &divide; 1,000 &times; daily feeding percentage</p></div></div>${table(['Tank','Saved fish','Configured average weight','Daily rate','Daily requirement','Share of daily feed'], tanks.map(t => `<tr><td><strong>${escapeHTML(t.name)}</strong><small>${escapeHTML(t.tank_id)}</small></td><td>${fmt(t.current_count)}</td><td>${fmt(t.avg_weight_g,3)} g</td><td>${fmt(t.feed_rate_pct*100,1)}%</td><td><strong>${fmt(t.daily_feed_kg,3)} kg</strong></td><td>${total ? fmt(t.daily_feed_kg/total*100,1)+'%' : '\u2014'}</td></tr>`), 'Add a tank to build your feed plan.')}</section>` +
    `<div class="grid-two"><section class="panel"><div class="panel-header"><h2>Feed planning insights</h2></div>${tanks.filter(t => t.current_count>0).sort((a,b) => b.daily_feed_kg-a.daily_feed_kg).map(t => `<div class="labelled-value"><div><strong>${escapeHTML(t.name)}</strong><small>${fmt(t.biomass_kg,3)} kg estimated biomass &middot; ${fmt(t.feed_rate_pct*100,1)}% daily rate</small></div><strong>${fmt(t.daily_feed_kg*7,3)} kg / 7 days</strong></div>`).join('') || empty('No stocked tanks yet','Saved stock provides the basis for feed estimates.')}</section><section class="panel feed-guidance"><h2>Understand your estimate</h2><p>Requirements use saved population. In Production Mode, validated whole-tank cameras automatically reconcile this population.</p><p>Configured average weight is a manual feeding input. Review it when fish grow, and adjust the feeding percentage in Tank Management.</p><p>The seven-day estimate uses today's settings; it does not predict growth or future stock changes.</p></section></div>`;
}
function dispersalTable(records) { return table(['Date', 'Reference', 'Tank', 'Recipient', 'Activity', 'Fish', 'Sales revenue'], records.map(d => `<tr><td>${escapeHTML(d.timestamp)}</td><td>${escapeHTML(d.dispersal_id)}</td><td>${escapeHTML(State.report?.tanks.find(t => t.tank_id === d.tank_id)?.name || d.tank_id)}</td><td>${escapeHTML(d.recipient)}</td><td>${d.type === 'sale' ? 'Sale' : 'Transfer'}</td><td>${fmt(d.count)}</td><td>${d.type === 'sale' ? money(d.total_revenue_php) : '—'}</td></tr>`), 'No dispersals recorded.'); }
function renderDispersals() { const s = State.report?.summary || {}; return heading('Dispersal Management', 'Record sales or transfers. Confirm quantities before fish are deducted from saved stock.', action('Record dispersal', 'dispersal-new', '', '', 'plus')) + `<div class="kpis">${kpi('Fish dispersed', fmt(s.total_dispersed_count), 'All recorded sales and transfers', 'dispersals')}${kpi('Recorded sales revenue', money(s.dispersal_earnings), 'Sales only; transfers do not earn revenue', 'analytics')}${kpi('Saved population', fmt(s.total_population), 'Fish remaining in active tanks', 'fish')}${kpi('Confirmed deaths today', fmt(s.confirmed_mortality_count), 'Deaths are recorded separately from dispersal', 'mortality', 'red')}</div><section class="panel"><div class="panel-header"><h2>Dispersal ledger</h2>${download('Export farm report', '/api/reports/export/csv')}</div>${dispersalTable(State.report?.dispersals || [])}</section>`; }
function chart(labels, datasets, description, suffix = '') {
  const values = datasets.flatMap(d => d.values.filter(v => v != null)); if (!values.length) return empty('No recorded chart data', 'Missing records are left blank. Enable validated Production monitoring to build trends.');
  const w = 720, h = 225, left = 48, top = 15, bottom = 34, right = 12, plotW = w - left - right, plotH = h - top - bottom, max = Math.max(1, ...values) * 1.08;
  const x = i => left + (labels.length === 1 ? plotW / 2 : i * plotW / (labels.length - 1)); const y = v => top + plotH - (v / max) * plotH;
  let svg = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${escapeHTML(description)}"><title>${escapeHTML(description)}</title>`;
  for (let i = 0; i <= 4; i++) { const yy = top + plotH * i / 4; svg += `<line x1="${left}" y1="${yy}" x2="${w - right}" y2="${yy}" stroke="#e7eee5"/><text x="${left - 8}" y="${yy + 4}" text-anchor="end" fill="#718274" font-size="10">${fmt(max * (1 - i / 4), suffix ? 1 : 0)}${suffix}</text>`; }
  labels.forEach((label, i) => { if (i === 0 || i === labels.length - 1 || i % Math.max(1, Math.floor(labels.length / 5)) === 0) svg += `<text x="${x(i)}" y="${h - 8}" text-anchor="middle" fill="#718274" font-size="10">${escapeHTML(label)}</text>`; });
  datasets.forEach(d => { let segment = []; const draw = () => { if (segment.length) svg += `<polyline points="${segment.join(' ')}" fill="none" stroke="${d.color}" stroke-width="2"/>`; segment = []; }; d.values.forEach((v, i) => { if (v == null) { draw(); return; } segment.push(`${x(i)},${y(v)}`); }); draw(); d.values.forEach((v, i) => { if (v != null) svg += `<circle cx="${x(i)}" cy="${y(v)}" r="3.5" fill="${d.color}" stroke="white" stroke-width="1.5" tabindex="0"><title>${escapeHTML(d.name)} · ${escapeHTML(labels[i])}: ${fmt(v, 2)}${suffix}</title></circle>`; }); });
  return `<div class="charts">${svg}</svg></div><div class="chart-legend">${datasets.map(d => `<span><i class="legend-dot" style="background:${d.color}"></i>${escapeHTML(d.name)}</span>`).join('')}</div><p class="chart-caption">Hover a point for its recorded value. Gaps indicate unavailable records.</p>`;
}
function renderAnalytics() {
  const m = State.mortality || { summary: {}, records: [], tank_series: {}, timeline: [] }, s = m.summary;
  const params = new URLSearchParams({ tank_id: State.filters.tank_id, days: State.filters.days, severity: State.filters.severity });
  const metric = State.filters.metric;
  const datasets = Object.values(m.tank_series).map(t => ({ name: t.tank_name, color: t.color, values: m.timeline_dates.map(day => { const row = m.records.find(r => r.tank_id === t.tank_id && r.date === day); return row ? (metric === 'population' ? row.population : metric === 'mortality_count' ? row.mortality_count : row.mortality_rate_pct) : null; }) }));
  return heading('Analytics & Reports', 'Review recorded population and losses, feeding requirements, and dispersal records in one place.', (s.estimated_mortality_count ? action('Review estimated losses','review-losses','','secondary','mortality') : '')+download('Export farm report', '/api/reports/export/csv')) +
    `<section class="panel"><div class="panel-header"><div><h2>Population & mortality</h2><p>Daily loss rate includes estimated losses and confirmed deaths ÷ (opening fish + incoming fish that day) × 100</p></div>${download('Export filtered mortality', '/api/analytics/mortality/export/csv?' + params)}</div><div class="filters"><label>Tank<select id="filterTank">${tankOptions(State.filters.tank_id, true)}</select></label><label>Period<select id="filterDays">${[7, 14, 30, 90].map(n => `<option value="${n}" ${n === Number(State.filters.days) ? 'selected' : ''}>${n} days</option>`).join('')}</select></label><label>Record filter<select id="filterSeverity">${[['all', 'All records'], ['normal', 'Below 1%'], ['elevated', '1% to 3%'], ['critical', 'Above 3%'], ['unrecorded', 'Rate unavailable']].map(([v, l]) => `<option value="${v}" ${v === State.filters.severity ? 'selected' : ''}>${l}</option>`).join('')}</select></label><label>Chart<select id="filterMetric">${[['mortality_rate', 'Daily loss rate'], ['population', 'Saved population'], ['mortality_count', 'Losses including estimates']].map(([v, l]) => `<option value="${v}" ${v === metric ? 'selected' : ''}>${l}</option>`).join('')}</select></label></div>${chart(m.timeline, datasets, metric === 'population' ? 'Recorded tank population' : metric === 'mortality_count' ? 'Daily losses including estimates' : 'Daily mortality rates', metric === 'mortality_rate' ? '%' : '')}</section>` +
    `<div class="kpis">${kpi('Losses including estimates', fmt(s.total_mortalities), 'Within selected dates and filters', 'mortality', 'red')}${kpi('Weighted daily mortality', pct(s.avg_mortality_rate_pct), 'Recorded deaths ÷ recorded fish-days', 'analytics')}${kpi('Daily check coverage', `${s.recorded_days || 0} / ${s.expected_days || 0}`, 'Recorded tank-days / possible tank-days', 'tanks', 'amber')}${kpi('Highest recorded rate', pct(s.peak_rate_pct), s.peak_date === 'N/A' ? 'No recorded rate' : `${s.peak_tank} · ${s.peak_date}`, 'mortality')}</div>` +
    `<section class="panel"><div class="panel-header"><h2>Daily mortality records</h2></div>${table(['Date', 'Tank', 'Opening fish', 'Incoming fish', 'Estimated losses', 'Confirmed deaths', 'Daily loss rate', 'Observation status'], m.records.map(r => `<tr><td>${escapeHTML(r.date)}</td><td>${escapeHTML(r.tank_name)}</td><td>${fmt(r.opening_population)}</td><td>${fmt(r.additions)}</td><td>${r.mortality_count == null ? 'Unavailable' : fmt(r.estimated_mortality_count)}</td><td>${fmt(r.confirmed_mortality_count)}</td><td>${pct(r.mortality_rate_pct)}</td><td><span class="pill ${r.severity === 'critical' ? 'red' : r.severity === 'elevated' ? 'amber' : r.severity === 'unrecorded' ? 'neutral' : ''}">${escapeHTML(r.status_label)}</span></td></tr>`))}<p class="chart-caption">A saved inventory correction makes that day's denominator unavailable. Historical openings are never invented.</p></section>` +
    `<section class="panel"><div class="panel-header"><h2>Saved tank inventory & feed estimates</h2></div>${table(['Tank', 'Fish', 'Biomass estimate', 'Feed rate', 'Daily feed estimate'], (State.report?.tanks || []).map(t => `<tr><td>${escapeHTML(t.name)}</td><td>${fmt(t.current_count)}</td><td>${fmt(t.biomass_kg, 3)} kg</td><td>${fmt(t.feed_rate_pct * 100, 1)}%</td><td>${fmt(t.daily_feed_kg, 3)} kg</td></tr>`))}</section><section class="panel"><div class="panel-header"><h2>Dispersal report</h2></div>${dispersalTable(State.report?.dispersals || [])}</section>`;
}
function renderAccounts() { return heading('Farms & Accounts', 'Each farmer has a private farm. Admin can view and manage all farms.', action('Create farmer account', 'account-new', '', '', 'plus')) + `<section class="panel"><div class="panel-header"><h2>Farmer accounts</h2></div>${table(['Farmer', 'Username', 'Farm', 'Access', 'Manage'], State.accounts.filter(u => u.role === 'user').map(u => `<tr><td><strong>${escapeHTML(u.display_name)}</strong></td><td>${escapeHTML(u.username)}</td><td>${escapeHTML(u.farm_name)}</td><td><span class="pill ${u.active ? '' : 'neutral'}">${u.active ? 'Active' : 'Disabled'}</span>${u.must_change_password ? '<small>Password change required</small>' : ''}</td><td>${action('Open farm', 'farm', u.farm_id, 'secondary small')}${action('Manage access', 'account-edit', u.id, 'secondary small')}</td></tr>`), 'No farmer accounts yet.')}</section><section class="panel"><div class="panel-header"><h2>Recent Admin & farm activity</h2><p>Account passwords and camera credentials are not recorded here.</p></div>${table(['Recorded', 'Account', 'Farm', 'Action'], State.audit.map(a => `<tr><td>${escapeHTML(a.timestamp)}</td><td>${escapeHTML(a.username || 'Local setup')}</td><td>${escapeHTML(a.farm_name || '—')}</td><td>${escapeHTML(a.action)}<small>${escapeHTML(a.detail)}</small></td></tr>`), 'No audited activity yet.')}</section>`; }
function renderBenchmarks() {
  const batches = Array.isArray(State.benchmarks) ? State.benchmarks : State.benchmarks?.batches || [];
  return heading('Benchmarks', 'Admin-only counting evaluation. Individual engines use anonymous labels and fixed calibrated settings.', download('Export benchmarks', '/api/evaluation-benchmarks/export/csv')) +
    `<section class="panel"><div class="panel-header"><div><h2>Check counting against a known image</h2><p>Provide a manually verified count. Optional annotations allow box-matching metrics.</p></div>${action('Evaluate an image', 'benchmark-new', '', '', 'benchmarks')}</div></section><section class="panel"><div class="panel-header"><h2>Evaluation history</h2>${action('Refresh', 'refresh', '', 'secondary small')}</div>${batches.length ? batches.map(b => { const engines = b.models_list || Object.values(b.models || {}); return `<div class="section-header"><h3>${escapeHTML(b.source_name || 'Recorded image')}</h3><span class="muted">${escapeHTML(b.timestamp || b.batch_id)}</span></div>${table(['Engine', 'Actual fish', 'Counted fish', 'Count error', 'Precision', 'Recall', 'F1', 'Detection time'], engines.map(e => `<tr><td>${escapeHTML(e.model_name)}</td><td>${fmt(e.actual_count)}</td><td>${fmt(e.predicted_count)}</td><td>${fmt(e.mae, 2)}</td><td>${fmt(e.precision * 100, 1)}%</td><td>${fmt(e.recall * 100, 1)}%</td><td>${fmt((e.f1_score ?? e.f1) * 100, 1)}%</td><td>${fmt(e.inference_ms, 1)} ms</td></tr>`))}`; }).join('') : empty('No evaluations recorded', 'Evaluate an image with a known fish count to compare counting results.')}</section>`;
}
function fieldHelp(label,help) { return help ? `<button type="button" class="field-help" aria-label="Help for ${escapeHTML(label)}" data-tooltip="${escapeHTML(help)}">?</button>` : ''; }
function field(name, label, value = '', type = 'text', extra = '', help = '') { return `<label class="field"><span class="field-title">${escapeHTML(label)}${fieldHelp(label,help)}</span><input name="${name}" aria-label="${escapeHTML(label)}" type="${type}" value="${escapeHTML(value)}" ${extra}></label>`; }
function select(name, label, options, help = '') { return `<label class="field"><span class="field-title">${escapeHTML(label)}${fieldHelp(label,help)}</span><select name="${name}" aria-label="${escapeHTML(label)}" ${options.includes('<option value="">') ? '' : 'required'}>${options}</select></label>`; }
let submitForm = null;
function openForm(title, content, submit, label = 'Save record', required = false) { $('dialogForm').oninput=$('dialogForm').onchange=null; delete $('dialogForm').dataset.tankEdit; delete $('dialogForm').dataset.unsaved; $('dialogTitle').textContent = title; $('dialogBody').innerHTML = content; $('formError').textContent = ''; $('saveDialog').textContent = label; $('saveDialog').classList.remove('danger'); $('saveDialog').disabled = false; $('cancelDialog').hidden = required; $('closeDialog').hidden = required; $('formDialog').dataset.required = String(required); submitForm = submit; $('formDialog').showModal(); }
async function saveOperation(path, data, method = 'POST') { await api(path, { method, body: data }); toast('Record saved.'); await refresh(); }
async function tankForm(id) {
  const tank=(State.report?.tanks || []).find(t=>t.tank_id===id), farm=State.farm, generation=State.generation;
  const production=State.production.enabled, config=State.monitoring.tanks.find(t=>t.tank_id===id);
  State.media=production ? [] : await api('/api/media');
  if (farm!==State.farm || generation!==State.generation) return;
  const number=Math.max(0,...(State.report?.tanks || []).map(t=>Number(/^TANK-(\d+)$/.exec(t.tank_id)?.[1] || 0)))+1;
  const suffix=String(number).padStart(2,'0'), suggestedName='Tank '+suffix;
  const liveSource=config?.live_source || (sourceKind(tank)==='camera' ? tank?.camera_source : '') || '';
  const initial=production ? liveSource : tank?.camera_source || '';
  const initialKind=!initial ? 'none' : /^\d+$/.test(initial) ? 'usb' : /^rtsps?:\/\//.test(initial) ? 'rtsp' : 'saved';
  const options=[['none','No source'],['usb','USB camera'],['rtsp','Network camera'],...(!production ? [['upload','Upload video'],['saved','Saved video']] : [])].map(([key,label])=>`<option value="${key}" ${key===initialKind ? 'selected' : ''}>${label}</option>`).join('');
  const remembered=config?.video_source || (initialKind==='saved' ? initial : '');
  const sourceFields=`<section class="source-picker">${select('source_type','Source',options,'Choose a camera connected to the system PC, a network camera, or a test video. The same physical camera is used for automatic monitoring.')}
    <div data-source-controls="usb">${select('local_camera','Camera number',Array.from({length:21},(_,i)=>`<option value="${i}" ${String(i)===liveSource ? 'selected' : ''}>Camera ${i}</option>`).join(''),'Numbers identify cameras connected to the system PC. Check the preview to choose the right tank view.')}</div>
    <div data-source-controls="rtsp">${field('rtsp_address','Network camera address',/^rtsps?:\/\//.test(liveSource) ? liveSource : '','text','placeholder="rtsp://192.168.1.100:554/stream"','Enter the RTSP address supplied with your network camera. Changing the physical camera requires a new whole-tank validation.')}</div>
    ${!production ? `<div data-source-controls="upload"><label class="field"><span class="field-title">Video file${fieldHelp('Video file','Upload a test video. It does not replace the remembered physical camera or change saved stock.')}</span><input name="video_upload" aria-label="Video file" type="file" accept=".mp4,.avi,.mov,.mkv,.webm"><small data-upload-status>Choose a file, then save the tank to upload it.</small></label></div>
    <div data-source-controls="saved">${select('video_source','Saved video','<option value="">Choose an uploaded video</option>'+State.media.map(m=>`<option value="${escapeHTML(m.source)}" ${m.source===remembered ? 'selected' : ''}>${escapeHTML(m.name)}</option>`).join(''),'Only videos belonging to this farm are shown. Your physical camera remains available for automatic monitoring.')}</div>` : ''}</section>`;
  openForm(tank ? 'Edit tank' : 'Add a tank',`<div class="form-grid">${field('name','Tank name',tank?.name || suggestedName,'text','required maxlength="120"','A name is filled automatically. You can change it to identify the location or purpose of this tank.')}${field('tank_id','Tank code',tank?.tank_id || 'TANK-'+suffix,'text','readonly','Permanent code used to keep this tank’s records together. New codes are assigned when saved and are never reused from archived tanks.')}${field('max_capacity','Tank capacity (fish)',tank?.max_capacity ?? 1000,'number','required min="1" step="1"','The planned maximum number of fish. Camera-view occupancy compares visible fish with this number; it is not a water-quality measurement.')}${field('current_count',tank ? 'Saved population (correction only)' : 'Known initial population',tank?.current_count ?? 0,'number','required min="0" step="1"','Enter an independently verified count. Correcting an existing population requires a reason and a new camera validation; record sales or transfers as dispersals.')}${field('avg_weight_g','Configured average weight (g)',tank?.avg_weight_g ?? 2.5,'number','required min="0.001" step="0.001"','Weigh a representative sample and enter the average in grams. Camera images do not measure fish weight. Review this input as fish grow.')}${field('feed_percent','Daily feed (% of biomass)',(tank?.feed_rate_pct ?? .05)*100,'number','required min="0.1" max="15" step="0.1"','The percentage of estimated fish biomass to feed each day. For example, 5 means 5%. Use your farm’s feeding guidance.')}</div>`+sourceFields+
    (tank ? select('status','Tank status',['active','quarantine'].map(v=>`<option value="${v}" ${v===tank.status ? 'selected' : ''}>${v==='inactive' ? 'Archived' : v[0].toUpperCase()+v.slice(1)}</option>`).join(''),'Active and quarantine tanks can be monitored. Use Delete tank to remove a tank while preserving its history.')+field('adjustment_note','Reason for a population correction','','text','maxlength="500"','Explain a verified correction. Leave this blank when only changing camera or feeding settings.') : '')+(tank ? `<section class="camera-setup" data-camera-setup="${escapeHTML(id)}"><h3>Camera accuracy</h3><p data-camera-check-status role="status"></p>${action('Check camera accuracy','validate-census',id,'secondary','camera')}<small>Check once against known stock before automatic updates. Repeat after camera changes or a population correction.</small></section><div class="tank-delete-action">${action('Delete tank','tank-delete',id,'danger','trash')}</div>` : ''),async form=>{
      const data=Object.fromEntries(new FormData(form)); data.feed_rate_pct=Number(data.feed_percent)/100; delete data.feed_percent;
      if (tank && Number(data.current_count)===tank.current_count) delete data.current_count;
      delete data.tank_id; if(!tank && data.name===suggestedName)delete data.name;
      const kind=form.elements.source_type.value;
      let value=kind==='usb' ? form.elements.local_camera.value : kind==='rtsp' ? form.elements.rtsp_address.value.trim() : kind==='saved' ? form.elements.video_source.value : '';
      if(kind==='upload') { const file=form.elements.video_upload.files[0]; if(!file)throw new Error('Choose a video file first.'); const upload=new FormData(); upload.append('file',file); form.querySelector('[data-upload-status]').textContent='Uploading video…'; try { value=(await api('/api/tanks/upload-video',{method:'POST',body:upload})).filepath; } catch(error) { form.querySelector('[data-upload-status]').textContent='Upload failed. Choose a file and try again.'; throw error; } }
      const sourceChanged=kind!==initialKind || value!==initial;
      if(sourceChanged || !tank)data.source={type:['saved','upload'].includes(kind) ? 'video' : kind,value};
      for(const key of ['source_type','local_camera','rtsp_address','video_source','video_upload'])delete data[key];
      if (farm!==State.farm) return;
      const restart=!production && tank && isMonitoring(tankStream(id)) && (sourceChanged || Number(data.max_capacity)!==tank.max_capacity);
      if (restart || (!production && tank && data.status==='inactive')) await stopTankMonitoring(id);
      const saved=await api(tank ? '/api/tanks/'+encodeURIComponent(id) : '/api/tanks',{method:tank ? 'PUT' : 'POST',body:data});
      if (farm!==State.farm) return;
      toast(tank ? 'Tank settings saved.' : saved.name+' created ('+saved.tank_id+').'); await refresh();
      if (restart && data.status!=='inactive' && saved.camera_source) startTankMonitoring(id);
    },tank ? 'Save tank' : 'Create tank');
  const form=$('dialogForm');
  const update=()=>form.querySelectorAll('[data-source-controls]').forEach(group=>{ const active=group.dataset.sourceControls===form.elements.source_type.value; group.hidden=!active; group.querySelectorAll('input,select').forEach(input=>{input.disabled=!active;input.required=active;}); });
  form.elements.source_type.onchange=update; update();
  form.dataset.tankEdit=id || '';
  const initialValues=JSON.stringify([...new FormData(form)].map(([k,v])=>[k,v instanceof File ? v.name : v]));
  const check=()=>{ form.dataset.unsaved=String(initialValues!==JSON.stringify([...new FormData(form)].map(([k,v])=>[k,v instanceof File ? v.name : v]))); updateCameraSetup(); };
  form.oninput=check; form.onchange=check; check();
}
function updateCameraSetup() {
  const root=document.querySelector('[data-camera-setup]'); if(!root)return;
  const config=monitorConfig(root.dataset.cameraSetup),button=root.querySelector('[data-action=validate-census]');
  const pending=$('dialogForm').dataset.unsaved==='true';
  root.querySelector('[data-camera-check-status]').textContent=config?.status==='validating' ? config.message || 'Checking camera accuracy...' : config?.validated ? 'Camera accuracy checked. Ready for automatic updates.' : config?.live_source ? config.message && config.message!=='Production Mode is off.' ? config.message : 'Accuracy check needed before automatic updates.' : 'Choose and save a physical camera first.';
  button.disabled=pending || !config?.live_source || config.status==='validating';
  button.dataset.tooltip=pending ? 'Save your changes before checking camera accuracy.' : config?.validated ? 'Repeat if the camera position or whole-tank view has changed.' : 'Compare the whole-tank camera view with an independently verified fish count.';
}
async function deleteTankForm(id) {
  const latest=await api('/api/tanks/'+encodeURIComponent(id));
  openForm('Delete '+latest.name+'?',`<p>Remove <strong>${escapeHTML(latest.name)}</strong> from active monitoring?</p><p>${fmt(latest.current_count)} saved fish will be removed from active inventory. This is recorded separately from deaths and dispersals. Historical records are kept.</p>`,async()=>{
    try {await api('/api/tanks/'+encodeURIComponent(id)+'?'+new URLSearchParams({remove_stock:'true',expected_count:String(latest.current_count)}),{method:'DELETE'});}
    catch(error){if(error.status===409){await refresh();const current=await api('/api/tanks/'+encodeURIComponent(id));latest.current_count=current.current_count; $('dialogBody').querySelectorAll('p')[1].textContent=fmt(current.current_count)+' saved fish will be removed from active inventory. History is kept; this does not record deaths or dispersal.';}throw error;}
    const entry=tankStream(id); if(entry){finishMonitoring(entry,'stopped','Tank removed.');State.streams.delete(entry.key);}
    if(State.countTank===id)closeInspector(false);
    toast('Tank deleted. Historical records are kept.'); await refresh();
  },'Delete tank');
  $('saveDialog').classList.add('danger');
}
function productionForm() {
  const enabled=!State.production.enabled;
  openForm(enabled ? 'Turn Production Mode on' : 'Turn Production Mode off',`<p>${enabled ? 'The installation will use only live cameras. Validated tanks will monitor without an open browser and automatically update population, estimated mortality and feeding requirements.' : 'Background production monitoring will stop. Demonstration sources will become available again; inventory records will be retained.'}</p>`,async()=>{
    if (enabled) await Promise.all([...State.streams.values()].filter(isMonitoring).map(e=>stopTankMonitoring(e.tankId)));
    stopAllMonitoring(true);
    State.production=await api('/api/admin/production',{method:'PUT',body:{enabled}});
    toast(enabled ? 'Production Mode enabled.' : 'Production Mode disabled.');
    await refresh();
  },enabled ? 'Enable Production Mode' : 'Disable Production Mode');
}
async function censusForm(id) {
  const tank=State.report.tanks.find(t=>t.tank_id===id), config=State.monitoring.tanks.find(t=>t.tank_id===id);
  if (!config?.live_source) { toast('Choose a physical camera in Edit Tank first.',true); return; }
  openForm('Check camera accuracy',`<p>Independently verify that <strong>${escapeHTML(tank.name)}</strong> contains <strong>${fmt(tank.current_count)} fish</strong>. Correct saved population in tank settings first if needed.</p><p>Validation observes at least one minute of clear, fresh live frames. Keep the camera fixed and avoid moving fish during this check.</p><label class="field"><span><input type="checkbox" name="whole_view" required> The live view shows the entire tank population.</span></label>`,async()=>{
    if (!State.production.enabled) await stopTankMonitoring(id);
    await api('/api/tanks/'+encodeURIComponent(id)+'/validate-census',{method:'POST',body:{known_count:tank.current_count,whole_view:true}});
    toast('Camera accuracy check started. Open Edit Tank to see progress.'); await refresh();
  },'Start accuracy check');
}
async function reviewLosses() {
  const farm=State.farm, activeIds=new Set(monitorTanks().map(t=>t.tank_id)), events=(await api('/api/census-events?unresolved=true&limit=2000')).filter(e=>e.kind==='estimated_mortality' && activeIds.has(e.tank_id));
  if (farm!==State.farm) return;
  if (!events.length) { toast('No estimated camera losses remain to review.'); return; }
  openForm('Review estimated losses',`<p>Use this optional review only when you can confirm that an estimated loss represents deaths. Population has already been updated; confirmation does not deduct the same fish again.</p>`+select('census_event_id','Camera loss',events.map(e=>`<option value="${escapeHTML(e.id)}">${escapeHTML(e.tank_id)}: ${fmt(e.remaining)} fish - ${escapeHTML(e.timestamp)}</option>`).join(''))+field('count','Fish confirmed dead','','number','required min="1" step="1"')+field('note','Observation (optional)','','text','maxlength="500"'),async form=>{
    const data=Object.fromEntries(new FormData(form)), event=events.find(e=>e.id===data.census_event_id);
    if (!event || Number(data.count)>event.remaining) throw new Error('Confirm only fish included in the selected camera loss.');
    await saveOperation('/api/tanks/'+encodeURIComponent(event.tank_id)+'/mortality',{...data,operation_id:form.dataset.operation});
  },'Confirm classification');
  $('dialogForm').dataset.operation=crypto.randomUUID();
}

function stockForm(id, estimate = '') { const tank = State.report.tanks.find(t => t.tank_id === id); openForm('Stock fish', `${field('count', 'Newly added fish', estimate, 'number', 'required min="1" step="1"')}${field('note', 'Stocking note', '', 'text', 'maxlength="500"')}`, async form => saveOperation('/api/tanks/' + encodeURIComponent(id) + '/stock', { ...Object.fromEntries(new FormData(form)), operation_id: form.dataset.operation }), 'Confirm stocking'); $('dialogForm').dataset.operation = crypto.randomUUID(); }
function mortalityForm(id) { const tank = State.report.tanks.find(t => t.tank_id === id); openForm('Daily mortality check', `${field('count', 'New confirmed deaths', 0, 'number', `required min="0" max="${tank.current_count}" step="1"`)}${field('note', 'Observation or cause (optional)', '', 'text', 'maxlength="500"')}`, async form => saveOperation('/api/tanks/' + encodeURIComponent(id) + '/mortality', { ...Object.fromEntries(new FormData(form)), operation_id: form.dataset.operation }), 'Save today’s check'); $('dialogForm').dataset.operation = crypto.randomUUID(); }
function dispersalForm(selectedTank='') {
  if (!monitorTanks().length) { toast('Create and stock a tank first.',true); return; }
  const reference='DISP-'+new Date().toISOString().slice(0,10).replaceAll('-','')+'-'+crypto.randomUUID().slice(0,8).toUpperCase();
  openForm('Record dispersal',select('tank_id','Source tank',tankOptions(selectedTank))+`<div class="form-grid">${field('dispersal_id','Dispersal reference',reference,'text','required maxlength="120"','Filled automatically. Keep this reference for your records.')}${select('type','Activity','<option value="sale">Sale</option><option value="transfer">Transfer</option>')}${field('count','Fish to disperse','','number','required min="1" step="1"')}${field('recipient','Buyer or recipient','','text','required maxlength="120"')}</div><div class="form-grid" data-sale-fields>${field('unit_price_php','Unit sale price (PHP)',0,'number','required min="0" step="0.01"')}${select('price_unit','Price basis','<option value="per_fish">Per fish</option><option value="per_kg">Per kilogram</option>')}</div>`+field('batch_code','Batch reference (optional)','','text','maxlength="120"'),async form=>saveOperation('/api/dispersal/commit',Object.fromEntries(new FormData(form))),'Confirm dispersal');
  const form=$('dialogForm'),priceFields=form.querySelector('[data-sale-fields]');
  form.elements.type.onchange=()=>{ const sale=form.elements.type.value==='sale'; priceFields.hidden=!sale; priceFields.querySelectorAll('input,select').forEach(el=>el.disabled=!sale); };
}
function accountForm(id) {
  const user = State.accounts.find(u => u.id === id);
  if (user) openForm('Manage farmer access', `${select('active', 'Account access', `<option value="1" ${user.active ? 'selected' : ''}>Active</option><option value="0" ${!user.active ? 'selected' : ''}>Disabled</option>`)}${field('password', 'New temporary password (optional)', '', 'password', 'minlength="12" maxlength="256" autocomplete="new-password"', 'A reset signs the farmer out and requires them to change this password.')}`, async form => { const data = Object.fromEntries(new FormData(form)); data.active = data.active === '1'; await saveOperation('/api/admin/accounts/' + encodeURIComponent(id), data, 'PUT'); });
  else openForm('Create farmer account', field('display_name', 'Farmer name', '', 'text', 'required maxlength="120"') + field('farm_name', 'Farm name', '', 'text', 'required maxlength="120"') + field('username', 'Username', '', 'text', 'required minlength="3" maxlength="80" autocomplete="off"') + field('password', 'Initial password', '', 'password', 'required minlength="12" maxlength="256" autocomplete="new-password"', 'Give this password to the farmer. They must change it at first sign-in.'), async form => { await saveOperation('/api/admin/accounts', Object.fromEntries(new FormData(form))); State.farms = await api('/api/admin/farms'); updateFarmSelector(); }, 'Create private farm & account');
}
function passwordForm(required = false) { openForm('Set your password', field('current_password', 'Current password', '', 'password', 'required autocomplete="current-password"') + field('new_password', 'New password', '', 'password', 'required minlength="12" maxlength="256" autocomplete="new-password"', 'Use at least 12 characters.') + field('confirm_password', 'Confirm new password', '', 'password', 'required minlength="12" autocomplete="new-password"'), async form => { const data = Object.fromEntries(new FormData(form)); if (data.new_password !== data.confirm_password) throw new Error('New passwords do not match.'); await api('/api/auth/password', { method: 'POST', body: data }); location.href = '/login'; }, 'Change password & sign in', required); }
function benchmarkForm() { openForm('Evaluate counting', `<label class="field">Image with a verified count<input name="image_file" type="file" accept="image/*" required></label>${field('actual_count', 'Manually verified fish count', '', 'number', 'min="0" step="1"')}<label class="field">Optional box annotations<input name="annotation_file" type="file" accept=".txt"><small>Matching normalized box labels for this image. A known count or labels is required.</small></label>`, async form => { const data = new FormData(form); if (!form.elements.annotation_file.files.length && !form.elements.actual_count.value) throw new Error('Enter the verified count or provide annotations.'); if (!form.elements.annotation_file.files.length) data.delete('annotation_file'); if (!form.elements.actual_count.value) data.delete('actual_count'); await api('/api/evaluate-sample', { method: 'POST', body: data }); toast('Evaluation saved.'); await refresh(); }, 'Evaluate & save'); }
async function selectFarm(id) { stopSampleCounting(); stopAllMonitoring(true); closeInspector(false); $('formDialog').close(); State.generation++; State.farm = id; State.report = State.mortality = null; $('farmSelector').value = id; State.view = 'dashboard'; location.hash = 'dashboard'; await refresh(); }
async function handleAction(name, id) {
  switch (name) { case 'display-clean': return setDisplayPreferences(Object.fromEntries(Object.keys(displayDefaults).map(key=>[key,false]))); case 'display-reset': return setDisplayPreferences(displayDefaults); case 'tank-view': if(id==='grid')return closeInspector(); return openCounting(State.countTank); case 'review-losses': return reviewLosses(); case 'production-toggle': return productionForm(); case 'validate-census': return censusForm(id); case 'view': return navigate(id); case 'refresh': return refresh(); case 'farm': return selectFarm(id); case 'tank-new': return tankForm(); case 'tank-edit': return tankForm(id); case 'tank-delete': return deleteTankForm(id); case 'stock': return stockForm(id); case 'mortality': return mortalityForm(id); case 'monitor-toggle': return toggleTankMonitoring(id); case 'monitor-all': return toggleAllMonitoring(); case 'dispersal-new': return dispersalForm(id); case 'account-new': return accountForm(); case 'account-edit': return accountForm(id); case 'benchmark-new': return benchmarkForm(); case 'count': return openCounting(id); }
}
function stopSampleCounting() { State.sampleSequence++; State.sampleMode=false; State.samplePreview=null; }
function clearSamplePreview() { State.estimate=null; $('useCount').disabled=true; $('countValue').textContent=$('occupancyValue').textContent=$('crossingValue').textContent='—'; $('countPreview').hidden=true; $('countOverlay').hidden=true; $('countPlaceholder').hidden=false; }
function inspectorOpen() { return State.view==='tanks' && State.tankView==='inspect' && !$('countDialog').hidden; }
function closeInspector(update=true) { const selected=State.countTank; stopSampleCounting(); State.tankView='grid'; $('countDialog').hidden=true; $('monitoringDock').append($('countDialog')); if(update && State.view==='tanks'){render();document.querySelector(`[data-open-tank="${CSS.escape(selected || '')}"]`)?.focus({preventScroll:true});} }
function mountInspector() {
  const host=$('tankInspectorHost'); $('countDialog').hidden=!host;
  if(!host)return;
  const tank=monitorTanks().find(t=>t.tank_id===State.countTank); if(!tank)return;
  host.append($('countDialog')); $('countTitle').textContent=tank.name;
  $('inspectedTank').innerHTML=monitorTanks().map(t=>`<option value="${escapeHTML(t.tank_id)}" ${t.tank_id===State.countTank ? 'selected' : ''}>${escapeHTML(t.name)} · ${escapeHTML(t.tank_id)}</option>`).join('');
  $('inspectorActions').innerHTML=action('Edit Tank','tank-edit',tank.tank_id,'secondary','edit')+action('Record dispersal','dispersal-new',tank.tank_id,'secondary','dispersals');
  if(!State.sampleMode)clearSamplePreview();
}
function openCounting(id) {
  if(!id || !monitorTanks().some(t=>t.tank_id===id))return;
  stopSampleCounting(); State.countTank=id; State.estimate=null;
  State.tankView='inspect'; State.view='tanks'; location.hash='tanks';
  $('countPhoto').value=''; $('countVideo').value=''; render(); $('countTitle').tabIndex=-1; $('countTitle').focus({preventScroll:true});
}
function showCountingResult(data) {
  State.samplePreview=data; State.estimate=data.count; $('countValue').textContent=fmt(data.count);
  const tank=State.report.tanks.find(t=>t.tank_id===State.countTank);
  $('occupancyValue').textContent=fmt(data.count/Math.max(1,tank.max_capacity)*100,1)+'%';
  $('crossingValue').textContent='\u2014';
  const frame=updateCameraPreview($('countPreview'),$('countOverlay'),{status:'running',key:'photo:'+State.sampleSequence,data:{...data,frame_idx:State.sampleSequence}});
  $('countPlaceholder').hidden=!!frame; $('countPlaceholder').textContent=frame ? '' : 'Preview unavailable. Reload the app and count the photo again.';
  $('useCount').disabled=!(data.count>0);
}
async function countPhoto(file) {
  if (!file) return;
  stopSampleCounting(); const sequence=State.sampleSequence, id=State.countTank, farm=State.farm;
  State.sampleMode=true; clearSamplePreview(); $('countStatus').textContent='Counting your photo…';
  await stopTankMonitoring(id);
  if (sequence!==State.sampleSequence || farm!==State.farm) return;
  const data=new FormData(); data.append('file',file); data.append('tank_id',id);
  try {
    const result=await api('/api/upload/image',{method:'POST',body:data});
    if (sequence!==State.sampleSequence || farm!==State.farm || !inspectorOpen()) return;
    showCountingResult(result); $('countStatus').textContent='Photo estimate. Review it before saving.';
  } catch (error) { if (sequence===State.sampleSequence) $('countStatus').textContent=error.message; }
  updateMonitoringViews();
}
async function countVideo(file) {
  if (!file) return;
  stopSampleCounting(); const sequence=State.sampleSequence, id=State.countTank, farm=State.farm;
  State.sampleMode=true; clearSamplePreview(); $('countStatus').textContent='Uploading this tank video…';
  await stopTankMonitoring(id);
  if (sequence!==State.sampleSequence || farm!==State.farm) return;
  const upload=new FormData(); upload.append('file',file);
  try {
    const result=await api('/api/tanks/upload-video',{method:'POST',body:upload});
    if (sequence!==State.sampleSequence || farm!==State.farm) return;
    await api('/api/tanks/'+encodeURIComponent(id),{method:'PUT',body:{camera_source:result.filepath}});
    if (sequence!==State.sampleSequence || farm!==State.farm) return;
    State.report.tanks.find(t => t.tank_id===id).camera_source=result.filepath;
    State.sampleMode=false; startTankMonitoring(id); await refresh(true);
  } catch (error) { if (sequence===State.sampleSequence) $('countStatus').textContent=error.message; }
  updateMonitoringViews();
}

$('navigation').addEventListener('click', event => { const button = event.target.closest('[data-view]'); if (button) navigate(button.dataset.view); });
function tankFromEvent(event) {
  if(event.target.closest('button,a,input,select,label,summary,details'))return null;
  return event.target.closest('[data-open-tank]');
}
$('main').addEventListener('click',event=>{ const button=event.target.closest('[data-action]'); if(button)Promise.resolve(handleAction(button.dataset.action,button.dataset.id)).catch(error=>toast(error.message,true)); else {const tank=tankFromEvent(event); if(tank)openCounting(tank.dataset.openTank);} });
$('main').addEventListener('keydown',event=>{ if(['Enter',' '].includes(event.key) && event.target.matches('[data-open-tank]')) {event.preventDefault();openCounting(event.target.dataset.openTank);} });
$('dialogBody').addEventListener('click',event=>{const button=event.target.closest('[data-action]');if(button){event.preventDefault();Promise.resolve(handleAction(button.dataset.action,button.dataset.id)).catch(error=>toast(error.message,true));}});
$('main').addEventListener('input',event=>{if(event.target.matches('[data-display-choice]'))setDisplayPreferences({[event.target.dataset.displayChoice]:event.target.checked});});
$('main').addEventListener('change', event => { const key = { filterTank: 'tank_id', filterDays: 'days', filterSeverity: 'severity', filterMetric: 'metric' }[event.target.id]; if (key) { State.filters[key] = event.target.value; if (key === 'metric') render(); else refresh(); } });
$('dialogForm').addEventListener('submit', async event => { event.preventDefault(); const button = $('saveDialog'); button.disabled = true; $('formError').textContent = ''; try { await submitForm(event.target); $('formDialog').close(); } catch (error) { $('formError').textContent = error.message; } finally { button.disabled = false; } });
$('formDialog').addEventListener('cancel', event => { if ($('formDialog').dataset.required === 'true') event.preventDefault(); });
$('closeDialog').onclick = $('cancelDialog').onclick = () => $('formDialog').close();
$('closeCount').onclick = () => closeInspector(); $('inspectedTank').onchange=event=>openCounting(event.target.value);
$('countPhoto').onchange = event => countPhoto(event.target.files[0]); $('countVideo').onchange = event => countVideo(event.target.files[0]); $('toggleCamera').onclick = () => { if (State.sampleMode) { stopSampleCounting(); startTankMonitoring(State.countTank); } else toggleTankMonitoring(State.countTank); updateMonitoringViews(); };
$('useCount').onclick = () => { if (!State.sampleMode && !freshMonitoring(tankStream(State.countTank))) { updateMonitoringViews(); toast('Wait for a current count before reviewing stocking.',true); return; } const id = State.countTank, estimate = State.estimate; closeInspector(); stockForm(id, estimate); };
$('farmSelector').onchange = event => selectFarm(event.target.value);
$('passwordButton').onclick = () => passwordForm(); $('logoutButton').onclick = async () => { try { stopSampleCounting(); stopAllMonitoring(true); await api('/api/auth/logout', { method: 'POST' }); location.href = '/login'; } catch (error) { toast(error.message, true); } };
$('menuButton').onclick = () => { document.body.classList.toggle('menu-open'); $('menuButton').setAttribute('aria-expanded', String(document.body.classList.contains('menu-open'))); };
$('monitoringStatus').onclick=() => navigate('tanks');
window.addEventListener('pagehide',() => stopAllMonitoring(true));
document.addEventListener('pointerdown',event=>{const menu=document.querySelector('.display-options');if(menu?.open && !menu.contains(event.target))menu.open=false;});
document.addEventListener('keydown',event=>{if(event.key==='Escape'){const menu=document.querySelector('.display-options');if(menu?.open){menu.open=false;menu.querySelector('summary').focus();}}});
start();
