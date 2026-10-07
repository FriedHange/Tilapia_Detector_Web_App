/* Simple help on hover, focus, or a touch-and-hold; actions keep normal click behavior. */
'use strict';
(() => {
  const help={
    'production-toggle':'Switch the whole installation between demonstrations and automatic live-camera monitoring.',
    'validate-census':'Check a whole-tank camera against a known fish population before allowing automatic updates.',
    'review-losses':'Optionally confirm camera-estimated losses as deaths. Population will not be deducted again.',
    'monitor-toggle':'Start or pause this tank camera. In Production Mode this controls background monitoring.',
    'monitor-all':'Start or pause monitoring for all eligible tanks in the selected farm.',
    'count':'Open the live view for this tank. Other cameras keep monitoring.',
    'tank-view':'Return to all tank cameras. Monitoring continues.',
    'display-clean':'Hide every camera overlay in your view. Counting continues unchanged.',
    'display-reset':'Restore boxes, fish labels and motion trails in your view.',
    'tank-edit':'Change the camera, capacity, feeding inputs, or correct the saved population.',
    'tank-delete':'Remove this tank from active monitoring and inventory while keeping its history.',
    'tank-new':'Create a tank with its known starting population and feeding settings.',
    'dispersal-new':'Record fish sold or transferred. The system matches camera changes to avoid deducting fish twice.',
    'account-new':'Create a farmer account with a private farm.',
    'account-edit':'Enable access or reset a farmer account password.',
    'benchmark-new':'Compare counting results with a manually verified image.',
    'farm':'Open this farm and its records.', 'refresh':'Load the latest records and monitoring status.',
    'view':'Open this part of the farm workspace.'
  };
  const ids={loginButton:'Sign in to your farm account.',logoutButton:'Sign out of this workspace. Production cameras keep monitoring.',passwordButton:'Change your sign-in password.',
    menuButton:'Show or hide the navigation menu.',monitoringStatus:'Open live cameras in Tank Management.',
    toggleCamera:'Start or pause monitoring for this tank.',closeCount:'Return to the camera grid. All running cameras keep monitoring.',
    closeDialog:'Close this form without saving.',cancelDialog:'Cancel these changes.',saveDialog:'Save the information in this form.'};
  const tip=document.createElement('div'); tip.id='action-help'; tip.className='action-tooltip'; tip.role='tooltip'; tip.hidden=true;
  document.body.append(tip);
  const selector='button,a.button,label.button'; let active=null,timer=null,held=false;
  function explain(element) { return element.dataset.tooltip || help[element.dataset.action] || ids[element.id] ||
    (element.dataset.view ? 'Open '+element.textContent.trim()+'.' : element.matches('a.button') ? 'Download the selected records.' :
    element.querySelector('input[type="file"]') ? 'Choose a file for a demonstration count. This does not update population.' :
    element.type==='submit' ? 'Submit the information in this form.' : element.getAttribute('aria-label') || element.textContent.trim()+' action.'); }
  function prepare() { if (active && !active.isConnected) hide(); document.querySelectorAll(selector).forEach(el=>{ if (!el.dataset.tooltip) el.dataset.tooltip=explain(el); }); }
  function hide() { if (active) { const ids=(active.getAttribute('aria-describedby') || '').split(' ').filter(v=>v && v!==tip.id); if (ids.length) active.setAttribute('aria-describedby',ids.join(' ')); else active.removeAttribute('aria-describedby'); } active=null; tip.hidden=true; }
  function show(el) { if (!el || !el.isConnected) return; hide(); active=el; tip.textContent=explain(el); (el.closest('dialog[open]') || document.body).append(tip); tip.hidden=false;
    el.setAttribute('aria-describedby',[el.getAttribute('aria-describedby'),tip.id].filter(Boolean).join(' '));
    const r=el.getBoundingClientRect(), bounds=tip.getBoundingClientRect();
    tip.style.left=Math.max(8,Math.min(window.innerWidth-bounds.width-8,r.left))+'px';
    tip.style.top=(r.bottom+bounds.height+12<window.innerHeight ? r.bottom+8 : Math.max(8,r.top-bounds.height-8))+'px'; }
  prepare(); new MutationObserver(prepare).observe(document.body,{childList:true,subtree:true});
  document.addEventListener('pointerover',e=>{ if (e.pointerType==='mouse') { const el=e.target.closest(selector); if (el && !el.contains(e.relatedTarget)) show(el); } });
  document.addEventListener('pointerout',e=>{ if (active && active.contains(e.target) && !active.contains(e.relatedTarget) && document.activeElement!==active) hide(); });
  document.addEventListener('focusin',e=>{ const el=e.target.closest(selector); if (el) show(el); else hide(); });
  document.addEventListener('focusout',hide);
  document.addEventListener('pointerdown',e=>{ held=false; if (e.pointerType==='touch') { const el=e.target.closest(selector); if (el) timer=setTimeout(()=>{held=true;show(el);},500); } });
  document.addEventListener('pointerup',()=>clearTimeout(timer)); document.addEventListener('pointercancel',()=>clearTimeout(timer));
  document.addEventListener('click',e=>{ if (held) { e.preventDefault();e.stopImmediatePropagation();held=false; } else hide(); },true);
  document.addEventListener('keydown',e=>{ if (e.key==='Escape') hide(); });
  const moved=()=>{ if (active && document.activeElement===active) show(active); else hide(); };
  window.addEventListener('resize',moved); document.addEventListener('scroll',moved,true);
})();
