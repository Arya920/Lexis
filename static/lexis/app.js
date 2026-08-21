/* ============================================================
   LEXIS — KNOWLEDGE ASSISTANT  |  app.js
   React 18 · Babel standalone · No build step required
   Flask endpoints: /chat /upload /files /remove-file
                    /upload-dataset /datasets /remove-dataset
                    /agent/visualize /agent/analyze
   ============================================================ */

const { useState, useEffect, useRef, useCallback } = React;

/* ── Utilities ─────────────────────────────────────────────── */
const ts  = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
const uid = () => (Date.now() + Math.random()).toString(36);

function extOf(n) { const m = n.match(/\.([^.]+)$/); return m ? m[1].toLowerCase() : ''; }
function iconFor(e) {
  if (e === 'pdf')                      return ['fa-file-pdf',   'pdf'];
  if (['doc','docx'].includes(e))       return ['fa-file-word',  'doc'];
  if (['txt','md'].includes(e))         return ['fa-file-lines', 'txt'];
  if (['csv','xlsx','xls'].includes(e)) return ['fa-file-excel', 'csv'];
  return ['fa-file', 'other'];
}
function basename(n) { const p = n.split('.'); if (p.length > 1) p.pop(); return p.join('.'); }
function shortPreview(msgs) {
  const f = msgs.find(m => m.sender === 'user');
  if (!f) return 'New conversation';
  const clean = f.text.replace(/\/[\w-]+/g, '').trim();
  return (clean || f.text).slice(0, 38) + ((clean || f.text).length > 38 ? '…' : '');
}

function fmtVal(v) {
  if (v === null || v === undefined) return '—';
  const n = parseFloat(v);
  if (isNaN(n)) return String(v);
  if (Math.abs(n) >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M';
  if (Math.abs(n) >= 1_000)     return (n / 1_000).toFixed(1) + 'K';
  if (n % 1 === 0)              return n.toLocaleString();
  return n.toFixed(2);
}

/* ── Agent definitions ─────────────────────────────────────── */
const AGENTS = [
  { name: 'web-search-agent',          icon: 'fa-globe',        desc: 'Search & retrieve live web results',    needsDataset: false },
  { name: 'data-analysis-agent',       icon: 'fa-chart-bar',    desc: 'Analyse datasets with deep insights',   needsDataset: true  },
  { name: 'create-visualization-agent',icon: 'fa-chart-pie',    desc: 'Generate interactive charts & graphs',  needsDataset: true  },
  { name: 'summarization-agent',       icon: 'fa-compress-alt', desc: 'Summarise long documents quickly',      needsDataset: false },
  { name: 'code-agent',                icon: 'fa-code',         desc: 'Write, debug and explain code',         needsDataset: false },
];

const SUGGESTIONS = [
  'Summarise the key findings',
  'What does the policy say about…',
  'List all action items',
  'Compare the two documents',
];

const MODELS = [
  { id: 'llama-3.1-8b-instant',    label: 'Llama 3.1 · 8B',    tag: 'fast'    },
  { id: 'llama-3.3-70b-versatile', label: 'Llama 3.3 · 70B',   tag: 'smart'   },
];

const LS = {
  get: (k, fb) => { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : fb; } catch { return fb; } },
  set: (k, v)  => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

/* ── BubbleContent ──────────────────────────────────────────── */
function BubbleContent({ text }) {
  const parts = [];
  const re = /(\/[\w-]+)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    const isAgent = AGENTS.some(a => '/' + a.name === m[0]);
    if (!isAgent) continue;
    if (m.index > last) parts.push({ type: 'text', val: text.slice(last, m.index) });
    parts.push({ type: 'agent', val: m[0] });
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push({ type: 'text', val: text.slice(last) });
  if (!parts.length) parts.push({ type: 'text', val: text });
  return (
    <span>
      {parts.map((t, i) =>
        t.type === 'agent'
          ? <span key={i} className="agent-token">{t.val}</span>
          : <span key={i} style={{ whiteSpace: 'pre-wrap' }}>{t.val}</span>
      )}
    </span>
  );
}

/* ── Agent autocomplete dropdown ────────────────────────────── */
function AgentDropdown({ filter, onSelect, focusIdx }) {
  const q    = filter.replace('/', '').toLowerCase();
  const list = AGENTS.filter(a => a.name.includes(q));
  if (!list.length) return null;
  return (
    <div className="agent-dropdown">
      <div className="agent-dropdown-hdr">Agents — ↑↓ navigate · ↵ or Tab select</div>
      {list.map((a, i) => (
        <div key={a.name} className={`agent-opt ${i === focusIdx ? 'focused' : ''}`}
             onMouseDown={e => { e.preventDefault(); onSelect(a); }}>
          <div className="agent-opt-icon"><i className={`fas ${a.icon}`} /></div>
          <div>
            <div className="agent-opt-name">/{a.name}</div>
            <div className="agent-opt-desc">{a.desc}</div>
          </div>
          {a.needsDataset && <span className="agent-opt-badge">needs dataset</span>}
        </div>
      ))}
    </div>
  );
}

/* ── Plotly chart bubble ─────────────────────────────────────── */
function ChartBubble({ figure, summary, filename, rows, columns }) {
  const containerRef = useRef(null);
  const [plotError, setPlotError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!containerRef.current || !figure) return;
    if (typeof Plotly === 'undefined') {
      setPlotError('Plotly.js not loaded. Add the Plotly CDN script to index.html.');
      setIsLoading(false);
      return;
    }
    try {
      Plotly.newPlot(
        containerRef.current,
        figure.data,
        figure.layout || {},
        figure.config || { responsive: true, displayModeBar: true }
      ).then(() => setIsLoading(false));
    } catch (err) {
      setPlotError(`Chart render error: ${err.message}`);
      setIsLoading(false);
    }
    return () => {
      if (containerRef.current && typeof Plotly !== 'undefined') {
        try { Plotly.purge(containerRef.current); } catch (_) {}
      }
    };
  }, [figure]);

  return (
    <div className="chart-bubble">
      <div className="chart-meta">
        <span className="chart-meta-icon"><i className="fas fa-table" /></span>
        <span className="chart-meta-filename">{filename}</span>
        <span className="chart-meta-stat">{rows?.toLocaleString()} rows</span>
        <span className="chart-meta-stat">{columns?.length} cols</span>
      </div>
      <div className="chart-container">
        {isLoading && !plotError && (
          <div className="chart-loading"><i className="fas fa-spinner fa-spin" /><span>Rendering chart…</span></div>
        )}
        {plotError && (
          <div className="chart-error"><i className="fas fa-triangle-exclamation" /><span>{plotError}</span></div>
        )}
        <div ref={containerRef} style={{ width: '100%', minHeight: 360, display: plotError ? 'none' : 'block' }} />
      </div>
      {summary && (
        <div className="chart-summary"><i className="fas fa-circle-info" />{summary}</div>
      )}
    </div>
  );
}

/* ── AnalysisBubble ──────────────────────────────────────────── */
function AnalysisBubble({ headline, narrative, keyFindings, recommendation, statsTable, primaryTable, operations, filename, rows }) {
  const [activeTab, setActiveTab] = useState('insights');
  const hasTable = primaryTable && primaryTable.rows && primaryTable.rows.length > 0;
  const hasStats = statsTable && statsTable.length > 0;

  return (
    <div className="analysis-bubble">
      <div className="ab-header">
        <div className="ab-header-left">
          <div className="ab-agent-badge">
            <i className="fas fa-chart-bar" />
            data-analysis-agent
          </div>
          <div className="ab-meta">
            <span><i className="fas fa-table" /> {filename}</span>
            <span><i className="fas fa-database" /> {rows?.toLocaleString()} rows</span>
          </div>
        </div>
        <div className="ab-ops-pills">
          {(operations || []).map(op => (
            <span key={op.op_id} className={`ab-op-pill ${op.status === 'error' ? 'error' : ''}`} title={op.label}>
              {op.status === 'error' ? <i className="fas fa-xmark" /> : <i className="fas fa-check" />}
              {op.op_id}
            </span>
          ))}
        </div>
      </div>
      <div className="ab-headline">{headline}</div>
      {hasStats && (
        <div className="ab-stats-row">
          {statsTable.slice(0, 6).map((s, i) => (
            <div key={i} className="ab-stat-card">
              <div className="ab-stat-label">{s.label}</div>
              <div className="ab-stat-value">{s.value}</div>
              {s.note && <div className="ab-stat-note">{s.note}</div>}
            </div>
          ))}
        </div>
      )}
      <div className="ab-tabs">
        <button className={`ab-tab ${activeTab === 'insights' ? 'active' : ''}`} onClick={() => setActiveTab('insights')}>
          <i className="fas fa-lightbulb" /> Insights
        </button>
        {hasTable && (
          <button className={`ab-tab ${activeTab === 'table' ? 'active' : ''}`} onClick={() => setActiveTab('table')}>
            <i className="fas fa-table-cells" /> {primaryTable.label}
          </button>
        )}
      </div>
      {activeTab === 'insights' && (
        <div className="ab-tab-content">
          <div className="ab-narrative">{narrative}</div>
          {keyFindings && keyFindings.length > 0 && (
            <div className="ab-findings">
              <div className="ab-findings-label"><i className="fas fa-star" /> Key findings</div>
              <ul className="ab-findings-list">
                {keyFindings.map((f, i) => (
                  <li key={i}><i className="fas fa-arrow-right" /><span>{f}</span></li>
                ))}
              </ul>
            </div>
          )}
          {recommendation && (
            <div className="ab-recommendation">
              <i className="fas fa-circle-arrow-right" />
              <div>
                <div className="ab-recommendation-label">Recommendation</div>
                <div className="ab-recommendation-text">{recommendation}</div>
              </div>
            </div>
          )}
        </div>
      )}
      {activeTab === 'table' && hasTable && (
        <div className="ab-tab-content ab-table-wrap">
          <table className="ab-table">
            <thead>
              <tr>{primaryTable.columns.map(col => <th key={col}>{col}</th>)}</tr>
            </thead>
            <tbody>
              {primaryTable.rows.map((row, i) => (
                <tr key={i}>
                  {primaryTable.columns.map(col => (
                    <td key={col}>{
                      row[col] === null || row[col] === undefined ? '—'
                        : typeof row[col] === 'number' ? fmtVal(row[col])
                        : String(row[col])
                    }</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ── Dataset Upload Modal ───────────────────────────────────── */
function DatasetUploadModal({ agent, datasets, activeDataset, onClose, onUploaded, onSelectExisting, onRemove }) {
  const [uploading, setUploading] = useState(false);
  const [dragOver,  setDragOver]  = useState(false);
  const fileRef = useRef(null);

  const handleFile = async (file) => {
    if (!file) return;
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['csv', 'xlsx', 'xls'].includes(ext)) {
      alert('Only CSV and Excel files (.csv, .xlsx, .xls) are supported.');
      return;
    }
    setUploading(true);
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res  = await fetch('/lexis/upload-dataset', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Upload failed');
      onUploaded(data.filename);
    } catch (err) { alert(`Upload failed: ${err.message}`); }
    finally { setUploading(false); }
  };

  const onDrop = e => {
    e.preventDefault(); setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  return (
    <div className="backdrop" onClick={onClose}>
      <div className="modal modal-dataset" onClick={e => e.stopPropagation()}>
        <div className="modal-dataset-header">
          <div className="modal-icon agent-icon">
            <i className={`fas ${agent?.icon || 'fa-chart-pie'}`} />
          </div>
          <div>
            <div className="modal-title">{agent ? `/${agent.name}` : 'Agent files'}</div>
            <div className="modal-subtitle">{agent?.desc || 'Upload, select, or delete CSV and Excel files for agents.'}</div>
          </div>
          <button className="modal-close-btn" onClick={onClose}><i className="fas fa-xmark" /></button>
        </div>
        {datasets.length > 0 && (
          <div className="modal-section">
            <div className="modal-section-label">Manage agent files</div>
            <div className="existing-datasets">
              {datasets.map(f => (
                <div key={f} className={`existing-dataset-row ${activeDataset === f ? 'active' : ''}`}>
                  <button className="existing-dataset-btn" onClick={() => onSelectExisting(f)}>
                    <i className="fas fa-file-excel" />
                    <span>{f}</span>
                    {activeDataset === f && <i className="fas fa-circle-check existing-selected" />}
                  </button>
                  <button className="existing-dataset-remove" onClick={() => onRemove(f)} title="Delete file">
                    <i className="fas fa-trash-can" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
        {datasets.length > 0 && <div className="modal-divider"><span>or upload a new one</span></div>}
        <div
          className={`drop-zone ${dragOver ? 'drag-over' : ''} ${uploading ? 'uploading' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => !uploading && fileRef.current?.click()}
        >
          <input ref={fileRef} type="file" accept=".csv,.xlsx,.xls" style={{ display: 'none' }}
                 onChange={e => handleFile(e.target.files[0])} />
          {uploading ? (
            <><i className="fas fa-spinner fa-spin drop-zone-icon" /><div className="drop-zone-text">Uploading…</div></>
          ) : (
            <>
              <i className="fas fa-cloud-arrow-up drop-zone-icon" />
              <div className="drop-zone-text">Drop CSV or Excel file here</div>
              <div className="drop-zone-sub">or click to browse · .csv .xlsx .xls</div>
            </>
          )}
        </div>
        <div className="modal-note">
          <i className="fas fa-circle-info" />
          This dataset will <strong>not</strong> be indexed into the RAG knowledge base.
          It is only available to agents.
        </div>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   PAGE COMPONENTS
   ════════════════════════════════════════════════════════════ */

/* ── Settings Page ──────────────────────────────────────────── */
function SettingsPage({ light, setLight, selectedModel, switchModel, modelSwitching }) {
  const [notifEmail,  setNotifEmail]  = useState(true);
  const [notifInApp,  setNotifInApp]  = useState(true);
  const [autoSave,    setAutoSave]    = useState(true);
  const [compactView, setCompactView] = useState(false);
  const [lang,        setLang]        = useState('en');
  const [saved,       setSaved]       = useState(false);

  const save = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const ToggleRow = ({ label, desc, value, onChange }) => (
    <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', padding:'13px 0', borderBottom:'1px solid var(--border-subtle)' }}>
      <div>
        <div style={{ fontSize:13.5, color:'var(--text-primary)', fontWeight:500 }}>{label}</div>
        {desc && <div style={{ fontSize:12, color:'var(--text-tertiary)', marginTop:2 }}>{desc}</div>}
      </div>
      <button
        onClick={() => onChange(!value)}
        style={{
          width:38, height:22, borderRadius:11, border:'1px solid',
          borderColor: value ? 'var(--accent-bdr)' : 'var(--border-default)',
          background: value ? 'var(--accent-dim)' : 'var(--bg-hover)',
          position:'relative', flexShrink:0, transition:'all 180ms ease', cursor:'pointer'
        }}
      >
        <span style={{
          position:'absolute', width:14, height:14, borderRadius:'50%',
          background: value ? 'var(--accent)' : 'var(--text-tertiary)',
          top:3, left: value ? 20 : 3, transition:'all 180ms ease'
        }} />
      </button>
    </div>
  );

  return (
    <div style={{ flex:1, overflowY:'auto', padding:'32px 24px 48px' }}>
      <div style={{ maxWidth:680, margin:'0 auto' }}>

        {/* Header */}
        <div style={{ marginBottom:32 }}>
          <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:6 }}>
            <div style={{ width:36, height:36, borderRadius:'var(--r-sm)', background:'var(--accent-dim)', border:'1px solid var(--accent-bdr)', display:'flex', alignItems:'center', justifyContent:'center', color:'var(--accent)', fontSize:14 }}>
              <i className="fas fa-gear" />
            </div>
            <h1 style={{ fontSize:22, fontWeight:600, letterSpacing:'-0.03em', color:'var(--text-primary)' }}>Settings</h1>
          </div>
          <p style={{ fontSize:13, color:'var(--text-tertiary)', marginLeft:46 }}>Manage your workspace preferences and configuration.</p>
        </div>

        {/* Appearance */}
        <div style={{ background:'var(--bg-surface)', border:'1px solid var(--border-subtle)', borderRadius:'var(--r-lg)', padding:'20px 22px', marginBottom:16 }}>
          <div style={{ display:'flex', alignItems:'center', gap:7, marginBottom:18 }}>
            <i className="fas fa-palette" style={{ color:'var(--accent)', fontSize:11 }} />
            <span style={{ fontFamily:'var(--font-mono)', fontSize:9.5, letterSpacing:'0.1em', textTransform:'uppercase', color:'var(--text-tertiary)' }}>Appearance</span>
          </div>
          <ToggleRow label="Dark mode" desc="Use dark theme across the interface" value={!light} onChange={v => setLight(!v)} />
          <ToggleRow label="Compact view" desc="Reduce spacing between messages" value={compactView} onChange={setCompactView} />
        </div>

        {/* Model */}
        <div style={{ background:'var(--bg-surface)', border:'1px solid var(--border-subtle)', borderRadius:'var(--r-lg)', padding:'20px 22px', marginBottom:16 }}>
          <div style={{ display:'flex', alignItems:'center', gap:7, marginBottom:18 }}>
            <i className="fas fa-microchip" style={{ color:'var(--accent)', fontSize:11 }} />
            <span style={{ fontFamily:'var(--font-mono)', fontSize:9.5, letterSpacing:'0.1em', textTransform:'uppercase', color:'var(--text-tertiary)' }}>Language Model</span>
          </div>
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:10 }}>
            {MODELS.map(m => (
              <button key={m.id} onClick={() => switchModel(m.id)} disabled={modelSwitching}
                style={{
                  padding:'12px 14px', borderRadius:'var(--r-md)', textAlign:'left', cursor:'pointer',
                  border:'1px solid', transition:'all 150ms ease',
                  borderColor: selectedModel === m.id ? 'var(--accent-bdr)' : 'var(--border-default)',
                  background:  selectedModel === m.id ? 'var(--accent-dim)' : 'var(--bg-hover)',
                  opacity: modelSwitching ? 0.6 : 1
                }}>
                <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:4 }}>
                  <span style={{ fontSize:13, fontWeight:600, color:'var(--text-primary)' }}>{m.label}</span>
                  {selectedModel === m.id && <i className="fas fa-circle-check" style={{ color:'var(--accent)', fontSize:11 }} />}
                </div>
                <span style={{ fontFamily:'var(--font-mono)', fontSize:9, color:'var(--text-tertiary)', padding:'1px 6px', borderRadius:3, background:'var(--bg-active)', border:'1px solid var(--border-subtle)' }}>{m.tag}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Language */}
        <div style={{ background:'var(--bg-surface)', border:'1px solid var(--border-subtle)', borderRadius:'var(--r-lg)', padding:'20px 22px', marginBottom:16 }}>
          <div style={{ display:'flex', alignItems:'center', gap:7, marginBottom:18 }}>
            <i className="fas fa-language" style={{ color:'var(--accent)', fontSize:11 }} />
            <span style={{ fontFamily:'var(--font-mono)', fontSize:9.5, letterSpacing:'0.1em', textTransform:'uppercase', color:'var(--text-tertiary)' }}>Language & Region</span>
          </div>
          <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between' }}>
            <div>
              <div style={{ fontSize:13.5, color:'var(--text-primary)', fontWeight:500 }}>Interface language</div>
              <div style={{ fontSize:12, color:'var(--text-tertiary)', marginTop:2 }}>Language used in menus and labels</div>
            </div>
            <select value={lang} onChange={e => setLang(e.target.value)}
              style={{ background:'var(--bg-hover)', border:'1px solid var(--border-default)', borderRadius:'var(--r-sm)', color:'var(--text-primary)', fontFamily:'var(--font-mono)', fontSize:12, padding:'6px 10px', outline:'none', cursor:'pointer' }}>
              <option value="en">English</option>
              <option value="fr">Français</option>
              <option value="de">Deutsch</option>
              <option value="es">Español</option>
              <option value="ja">日本語</option>
            </select>
          </div>
        </div>

        {/* Notifications */}
        <div style={{ background:'var(--bg-surface)', border:'1px solid var(--border-subtle)', borderRadius:'var(--r-lg)', padding:'20px 22px', marginBottom:16 }}>
          <div style={{ display:'flex', alignItems:'center', gap:7, marginBottom:18 }}>
            <i className="fas fa-bell" style={{ color:'var(--accent)', fontSize:11 }} />
            <span style={{ fontFamily:'var(--font-mono)', fontSize:9.5, letterSpacing:'0.1em', textTransform:'uppercase', color:'var(--text-tertiary)' }}>Notifications</span>
          </div>
          <ToggleRow label="Email notifications" desc="Get summaries and alerts via email" value={notifEmail} onChange={setNotifEmail} />
          <ToggleRow label="In-app notifications" desc="Show alerts inside Lexis" value={notifInApp} onChange={setNotifInApp} />
          <ToggleRow label="Auto-save conversations" desc="Automatically persist chat history" value={autoSave} onChange={setAutoSave} />
        </div>

        {/* Danger zone */}
        <div style={{ background:'var(--red-dim)', border:'1px solid var(--red-bdr)', borderRadius:'var(--r-lg)', padding:'20px 22px', marginBottom:24 }}>
          <div style={{ display:'flex', alignItems:'center', gap:7, marginBottom:14 }}>
            <i className="fas fa-triangle-exclamation" style={{ color:'var(--red)', fontSize:11 }} />
            <span style={{ fontFamily:'var(--font-mono)', fontSize:9.5, letterSpacing:'0.1em', textTransform:'uppercase', color:'var(--red)' }}>Danger Zone</span>
          </div>
          <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between' }}>
            <div>
              <div style={{ fontSize:13.5, color:'var(--text-primary)', fontWeight:500 }}>Clear all data</div>
              <div style={{ fontSize:12, color:'var(--text-tertiary)', marginTop:2 }}>Permanently delete conversations and uploaded files</div>
            </div>
            <button style={{ padding:'7px 14px', borderRadius:'var(--r-sm)', border:'1px solid var(--red-bdr)', background:'transparent', color:'var(--red)', fontSize:12.5, fontWeight:500, cursor:'pointer', transition:'all 150ms ease' }}
              onMouseOver={e => e.currentTarget.style.background='rgba(248,113,113,0.15)'}
              onMouseOut={e => e.currentTarget.style.background='transparent'}>
              Clear data
            </button>
          </div>
        </div>

        {/* Save */}
        <div style={{ display:'flex', justifyContent:'flex-end' }}>
          <button onClick={save}
            style={{ display:'flex', alignItems:'center', gap:8, padding:'9px 20px', borderRadius:'var(--r-sm)', border:'1px solid var(--accent-bdr)', background: saved ? 'var(--green-dim)' : 'var(--accent-dim)', color: saved ? 'var(--green)' : 'var(--accent)', fontSize:13, fontWeight:500, cursor:'pointer', transition:'all 200ms ease' }}>
            <i className={`fas ${saved ? 'fa-check' : 'fa-floppy-disk'}`} />
            {saved ? 'Saved!' : 'Save changes'}
          </button>
        </div>

      </div>
    </div>
  );
}

/* ── Integrations Page ──────────────────────────────────────── */
function IntegrationsPage() {
  const [connected, setConnected] = useState({});
  const [connecting, setConnecting] = useState({});

  const integrations = [
    { id:'slack',    name:'Slack',         icon:'fa-slack',         fab:true,  desc:'Send Lexis answers directly to Slack channels or DMs.',          color:'#4A154B', badge:'Popular' },
    { id:'gdrive',   name:'Google Drive',  icon:'fa-google-drive',  fab:true,  desc:'Index documents straight from your Drive folders.',              color:'#1FA463', badge:null },
    { id:'notion',   name:'Notion',        icon:'fa-n',             fab:false, desc:'Pull pages and databases into the Lexis knowledge base.',        color:'#000',    badge:'New' },
    { id:'teams',    name:'MS Teams',      icon:'fa-microsoft',     fab:true,  desc:'Bring Lexis AI answers into Teams conversations.',               color:'#6264A7', badge:null },
    { id:'github',   name:'GitHub',        icon:'fa-github',        fab:true,  desc:'Answer questions about your repos, PRs, and issues.',            color:'#333',    badge:null },
    { id:'jira',     name:'Jira',          icon:'fa-jira',          fab:true,  desc:'Link Jira tickets and sprint data to your knowledge base.',      color:'#0052CC', badge:null },
    { id:'dropbox',  name:'Dropbox',       icon:'fa-dropbox',       fab:true,  desc:'Sync files from Dropbox for instant RAG indexing.',             color:'#0061FF', badge:null },
    { id:'salesforce',name:'Salesforce',   icon:'fa-salesforce',    fab:true,  desc:'Query CRM data and customer records conversationally.',          color:'#00A1E0', badge:'Beta' },
    { id:'confluence',name:'Confluence',   icon:'fa-confluence',    fab:true,  desc:'Index Confluence spaces for team knowledge retrieval.',          color:'#172B4D', badge:null },
  ];

  const handleConnect = async (id) => {
    setConnecting(p => ({ ...p, [id]: true }));
    await new Promise(r => setTimeout(r, 1200));
    setConnecting(p => ({ ...p, [id]: false }));
    setConnected(p => ({ ...p, [id]: !p[id] }));
  };

  return (
    <div style={{ flex:1, overflowY:'auto', padding:'32px 24px 48px' }}>
      <div style={{ maxWidth:860, margin:'0 auto' }}>

        {/* Header */}
        <div style={{ marginBottom:32 }}>
          <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:6 }}>
            <div style={{ width:36, height:36, borderRadius:'var(--r-sm)', background:'var(--purple-dim)', border:'1px solid var(--purple-bdr)', display:'flex', alignItems:'center', justifyContent:'center', color:'var(--purple)', fontSize:14 }}>
              <i className="fas fa-plug" />
            </div>
            <h1 style={{ fontSize:22, fontWeight:600, letterSpacing:'-0.03em', color:'var(--text-primary)' }}>Integrations</h1>
          </div>
          <p style={{ fontSize:13, color:'var(--text-tertiary)', marginLeft:46 }}>
            Connect your favourite tools to supercharge Lexis with live data from your existing workflows.
          </p>
        </div>

        {/* Stats row */}
        <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:12, marginBottom:28 }}>
          {[
            { label:'Connected', value: Object.values(connected).filter(Boolean).length, icon:'fa-circle-check', color:'var(--green)' },
            { label:'Available', value: integrations.length, icon:'fa-plug', color:'var(--accent)' },
            { label:'Coming soon', value: '12+', icon:'fa-clock', color:'var(--text-tertiary)' },
          ].map(s => (
            <div key={s.label} style={{ background:'var(--bg-surface)', border:'1px solid var(--border-subtle)', borderRadius:'var(--r-md)', padding:'14px 18px', display:'flex', alignItems:'center', gap:12 }}>
              <i className={`fas ${s.icon}`} style={{ fontSize:16, color:s.color }} />
              <div>
                <div style={{ fontSize:20, fontWeight:700, color:'var(--text-primary)', letterSpacing:'-0.03em' }}>{s.value}</div>
                <div style={{ fontSize:11, color:'var(--text-tertiary)', fontFamily:'var(--font-mono)' }}>{s.label}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Integration cards grid */}
        <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(260px, 1fr))', gap:12 }}>
          {integrations.map(intg => {
            const isConnected  = !!connected[intg.id];
            const isConnecting = !!connecting[intg.id];
            return (
              <div key={intg.id}
                style={{ background:'var(--bg-surface)', border:`1px solid ${isConnected ? 'var(--green-bdr)' : 'var(--border-subtle)'}`, borderRadius:'var(--r-lg)', padding:'18px', display:'flex', flexDirection:'column', gap:12, transition:'all 200ms ease', position:'relative', overflow:'hidden' }}>

                {/* Badge */}
                {intg.badge && (
                  <span style={{ position:'absolute', top:12, right:12, fontFamily:'var(--font-mono)', fontSize:8.5, padding:'2px 7px', borderRadius:3, background: intg.badge==='New' ? 'var(--green-dim)' : intg.badge==='Beta' ? 'var(--amber-dim)' : 'var(--accent-dim)', color: intg.badge==='New' ? 'var(--green)' : intg.badge==='Beta' ? 'var(--amber)' : 'var(--accent)', border:'1px solid', borderColor: intg.badge==='New' ? 'var(--green-bdr)' : intg.badge==='Beta' ? 'var(--amber-bdr)' : 'var(--accent-bdr)' }}>
                    {intg.badge}
                  </span>
                )}

                {/* Icon + Name */}
                <div style={{ display:'flex', alignItems:'center', gap:10 }}>
                  <div style={{ width:38, height:38, borderRadius:'var(--r-sm)', background: isConnected ? 'var(--green-dim)' : 'var(--bg-hover)', border:`1px solid ${isConnected ? 'var(--green-bdr)' : 'var(--border-default)'}`, display:'flex', alignItems:'center', justifyContent:'center', fontSize:16, color: isConnected ? 'var(--green)' : 'var(--text-secondary)', transition:'all 200ms ease', flexShrink:0 }}>
                    <i className={`${intg.fab ? 'fab' : 'fas'} ${intg.icon}`} />
                  </div>
                  <div>
                    <div style={{ fontSize:14, fontWeight:600, color:'var(--text-primary)' }}>{intg.name}</div>
                    {isConnected && <div style={{ fontFamily:'var(--font-mono)', fontSize:9, color:'var(--green)', marginTop:1 }}>● connected</div>}
                  </div>
                </div>

                {/* Description */}
                <p style={{ fontSize:12.5, color:'var(--text-secondary)', lineHeight:1.65, flex:1 }}>{intg.desc}</p>

                {/* Button */}
                <button onClick={() => handleConnect(intg.id)} disabled={isConnecting}
                  style={{ display:'flex', alignItems:'center', justifyContent:'center', gap:7, padding:'8px 0', borderRadius:'var(--r-sm)', border:'1px solid', fontSize:12.5, fontWeight:500, cursor: isConnecting ? 'not-allowed' : 'pointer', transition:'all 150ms ease', opacity: isConnecting ? 0.7 : 1,
                    borderColor: isConnected ? 'var(--red-bdr)' : 'var(--accent-bdr)',
                    background: isConnected ? 'var(--red-dim)' : 'var(--accent-dim)',
                    color: isConnected ? 'var(--red)' : 'var(--accent)'
                  }}>
                  {isConnecting
                    ? <><i className="fas fa-spinner fa-spin" /> Connecting…</>
                    : isConnected
                      ? <><i className="fas fa-plug-circle-xmark" /> Disconnect</>
                      : <><i className="fas fa-plug" /> Connect</>
                  }
                </button>
              </div>
            );
          })}
        </div>

        {/* Footer note */}
        <div style={{ marginTop:28, padding:'14px 18px', borderRadius:'var(--r-md)', background:'var(--bg-surface)', border:'1px solid var(--border-subtle)', display:'flex', alignItems:'flex-start', gap:10 }}>
          <i className="fas fa-circle-info" style={{ color:'var(--accent)', fontSize:12, flexShrink:0, marginTop:1 }} />
          <span style={{ fontSize:12.5, color:'var(--text-tertiary)', lineHeight:1.65 }}>
            Need an integration that's not listed? Contact your Lexis administrator or{' '}
            <span style={{ color:'var(--accent)', cursor:'pointer' }}>submit a request</span>.
            Custom integrations via the Lexis API are available on Enterprise plans.
          </span>
        </div>

      </div>
    </div>
  );
}

/* ── Help Page ──────────────────────────────────────────────── */
function HelpPage() {
  const [openFaq, setOpenFaq] = useState(null);
  const [ticketForm, setTicketForm] = useState({ subject:'', message:'' });
  const [submitted, setSubmitted] = useState(false);

  const faqs = [
    { q:'How do I upload documents to the knowledge base?', a:'Use the "Upload document" button in the left sidebar. Supported formats include PDF, DOCX, TXT, MD, and CSV. Files are automatically chunked, embedded, and indexed for RAG retrieval.' },
    { q:'How do I invoke an agent?', a:'Type / in the chat input to open the agent picker. Use ↑↓ to navigate, then press Enter or Tab to select. The agent name will appear as a token in your message.' },
    { q:'What datasets can I use with the data agents?', a:'The data-analysis-agent and create-visualization-agent support CSV and Excel files (.xlsx, .xls). Upload via the + button in the chat bar, then select a file before sending your query.' },
    { q:'Can I switch language models mid-conversation?', a:'Yes. Use the model selector in the input bar (the chip labelled with the model name) to switch between Llama 3.1 8B and Llama 3.3 70B at any time.' },
    { q:'How is RAG mode different from direct mode?', a:'In RAG mode, Lexis retrieves relevant chunks from your indexed documents and passes them as context to the model. In direct mode, the model answers from its own training data only.' },
    { q:'Is my data stored securely?', a:'Uploaded files are stored on the server running Lexis. Enterprise deployments can be configured with encrypted storage and private inference endpoints. Contact your administrator for details.' },
  ];

  const resources = [
    { icon:'fa-book', label:'Documentation', desc:'Full API and configuration reference', color:'var(--accent)' },
    { icon:'fa-graduation-cap', label:'Getting started', desc:'Step-by-step guide for new users', color:'var(--green)' },
    { icon:'fa-video', label:'Video tutorials', desc:'Walkthroughs for agents and RAG', color:'var(--purple)' },
    { icon:'fa-code-branch', label:'Changelog', desc:'What\'s new in each release', color:'var(--amber)' },
  ];

  const handleSubmit = () => {
    if (!ticketForm.subject.trim() || !ticketForm.message.trim()) return;
    setSubmitted(true);
    setTimeout(() => { setSubmitted(false); setTicketForm({ subject:'', message:'' }); }, 3000);
  };

  return (
    <div style={{ flex:1, overflowY:'auto', padding:'32px 24px 48px' }}>
      <div style={{ maxWidth:780, margin:'0 auto' }}>

        {/* Header */}
        <div style={{ marginBottom:32 }}>
          <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:6 }}>
            <div style={{ width:36, height:36, borderRadius:'var(--r-sm)', background:'var(--green-dim)', border:'1px solid var(--green-bdr)', display:'flex', alignItems:'center', justifyContent:'center', color:'var(--green)', fontSize:14 }}>
              <i className="fas fa-life-ring" />
            </div>
            <h1 style={{ fontSize:22, fontWeight:600, letterSpacing:'-0.03em', color:'var(--text-primary)' }}>Help & Support</h1>
          </div>
          <p style={{ fontSize:13, color:'var(--text-tertiary)', marginLeft:46 }}>Everything you need to get the most out of Lexis.</p>
        </div>

        {/* Quick resources */}
        <div style={{ display:'grid', gridTemplateColumns:'repeat(2,1fr)', gap:10, marginBottom:28 }}>
          {resources.map(r => (
            <button key={r.label}
              style={{ display:'flex', alignItems:'center', gap:12, padding:'14px 16px', borderRadius:'var(--r-md)', background:'var(--bg-surface)', border:'1px solid var(--border-subtle)', textAlign:'left', cursor:'pointer', transition:'all 150ms ease' }}
              onMouseOver={e => { e.currentTarget.style.borderColor='var(--accent-bdr)'; e.currentTarget.style.background='var(--accent-dim)'; }}
              onMouseOut={e => { e.currentTarget.style.borderColor='var(--border-subtle)'; e.currentTarget.style.background='var(--bg-surface)'; }}>
              <div style={{ width:34, height:34, borderRadius:'var(--r-sm)', background:'var(--bg-hover)', display:'flex', alignItems:'center', justifyContent:'center', fontSize:14, color:r.color, flexShrink:0 }}>
                <i className={`fas ${r.icon}`} />
              </div>
              <div>
                <div style={{ fontSize:13.5, fontWeight:600, color:'var(--text-primary)' }}>{r.label}</div>
                <div style={{ fontSize:12, color:'var(--text-tertiary)', marginTop:1 }}>{r.desc}</div>
              </div>
              <i className="fas fa-arrow-up-right-from-square" style={{ marginLeft:'auto', fontSize:10, color:'var(--text-tertiary)', flexShrink:0 }} />
            </button>
          ))}
        </div>

        {/* FAQ */}
        <div style={{ background:'var(--bg-surface)', border:'1px solid var(--border-subtle)', borderRadius:'var(--r-lg)', overflow:'hidden', marginBottom:24 }}>
          <div style={{ padding:'16px 20px', borderBottom:'1px solid var(--border-subtle)', display:'flex', alignItems:'center', gap:7 }}>
            <i className="fas fa-circle-question" style={{ color:'var(--accent)', fontSize:12 }} />
            <span style={{ fontFamily:'var(--font-mono)', fontSize:9.5, letterSpacing:'0.1em', textTransform:'uppercase', color:'var(--text-tertiary)' }}>Frequently asked questions</span>
          </div>
          {faqs.map((faq, i) => (
            <div key={i} style={{ borderBottom: i < faqs.length-1 ? '1px solid var(--border-subtle)' : 'none' }}>
              <button onClick={() => setOpenFaq(openFaq === i ? null : i)}
                style={{ width:'100%', display:'flex', alignItems:'center', justifyContent:'space-between', padding:'14px 20px', background:'transparent', cursor:'pointer', textAlign:'left', gap:12 }}>
                <span style={{ fontSize:13.5, color:'var(--text-primary)', fontWeight:500, flex:1 }}>{faq.q}</span>
                <i className={`fas fa-chevron-${openFaq === i ? 'up' : 'down'}`} style={{ fontSize:10, color:'var(--text-tertiary)', flexShrink:0, transition:'transform 200ms ease' }} />
              </button>
              {openFaq === i && (
                <div style={{ padding:'0 20px 16px', fontSize:13, color:'var(--text-secondary)', lineHeight:1.75 }}>
                  {faq.a}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Contact */}
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12, marginBottom:24 }}>
          <div style={{ background:'var(--bg-surface)', border:'1px solid var(--border-subtle)', borderRadius:'var(--r-lg)', padding:'18px 20px' }}>
            <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:10 }}>
              <i className="fas fa-envelope" style={{ color:'var(--accent)', fontSize:13 }} />
              <span style={{ fontSize:14, fontWeight:600, color:'var(--text-primary)' }}>Email support</span>
            </div>
            <p style={{ fontSize:12.5, color:'var(--text-secondary)', lineHeight:1.65, marginBottom:14 }}>
              Reach our support team for account and billing queries.
            </p>
            <a href="mailto:support@lexis.ai" style={{ fontFamily:'var(--font-mono)', fontSize:11.5, color:'var(--accent)', textDecoration:'none' }}>
              support@lexis.ai
            </a>
          </div>
          <div style={{ background:'var(--bg-surface)', border:'1px solid var(--border-subtle)', borderRadius:'var(--r-lg)', padding:'18px 20px' }}>
            <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:10 }}>
              <i className="fas fa-comments" style={{ color:'var(--purple)', fontSize:13 }} />
              <span style={{ fontSize:14, fontWeight:600, color:'var(--text-primary)' }}>Community</span>
            </div>
            <p style={{ fontSize:12.5, color:'var(--text-secondary)', lineHeight:1.65, marginBottom:14 }}>
              Join other Lexis users to share tips and workflows.
            </p>
            <span style={{ fontFamily:'var(--font-mono)', fontSize:11.5, color:'var(--purple)', cursor:'pointer' }}>
              community.lexis.ai ↗
            </span>
          </div>
        </div>

        {/* Ticket form */}
        <div style={{ background:'var(--bg-surface)', border:'1px solid var(--border-subtle)', borderRadius:'var(--r-lg)', padding:'20px 22px' }}>
          <div style={{ display:'flex', alignItems:'center', gap:7, marginBottom:18 }}>
            <i className="fas fa-ticket" style={{ color:'var(--amber)', fontSize:11 }} />
            <span style={{ fontFamily:'var(--font-mono)', fontSize:9.5, letterSpacing:'0.1em', textTransform:'uppercase', color:'var(--text-tertiary)' }}>Submit a support ticket</span>
          </div>
          {submitted ? (
            <div style={{ display:'flex', alignItems:'center', justifyContent:'center', flexDirection:'column', gap:10, padding:'24px 0', color:'var(--green)' }}>
              <i className="fas fa-circle-check" style={{ fontSize:28 }} />
              <span style={{ fontSize:14, fontWeight:500 }}>Ticket submitted — we'll be in touch shortly.</span>
            </div>
          ) : (
            <>
              <input value={ticketForm.subject} onChange={e => setTicketForm(p => ({...p, subject:e.target.value}))}
                placeholder="Subject"
                style={{ width:'100%', padding:'9px 12px', borderRadius:'var(--r-sm)', border:'1px solid var(--border-default)', background:'var(--bg-input)', color:'var(--text-primary)', fontSize:13.5, marginBottom:10, outline:'none', boxSizing:'border-box' }} />
              <textarea value={ticketForm.message} onChange={e => setTicketForm(p => ({...p, message:e.target.value}))}
                placeholder="Describe your issue in detail…" rows={4}
                style={{ width:'100%', padding:'9px 12px', borderRadius:'var(--r-sm)', border:'1px solid var(--border-default)', background:'var(--bg-input)', color:'var(--text-primary)', fontSize:13.5, resize:'vertical', outline:'none', boxSizing:'border-box', marginBottom:14 }} />
              <div style={{ display:'flex', justifyContent:'flex-end' }}>
                <button onClick={handleSubmit}
                  style={{ display:'flex', alignItems:'center', gap:7, padding:'8px 18px', borderRadius:'var(--r-sm)', border:'1px solid var(--amber-bdr)', background:'var(--amber-dim)', color:'var(--amber)', fontSize:13, fontWeight:500, cursor:'pointer' }}>
                  <i className="fas fa-paper-plane" /> Send ticket
                </button>
              </div>
            </>
          )}
        </div>

      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════
   Main App
   ════════════════════════════════════════════════════════════ */
function LexisLogo() {
  return (
    <div className="lexis-logo" aria-hidden="true">
      <span className="lexis-logo-core" />
      <span className="lexis-logo-orbit orbit-a" />
      <span className="lexis-logo-orbit orbit-b" />
    </div>
  );
}

function App() {
  /* ── State ── */
  const [sessions,    setSessions]    = useState(() => LS.get('lexis_sessions', []));
  const [activeId,    setActiveId]    = useState(() => LS.get('lexis_active', null));
  const [input,       setInput]       = useState('');
  const [typing,      setTyping]      = useState(false);
  const [sidebar,     setSidebar]     = useState(true);
  const [light,       setLight]       = useState(() => LS.get('lexis_light', false));
  const [modal,       setModal]       = useState(null);
  const [toast,       setToast]       = useState(null);

  /* ── Active page: 'chat' | 'settings' | 'integrations' | 'help' ── */
  const [currentPage, setCurrentPage] = useState('chat');

  const [docs,        setDocs]        = useState([]);
  const [docsLoading, setDocsLoading] = useState(true);
  const [deletingDoc, setDeletingDoc] = useState(false);

  const [datasets,        setDatasets]        = useState([]);
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [activeDataset,   setActiveDataset]   = useState(null);

  const [ragMode,   setRagMode]   = useState(true);
  const [webSearch, setWebSearch] = useState(false);
  const [agentMode, setAgentMode] = useState(false);

  const [showDrop,    setShowDrop]    = useState(false);
  const [agentFilter, setAgentFilter] = useState('');
  const [dropFocus,   setDropFocus]   = useState(0);
  const [activeAgent, setActiveAgent] = useState(null);

  const [showDatasetModal, setShowDatasetModal] = useState(false);
  const [pendingAgent,     setPendingAgent]      = useState(null);

  const [selectedModel,    setSelectedModel]    = useState(MODELS[0].id);
  const [modelSwitching,   setModelSwitching]   = useState(false);

  const endRef   = useRef(null);
  const taRef    = useRef(null);
  const toastTmr = useRef(null);

  /* ── Derived ── */
  const currentSession = sessions.find(s => s.id === activeId) || null;
  const msgs           = currentSession ? currentSession.msgs : [];
  const isAgentic      = agentMode || !!activeAgent;
  const isHome         = msgs.length === 0 && !typing && currentPage === 'chat';
  const modeLabel      = agentMode
    ? (activeAgent ? activeAgent.name : 'agentic')
    : (ragMode ? (webSearch ? 'RAG+Web' : 'RAG') : (webSearch ? 'web' : 'direct'));

  /* ── Persistence ── */
  useEffect(() => { LS.set('lexis_sessions', sessions); }, [sessions]);
  useEffect(() => { LS.set('lexis_active',   activeId); }, [activeId]);
  useEffect(() => { LS.set('lexis_light',    light);    }, [light]);
  useEffect(() => { document.body.classList.toggle('light', light); }, [light]);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [msgs, typing]);

  useEffect(() => {
    if (!activeId || !sessions.find(s => s.id === activeId)) {
      if (sessions.length > 0) setActiveId(sessions[0].id);
      else { const id = uid(); setSessions([{ id, createdAt: ts(), msgs: [] }]); setActiveId(id); }
    }
  }, []);

  const setMsgs = fn => setSessions(prev =>
    prev.map(s => s.id === activeId
      ? { ...s, msgs: typeof fn === 'function' ? fn(s.msgs) : fn }
      : s)
  );

  /* ── Data loaders ── */
  const loadDocs = useCallback(async () => {
    setDocsLoading(true);
    try {
      const res  = await fetch('/lexis/files');
      const data = await res.json();
      setDocs((data.files || []).map((name, i) => ({ id: i, name })));
    } catch { setDocs([]); }
    finally { setDocsLoading(false); }
  }, []);
  useEffect(() => { loadDocs(); }, [loadDocs]);

  const loadDatasets = useCallback(async () => {
    setDatasetsLoading(true);
    try {
      const res  = await fetch('/lexis/datasets');
      const data = await res.json();
      setDatasets(data.files || []);
    } catch { setDatasets([]); }
    finally { setDatasetsLoading(false); }
  }, []);
  useEffect(() => { loadDatasets(); }, [loadDatasets]);

  useEffect(() => {
    fetch('/lexis/model')
      .then(r => r.json())
      .then(d => {
        if (d.model && MODELS.find(m => m.id === d.model)) setSelectedModel(d.model);
      })
      .catch(() => {});
  }, []);

  /* ── Toast ── */
  const notify = (text, kind = 'ok') => {
    clearTimeout(toastTmr.current);
    setToast({ text, kind });
    toastTmr.current = setTimeout(() => setToast(null), 2800);
  };

  /* ── Switch model ── */
  const switchModel = async (modelId) => {
    if (modelId === selectedModel || modelSwitching) return;
    setModelSwitching(true);
    try {
      const res  = await fetch('/lexis/model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelId }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Switch failed');
      setSelectedModel(modelId);
      notify(`Model → ${MODELS.find(m => m.id === modelId)?.label || modelId}`);
    } catch (err) {
      notify(`Failed: ${err.message}`, 'err');
    } finally {
      setModelSwitching(false);
    }
  };

  /* ── Sessions ── */
  const newChat = () => {
    const id = uid();
    setSessions(prev => [{ id, createdAt: ts(), msgs: [] }, ...prev]);
    setActiveId(id); setInput(''); setActiveAgent(null); setCurrentPage('chat');
  };
  const switchSession = id => { setActiveId(id); setInput(''); setActiveAgent(null); setCurrentPage('chat'); };
  const deleteSession = id => {
    setSessions(prev => prev.filter(s => s.id !== id));
    if (activeId === id) {
      const rem = sessions.filter(s => s.id !== id);
      if (rem.length) setActiveId(rem[0].id);
      else { const nid = uid(); setSessions([{ id: nid, createdAt: ts(), msgs: [] }]); setActiveId(nid); }
    }
  };

  /* ── Navigate to page (also resets to chat) ── */
  const navigate = (page) => setCurrentPage(page);

  /* ── Input & agent autocomplete ── */
  const handleInputChange = e => {
    const val = e.target.value;
    setInput(val);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 140) + 'px';
    const cur   = e.target.selectionStart;
    const before = val.slice(0, cur);
    const slash  = before.match(/(\/[\w-]*)$/);
    if (slash) { setAgentFilter(slash[1]); setShowDrop(true); setDropFocus(0); }
    else        { setShowDrop(false); setAgentFilter(''); }
    const found = AGENTS.find(a => val.includes('/' + a.name));
    setActiveAgent(found || null);
  };

  const selectAgent = agent => {
    const cur    = taRef.current?.selectionStart ?? input.length;
    const before = input.slice(0, cur);
    const after  = input.slice(cur);
    const si     = before.lastIndexOf('/');
    const newVal = before.slice(0, si) + '/' + agent.name + ' ' + after;
    setInput(newVal);
    setShowDrop(false); setAgentFilter(''); setActiveAgent(agent);
    if (agent.needsDataset) { setPendingAgent(agent); setShowDatasetModal(true); }
    setTimeout(() => {
      const pos = before.slice(0, si).length + agent.name.length + 2;
      taRef.current?.setSelectionRange(pos, pos);
      taRef.current?.focus();
    }, 0);
  };

  const handleDatasetSelected = (filename) => {
    setActiveDataset(prev => prev === filename ? null : filename);
    setShowDatasetModal(false);
    setPendingAgent(null);
    notify(`Dataset "${filename}" selected`);
    loadDatasets();
    taRef.current?.focus();
  };

  const onKey = e => {
    if (showDrop) {
      const list = AGENTS.filter(a => a.name.includes(agentFilter.replace('/', '').toLowerCase()));
      if (e.key === 'ArrowDown') { e.preventDefault(); setDropFocus(f => Math.min(f+1, list.length-1)); return; }
      if (e.key === 'ArrowUp')   { e.preventDefault(); setDropFocus(f => Math.max(f-1, 0)); return; }
      if (e.key === 'Tab' || (e.key === 'Enter' && list[dropFocus])) { e.preventDefault(); selectAgent(list[dropFocus]); return; }
      if (e.key === 'Escape') { setShowDrop(false); return; }
    }
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  };

  /* ── Send ── */
  const send = async () => {
    const text = input.trim();
    if (!text || typing) return;
    setInput('');
    setShowDrop(false);
    if (taRef.current) taRef.current.style.height = 'auto';

    const usedAgent = activeAgent;
    setActiveAgent(null);

    setMsgs(p => [...p, { id: uid(), sender: 'user', text, time: ts() }]);
    setTyping(true);

    try {
      if (usedAgent?.name === 'create-visualization-agent') {
        if (!activeDataset) {
          setPendingAgent(usedAgent); setShowDatasetModal(true);
          setMsgs(p => [...p, { id: uid(), sender: 'ai', time: ts(), text: 'Please select or upload a dataset first using the panel that just opened.' }]);
          setTyping(false); return;
        }
        const res  = await fetch('/lexis/agent/visualize', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, filename: activeDataset }),
        });
        const data = await res.json();
        if (data.success) {
          setMsgs(p => [...p, { id: uid(), sender: 'ai', agent: usedAgent.name, time: ts(), msgType: 'chart', figure: data.figure, summary: data.summary, filename: data.filename, rows: data.rows, columns: data.columns, text: data.summary || 'Chart generated.' }]);
        } else {
          setMsgs(p => [...p, { id: uid(), sender: 'ai', time: ts(), text: `⚠ Visualization error: ${data.error}` }]);
        }

      } else if (usedAgent?.name === 'data-analysis-agent') {
        if (!activeDataset) {
          setPendingAgent(usedAgent); setShowDatasetModal(true);
          setMsgs(p => [...p, { id: uid(), sender: 'ai', time: ts(), text: 'Please select or upload a dataset first using the panel that just opened.' }]);
          setTyping(false); return;
        }
        const res  = await fetch('/lexis/agent/analyze', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, filename: activeDataset }),
        });
        const data = await res.json();
        if (data.success) {
          setMsgs(p => [...p, { id: uid(), sender: 'ai', agent: usedAgent.name, time: ts(), msgType: 'analysis', headline: data.headline, narrative: data.narrative, keyFindings: data.key_findings, recommendation: data.recommendation, statsTable: data.stats_table, primaryTable: data.primary_table, operations: data.operations, filename: data.filename, rows: data.rows, columns: data.columns, text: data.headline || 'Analysis complete.' }]);
        } else {
          setMsgs(p => [...p, { id: uid(), sender: 'ai', time: ts(), text: `⚠ Analysis error: ${data.error}` }]);
        }

      } else {
        const res  = await fetch('/lexis/chat', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, rag: ragMode, web_search: webSearch, agent_mode: agentMode, agent: usedAgent?.name || null }),
        });
        const data = await res.json();
        setMsgs(p => [...p, { id: uid(), sender: 'ai', text: data.response, sources: data.sources || [], agent: usedAgent?.name || null, time: ts() }]);
      }
    } catch (err) {
      setMsgs(p => [...p, { id: uid(), sender: 'ai', time: ts(), text: '⚠ Something went wrong — please try again.' }]);
    } finally {
      setTyping(false);
    }
  };

  /* ── RAG document upload ── */
  const upload = async e => {
    const file = e.target.files[0]; if (!file) return;
    const fd = new FormData(); fd.append('file', file);
    try {
      const res = await fetch('/lexis/upload', { method: 'POST', body: fd });
      if (!res.ok) throw new Error();
      await loadDocs(); notify(`"${file.name}" indexed`);
    } catch { notify('Upload failed — please retry', 'err'); }
    e.target.value = '';
  };

  /* ── Remove RAG doc ── */
  const confirmRemoveDoc = async () => {
    const doc = modal?.doc; if (!doc) return;
    setDeletingDoc(true);
    try {
      setDocs(prev => prev.filter(d => d.name !== doc.name));
      const res  = await fetch('/lexis/remove-file', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: doc.name }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Delete failed');
      await loadDocs(); setModal(null); notify(`"${doc.name}" removed`);
    } catch { await loadDocs(); notify('Failed to remove', 'err'); }
    finally { setDeletingDoc(false); }
  };

  /* ── Remove dataset ── */
  const removeDataset = async (filename) => {
    try {
      await fetch('/lexis/remove-dataset', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      });
      if (activeDataset === filename) setActiveDataset(null);
      await loadDatasets(); notify(`Dataset "${filename}" removed`);
    } catch { notify('Failed to remove dataset', 'err'); }
  };

  /* ── Sidebar nav item helper ── */
  const SbNavItem = ({ page, icon, label, title }) => {
    const active = currentPage === page;
    return (
      <button
        className="sb-nav-item"
        title={title || label}
        onClick={() => navigate(page)}
        style={active ? { background:'var(--accent-dim)', borderColor:'var(--accent-bdr)', color:'var(--accent)' } : {}}
      >
        <i className={`fas ${icon}`} style={active ? { color:'var(--accent)' } : {}} />
        <span>{label}</span>
      </button>
    );
  };

  /* ════════════════════════ RENDER ════════════════════════ */
  return (
    <div className="shell">

      {/* ════════════ SIDEBAR ════════════ */}
      <aside className={`sidebar ${sidebar ? 'expanded' : 'collapsed'}`}>
        <div className="sb-inner">

          <div className="sb-brand">
            <LexisLogo />
            <div className="brand-info">
              <div className="brand-name">Lexis</div>
              <div className="brand-sub">Knowledge AI · TCS ❤️ · v1.0.5β</div>
            </div>
            <button className="sb-collapse-btn" onClick={() => setSidebar(p => !p)} title={sidebar ? 'Collapse sidebar' : 'Expand sidebar'}>
              <i className={`fas fa-${sidebar ? 'angles-left' : 'angles-right'}`} />
            </button>
          </div>

          <button className="new-chat-btn" onClick={newChat}>
            <i className="fas fa-plus" /> <span>New chat</span>
          </button>

          <div className="sb-nav">
            <SbNavItem page="chat"         icon="fa-comment-dots" label="Chat"        title="Back to chat" />
            <button className="sb-nav-item" title="Team chats"><i className="fas fa-users" /><span>Team chats</span></button>
            <button className="sb-nav-item" title="Dashboards"><i className="fas fa-chart-line" /><span>Dashboards</span></button>
            <button className="sb-nav-item" title="Prompt lab"><i className="fas fa-flask" /><span>Prompt lab</span></button>
            <button className="sb-nav-item" title="My Assistant"><i className="fas fa-robot" /><span>My Assistant</span></button>
          </div>

          {sessions.length > 0 && (
            <>
              <div className="sb-section-label">History</div>
              <div className="history-list">
                {sessions.map(s => (
                  <div key={s.id} className={`hist-item ${s.id === activeId && currentPage === 'chat' ? 'active' : ''}`}
                       onClick={() => switchSession(s.id)}>
                    <div className="hist-dot" />
                    <div className="hist-label">{shortPreview(s.msgs)}</div>
                    <div className="hist-time">{s.createdAt}</div>
                    <button className="hist-del" onClick={ev => { ev.stopPropagation(); deleteSession(s.id); }} title="Delete">
                      <i className="fas fa-xmark" />
                    </button>
                  </div>
                ))}
              </div>
            </>
          )}

          <div className="sb-sep" />

          <div className="sb-section-label">Workspace</div>
          <input type="file" id="sb-file" style={{ display: 'none' }} onChange={upload}
                 accept=".pdf,.doc,.docx,.txt,.md,.csv,.xlsx" />
          <label htmlFor="sb-file" className="sb-btn upload">
            <i className="fas fa-arrow-up-from-bracket" /> <span>Upload document</span>
          </label>
          <button className="sb-btn danger" onClick={() => setModal({ type: 'clear' })}>
            <i className="fas fa-trash-can" /> <span>Clear conversation</span>
          </button>

          <div className="sb-sep" />

          {/* Knowledge base — only shown in sidebar (not a page) */}
          <div className="kb-section">
            <div className="sb-section-label" style={{ paddingBottom: 0 }}>Knowledge base</div>
            <div className="kb-header">
              <span style={{ fontSize: 11, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>indexed docs</span>
              <span className="kb-count">{docs.length}</span>
            </div>
            <div className="doc-list">
              {docsLoading && [0,1,2].map(i => <div key={i} className="shimmer" />)}
              {!docsLoading && docs.length === 0 && (
                <div className="doc-empty"><i className="fas fa-folder-open" />No documents indexed yet.</div>
              )}
              {!docsLoading && docs.map(d => {
                const ext = extOf(d.name); const [ico, cls] = iconFor(ext);
                return (
                  <div className="doc-item" key={d.id}>
                    <div className={`doc-icon ${cls}`}><i className={`fas ${ico}`} /></div>
                    <div className="doc-info">
                      <div className="doc-name" title={d.name}>{d.name}</div>
                      <div className="doc-ext">.{ext || 'file'}</div>
                    </div>
                    <div className="doc-pulse" title="Indexed" />
                    <button className="doc-remove" title="Remove" onClick={() => setModal({ type: 'remove-doc', doc: d })}>
                      <i className="fas fa-trash-can" />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Footer nav — Settings / Integrations / Help */}
          <div className="sb-footer">
            <button
              className="sb-footer-link"
              title="Settings"
              onClick={() => navigate('settings')}
              style={currentPage === 'settings' ? { background:'var(--accent-dim)', borderColor:'var(--accent-bdr)', color:'var(--accent)' } : {}}
            >
              <i className="fas fa-gear" style={currentPage === 'settings' ? { color:'var(--accent)' } : {}} />
              <span>Settings</span>
            </button>
            <button
              className="sb-footer-link"
              title="Integrations"
              onClick={() => navigate('integrations')}
              style={currentPage === 'integrations' ? { background:'var(--purple-dim)', borderColor:'var(--purple-bdr)', color:'var(--purple)' } : {}}
            >
              <i className="fas fa-plug" style={currentPage === 'integrations' ? { color:'var(--purple)' } : {}} />
              <span>Integrations</span>
            </button>
            <button
              className="sb-footer-link"
              title="Help and Support"
              onClick={() => navigate('help')}
              style={currentPage === 'help' ? { background:'var(--green-dim)', borderColor:'var(--green-bdr)', color:'var(--green)' } : {}}
            >
              <i className="fas fa-life-ring" style={currentPage === 'help' ? { color:'var(--green)' } : {}} />
              <span>Help & Support</span>
            </button>
            <button className="theme-toggle" onClick={() => setLight(p => !p)}>
              <div className="theme-toggle-label">
                <i className={`fas fa-${light ? 'sun' : 'moon'}`} />
                {light ? 'Light mode' : 'Dark mode'}
              </div>
              <div className="toggle-pill" />
            </button>
          </div>

        </div>
      </aside>

      {/* ════════════ MAIN ════════════ */}
      <main className={`main ${isHome ? 'home' : ''}`} style={{ display:'flex', flexDirection:'column' }}>

        {/* ── Topbar (always visible, content adapts per page) ── */}
        <div className="topbar">
          <button className="tb-icon-btn" onClick={() => setSidebar(p => !p)} title="Toggle sidebar">
            <i className="fas fa-bars" />
          </button>
          <div className="tb-center">
            {currentPage === 'chat' ? (
              <>
                <span className="tb-title">
                  {currentSession && msgs.length > 0 ? shortPreview(msgs) : 'Conversation'}
                </span>
                <div className="tb-badges">
                  <span className={`tb-badge ${ragMode ? 'green' : 'muted'}`}>{ragMode ? 'RAG' : 'direct'}</span>
                  {webSearch && <span className="tb-badge amber">web</span>}
                  {isAgentic  && <span className="tb-badge purple">agentic</span>}
                  <span className="tb-badge accent" title={`Model: groq:${selectedModel}`}>
                    <i className="fas fa-microchip" style={{ marginRight: 3 }} />
                    {MODELS.find(m => m.id === selectedModel)?.label || selectedModel}
                  </span>
                  {activeDataset && (
                    <span className="tb-badge amber" title={`Active: ${activeDataset}`}>
                      <i className="fas fa-table" style={{ marginRight: 3 }} />{activeDataset}
                    </span>
                  )}
                  <span className="tb-badge muted">{docs.length} docs</span>
                </div>
              </>
            ) : (
              <>
                {/* Breadcrumb for sub-pages */}
                <button
                  onClick={() => navigate('chat')}
                  style={{ display:'flex', alignItems:'center', gap:6, fontFamily:'var(--font-mono)', fontSize:11, color:'var(--text-tertiary)', background:'none', border:'none', cursor:'pointer', padding:0 }}
                >
                  <i className="fas fa-comment-dots" style={{ fontSize:10 }} /> Chat
                </button>
                <i className="fas fa-chevron-right" style={{ fontSize:8, color:'var(--border-strong)' }} />
                <span className="tb-title" style={{ textTransform:'capitalize' }}>{currentPage}</span>
              </>
            )}
          </div>
          {currentPage === 'chat' && (
            <div className="tb-right">
              <div className="tb-status">
                <div className={`status-dot ${typing ? 'thinking' : ''}`} />
                {typing ? 'thinking' : 'ready'}
              </div>
            </div>
          )}
        </div>

        {/* ── Page routing ── */}
        {currentPage === 'settings' && (
          <SettingsPage
            light={light}
            setLight={setLight}
            selectedModel={selectedModel}
            switchModel={switchModel}
            modelSwitching={modelSwitching}
          />
        )}

        {currentPage === 'integrations' && <IntegrationsPage />}

        {currentPage === 'help' && <HelpPage />}

        {currentPage === 'chat' && (
          <>
            {/* Messages */}
            <div className="messages">
              {msgs.map(m => {
                const isAgentMsg = m.sender === 'ai' && m.agent;
                const agentDef   = AGENTS.find(a => a.name === m.agent);
                const isChart    = m.msgType === 'chart';
                const isAnalysis = m.msgType === 'analysis';
                const isWide     = isChart || isAnalysis;

                return (
                  <div key={m.id} className={`msg-row ${m.sender}`}>
                    {m.sender === 'ai' && (
                      <div className={`ai-avatar ${isAgentMsg ? 'agent' : ''}`}>
                        <i className={`fas ${isAgentMsg ? (agentDef?.icon || 'fa-robot') : 'fa-circle-nodes'}`} />
                      </div>
                    )}
                    <div className={`msg-content ${isWide ? 'msg-content-wide' : ''}`}>
                      <div className="msg-sender">
                        {m.sender === 'ai' ? (m.agent || 'lexis') : 'you'}
                      </div>
                      {isChart ? (
                        <ChartBubble figure={m.figure} summary={m.summary} filename={m.filename} rows={m.rows} columns={m.columns} />
                      ) : isAnalysis ? (
                        <AnalysisBubble headline={m.headline} narrative={m.narrative} keyFindings={m.keyFindings} recommendation={m.recommendation} statsTable={m.statsTable} primaryTable={m.primaryTable} operations={m.operations} filename={m.filename} rows={m.rows} />
                      ) : (
                        <div className="bubble"><BubbleContent text={m.text} /></div>
                      )}
                      {!isWide && m.sender === 'ai' && m.sources?.length > 0 && (
                        <div className="sources">
                          {m.sources.map((src, i) => (
                            <span key={i} className="source-chip">
                              <i className="fas fa-file" /> {src.label}
                            </span>
                          ))}
                        </div>
                      )}
                      <div className="msg-time">{m.time}</div>
                    </div>
                  </div>
                );
              })}

              {typing && (
                <div className="typing-row">
                  <div className={`ai-avatar ${isAgentic ? 'agent' : ''}`}>
                    <i className="fas fa-circle-nodes" />
                  </div>
                  <div className="typing-bub"><span /><span /><span /></div>
                </div>
              )}
              <div ref={endRef} />
            </div>

            {/* Input bar */}
            <div className="input-bar">
              <div className="input-outer">
                {isHome && (
                  <div className="home-hero">
                    <div className="home-orb-wrap">
                      <div className="home-beam" />
                      <LexisLogo />
                    </div>
                    <div className="home-title">Good morning, ready to start?</div>
                  </div>
                )}

                <div className="mode-strip">
                  <button className={`mode-chip ${ragMode ? 'on-green' : ''}`} onClick={() => setRagMode(p => !p)} title="Toggle RAG">
                    <i className="fas fa-database" /> RAG
                  </button>
                  <button className={`mode-chip ${webSearch ? 'on-amber' : ''}`} onClick={() => setWebSearch(p => !p)} title="Toggle web search">
                    <i className="fas fa-globe" /> Web
                  </button>
                  <button className={`mode-chip ${agentMode ? 'on-purple' : ''}`}
                    onClick={() => {
                      const next = !agentMode; setAgentMode(next);
                      if (next && !input.startsWith('/')) {
                        setInput(prev => prev ? prev : '/');
                        setTimeout(() => { taRef.current?.focus(); taRef.current?.setSelectionRange(1,1); }, 30);
                      }
                    }} title="Toggle agentic mode">
                    <i className="fas fa-robot" /> Agentic
                  </button>
                  <div className="model-selector-wrap" title="Switch LLM">
                    <i className={`fas fa-microchip model-sel-icon ${modelSwitching ? 'fa-spin' : ''}`} />
                    <select className="model-selector" value={selectedModel} disabled={modelSwitching} onChange={e => switchModel(e.target.value)}>
                      {MODELS.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
                    </select>
                    <i className="fas fa-chevron-down model-sel-caret" />
                  </div>
                  <div className="strip-info">{modeLabel} · {docs.length} docs</div>
                </div>

                <div className="input-wrap">
                  {showDrop && <AgentDropdown filter={agentFilter} onSelect={selectAgent} focusIdx={dropFocus} />}
                  <div className={`input-box ${isAgentic ? 'agentic' : ''}`}>
                    <button className="attach-btn" onClick={() => { setPendingAgent(null); setShowDatasetModal(true); }} title="Upload or manage CSV and Excel files">
                      <i className="fas fa-plus" />
                    </button>
                    <textarea ref={taRef} className="msg-ta" value={input}
                      onChange={handleInputChange} onKeyDown={onKey}
                      onBlur={() => setTimeout(() => setShowDrop(false), 160)}
                      placeholder={agentMode ? 'Type / to invoke an agent, or just ask…' : ragMode ? 'Ask anything about your documents…' : 'Chat directly with the AI…'}
                      rows={1} />
                    <button className={`send-btn ${isAgentic ? 'agentic' : ''}`} onClick={send}
                            disabled={!input.trim() || typing} title="Send (Enter)">
                      <i className="fas fa-arrow-up" />
                    </button>
                  </div>
                </div>

                {isHome && (
                  <div className="home-cards">
                    <button className="home-card" onClick={() => { setInput('What interest rate applies to a Personal Temporary Overdraft?'); taRef.current?.focus(); }}>
                      <span className="home-card-icon"><i className="fas fa-file-lines" /></span>
                      <span className="home-card-title">RAG Insight</span>
                      <span className="home-card-copy">Mortgage Lending Guidance</span>
                    </button>
                    <button className="home-card" onClick={() => { setInput('/create-visualization-agent Create a stacked bar chart showing total complaints by State, split by Issue.'); setAgentMode(true); taRef.current?.focus(); }}>
                      <span className="home-card-icon"><i className="fas fa-chart-simple" /></span>
                      <span className="home-card-title">Dashboard Designer</span>
                      <span className="home-card-copy">Create a chart for complaint data</span>
                    </button>
                    <button className="home-card" onClick={() => { setInput('/data-analysis-agent Which Issue category has the highest complaint count?'); setAgentMode(true); taRef.current?.focus(); }}>
                      <span className="home-card-icon"><i className="fas fa-magnifying-glass-chart" /></span>
                      <span className="home-card-title">Advanced Analysis</span>
                      <span className="home-card-copy">Explore patterns and insights.</span>
                    </button>
                  </div>
                )}

                <div className="input-meta">
                  <span className="input-hint">↵ send · shift+↵ newline{agentMode ? ' · / invoke agent' : ''}</span>
                  <span className="input-chars">{input.length > 0 ? `${input.length} chars` : ''}</span>
                </div>
              </div>
            </div>
          </>
        )}

      </main>

      {/* ════════════ MODALS ════════════ */}

      {showDatasetModal && (
        <DatasetUploadModal
          agent={pendingAgent}
          datasets={datasets}
          activeDataset={activeDataset}
          onClose={() => { setShowDatasetModal(false); setPendingAgent(null); }}
          onUploaded={handleDatasetSelected}
          onSelectExisting={handleDatasetSelected}
          onRemove={removeDataset}
        />
      )}

      {modal?.type === 'clear' && (
        <div className="backdrop" onClick={() => setModal(null)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-icon err"><i className="fas fa-trash-can" /></div>
            <div className="modal-title">Clear this conversation?</div>
            <div className="modal-body">All messages will be permanently removed. Knowledge base remains unaffected.</div>
            <div className="modal-row">
              <button className="m-btn" onClick={() => setModal(null)}>Cancel</button>
              <button className="m-btn danger" onClick={() => { setMsgs([]); setModal(null); notify('Conversation cleared'); }}>
                Clear chat
              </button>
            </div>
          </div>
        </div>
      )}

      {modal?.type === 'remove-doc' && (
        <div className="backdrop" onClick={() => !deletingDoc && setModal(null)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-icon warn"><i className="fas fa-file-xmark" /></div>
            <div className="modal-title">{deletingDoc ? 'Removing file…' : 'Remove from knowledge base?'}</div>
            <div className="modal-body">
              {deletingDoc ? 'Please wait…' : 'This will unindex the file. The model will no longer retrieve information from it.'}
            </div>
            <div className="modal-fname">{modal.doc?.name}</div>
            <div className="modal-row">
              {!deletingDoc ? (
                <>
                  <button className="m-btn" onClick={() => setModal(null)}>Cancel</button>
                  <button className="m-btn danger" onClick={confirmRemoveDoc}>Remove file</button>
                </>
              ) : (
                <button className="m-btn" disabled>
                  <i className="fas fa-spinner fa-spin" style={{ marginRight: 6 }} />Deleting…
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {toast && (
        <div className="toast-wrap">
          <div className="toast"><div className={`tdot ${toast.kind}`} />{toast.text}</div>
        </div>
      )}

    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);