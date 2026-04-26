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

/* Format a raw number value for display */
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

/* ── Available Groq models ──────────────────────────────────── */
const MODELS = [
  { id: 'llama-3.1-8b-instant',    label: 'Llama 3.1 · 8B',    tag: 'fast'    },
  { id: 'llama-3.3-70b-versatile', label: 'Llama 3.3 · 70B',   tag: 'smart'   },
];

/* ── Local storage helper ───────────────────────────────────── */
const LS = {
  get: (k, fb) => { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : fb; } catch { return fb; } },
  set: (k, v)  => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

/* ── BubbleContent: highlights /agent-name tokens ──────────── */
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

/* ════════════════════════════════════════════════════════════
   AnalysisBubble — renders the data analysis agent response
   ════════════════════════════════════════════════════════════ */
function AnalysisBubble({ headline, narrative, keyFindings, recommendation, statsTable, primaryTable, operations, filename, rows }) {
  const [activeTab, setActiveTab] = useState('insights'); // 'insights' | 'table' | 'ops'
  const hasTable = primaryTable && primaryTable.rows && primaryTable.rows.length > 0;
  const hasStats = statsTable && statsTable.length > 0;

  return (
    <div className="analysis-bubble">

      {/* ── Header row ── */}
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
        {/* Ops pills */}
        <div className="ab-ops-pills">
          {(operations || []).map(op => (
            <span key={op.op_id} className={`ab-op-pill ${op.status === 'error' ? 'error' : ''}`} title={op.label}>
              {op.status === 'error'
                ? <i className="fas fa-xmark" />
                : <i className="fas fa-check" />
              }
              {op.op_id}
            </span>
          ))}
        </div>
      </div>

      {/* ── Headline ── */}
      <div className="ab-headline">{headline}</div>

      {/* ── Stats cards ── */}
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

      {/* ── Tab nav ── */}
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

      {/* ── Insights tab ── */}
      {activeTab === 'insights' && (
        <div className="ab-tab-content">

          {/* Narrative */}
          <div className="ab-narrative">{narrative}</div>

          {/* Key findings */}
          {keyFindings && keyFindings.length > 0 && (
            <div className="ab-findings">
              <div className="ab-findings-label">
                <i className="fas fa-star" /> Key findings
              </div>
              <ul className="ab-findings-list">
                {keyFindings.map((f, i) => (
                  <li key={i}><i className="fas fa-arrow-right" /><span>{f}</span></li>
                ))}
              </ul>
            </div>
          )}

          {/* Recommendation */}
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

      {/* ── Table tab ── */}
      {activeTab === 'table' && hasTable && (
        <div className="ab-tab-content ab-table-wrap">
          <table className="ab-table">
            <thead>
              <tr>
                {primaryTable.columns.map(col => (
                  <th key={col}>{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {primaryTable.rows.map((row, i) => (
                <tr key={i}>
                  {primaryTable.columns.map(col => (
                    <td key={col}>{
                      row[col] === null || row[col] === undefined
                        ? '—'
                        : typeof row[col] === 'number'
                          ? fmtVal(row[col])
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
function DatasetUploadModal({ agent, datasets, onClose, onUploaded, onSelectExisting }) {
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
      const res  = await fetch('/upload-dataset', { method: 'POST', body: fd });
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
            <div className="modal-title">/{agent?.name}</div>
            <div className="modal-subtitle">{agent?.desc}</div>
          </div>
          <button className="modal-close-btn" onClick={onClose}><i className="fas fa-xmark" /></button>
        </div>

        {datasets.length > 0 && (
          <div className="modal-section">
            <div className="modal-section-label">Use an existing dataset</div>
            <div className="existing-datasets">
              {datasets.map(f => (
                <button key={f} className="existing-dataset-btn" onClick={() => onSelectExisting(f)}>
                  <i className="fas fa-file-excel" />
                  <span>{f}</span>
                  <i className="fas fa-arrow-right existing-arrow" />
                </button>
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
   Main App
   ════════════════════════════════════════════════════════════ */
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
      const res  = await fetch('/files');
      const data = await res.json();
      setDocs((data.files || []).map((name, i) => ({ id: i, name })));
    } catch { setDocs([]); }
    finally { setDocsLoading(false); }
  }, []);
  useEffect(() => { loadDocs(); }, [loadDocs]);

  const loadDatasets = useCallback(async () => {
    setDatasetsLoading(true);
    try {
      const res  = await fetch('/datasets');
      const data = await res.json();
      setDatasets(data.files || []);
    } catch { setDatasets([]); }
    finally { setDatasetsLoading(false); }
  }, []);
  useEffect(() => { loadDatasets(); }, [loadDatasets]);

  /* ── Load current model from backend ── */
  useEffect(() => {
    fetch('/model')
      .then(r => r.json())
      .then(d => {
        if (d.model && MODELS.find(m => m.id === d.model)) {
          setSelectedModel(d.model);
        }
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
      const res  = await fetch('/model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelId }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Switch failed');
      setSelectedModel(modelId);
      const m = MODELS.find(m => m.id === modelId);
      notify(`Model → ${m?.label || modelId}`);
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
    setActiveId(id); setInput(''); setActiveAgent(null);
  };
  const switchSession = id => { setActiveId(id); setInput(''); setActiveAgent(null); };
  const deleteSession = id => {
    setSessions(prev => prev.filter(s => s.id !== id));
    if (activeId === id) {
      const rem = sessions.filter(s => s.id !== id);
      if (rem.length) setActiveId(rem[0].id);
      else { const nid = uid(); setSessions([{ id: nid, createdAt: ts(), msgs: [] }]); setActiveId(nid); }
    }
  };

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
    setActiveDataset(filename);
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

  /* ════════════════════════════════════════════════════════════
     SEND — routes to the correct agent endpoint
     ════════════════════════════════════════════════════════════ */
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

      /* ══ visualization agent ══════════════════════════════ */
      if (usedAgent?.name === 'create-visualization-agent') {
        if (!activeDataset) {
          setPendingAgent(usedAgent); setShowDatasetModal(true);
          setMsgs(p => [...p, { id: uid(), sender: 'ai', time: ts(),
            text: 'Please select or upload a dataset first using the panel that just opened.' }]);
          setTyping(false); return;
        }
        const res  = await fetch('/agent/visualize', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, filename: activeDataset }),
        });
        const data = await res.json();
        if (data.success) {
          setMsgs(p => [...p, {
            id: uid(), sender: 'ai', agent: usedAgent.name, time: ts(),
            msgType: 'chart', figure: data.figure, summary: data.summary,
            filename: data.filename, rows: data.rows, columns: data.columns,
            text: data.summary || 'Chart generated.',
          }]);
        } else {
          setMsgs(p => [...p, { id: uid(), sender: 'ai', time: ts(),
            text: `⚠ Visualization error: ${data.error}` }]);
        }

      /* ══ data analysis agent ══════════════════════════════ */
      } else if (usedAgent?.name === 'data-analysis-agent') {
        if (!activeDataset) {
          setPendingAgent(usedAgent); setShowDatasetModal(true);
          setMsgs(p => [...p, { id: uid(), sender: 'ai', time: ts(),
            text: 'Please select or upload a dataset first using the panel that just opened.' }]);
          setTyping(false); return;
        }
        const res  = await fetch('/agent/analyze', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, filename: activeDataset }),
        });
        const data = await res.json();
        if (data.success) {
          setMsgs(p => [...p, {
            id: uid(), sender: 'ai', agent: usedAgent.name, time: ts(),
            msgType:        'analysis',
            headline:       data.headline,
            narrative:      data.narrative,
            keyFindings:    data.key_findings,
            recommendation: data.recommendation,
            statsTable:     data.stats_table,
            primaryTable:   data.primary_table,
            operations:     data.operations,
            filename:       data.filename,
            rows:           data.rows,
            columns:        data.columns,
            text:           data.headline || 'Analysis complete.',
          }]);
        } else {
          setMsgs(p => [...p, { id: uid(), sender: 'ai', time: ts(),
            text: `⚠ Analysis error: ${data.error}` }]);
        }

      /* ══ all other agents / direct / RAG ═════════════════ */
      } else {
        const res  = await fetch('/chat', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: text, rag: ragMode, web_search: webSearch,
            agent_mode: agentMode, agent: usedAgent?.name || null,
          }),
        });
        const data = await res.json();
        setMsgs(p => [...p, {
          id: uid(), sender: 'ai', text: data.response,
          sources: data.sources || [], agent: usedAgent?.name || null, time: ts(),
        }]);
      }

    } catch (err) {
      setMsgs(p => [...p, { id: uid(), sender: 'ai', time: ts(),
        text: '⚠ Something went wrong — please try again.' }]);
    } finally {
      setTyping(false);
    }
  };

  /* ── RAG document upload ── */
  const upload = async e => {
    const file = e.target.files[0]; if (!file) return;
    const fd = new FormData(); fd.append('file', file);
    try {
      const res = await fetch('/upload', { method: 'POST', body: fd });
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
      const res  = await fetch('/remove-file', {
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
      await fetch('/remove-dataset', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      });
      if (activeDataset === filename) setActiveDataset(null);
      await loadDatasets(); notify(`Dataset "${filename}" removed`);
    } catch { notify('Failed to remove dataset', 'err'); }
  };

  /* ════════════════════════ RENDER ════════════════════════ */
  return (
    <div className="shell">

      {/* ════════════ SIDEBAR ════════════ */}
      <aside className={`sidebar ${sidebar ? '' : 'closed'}`}>
        <div className="sb-inner">

          <div className="sb-brand">
            <div className="brand-icon"><i className="fas fa-circle-nodes" /></div>
            <div className="brand-info">
              <div className="brand-name">Lexis</div>
              <div className="brand-sub"> Knowledge AI · TCS ❤️ · v1.0.4β </div>
            </div>
          </div>

          <button className="new-chat-btn" onClick={newChat}>
            <i className="fas fa-plus" /> New conversation
          </button>

          {sessions.length > 0 && (
            <>
              <div className="sb-section-label">History</div>
              <div className="history-list">
                {sessions.map(s => (
                  <div key={s.id} className={`hist-item ${s.id === activeId ? 'active' : ''}`}
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
            <i className="fas fa-arrow-up-from-bracket" /> Upload document
          </label>
          <button className="sb-btn danger" onClick={() => setModal({ type: 'clear' })}>
            <i className="fas fa-trash-can" /> Clear conversation
          </button>

          <div className="sb-sep" />

          {/* RAG Knowledge Base */}
          <div className="kb-section" style={{ maxHeight: '28%' }}>
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
                      <div className="doc-name" title={d.name}>{basename(d.name)}</div>
                      <div className="doc-ext">.{ext || 'file'}</div>
                    </div>
                    <div className="doc-pulse" title="Indexed" />
                    <button className="doc-remove" title="Remove"
                            onClick={() => setModal({ type: 'remove-doc', doc: d })}>
                      <i className="fas fa-trash-can" />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="sb-sep" />

          {/* Agent Datasets */}
          <div className="kb-section" style={{ flex: 1 }}>
            <div className="sb-section-label" style={{ paddingBottom: 0 }}>Agent datasets</div>
            <div className="kb-header">
              <span style={{ fontSize: 11, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>csv / excel</span>
              <span className="kb-count" style={{ background: 'var(--amber-dim)', color: 'var(--amber)', borderColor: 'var(--amber-bdr)' }}>
                {datasets.length}
              </span>
            </div>
            <button className="sb-btn" style={{ marginBottom: 4 }}
                    onClick={() => { setPendingAgent(null); setShowDatasetModal(true); }}>
              <i className="fas fa-table-cells-large" /> Upload dataset
            </button>
            <div className="doc-list">
              {datasetsLoading && [0,1].map(i => <div key={i} className="shimmer" />)}
              {!datasetsLoading && datasets.length === 0 && (
                <div className="doc-empty">
                  <i className="fas fa-chart-pie" />No datasets uploaded.<br />
                  Upload CSV or Excel files<br />for agent use.
                </div>
              )}
              {!datasetsLoading && datasets.map(f => (
                <div className={`doc-item ${activeDataset === f ? 'dataset-active' : ''}`} key={f}
                     onClick={() => setActiveDataset(activeDataset === f ? null : f)}
                     style={{ cursor: 'pointer' }} title="Click to select for agents">
                  <div className="doc-icon csv"><i className="fas fa-file-excel" /></div>
                  <div className="doc-info">
                    <div className="doc-name" title={f}>{basename(f)}</div>
                    <div className="doc-ext">.{extOf(f) || 'file'}</div>
                  </div>
                  {activeDataset === f
                    ? <div style={{ width:8,height:8,borderRadius:'50%',background:'var(--amber)',boxShadow:'0 0 6px var(--amber)',flexShrink:0 }} />
                    : <div style={{ width:8,height:8,borderRadius:'50%',background:'var(--border-default)',flexShrink:0 }} />
                  }
                  <button className="doc-remove" title="Remove dataset"
                          onClick={ev => { ev.stopPropagation(); removeDataset(f); }}>
                    <i className="fas fa-trash-can" />
                  </button>
                </div>
              ))}
            </div>
            {activeDataset && (
              <div className="active-dataset-pill">
                <i className="fas fa-circle-check" />
                <span>{activeDataset}</span>
                <button onClick={() => setActiveDataset(null)} title="Deselect">
                  <i className="fas fa-xmark" />
                </button>
              </div>
            )}
          </div>

          <div className="sb-footer">
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
      <main className="main">

        <div className="topbar">
          <button className="tb-icon-btn" onClick={() => setSidebar(p => !p)} title="Toggle sidebar">
            <i className="fas fa-sidebar" />
          </button>
          <div className="tb-center">
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
          </div>
          <div className="tb-right">
            <div className="tb-status">
              <div className={`status-dot ${typing ? 'thinking' : ''}`} />
              {typing ? 'thinking' : 'ready'}
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="messages">
          {msgs.length === 0 && !typing && (
            <div className="empty">
              <div className="empty-icon"><i className="fas fa-circle-nodes" /></div>
              <div className="empty-title">Lexis</div>
              <div className="empty-sub">
                {agentMode
                  ? 'Agentic mode active — type / to invoke an agent.'
                  : ragMode
                    ? 'Upload documents in the sidebar, then query your knowledge base.'
                    : 'RAG mode off — chatting directly with the language model.'}
              </div>
              <div className="mode-pill">
                <i className={`fas fa-${agentMode ? 'robot' : ragMode ? 'database' : 'comment'}`} />
                {modeLabel} mode
              </div>
              <div className="chips">
                {SUGGESTIONS.map(s => (
                  <div className="chip" key={s} onClick={() => { setInput(s); taRef.current?.focus(); }}>{s}</div>
                ))}
              </div>
            </div>
          )}

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
                    <ChartBubble figure={m.figure} summary={m.summary}
                      filename={m.filename} rows={m.rows} columns={m.columns} />

                  ) : isAnalysis ? (
                    <AnalysisBubble
                      headline={m.headline}
                      narrative={m.narrative}
                      keyFindings={m.keyFindings}
                      recommendation={m.recommendation}
                      statsTable={m.statsTable}
                      primaryTable={m.primaryTable}
                      operations={m.operations}
                      filename={m.filename}
                      rows={m.rows}
                    />

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

              {/* ── Model selector ── */}
              <div className="model-selector-wrap" title="Switch LLM">
                <i className={`fas fa-microchip model-sel-icon ${modelSwitching ? 'fa-spin' : ''}`} />
                <select
                  className="model-selector"
                  value={selectedModel}
                  disabled={modelSwitching}
                  onChange={e => switchModel(e.target.value)}
                >
                  {MODELS.map(m => (
                    <option key={m.id} value={m.id}>{m.label}</option>
                  ))}
                </select>
                <i className="fas fa-chevron-down model-sel-caret" />
              </div>
              {activeDataset && (
                <div className="strip-dataset">
                  <i className="fas fa-table" />
                  <span>{activeDataset}</span>
                  <button onClick={() => setActiveDataset(null)} title="Deselect dataset">
                    <i className="fas fa-xmark" />
                  </button>
                </div>
              )}
              <div className="strip-info">{modeLabel} · {docs.length} docs</div>
            </div>

            <div className="input-wrap">
              {showDrop && <AgentDropdown filter={agentFilter} onSelect={selectAgent} focusIdx={dropFocus} />}
              <div className={`input-box ${isAgentic ? 'agentic' : ''}`}>
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

            <div className="input-meta">
              <span className="input-hint">↵ send · shift+↵ newline{agentMode ? ' · / invoke agent' : ''}</span>
              <span className="input-chars">{input.length > 0 ? `${input.length} chars` : ''}</span>
            </div>
          </div>
        </div>
      </main>

      {/* ════════════ MODALS ════════════ */}

      {showDatasetModal && (
        <DatasetUploadModal
          agent={pendingAgent || AGENTS.find(a => a.needsDataset)}
          datasets={datasets}
          onClose={() => { setShowDatasetModal(false); setPendingAgent(null); }}
          onUploaded={handleDatasetSelected}
          onSelectExisting={handleDatasetSelected}
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
              <button className="m-btn danger"
                      onClick={() => { setMsgs([]); setModal(null); notify('Conversation cleared'); }}>
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