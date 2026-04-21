/* ============================================================
   LEXIS — KNOWLEDGE ASSISTANT  |  app.js
   React 18 · Babel standalone · No build step required
   All Flask endpoints untouched: /chat /upload /files /remove-file
   ============================================================ */

const { useState, useEffect, useRef, useCallback } = React;

/* ── Utilities ─────────────────────────────────────────────── */
const ts  = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
const uid = () => (Date.now() + Math.random()).toString(36);

function extOf(n) {
  const m = n.match(/\.([^.]+)$/);
  return m ? m[1].toLowerCase() : '';
}

function iconFor(e) {
  if (e === 'pdf')                      return ['fa-file-pdf',   'pdf'];
  if (['doc','docx'].includes(e))       return ['fa-file-word',  'doc'];
  if (['txt','md'].includes(e))         return ['fa-file-lines', 'txt'];
  if (['csv','xlsx','xls'].includes(e)) return ['fa-file-excel', 'csv'];
  return ['fa-file', 'other'];
}

function basename(n) {
  const p = n.split('.');
  if (p.length > 1) p.pop();
  return p.join('.');
}

function shortPreview(msgs) {
  const f = msgs.find(m => m.sender === 'user');
  if (!f) return 'New conversation';
  const clean = f.text.replace(/\/[\w-]+/g, '').trim();
  return (clean || f.text).slice(0, 38) + ((clean || f.text).length > 38 ? '…' : '');
}

/* ── Agents ─────────────────────────────────────────────────── */
const AGENTS = [
  { name: 'web-search-agent',          icon: 'fa-globe',        desc: 'Search & retrieve live web results' },
  { name: 'data-analysis-agent',       icon: 'fa-chart-bar',    desc: 'Analyse datasets, tables & numbers' },
  { name: 'create-visualization-agent',icon: 'fa-chart-pie',    desc: 'Generate charts and visual reports' },
  { name: 'summarization-agent',       icon: 'fa-compress-alt', desc: 'Summarise long documents quickly'   },
  { name: 'code-agent',                icon: 'fa-code',         desc: 'Write, debug and explain code'      },
];

const SUGGESTIONS = [
  'Summarise the key findings',
  'What does the policy say about…',
  'List all action items',
  'Compare the two documents',
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
  const q = filter.replace('/', '').toLowerCase();
  const list = AGENTS.filter(a => a.name.includes(q));
  if (!list.length) return null;
  return (
    <div className="agent-dropdown">
      <div className="agent-dropdown-hdr">Agents — ↑↓ navigate · ↵ or Tab select</div>
      {list.map((a, i) => (
        <div
          key={a.name}
          className={`agent-opt ${i === focusIdx ? 'focused' : ''}`}
          onMouseDown={e => { e.preventDefault(); onSelect(a); }}
        >
          <div className="agent-opt-icon"><i className={`fas ${a.icon}`} /></div>
          <div>
            <div className="agent-opt-name">/{a.name}</div>
            <div className="agent-opt-desc">{a.desc}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── Main App ───────────────────────────────────────────────── */
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

  /* Mode toggles */
  const [ragMode,    setRagMode]    = useState(true);
  const [webSearch,  setWebSearch]  = useState(false);
  const [agentMode,  setAgentMode]  = useState(false);

  /* Agent autocomplete */
  const [showDrop,    setShowDrop]    = useState(false);
  const [agentFilter, setAgentFilter] = useState('');
  const [dropFocus,   setDropFocus]   = useState(0);
  const [activeAgent, setActiveAgent] = useState(null);

  const endRef   = useRef(null);
  const taRef    = useRef(null);
  const toastTmr = useRef(null);

  /* ── Derived state ── */
  const currentSession = sessions.find(s => s.id === activeId) || null;
  const msgs = currentSession ? currentSession.msgs : [];
  const isAgentic = agentMode || !!activeAgent;
  const modeLabel = agentMode
    ? (activeAgent ? activeAgent.name : 'agentic')
    : (ragMode ? (webSearch ? 'RAG+Web' : 'RAG') : (webSearch ? 'web' : 'direct'));

  /* ── Persistence ── */
  useEffect(() => { LS.set('lexis_sessions', sessions); }, [sessions]);
  useEffect(() => { LS.set('lexis_active',   activeId); }, [activeId]);
  useEffect(() => { LS.set('lexis_light',    light); },   [light]);
  useEffect(() => { document.body.classList.toggle('light', light); }, [light]);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [msgs, typing]);

  /* ── Ensure a session always exists ── */
  useEffect(() => {
    if (!activeId || !sessions.find(s => s.id === activeId)) {
      if (sessions.length > 0) {
        setActiveId(sessions[0].id);
      } else {
        const id = uid();
        setSessions([{ id, createdAt: ts(), msgs: [] }]);
        setActiveId(id);
      }
    }
  }, []);

  /* ── setMsgs helper (scoped to active session) ── */
  const setMsgs = fn => setSessions(prev =>
    prev.map(s => s.id === activeId
      ? { ...s, msgs: typeof fn === 'function' ? fn(s.msgs) : fn }
      : s)
  );

  /* ── Load docs from Flask /files ── */
  const loadDocs = useCallback(async () => {
    setDocsLoading(true);
    try {
      const res  = await fetch('/files');
      if (!res.ok) throw new Error();
      const data = await res.json();
      setDocs((data.files || []).map((name, i) => ({ id: i, name })));
    } catch {
      setDocs([]);
    } finally {
      setDocsLoading(false);
    }
  }, []);

  useEffect(() => { loadDocs(); }, [loadDocs]);

  /* ── Toast ── */
  const notify = (text, kind = 'ok') => {
    clearTimeout(toastTmr.current);
    setToast({ text, kind });
    toastTmr.current = setTimeout(() => setToast(null), 2800);
  };

  /* ── Session management ── */
  const newChat = () => {
    const id = uid();
    setSessions(prev => [{ id, createdAt: ts(), msgs: [] }, ...prev]);
    setActiveId(id);
    setInput('');
    setActiveAgent(null);
  };

  const switchSession = id => { setActiveId(id); setInput(''); setActiveAgent(null); };

  const deleteSession = id => {
    setSessions(prev => prev.filter(s => s.id !== id));
    if (activeId === id) {
      const rem = sessions.filter(s => s.id !== id);
      if (rem.length) setActiveId(rem[0].id);
      else {
        const nid = uid();
        setSessions([{ id: nid, createdAt: ts(), msgs: [] }]);
        setActiveId(nid);
      }
    }
  };

  /* ── Input + agent autocomplete ── */
  const handleInputChange = e => {
    const val = e.target.value;
    setInput(val);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 140) + 'px';

    const cur    = e.target.selectionStart;
    const before = val.slice(0, cur);
    const slash  = before.match(/(\/[\w-]*)$/);
    if (slash) {
      setAgentFilter(slash[1]);
      setShowDrop(true);
      setDropFocus(0);
    } else {
      setShowDrop(false);
      setAgentFilter('');
    }
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
    setShowDrop(false);
    setAgentFilter('');
    setActiveAgent(agent);
    setTimeout(() => {
      const pos = before.slice(0, si).length + agent.name.length + 2;
      taRef.current?.setSelectionRange(pos, pos);
      taRef.current?.focus();
    }, 0);
  };

  const onKey = e => {
    if (showDrop) {
      const list = AGENTS.filter(a => a.name.includes(agentFilter.replace('/', '').toLowerCase()));
      if (e.key === 'ArrowDown') { e.preventDefault(); setDropFocus(f => Math.min(f + 1, list.length - 1)); return; }
      if (e.key === 'ArrowUp')   { e.preventDefault(); setDropFocus(f => Math.max(f - 1, 0)); return; }
      if (e.key === 'Tab' || (e.key === 'Enter' && list[dropFocus])) { e.preventDefault(); selectAgent(list[dropFocus]); return; }
      if (e.key === 'Escape') { setShowDrop(false); return; }
    }
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  };

  /* ── Send message (Flask /chat untouched) ── */
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
      const res  = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          rag: ragMode,
          web_search: webSearch,
          agent_mode: agentMode,
          agent: usedAgent?.name || null,
        }),
      });
      const data = await res.json();
      setMsgs(p => [...p, {
        id: uid(),
        sender: 'ai',
        text: data.response,
        sources: data.sources || [],
        agent: usedAgent?.name || null,
        time: ts(),
      }]);
    } catch {
      setMsgs(p => [...p, {
        id: uid(), sender: 'ai',
        text: '⚠ Something went wrong — please try again.',
        time: ts(),
      }]);
    } finally {
      setTyping(false);
    }
  };

  /* ── Upload (Flask /upload untouched) ── */
  const upload = async e => {
    const file = e.target.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await fetch('/upload', { method: 'POST', body: fd });
      if (!res.ok) throw new Error();
      await loadDocs();
      notify(`"${file.name}" indexed`);
    } catch {
      notify('Upload failed — please retry', 'err');
    }
    e.target.value = '';
  };

  /* ── Remove doc (Flask /remove-file untouched) ── */
  const confirmRemoveDoc = async () => {
    const doc = modal?.doc;
    if (!doc) return;
    setDeletingDoc(true);
    try {
      setDocs(prev => prev.filter(d => d.name !== doc.name));
      const res = await fetch('/remove-file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: doc.name }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Delete failed');
      await loadDocs();
      setModal(null);
      notify(`"${doc.name}" removed`);
    } catch (err) {
      await loadDocs();
      notify(`Failed to remove "${doc.name}"`, 'err');
    } finally {
      setDeletingDoc(false);
    }
  };

  /* ── Render ── */
  return (
    <div className="shell">

      {/* ════════════════════════ SIDEBAR ════════════════════════ */}
      <aside className={`sidebar ${sidebar ? '' : 'closed'}`}>
        <div className="sb-inner">

          {/* Brand */}
          <div className="sb-brand">
            <div className="brand-icon">
              <i className="fas fa-circle-nodes" />
            </div>
            <div className="brand-info">
              <div className="brand-name">Lexis</div>
              <div className="brand-sub">knowledge assistant · v2</div>
            </div>
          </div>

          {/* New conversation */}
          <button className="new-chat-btn" onClick={newChat}>
            <i className="fas fa-plus" /> New conversation
          </button>

          {/* Session history */}
          {sessions.length > 0 && (
            <>
              <div className="sb-section-label">History</div>
              <div className="history-list">
                {sessions.map(s => (
                  <div
                    key={s.id}
                    className={`hist-item ${s.id === activeId ? 'active' : ''}`}
                    onClick={() => switchSession(s.id)}
                  >
                    <div className="hist-dot" />
                    <div className="hist-label">{shortPreview(s.msgs)}</div>
                    <div className="hist-time">{s.createdAt}</div>
                    <button
                      className="hist-del"
                      onClick={ev => { ev.stopPropagation(); deleteSession(s.id); }}
                      title="Delete session"
                    >
                      <i className="fas fa-xmark" />
                    </button>
                  </div>
                ))}
              </div>
            </>
          )}

          <div className="sb-sep" />

          {/* Workspace */}
          <div className="sb-section-label">Workspace</div>

          <input
            type="file" id="sb-file"
            style={{ display: 'none' }}
            onChange={upload}
            accept=".pdf,.doc,.docx,.txt,.md,.csv,.xlsx"
          />
          <label htmlFor="sb-file" className="sb-btn upload">
            <i className="fas fa-arrow-up-from-bracket" /> Upload document
          </label>

          <button className="sb-btn danger" onClick={() => setModal({ type: 'clear' })}>
            <i className="fas fa-trash-can" /> Clear conversation
          </button>

          <div className="sb-sep" />

          {/* Knowledge base */}
          <div className="kb-section">
            <div className="sb-section-label" style={{ paddingBottom: 0 }}>Knowledge base</div>
            <div className="kb-header">
              <span style={{ fontSize: 11, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>
                indexed docs
              </span>
              <span className="kb-count">{docs.length}</span>
            </div>

            <div className="doc-list">
              {docsLoading && [0, 1, 2].map(i => <div key={i} className="shimmer" />)}

              {!docsLoading && docs.length === 0 && (
                <div className="doc-empty">
                  <i className="fas fa-folder-open" />
                  No documents indexed yet.<br />
                  Upload files to populate your knowledge base.
                </div>
              )}

              {!docsLoading && docs.map(d => {
                const ext = extOf(d.name);
                const [ico, cls] = iconFor(ext);
                return (
                  <div className="doc-item" key={d.id}>
                    <div className={`doc-icon ${cls}`}>
                      <i className={`fas ${ico}`} />
                    </div>
                    <div className="doc-info">
                      <div className="doc-name" title={d.name}>{basename(d.name)}</div>
                      <div className="doc-ext">.{ext || 'file'}</div>
                    </div>
                    <div className="doc-pulse" title="Indexed" />
                    <button
                      className="doc-remove"
                      title="Remove from knowledge base"
                      onClick={() => setModal({ type: 'remove-doc', doc: d })}
                    >
                      <i className="fas fa-trash-can" />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Theme toggle */}
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

      {/* ════════════════════════ MAIN PANEL ════════════════════════ */}
      <main className="main">

        {/* Topbar */}
        <div className="topbar">
          <button className="tb-icon-btn" onClick={() => setSidebar(p => !p)} title="Toggle sidebar">
            <i className="fas fa-sidebar" />
          </button>

          <div className="tb-center">
            <span className="tb-title">
              {currentSession && msgs.length > 0 ? shortPreview(msgs) : 'Conversation'}
            </span>
            <div className="tb-badges">
              <span className={`tb-badge ${ragMode ? 'green' : 'muted'}`}>
                {ragMode ? 'RAG' : 'direct'}
              </span>
              {webSearch && <span className="tb-badge amber">web</span>}
              {isAgentic  && <span className="tb-badge purple">agentic</span>}
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
              <div className="empty-icon">
                <i className="fas fa-circle-nodes" />
              </div>
              <div className="empty-title">Lexis</div>
              <div className="empty-sub">
                {agentMode
                  ? 'Agentic mode active — type / to invoke an agent, or describe what you need.'
                  : ragMode
                    ? 'Upload documents in the sidebar, then query your indexed knowledge base.'
                    : 'RAG mode off — chatting directly with the language model.'}
              </div>
              <div className="mode-pill">
                <i className={`fas fa-${agentMode ? 'robot' : ragMode ? 'database' : 'comment'}`} />
                {modeLabel} mode
              </div>
              <div className="chips">
                {SUGGESTIONS.map(s => (
                  <div className="chip" key={s} onClick={() => { setInput(s); taRef.current?.focus(); }}>
                    {s}
                  </div>
                ))}
              </div>
            </div>
          )}

          {msgs.map(m => {
            const isAgentMsg = m.sender === 'ai' && m.agent;
            const agentDef   = AGENTS.find(a => a.name === m.agent);
            return (
              <div key={m.id} className={`msg-row ${m.sender}`}>
                {m.sender === 'ai' && (
                  <div className={`ai-avatar ${isAgentMsg ? 'agent' : ''}`}>
                    <i className={`fas ${isAgentMsg ? (agentDef?.icon || 'fa-robot') : 'fa-circle-nodes'}`} />
                  </div>
                )}
                <div className="msg-content">
                  <div className="msg-sender">
                    {m.sender === 'ai' ? (m.agent || 'lexis') : 'you'}
                  </div>
                  <div className="bubble">
                    <BubbleContent text={m.text} />
                  </div>
                  {m.sender === 'ai' && m.sources?.length > 0 && (
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
              <div className={`ai-avatar ${agentMode ? 'agent' : ''}`}>
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

            {/* Mode strip */}
            <div className="mode-strip">
              <button
                className={`mode-chip ${ragMode ? 'on-green' : ''}`}
                onClick={() => setRagMode(p => !p)}
                title="Toggle RAG retrieval"
              >
                <i className="fas fa-database" /> RAG
              </button>
              <button
                className={`mode-chip ${webSearch ? 'on-amber' : ''}`}
                onClick={() => setWebSearch(p => !p)}
                title="Toggle web search"
              >
                <i className="fas fa-globe" /> Web
              </button>
              <button
                className={`mode-chip ${agentMode ? 'on-purple' : ''}`}
                onClick={() => {
                  const next = !agentMode;
                  setAgentMode(next);
                  if (next && !input.startsWith('/')) {
                    setInput(prev => prev ? prev : '/');
                    setTimeout(() => { taRef.current?.focus(); taRef.current?.setSelectionRange(1, 1); }, 30);
                  }
                }}
                title="Toggle agentic mode"
              >
                <i className="fas fa-robot" /> Agentic
              </button>
              <div className="strip-info">{modeLabel} · {docs.length} indexed</div>
            </div>

            {/* Input + agent dropdown */}
            <div className="input-wrap">
              {showDrop && (
                <AgentDropdown
                  filter={agentFilter}
                  onSelect={selectAgent}
                  focusIdx={dropFocus}
                />
              )}
              <div className={`input-box ${isAgentic ? 'agentic' : ''}`}>
                <textarea
                  ref={taRef}
                  className="msg-ta"
                  value={input}
                  onChange={handleInputChange}
                  onKeyDown={onKey}
                  onBlur={() => setTimeout(() => setShowDrop(false), 160)}
                  placeholder={
                    agentMode
                      ? 'Type / to invoke an agent, or just ask…'
                      : ragMode
                        ? 'Ask anything about your documents…'
                        : 'Chat directly with the AI…'
                  }
                  rows={1}
                />
                <button
                  className={`send-btn ${isAgentic ? 'agentic' : ''}`}
                  onClick={send}
                  disabled={!input.trim() || typing}
                  title="Send (Enter)"
                >
                  <i className="fas fa-arrow-up" />
                </button>
              </div>
            </div>

            <div className="input-meta">
              <span className="input-hint">
                ↵ send · shift+↵ newline{agentMode ? ' · / invoke agent' : ''}
              </span>
              <span className="input-chars">
                {input.length > 0 ? `${input.length} chars` : ''}
              </span>
            </div>
          </div>
        </div>

      </main>

      {/* ════════════════════════ MODALS ════════════════════════ */}

      {/* Clear conversation */}
      {modal?.type === 'clear' && (
        <div className="backdrop" onClick={() => setModal(null)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-icon err"><i className="fas fa-trash-can" /></div>
            <div className="modal-title">Clear this conversation?</div>
            <div className="modal-body">
              All messages in the current session will be permanently removed.
              Your indexed knowledge base documents remain unaffected.
            </div>
            <div className="modal-row">
              <button className="m-btn" onClick={() => setModal(null)}>Cancel</button>
              <button
                className="m-btn danger"
                onClick={() => { setMsgs([]); setModal(null); notify('Conversation cleared'); }}
              >
                Clear chat
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Remove document */}
      {modal?.type === 'remove-doc' && (
        <div className="backdrop" onClick={() => !deletingDoc && setModal(null)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-icon warn"><i className="fas fa-file-xmark" /></div>
            <div className="modal-title">
              {deletingDoc ? 'Removing file…' : 'Remove from knowledge base?'}
            </div>
            <div className="modal-body">
              {deletingDoc
                ? 'Please wait while we update the knowledge base.'
                : 'This will unindex the file. The model will no longer retrieve information from it.'}
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
                  <i className="fas fa-spinner fa-spin" style={{ marginRight: 6 }} />
                  Deleting…
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ════════════════════════ TOAST ════════════════════════ */}
      {toast && (
        <div className="toast-wrap">
          <div className="toast">
            <div className={`tdot ${toast.kind}`} />
            {toast.text}
          </div>
        </div>
      )}

    </div>
  );
}

/* ── Mount ── */
ReactDOM.createRoot(document.getElementById('root')).render(<App />);