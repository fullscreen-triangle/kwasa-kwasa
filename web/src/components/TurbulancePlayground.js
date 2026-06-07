import React, { useState, useRef, useCallback, useMemo, useEffect } from "react";
import {
  Files, Search, GitBranch, Play, Blocks, Settings, ChevronRight, ChevronDown,
  X, Circle, FileCode2, FileText, Folder, FolderOpen,
  AlertCircle, Bell, Check, Eye, Code2, Lightbulb,
} from "lucide-react";
import CodeMirror from "@uiw/react-codemirror";
import { vscodeDark } from "@uiw/codemirror-theme-vscode";
import { run, tbToString } from "../lib/turbulance";
import { examples } from "../lib/turbulance/examples";
import { turbulance } from "../lib/turbulance/codemirror";

/* ------------------------------------------------------------------ *
 *  THEME — VS Code dark. Retheme in one place.                        *
 * ------------------------------------------------------------------ */
const theme = {
  titlebar: "#3c3c3c", activitybar: "#333333", activitybarFg: "#858585",
  activitybarFgActive: "#ffffff", sidebar: "#252526", sidebarFg: "#cccccc",
  sidebarHeader: "#bbbbbb", editor: "#1e1e1e", editorFg: "#d4d4d4",
  tabBar: "#252526", tabActive: "#1e1e1e", tabInactive: "#2d2d2d",
  tabFg: "#969696", tabFgActive: "#ffffff", border: "#3c3c3c",
  accent: "#0e639c", accentBright: "#007acc", statusBar: "#007acc",
  statusFg: "#ffffff", panel: "#1e1e1e", gutter: "#858585",
  lineActive: "#2a2d2e", selection: "#264f78",
};

const verdictColor = (v) =>
  v === "Supported" ? "#4ec98a" : v === "Contradicted" ? "#f48771" : "#dcb360";

/* ------------------------------------------------------------------ *
 *  FILE SYSTEM — the tutorial scripts (flat .trb files).              *
 * ------------------------------------------------------------------ */
const initialFiles = examples.reduce((acc, ex) => {
  acc[`${ex.id}.trb`] = { type: "file", lang: "trb", content: ex.code, title: ex.title, description: ex.description };
  return acc;
}, {});
const firstFile = `${examples[0].id}.trb`;

/* ------------------------------------------------------------------ *
 *  Helpers                                                            *
 * ------------------------------------------------------------------ */
const fileIcon = (name) => {
  if (name.endsWith(".trb")) return { Icon: FileCode2, color: "#58E6D9" };
  if (name.endsWith(".md")) return { Icon: FileText, color: "#519aba" };
  return { Icon: FileText, color: "#858585" };
};
const langLabel = (lang) => ({ trb: "Turbulance", md: "Markdown" }[lang] || "Plain Text");
const getNode = (tree, path) => { let n = { children: tree }; for (const p of path) { n = n.children?.[p]; if (!n) return null; } return n; };

/* ------------------------------------------------------------------ *
 *  File tree                                                          *
 * ------------------------------------------------------------------ */
function Tree({ tree, expanded, toggle, activePath, openFile, depth = 0, path = [] }) {
  const entries = Object.entries(tree).sort((a, b) =>
    a[1].type !== b[1].type ? (a[1].type === "folder" ? -1 : 1) : a[0].localeCompare(b[0]));
  return (
    <>
      {entries.map(([name, node]) => {
        const fullPath = [...path, name];
        const key = fullPath.join("/");
        const isFolder = node.type === "folder";
        const isOpen = expanded.has(key);
        const isActive = activePath === key;
        const { Icon, color } = isFolder ? { Icon: isOpen ? FolderOpen : Folder, color: "#90a4ae" } : fileIcon(name);
        return (
          <div key={key}>
            <button
              onClick={() => (isFolder ? toggle(key) : openFile(fullPath))}
              className="flex w-full items-center gap-1 py-0.5 pr-2 text-left text-[13px] leading-relaxed transition-colors"
              style={{ paddingLeft: 8 + depth * 12, color: theme.sidebarFg, background: isActive ? theme.lineActive : "transparent" }}
              onMouseEnter={(e) => { if (!isActive) e.currentTarget.style.background = "#2a2d2e"; }}
              onMouseLeave={(e) => { if (!isActive) e.currentTarget.style.background = "transparent"; }}
            >
              {isFolder ? (isOpen ? <ChevronDown size={14} className="shrink-0 opacity-70" /> : <ChevronRight size={14} className="shrink-0 opacity-70" />) : <span className="w-[14px] shrink-0" />}
              <Icon size={15} className="shrink-0" style={{ color }} />
              <span className="truncate">{name}</span>
            </button>
            {isFolder && isOpen && (
              <Tree tree={node.children} expanded={expanded} toggle={toggle} activePath={activePath} openFile={openFile} depth={depth + 1} path={fullPath} />
            )}
          </div>
        );
      })}
    </>
  );
}

/* ------------------------------------------------------------------ *
 *  Editor (line-numbered textarea). Swap for CodeMirror later.        *
 * ------------------------------------------------------------------ */
function CodeEditor({ value, onChange, onCursor }) {
  return (
    <div className="min-h-0 flex-1 overflow-hidden" style={{ background: theme.editor }}>
      <CodeMirror
        value={value}
        height="100%"
        theme={vscodeDark}
        extensions={[turbulance()]}
        onChange={(val) => onChange(val)}
        onUpdate={(vu) => {
          const pos = vu.state.selection.main.head;
          const line = vu.state.doc.lineAt(pos);
          onCursor({ ln: line.number, col: pos - line.from + 1 });
        }}
        basicSetup={{
          lineNumbers: true,
          foldGutter: false,
          highlightActiveLine: true,
          autocompletion: false,
          indentOnInput: false,
          tabSize: 4,
        }}
        style={{ height: "100%", fontSize: "13px" }}
      />
    </div>
  );
}

/* ------------------------------------------------------------------ *
 *  Result rendering                                                   *
 * ------------------------------------------------------------------ */
function ScoreBar({ score, color }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full" style={{ background: "#333" }}>
      <div className="h-full rounded-full transition-all" style={{ width: `${Math.round(score * 100)}%`, background: color }} />
    </div>
  );
}

function ResultView({ result }) {
  if (!result) return <div className="px-3 pt-3 text-[12px]" style={{ color: "#5a5a5a" }}>Press Run to evaluate.</div>;
  const { error, output, propositions, points } = result;
  return (
    <div className="h-full overflow-y-auto p-3 text-[12px]" style={{ color: theme.editorFg }}>
      {error && (
        <div className="mb-3 flex items-start gap-2 rounded border px-3 py-2" style={{ borderColor: "#5a2d2d", background: "#3a1d1d", color: "#f48771" }}>
          <AlertCircle size={14} className="mt-0.5 shrink-0" />
          <span className="font-mono">{error.message}{error.line ? ` (line ${error.line})` : ""}</span>
        </div>
      )}
      {output && output.length > 0 && (
        <div className="mb-4">
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider" style={{ color: "#7a7a7a" }}>Console</div>
          <pre className="whitespace-pre-wrap rounded p-2 font-mono text-[12px] leading-relaxed" style={{ background: "#161616" }}>{output.join("\n")}</pre>
        </div>
      )}
      {propositions && propositions.length > 0 && (
        <div className="mb-4">
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider" style={{ color: "#7a7a7a" }}>Propositions</div>
          {propositions.map((p, i) => (
            <div key={i} className="mb-3 rounded border p-3" style={{ borderColor: theme.border, background: "#232323" }}>
              <div className="mb-2 flex items-center justify-between">
                <span className="font-semibold" style={{ color: "#fff" }}>{p.name}</span>
                <span className="rounded px-2 py-0.5 text-[11px] font-medium" style={{ background: verdictColor(p.verdict) + "22", color: verdictColor(p.verdict) }}>
                  {p.verdict} · {p.score.toFixed(2)}
                </span>
              </div>
              <ScoreBar score={p.score} color={verdictColor(p.verdict)} />
              <div className="mt-3 space-y-2">
                {p.motions.map((m, j) => (
                  <div key={j}>
                    <div className="mb-1 flex items-center justify-between">
                      <span style={{ color: "#cfcfcf" }}>{m.name}</span>
                      <span style={{ color: verdictColor(m.verdict) }}>{m.verdict} · {m.score.toFixed(2)}</span>
                    </div>
                    <ScoreBar score={m.score} color={verdictColor(m.verdict)} />
                    <div className="mt-0.5 text-[11px] italic" style={{ color: "#888" }}>{m.desc}</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
      {points && points.length > 0 && (
        <div className="mb-4">
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider" style={{ color: "#7a7a7a" }}>Resolved points</div>
          {points.map((pt, i) => (
            <div key={i} className="mb-2 rounded border px-3 py-2" style={{ borderColor: theme.border, background: "#232323" }}>
              <div style={{ color: "#fff" }}>{tbToString(pt.value)}</div>
              <div className="mt-1 flex items-center gap-2">
                <span className="text-[11px]" style={{ color: "#888" }}>confidence {pt.confidence.toFixed(2)}</span>
                <div className="flex-1"><ScoreBar score={pt.confidence} color="#58E6D9" /></div>
              </div>
            </div>
          ))}
        </div>
      )}
      {!error && (!output || !output.length) && (!propositions || !propositions.length) && (!points || !points.length) && (
        <div style={{ color: "#5a5a5a" }}>Ran with no output. Try a print(...) or a proposition.</div>
      )}
    </div>
  );
}

function OutputColumn({ result, onRun }) {
  const [tab, setTab] = useState("output");
  const tabs = [{ id: "output", label: "Output", Icon: Eye }, { id: "ast", label: "AST", Icon: Code2 }];
  return (
    <div className="flex min-w-0 flex-1 flex-col" style={{ background: theme.editor, borderLeft: `1px solid ${theme.border}` }}>
      <div className="flex h-9 shrink-0 items-center justify-between pr-2" style={{ background: theme.tabInactive }}>
        <div className="flex h-full">
          {tabs.map(({ id, label, Icon }) => {
            const active = tab === id;
            return (
              <button key={id} onClick={() => setTab(id)} className="relative flex items-center gap-1.5 px-3 text-[12px] transition-colors"
                style={{ color: active ? theme.tabFgActive : theme.tabFg, background: active ? theme.tabActive : "transparent" }}>
                <Icon size={13} /> {label}
                {active && <span className="absolute left-0 top-0 h-0.5 w-full" style={{ background: theme.accentBright }} />}
              </button>
            );
          })}
        </div>
        <button onClick={onRun} title="Run (Ctrl+Enter)" className="flex h-6 items-center gap-1 rounded px-2 text-[12px]" style={{ background: theme.accent, color: "#fff" }}>
          <Play size={12} /> Run
        </button>
      </div>
      <div className="min-h-0 flex-1">
        {tab === "output" && <ResultView result={result} />}
        {tab === "ast" && (
          <pre className="h-full overflow-auto p-3 font-mono text-[11px] leading-[1.45]" style={{ color: theme.editorFg }}>
            {result?.ast ? JSON.stringify(result.ast, null, 2) : "AST appears here after a run."}
          </pre>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 *  Main shell                                                         *
 * ------------------------------------------------------------------ */
export default function TurbulancePlayground() {
  const [files, setFiles] = useState(initialFiles);
  const [expanded, setExpanded] = useState(new Set());
  const [openTabs, setOpenTabs] = useState([[firstFile]]);
  const [activeTab, setActiveTab] = useState(firstFile);
  const [dirty, setDirty] = useState(new Set());
  const [sidebar, setSidebar] = useState(true);
  const [activity, setActivity] = useState("files");
  const [cursor, setCursor] = useState({ ln: 1, col: 1 });
  const [result, setResult] = useState(null);

  const [editorWidth, setEditorWidth] = useState(55);
  const splitRef = useRef(null);
  const dragging = useRef(false);

  const activePathArr = useMemo(() => openTabs.find((t) => t.join("/") === activeTab) || null, [openTabs, activeTab]);
  const activeNode = activePathArr ? getNode(files, activePathArr) : null;

  const doRun = useCallback(() => {
    if (!activeNode) return;
    setResult(run(activeNode.content));
  }, [activeNode]);

  // auto-run (debounced) on edit / file switch
  useEffect(() => {
    const t = setTimeout(() => { if (activeNode) setResult(run(activeNode.content)); }, 350);
    return () => clearTimeout(t);
  }, [activeNode]);

  // Ctrl/Cmd+Enter to run
  useEffect(() => {
    const h = (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); doRun(); } };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [doRun]);

  // splitter drag
  useEffect(() => {
    const move = (e) => {
      if (!dragging.current || !splitRef.current) return;
      const r = splitRef.current.getBoundingClientRect();
      setEditorWidth(Math.min(80, Math.max(25, ((e.clientX - r.left) / r.width) * 100)));
    };
    const up = () => { dragging.current = false; document.body.style.cursor = ""; };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
  }, []);

  const toggleFolder = useCallback((key) => setExpanded((p) => { const n = new Set(p); n.has(key) ? n.delete(key) : n.add(key); return n; }), []);
  const openFile = useCallback((pathArr) => {
    const key = pathArr.join("/");
    setOpenTabs((p) => (p.some((t) => t.join("/") === key) ? p : [...p, pathArr]));
    setActiveTab(key);
  }, []);
  const closeTab = useCallback((key, e) => {
    e.stopPropagation();
    setOpenTabs((p) => {
      const next = p.filter((t) => t.join("/") !== key);
      if (activeTab === key) setActiveTab(next.length ? next[next.length - 1].join("/") : null);
      return next;
    });
    setDirty((p) => { const n = new Set(p); n.delete(key); return n; });
  }, [activeTab]);

  const updateContent = useCallback((val) => {
    if (!activePathArr) return;
    setFiles((prev) => { const next = structuredClone(prev); const node = getNode(next, activePathArr); node.content = val; return next; });
    setDirty((p) => new Set(p).add(activeTab));
  }, [activePathArr, activeTab]);

  const activities = [
    { id: "files", Icon: Files, label: "Tutorials" },
    { id: "search", Icon: Search, label: "Search" },
    { id: "git", Icon: GitBranch, label: "Source Control" },
    { id: "run", Icon: Play, label: "Run" },
    { id: "ext", Icon: Blocks, label: "Extensions" },
  ];

  return (
    <div className="flex h-full w-full flex-col overflow-hidden rounded-lg text-sm shadow-2xl"
      style={{ background: theme.editor, color: theme.editorFg, border: `1px solid ${theme.border}` }}>
      {/* Title bar */}
      <div className="flex h-9 shrink-0 items-center justify-between px-3" style={{ background: theme.titlebar }}>
        <div className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full" style={{ background: "#ff5f56" }} />
          <span className="h-3 w-3 rounded-full" style={{ background: "#ffbd2e" }} />
          <span className="h-3 w-3 rounded-full" style={{ background: "#27c93f" }} />
        </div>
        <span className="text-xs" style={{ color: "#cccccc" }}>kwasa-kwasa — Turbulance Playground</span>
        <div className="w-12" />
      </div>

      <div className="flex min-h-0 flex-1">
        {/* Activity bar */}
        <div className="flex w-12 shrink-0 flex-col items-center justify-between py-2" style={{ background: theme.activitybar }}>
          <div className="flex flex-col items-center gap-1">
            {activities.map(({ id, Icon, label }) => {
              const active = activity === id;
              return (
                <button key={id} title={label}
                  onClick={() => { if (active) setSidebar((s) => !s); else { setActivity(id); setSidebar(true); } }}
                  className="relative flex h-11 w-12 items-center justify-center transition-colors"
                  style={{ color: active ? theme.activitybarFgActive : theme.activitybarFg }}>
                  {active && <span className="absolute left-0 top-1/2 h-6 w-0.5 -translate-y-1/2" style={{ background: "#ffffff" }} />}
                  <Icon size={24} strokeWidth={1.5} />
                </button>
              );
            })}
          </div>
          <button title="Settings" className="flex h-11 w-12 items-center justify-center" style={{ color: theme.activitybarFg }}>
            <Settings size={24} strokeWidth={1.5} />
          </button>
        </div>

        {/* Sidebar */}
        {sidebar && (
          <div className="flex w-60 shrink-0 flex-col overflow-hidden" style={{ background: theme.sidebar, borderRight: `1px solid ${theme.border}` }}>
            <div className="flex h-9 shrink-0 items-center px-4 text-[11px] font-medium uppercase tracking-wider" style={{ color: theme.sidebarHeader }}>
              {activities.find((a) => a.id === activity)?.label}
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto pb-2">
              {activity === "files" ? (
                <Tree tree={files} expanded={expanded} toggle={toggleFolder} activePath={activeTab} openFile={openFile} />
              ) : (
                <div className="px-4 py-6 text-[13px]" style={{ color: theme.tabFg }}>{activities.find((a) => a.id === activity)?.label} panel</div>
              )}
            </div>
          </div>
        )}

        {/* Editor + Output split */}
        <div ref={splitRef} className="flex min-w-0 flex-1">
          <div className="flex min-w-0 flex-col" style={{ width: `${editorWidth}%` }}>
            {/* tabs */}
            <div className="flex h-9 shrink-0 items-stretch overflow-x-auto" style={{ background: theme.tabInactive }}>
              {openTabs.map((pathArr) => {
                const key = pathArr.join("/");
                const name = pathArr[pathArr.length - 1];
                const active = key === activeTab;
                const isDirty = dirty.has(key);
                const { Icon, color } = fileIcon(name);
                return (
                  <div key={key} onClick={() => setActiveTab(key)}
                    className="group flex cursor-pointer items-center gap-2 border-r px-3 text-[13px]"
                    style={{ background: active ? theme.tabActive : theme.tabInactive, color: active ? theme.tabFgActive : theme.tabFg, borderColor: theme.border, borderTop: active ? `1px solid ${theme.accentBright}` : "1px solid transparent" }}>
                    <Icon size={15} style={{ color }} />
                    <span className="whitespace-nowrap">{name}</span>
                    <button onClick={(e) => closeTab(key, e)} className="flex h-5 w-5 items-center justify-center rounded" style={{ color: active ? theme.tabFgActive : theme.tabFg }}>
                      {isDirty ? <Circle size={9} fill="currentColor" className="group-hover:hidden" /> : null}
                      <X size={15} className={isDirty ? "hidden group-hover:block" : "opacity-0 group-hover:opacity-100"} />
                    </button>
                  </div>
                );
              })}
            </div>

            {/* description banner */}
            {activeNode?.description && (
              <div className="flex items-start gap-2 px-4 py-2 text-[12px]" style={{ background: "#202020", color: "#9a9a9a", borderBottom: `1px solid ${theme.border}` }}>
                <Lightbulb size={14} className="mt-0.5 shrink-0" style={{ color: "#58E6D9" }} />
                <span><span style={{ color: "#cfcfcf" }}>{activeNode.title}</span> — {activeNode.description}</span>
              </div>
            )}

            {activeNode ? (
              <CodeEditor value={activeNode.content} onChange={updateContent} onCursor={setCursor} />
            ) : (
              <div className="flex min-h-0 flex-1 items-center justify-center text-sm" style={{ background: theme.editor, color: "#5a5a5a" }}>Open a tutorial from the sidebar</div>
            )}
          </div>

          {/* Splitter */}
          <div onMouseDown={() => { dragging.current = true; document.body.style.cursor = "col-resize"; }}
            className="w-1 shrink-0 cursor-col-resize" style={{ background: theme.border }} title="Drag to resize" />

          {/* Output */}
          <OutputColumn result={result} onRun={doRun} />
        </div>
      </div>

      {/* Status bar */}
      <div className="flex h-6 shrink-0 items-center justify-between px-3 text-[12px]" style={{ background: theme.statusBar, color: theme.statusFg }}>
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1"><GitBranch size={13} /> turbulance</span>
          <span className="flex items-center gap-1">
            {result?.error ? <><AlertCircle size={13} /> 1</> : <><Check size={13} /> 0</>}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span>Ln {cursor.ln}, Col {cursor.col}</span>
          <span>Spaces: 4</span><span>UTF-8</span>
          <span>{activeNode ? langLabel(activeNode.lang) : "—"}</span>
          <Bell size={13} />
        </div>
      </div>
    </div>
  );
}
