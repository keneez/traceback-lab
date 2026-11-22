import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  api,
  Case,
  caseQuery,
  demoTitle,
  emptyQuery,
  Investigation,
  Job,
  labels,
  ModelStatus,
  shortRev,
  Snapshot,
  terminal,
} from "./data";
import { Empty, ErrorBox, Icon, IconName, ThemeButton } from "./ui";
import { ImportDialog } from "./ImportDialog";
import { Workbench } from "./Workbench";
import { ChecksPanel } from "./ChecksPanel";
import { HistoryView, LabView } from "./LibraryViews";
import "./fonts.css";
import "./style.css";

type Page = "work" | "history" | "lab";
const pages: [Page, string, IconName][] = [
  ["work", "Расследование", "search"],
  ["history", "История", "history"],
  ["lab", "Лаборатория", "lab"],
];
const basePath = () => window.location.pathname;
function runUrl(id?: string) {
  window.history.replaceState(
    null,
    "",
    basePath() + (id ? "?run=" + encodeURIComponent(id) : ""),
  );
}
function projectUrl(id: string, view: "query" | "checks") {
  window.history.replaceState(
    null,
    "",
    basePath() + "?" + new URLSearchParams({ project: id, view }),
  );
}
function matchesCase(item: Case, q: Investigation) {
  return (
    item.snapshot_id === q.snapshot_id &&
    item.title === q.title &&
    item.description === q.description
  );
}
function App() {
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]),
    [cases, setCases] = useState<Case[]>([]),
    [models, setModels] = useState<ModelStatus | null>(null);
  const [page, setPage] = useState<Page>("work"),
    [workspace, setWorkspace] = useState<"query" | "checks">("query");
  const [query, setQuery] = useState<Investigation>(emptyQuery()),
    [caseId, setCaseId] = useState(""),
    [submittedCase, setSubmittedCase] = useState<Case | undefined>();
  const [job, setJob] = useState<Job | null>(null),
    [starting, setStarting] = useState(false),
    [ready, setReady] = useState(false),
    [error, setError] = useState("");
  const [importing, setImporting] = useState(false),
    [traceOpen, setTraceOpen] = useState(false),
    [optionsOpen, setOptionsOpen] = useState(false);
  const request = useRef<AbortController | null>(null),
    jobId = useRef(""),
    active = useRef(false);
  const snapshot = snapshots.find((s) => s.id === query.snapshot_id);
  const running = starting || !!(job && !terminal(job.status));
  const update = <K extends keyof Investigation>(
    key: K,
    value: Investigation[K],
  ) => setQuery((q) => ({ ...q, [key]: value }));
  function stopActive() {
    request.current?.abort();
    request.current = null;
    if (active.current && jobId.current)
      void fetch("/v1/investigations/" + jobId.current, {
        method: "DELETE",
      }).catch(() => {});
    active.current = false;
    jobId.current = "";
  }
  function resetWork() {
    stopActive();
    setStarting(false);
    setJob(null);
    setError("");
    setSubmittedCase(undefined);
    setTraceOpen(false);
    runUrl();
  }
  async function track(task: Job, abort: AbortController) {
    if (abort.signal.aborted) return;
    setStarting(false);
    jobId.current = task.id;
    runUrl(task.id);
    while (!abort.signal.aborted) {
      setJob(task);
      if (terminal(task.status)) {
        active.current = false;
        if (task.status === "failed")
          setError(task.error || "Не удалось выполнить поиск");
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 280));
      if (abort.signal.aborted) return;
      task = await api<Job>("/v1/investigations/" + task.id, {
        signal: abort.signal,
      });
    }
  }
  async function execute(body: Investigation, example?: Case) {
    stopActive();
    const abort = new AbortController();
    request.current = abort;
    active.current = true;
    setJob(null);
    setStarting(true);
    setError("");
    setSubmittedCase(
      example && matchesCase(example, body) ? example : undefined,
    );
    runUrl();
    try {
      // If navigation happens during submission, retain the returned id long enough to cancel it.
      const task = await api<Job>("/v1/investigations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (abort.signal.aborted) {
        if (!terminal(task.status))
          void fetch("/v1/investigations/" + task.id, {
            method: "DELETE",
          }).catch(() => {});
        return;
      }
      await track(task, abort);
    } catch (e) {
      if (!abort.signal.aborted) {
        active.current = false;
        setStarting(false);
        setError((e as Error).message);
        setJob(null);
      }
    }
  }
  function chooseCase(item: Case, base = query) {
    const next = caseQuery(item, base);
    setQuery(next);
    setCaseId(item.id);
    setTraceOpen(!!next.traceback);
    setPage("work");
    setWorkspace("query");
    void execute(next, item);
  }
  async function restore(id: string, examples = cases) {
    resetWork();
    const abort = new AbortController();
    request.current = abort;
    setPage("work");
    setWorkspace("query");
    setStarting(true);
    try {
      const task = await api<Job>(
        "/v1/investigations/" + encodeURIComponent(id),
        { signal: abort.signal },
      );
      if (abort.signal.aborted) return;
      const q = task.request;
      const example = examples.find((item) => matchesCase(item, q));
      setQuery(q);
      setCaseId(example?.id || "");
      setSubmittedCase(example);
      setTraceOpen(!!q.traceback);
      active.current = !terminal(task.status);
      await track(task, abort);
    } catch (e) {
      if (!abort.signal.aborted) {
        setError((e as Error).message);
        setStarting(false);
        active.current = false;
      }
    }
  }
  useEffect(() => {
    const c = new AbortController();
    const params = new URLSearchParams(window.location.search);
    const refId = params.get("run");
    const projectId = params.get("project");
    Promise.all([
      api<Snapshot[]>("/v1/snapshots?compact=true", { signal: c.signal }),
      api<Case[]>("/v1/cases", { signal: c.signal }),
      api<ModelStatus>("/v1/models", { signal: c.signal }),
    ])
      .then(([snaps, examples, status]) => {
        if (c.signal.aborted) return;
        setSnapshots(snaps);
        setCases(examples);
        setModels(status);
        setReady(true);
        const base = {
          ...emptyQuery(snaps[0]?.id),
          mode: status.encoder_name === "UniXcoder" ? "hybrid" : "context",
        };
        setQuery(base);
        if (refId) {
          void restore(refId, examples);
        } else if (projectId && snaps.some((s) => s.id === projectId)) {
          setQuery({ ...base, snapshot_id: projectId });
          setWorkspace(params.get("view") === "checks" ? "checks" : "query");
        } else if (examples.length) {
          chooseCase(examples[0], base);
        }
      })
      .catch((e) => {
        if (!c.signal.aborted) setError(e.message);
      });
    return () => {
      c.abort();
      stopActive();
    };
  }, []);
  function chooseSnapshot(id: string, checks = false) {
    resetWork();
    projectUrl(id, checks ? "checks" : "query");
    setWorkspace(checks ? "checks" : "query");
    setPage("work");
    setCaseId("");
    setOptionsOpen(false);
    setQuery({
      ...emptyQuery(id),
      mode: query.mode,
      graph: query.graph,
      profile: query.profile,
      rerank: query.rerank,
    });
  }
  function cancel() {
    stopActive();
    setStarting(false);
    setError("");
    setJob((last) =>
      last ? { ...last, status: "cancelled", result: null } : null,
    );
  }
  const canSearch = !!(
    snapshot?.symbol_count &&
    (query.mode === "traceback"
      ? query.traceback.trim()
      : (query.title + query.description + query.traceback).trim())
  );
  const dirty =
    !!job &&
    JSON.stringify(Object.entries(job.request).sort()) !==
      JSON.stringify(Object.entries(query).sort());
  return (
    <div className="app">
      <a href="#main" className="skip">
        К содержимому
      </a>
      <header className="appHeader">
        <div className="headerInner">
          <button
            className="brand"
            aria-label="Traceback Lab, расследование"
            onClick={() => setPage("work")}
          >
            <span className="brandMark">
              <Icon name="terminal" size={25} />
            </span>
            <span>
              Traceback<span className="brandSub"> Lab</span>
            </span>
          </button>
          <nav aria-label="Разделы приложения">
            {pages.map(([id, t, icon]) => (
              <button
                key={id}
                aria-current={page === id ? "page" : undefined}
                onClick={() => setPage(id)}
              >
                <Icon name={icon} size={17} />
                {t}
              </button>
            ))}
          </nav>
          <ThemeButton />
        </div>
      </header>
      <main id="main">
        <div className="serviceLine">
          <span className="serviceLight" />
          {ready ? "Локальный сервис" : "Подключение к сервису"}
          {models?.encoder_name && (
            <>
              <i /> {models.encoder_name}
            </>
          )}
        </div>
        {error && <ErrorBox text={error} />}
        {page === "work" && (
          <>
            <div className="pageHeading">
              <div>
                <h1>Расследование бага</h1>
                <p>
                  Найдите в Python-проекте функции, которые стоит проверить.
                </p>
              </div>
              <button
                className="primary importLaunch"
                onClick={() => setImporting(true)}
              >
                <Icon name="plus" />
                Подключить код
              </button>
            </div>
            <div className="repoStrip">
              <span className="repoGlyph">
                <Icon name="folder" size={21} />
              </span>
              <div className="repoSelect">
                <label htmlFor="snapshot">Снимок проекта</label>
                <select
                  id="snapshot"
                  value={query.snapshot_id}
                  onChange={(e) => chooseSnapshot(e.target.value)}
                >
                  {snapshots.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.repository} · {shortRev(s)}
                    </option>
                  ))}
                </select>
              </div>
              {snapshot && (
                <div className="repoStats">
                  <span>
                    Файлов <b>{snapshot.file_count}</b>
                  </span>
                  <span>
                    Функций <b>{snapshot.symbol_count}</b>
                  </span>
                  <span>
                    Связей <b>{snapshot.edge_count}</b>
                  </span>
                </div>
              )}
            </div>
            <nav className="workspaceTabs" aria-label="Режим работы">
              {(
                [
                  ["query", "По симптому", "search"],
                  ["checks", "Проверки кода", "scan"],
                ] as const
              ).map(([id, t, icon]) => (
                <button
                  key={id}
                  aria-pressed={workspace === id}
                  onClick={() => {
                    setWorkspace(id);
                    if (id === "query" && job?.id) runUrl(job.id);
                    else if (snapshot) projectUrl(snapshot.id, id);
                  }}
                >
                  <Icon name={icon} size={16} />
                  {t}
                </button>
              ))}
            </nav>
            {workspace === "checks" && snapshot ? (
              <ChecksPanel key={snapshot.id} snapshot={snapshot} />
            ) : (
              workspace === "query" && (
                <>
                  <div className="inputGrid">
                    <form
                      className="panel composer"
                      onSubmit={(e) => {
                        e.preventDefault();
                        if (canSearch && !running)
                          void execute(
                            query,
                            cases.find((c) => c.id === caseId),
                          );
                      }}
                      onKeyDown={(e) => {
                        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                          e.preventDefault();
                          e.currentTarget.requestSubmit();
                        }
                      }}
                    >
                      <div className="panelHead">
                        <h2>
                          <Icon name="issue" />
                          Описание проблемы
                        </h2>
                      </div>
                      <div className="fields">
                        <label className="field" htmlFor="title">
                          Симптом
                        </label>
                        <input
                          id="title"
                          placeholder="Connection cannot be reused after timeout"
                          value={query.title}
                          onChange={(e) => update("title", e.target.value)}
                          maxLength={1000}
                        />
                        <label className="field" htmlFor="description">
                          Подробности <small>текст issue на английском</small>
                        </label>
                        <textarea
                          id="description"
                          placeholder="Что произошло и какого поведения вы ожидали?"
                          value={query.description}
                          onChange={(e) =>
                            update("description", e.target.value)
                          }
                          maxLength={60000}
                        />
                        <button
                          type="button"
                          className="traceToggle"
                          aria-expanded={traceOpen}
                          onClick={() => setTraceOpen(!traceOpen)}
                        >
                          <Icon name={traceOpen ? "down" : "plus"} size={14} />
                          {traceOpen ? "Стек вызовов" : "Добавить traceback"}
                        </button>
                        {traceOpen && (
                          <textarea
                            className="traceInput"
                            aria-label="Traceback"
                            placeholder={
                              'File "client.py", line 42, in request\nTimeoutError: ...'
                            }
                            value={query.traceback}
                            onChange={(e) =>
                              setQuery((q) => ({
                                ...q,
                                traceback: e.target.value,
                                trace_revision: null,
                              }))
                            }
                            maxLength={60000}
                          />
                        )}
                      </div>
                      {optionsOpen && (
                        <div className="options">
                          <label className="field">
                            Метод поиска
                            <select
                              aria-label="Метод поиска"
                              value={query.mode}
                              onChange={(e) => update("mode", e.target.value)}
                            >
                              {[
                                "context",
                                "bm25",
                                "hybrid",
                                "dense",
                                "traceback",
                              ].map((mode) => (
                                <option
                                  value={mode}
                                  key={mode}
                                  disabled={
                                    ["dense", "hybrid"].includes(mode) &&
                                    !models?.encoder_loaded
                                  }
                                >
                                  {labels[mode]}
                                </option>
                              ))}
                            </select>
                          </label>
                          <label className="field">
                            Бюджет
                            <select
                              aria-label="Бюджет поиска"
                              value={query.profile}
                              onChange={(e) =>
                                update("profile", e.target.value)
                              }
                            >
                              <option value="quality">50 кандидатов</option>
                              <option value="fast">20 кандидатов</option>
                            </select>
                          </label>
                          <label className="checkbox">
                            <input
                              type="checkbox"
                              checked={query.graph}
                              disabled={
                                !["context", "hybrid", "traceback"].includes(
                                  query.mode,
                                )
                              }
                              onChange={(e) =>
                                update("graph", e.target.checked)
                              }
                            />
                            Граф вызовов
                          </label>
                          {models?.reranker_loaded && (
                            <label className="checkbox">
                              <input
                                type="checkbox"
                                checked={query.rerank}
                                onChange={(e) =>
                                  update("rerank", e.target.checked)
                                }
                              />
                              CodeBERT
                            </label>
                          )}
                        </div>
                      )}
                      <div className="composerFoot">
                        <button
                          type="button"
                          className="configButton"
                          aria-expanded={optionsOpen}
                          onClick={() => setOptionsOpen(!optionsOpen)}
                        >
                          <Icon name="settings" size={17} />
                          {labels[query.mode]} <Icon name="down" size={12} />
                        </button>
                        {running ? (
                          <button
                            type="button"
                            className="secondary"
                            onClick={cancel}
                          >
                            <i className="spinner" /> Отменить
                          </button>
                        ) : (
                          <button className="primary" disabled={!canSearch}>
                            Найти функции <Icon name="arrow" size={16} />
                          </button>
                        )}
                      </div>
                    </form>
                    <aside className="panel examples">
                      <div className="panelHead">
                        <h2>
                          <Icon name="book" />
                          Исторические кейсы
                        </h2>
                      </div>
                      <div className="examplesList">
                        {cases.map((item) => (
                          <button
                            className={
                              "example " + (caseId === item.id ? "active" : "")
                            }
                            key={item.id}
                            onClick={() => chooseCase(item)}
                          >
                            <div>
                              <small>
                                {item.repository}{" "}
                                <span>#{item.id.split("-").at(-1)}</span>
                              </small>
                              <b>{demoTitle(item)}</b>
                            </div>
                            <Icon name="chevron" size={16} />
                          </button>
                        ))}
                      </div>
                      <p className="exampleNote">
                        <Icon name="patch" size={16} />
                        После поиска можно открыть реальное исправление.
                      </p>
                    </aside>
                  </div>
                  {running ? (
                    <div className="loadingStrip" role="status">
                      <i className="spinner" />
                      {labels[job?.status || "queued"]}…
                    </div>
                  ) : job?.result ? (
                    <Workbench
                      key={job.id}
                      job={job}
                      example={submittedCase}
                      dirty={dirty}
                    />
                  ) : (
                    <div className="panel emptyResults">
                      <Empty
                        icon="search"
                        title={
                          job?.status === "cancelled"
                            ? "Поиск отменён"
                            : snapshot && !snapshot.symbol_count
                              ? "В этом снимке нет функций"
                              : "Здесь появятся кандидаты"
                        }
                      >
                        {snapshot && !snapshot.symbol_count
                          ? "Откройте «Проверки кода» для анализа строк."
                          : "Укажите симптом, текст исключения или traceback и запустите поиск."}
                      </Empty>
                    </div>
                  )}
                </>
              )
            )}
          </>
        )}
        {page === "history" && (
          <HistoryView
            snapshots={snapshots}
            onOpen={(id) => void restore(id)}
          />
        )}
        {page === "lab" && <LabView models={models} />}
      </main>
      <footer className="pageFooter">
        <span>Traceback Lab</span>
        <span>v0.2.0</span>
      </footer>
      {importing && (
        <ImportDialog
          onClose={() => setImporting(false)}
          onImport={(s) => {
            setSnapshots((list) => [s, ...list.filter((x) => x.id !== s.id)]);
            chooseSnapshot(s.id, true);
            setImporting(false);
          }}
        />
      )}
    </div>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
