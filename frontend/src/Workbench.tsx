import { useEffect, useRef, useState } from "react";
import {
  api,
  Candidate,
  Case,
  CodeSymbol,
  Gold,
  Job,
  labels,
  Result,
} from "./data";
import { CodeView, CopyButton, Dialog, Empty, ErrorBox, Icon } from "./ui";

const rankOf = (rows: Candidate[], id: string) =>
  rows.find((r) => r.id === id)?.rank;
function rankText(rows: Candidate[], id: string) {
  const rank = rankOf(rows, id);
  return rank ? "#" + rank : "Вне выдачи";
}
function exportRun(
  job: Job,
  result: Result,
  gold: Gold | null,
  baseline: Result | null,
) {
  const query = job.request;
  const lines = [
    "# " + (query.title || "Расследование"),
    "",
    result.repository + " · " + result.revision,
    "",
    query.description,
    "",
    ...(query.traceback ? ["~~~text", query.traceback, "~~~", ""] : []),
    "## Кандидаты",
    "",
    ...result.candidates
      .slice(0, 5)
      .map(
        (r) =>
          `${r.rank}. ${r.qualname} — ${r.path}, строки ${r.start_line}–${r.end_line}`,
      ),
    "",
    "Метод: " +
      labels[query.mode] +
      ", модель: " +
      (result.methods.includes("dense")
        ? result.models?.encoder || "энкодер"
        : "без энкодера") +
      ".",
    ...(baseline
      ? [
          "",
          "## BM25",
          "",
          ...baseline.candidates
            .slice(0, 5)
            .map((r) => `${r.rank}. ${r.qualname} — ${r.path}`),
        ]
      : []),
    ...(gold
      ? [
          "",
          "## Историческое исправление",
          "",
          gold.fix_url,
          ...gold.positive_symbols.map(
            (s) => s.qualname + " · " + rankText(result.candidates, s.id),
          ),
        ]
      : []),
  ];
  const url = URL.createObjectURL(
    new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = "traceback-" + job.id.slice(0, 8) + ".md";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function Workbench({
  job,
  example,
  dirty,
}: {
  job: Job;
  example?: Case;
  dirty: boolean;
}) {
  const result = job.result!;
  const [selected, setSelected] = useState(result.candidates[0]?.id || ""),
    [viewId, setViewId] = useState(selected);
  const [limit, setLimit] = useState(5),
    [tab, setTab] = useState<"code" | "evidence">("code"),
    [symbol, setSymbol] = useState<CodeSymbol | null>(null),
    [sourceError, setSourceError] = useState("");
  const [gold, setGold] = useState<Gold | null>(null),
    [diff, setDiff] = useState(false),
    [baseline, setBaseline] = useState<Result | null>(null),
    [compare, setCompare] = useState(false);
  const [pending, setPending] = useState(""),
    [error, setError] = useState("");
  const life = useRef<AbortController | null>(null);
  useEffect(() => {
    const c = new AbortController();
    life.current = c;
    return () => c.abort();
  }, []);
  useEffect(() => {
    const c = new AbortController();
    setSymbol(null);
    setSourceError("");
    if (viewId)
      api<CodeSymbol>(`/v1/snapshots/${result.snapshot_id}/symbols/${viewId}`, {
        signal: c.signal,
      })
        .then((s) => {
          if (!c.signal.aborted) setSymbol(s);
        })
        .catch((e) => {
          if (!c.signal.aborted) setSourceError(e.message);
        });
    return () => c.abort();
  }, [result.snapshot_id, viewId]);
  const current = result.candidates.find((c) => c.id === selected);
  function select(id: string) {
    setSelected(id);
    setViewId(id);
    setTab("code");
  }
  async function action(kind: "baseline" | "gold") {
    if (kind === "gold" && gold) {
      setDiff(true);
      return;
    }
    if (kind === "baseline" && baseline) {
      setCompare(true);
      return;
    }
    if (pending) return;
    const signal = life.current?.signal;
    setPending(kind);
    setError("");
    try {
      if (kind === "baseline") {
        const res = await api<Result>(`/v1/investigations/${job.id}/baseline`, {
          signal,
        });
        if (!signal?.aborted) {
          setBaseline(res);
          setCompare(true);
        }
      } else if (example) {
        const res = await api<Gold>(`/v1/cases/${example.id}/reveal`, {
          method: "POST",
          signal,
        });
        if (!signal?.aborted) {
          setGold(res);
          setDiff(true);
        }
      }
    } catch (e) {
      if (!signal?.aborted) setError((e as Error).message);
    } finally {
      if (!signal?.aborted) setPending("");
    }
  }
  const targetWindow =
    current?.window && selected === viewId ? current.window : null;
  const traceLine = current?.trace.find((f) => f.symbol_id === viewId)?.line;
  let mark: [number, number] | undefined;
  if (symbol) {
    mark = targetWindow
      ? [targetWindow.start_line, targetWindow.end_line]
      : traceLine &&
          traceLine >= symbol.start_line &&
          traceLine <= symbol.end_line
        ? [traceLine, traceLine]
        : [symbol.start_line, symbol.start_line];
  }
  return (
    <section className="results" aria-label="Результаты поиска">
      <div className="sectionHeading">
        <div>
          <h2>
            Кандидаты <span className="count">{result.candidates.length}</span>
          </h2>
          <p>
            {dirty
              ? "Запрос изменён. Здесь показан предыдущий результат."
              : `${labels[job.request.mode]} · ${(result.timings.total_ms / 1000).toLocaleString("ru", { maximumFractionDigits: 2 })} с${job.cached ? " · из кэша" : ""}`}
          </p>
        </div>
        <div className="actions">
          <button
            className="secondary"
            disabled={!!pending}
            onClick={() => void action("baseline")}
          >
            <Icon name="compare" />
            {pending === "baseline" ? "Сравниваем…" : "Сравнить с BM25"}
          </button>
          {example && (
            <button
              className="secondary tinted"
              disabled={!!pending}
              onClick={() => void action("gold")}
            >
              <Icon name="patch" />
              {pending === "gold" ? "Загрузка…" : "Раскрыть исправление"}
            </button>
          )}
          <button
            className="iconButton framed"
            title="Скачать отчёт Markdown"
            aria-label="Скачать отчёт Markdown"
            onClick={() => exportRun(job, result, gold, baseline)}
          >
            <Icon name="download" />
          </button>
        </div>
      </div>
      {error && <ErrorBox text={error} />}
      {result.warnings.map((w) => (
        <p key={w} className="warningNote">
          {w}
        </p>
      ))}
      {!current ? (
        <div className="panel">
          <Empty icon="search" title="Совпадений не найдено">
            Добавьте имя функции, текст исключения или стек вызовов.
          </Empty>
        </div>
      ) : (
        <div className="workbench">
          <div className="candidatePane">
            <div className="paneTitle">
              <Icon name="list" size={16} />
              Рейтинг функций
            </div>
            <div className="resultList">
              {result.candidates.slice(0, limit).map((row) => (
                <button
                  key={row.id}
                  onClick={() => select(row.id)}
                  className={
                    "candidate " + (selected === row.id ? "chosen" : "")
                  }
                  aria-pressed={selected === row.id}
                >
                  <span className="rank">
                    {String(row.rank).padStart(2, "0")}
                  </span>
                  <span className="candidateText">
                    <b>{row.qualname}</b>
                    <small>{row.path}</small>
                  </span>
                  {gold?.positive_ids.includes(row.id) ? (
                    <span title="Изменена патчем" className="patchTick">
                      <Icon name="check" size={15} />
                    </span>
                  ) : (
                    <Icon name="chevron" size={13} />
                  )}
                </button>
              ))}
            </div>
            {limit < result.candidates.length && (
              <button
                className="more"
                onClick={() =>
                  setLimit(Math.min(limit + 10, result.candidates.length))
                }
              >
                Показать ещё {Math.min(10, result.candidates.length - limit)}{" "}
                <Icon name="down" size={14} />
              </button>
            )}
          </div>
          <div className="codePane">
            <div className="editorBar">
              <div className="editorTabs">
                <button
                  className={tab === "code" ? "active" : ""}
                  onClick={() => setTab("code")}
                >
                  <Icon name="code" size={15} />
                  Исходник
                </button>
                <button
                  className={tab === "evidence" ? "active" : ""}
                  onClick={() => setTab("evidence")}
                >
                  <Icon name="branch" size={15} />
                  Почему здесь
                </button>
              </div>
              {tab === "code" && symbol && (
                <CopyButton label="Скопировать функцию" text={symbol.code} />
              )}
            </div>
            {tab === "code" ? (
              <>
                <div className="codePath">
                  <Icon name="file" size={15} />
                  <span>{symbol?.path || current.path}</span>
                  {symbol && (
                    <small>
                      L{symbol.start_line}–{symbol.end_line}
                    </small>
                  )}
                  {viewId !== selected && (
                    <button
                      className="link"
                      onClick={() => setViewId(selected)}
                    >
                      К кандидату
                    </button>
                  )}
                </div>
                {sourceError ? (
                  <ErrorBox text={sourceError} />
                ) : symbol ? (
                  <CodeView
                    source={symbol.file_source}
                    from={Math.max(symbol.start_line - 2, 1)}
                    to={symbol.end_line + 2}
                    mark={mark}
                    focus={mark?.[0]}
                  />
                ) : (
                  <div className="loadingStrip">Открываем исходник…</div>
                )}
              </>
            ) : (
              <div className="evidence">
                <h3>{current.qualname}</h3>
                <div className="evidenceRow">
                  <span>Как попала в выдачу</span>
                  <b>{labels[current.origin]}</b>
                </div>
                {Object.entries(current.ranks).map(([method, r]) => (
                  <div key={method} className="evidenceRow">
                    <span>
                      {method === "dense"
                        ? "Нейросетевой поиск"
                        : "Лексический поиск"}
                    </span>
                    <b>#{r}</b>
                  </div>
                ))}
                {current.terms.length > 0 && (
                  <>
                    <h4>Совпавшие слова</h4>
                    <div className="chips">
                      {current.terms.map((t) => (
                        <span key={t}>{t}</span>
                      ))}
                    </div>
                  </>
                )}
                {current.graph && (
                  <button
                    className="link"
                    onClick={() => {
                      setViewId(current.graph!.seed);
                      setTab("code");
                    }}
                  >
                    <Icon name="branch" size={15} />
                    Вызов {current.graph.edge.expression} · L
                    {current.graph.edge.line}
                  </button>
                )}
                {result.frames.length > 0 && (
                  <>
                    <h4>Кадры стека</h4>
                    {result.frames.slice(-8).map((f, i) => (
                      <button
                        key={i}
                        className="frameRow"
                        disabled={!f.symbol_id}
                        onClick={() => {
                          if (f.symbol_id) setViewId(f.symbol_id);
                          setTab("code");
                        }}
                      >
                        {f.name} · L{f.line}
                        <span>{labels[f.status] || f.status}</span>
                      </button>
                    ))}
                  </>
                )}
                <p className="small muted">
                  {result.confidence.probability == null
                    ? "Вероятность попадания не калибрована. Ранг задаёт порядок проверки."
                    : `Первое место совпадёт с историческим исправлением с оценкой ${Math.round(result.confidence.probability * 100)}%. ${result.confidence.accepted ? "Порог принятия пройден." : "Ниже порога принятия."}`}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
      {compare && baseline && (
        <Dialog title="Сравнение с BM25" wide onClose={() => setCompare(false)}>
          <div className="dialogBody">
            <p className="muted">
              Один запрос и один снимок. Слева — сохранённый результат, справа —
              независимый поиск по словам.
            </p>
            <div className="compareGrid">
              {[
                [labels[job.request.mode], result],
                ["Только BM25", baseline],
              ].map(([name, res], i) => (
                <section key={i}>
                  <h3>{name as string}</h3>
                  {(res as Result).candidates.slice(0, 5).map((c) => (
                    <div className="compareRow" key={c.id}>
                      <span className="rank">{c.rank}</span>
                      <div>
                        <b>{c.qualname}</b>
                        <small>{c.path}</small>
                      </div>
                      {gold?.positive_ids.includes(c.id) && (
                        <Icon name="check" size={16} />
                      )}
                    </div>
                  ))}
                  {!(res as Result).candidates.length && (
                    <p className="small muted">Совпадений нет</p>
                  )}
                </section>
              ))}
            </div>
            {gold && (
              <div className="goldSummary">
                {gold.positive_symbols.map((s) => (
                  <p key={s.id}>
                    {s.qualname} · основной {rankText(result.candidates, s.id)}{" "}
                    · BM25 {rankText(baseline.candidates, s.id)}
                  </p>
                ))}
              </div>
            )}
          </div>
        </Dialog>
      )}
      {diff && gold && (
        <Dialog
          title="Историческое исправление"
          wide
          onClose={() => setDiff(false)}
        >
          <div className="patchMeta">
            <div>
              {gold.positive_symbols.map((s) => (
                <p key={s.id}>
                  <span className="patchRank">
                    {rankText(result.candidates, s.id)}
                  </span>{" "}
                  {s.qualname}
                </p>
              ))}
            </div>
            <a
              className="link"
              href={gold.fix_url}
              target="_blank"
              rel="noreferrer"
            >
              Открыть PR <Icon name="exit" size={15} />
            </a>
          </div>
          <div className="diffScroll">
            <pre>
              {gold.patch.split("\n").map((line, i) => (
                <div
                  key={i}
                  className={
                    line.startsWith("@@")
                      ? "hunk"
                      : line.startsWith("+") && !line.startsWith("+++")
                        ? "added"
                        : line.startsWith("-") && !line.startsWith("---")
                          ? "removed"
                          : ""
                  }
                >
                  {line || " "}
                </div>
              ))}
            </pre>
          </div>
        </Dialog>
      )}
    </section>
  );
}
