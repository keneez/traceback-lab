import { useEffect, useState } from "react";
import { api, Checks, Finding, Snapshot } from "./data";
import { CodeView, Empty, ErrorBox, Icon } from "./ui";

export function ChecksPanel({ snapshot }: { snapshot: Snapshot }) {
  const [report, setReport] = useState<Checks | null>(null),
    [error, setError] = useState(""),
    [choice, setChoice] = useState<Finding | null>(null),
    [source, setSource] = useState<string | null>(null),
    [sourceError, setSourceError] = useState("");
  const [level, setLevel] = useState("all");
  useEffect(() => {
    const c = new AbortController();
    setReport(null);
    setError("");
    setChoice(null);
    setLevel("all");
    api<Checks>(`/v1/snapshots/${snapshot.id}/checks`, { signal: c.signal })
      .then((data) => {
        if (!c.signal.aborted) {
          setReport(data);
          setChoice(data.findings[0] || null);
        }
      })
      .catch((e) => {
        if (!c.signal.aborted) setError(e.message);
      });
    return () => c.abort();
  }, [snapshot.id]);
  useEffect(() => {
    const c = new AbortController();
    setSource(null);
    setSourceError("");
    if (choice)
      api<{ source: string }>(
        `/v1/snapshots/${snapshot.id}/source?` +
          new URLSearchParams({ path: choice.path }),
        { signal: c.signal },
      )
        .then((data) => {
          if (!c.signal.aborted) setSource(data.source);
        })
        .catch((e) => {
          if (!c.signal.aborted) setSourceError(e.message);
        });
    return () => c.abort();
  }, [snapshot.id, choice?.path]);
  if (error) return <ErrorBox text={error} />;
  if (!report)
    return (
      <div className="loadingStrip" role="status">
        <i className="spinner" /> Разбираем синтаксис файлов…
      </div>
    );
  const visible = report.findings.filter(
    (f) => level === "all" || f.level === level,
  );
  return (
    <div className="checksSection">
      <div className="sectionHeading">
        <div>
          <h2>Статические проверки</h2>
          <p>
            Проверено файлов · {report.files_scanned} &nbsp; / &nbsp; Правил ·{" "}
            {report.rule_count}
          </p>
        </div>
        <label className="small">
          Уровень{" "}
          <select
            value={level}
            onChange={(e) => {
              const next = e.target.value;
              setLevel(next);
              setChoice(
                report.findings.find(
                  (f) => next === "all" || f.level === next,
                ) || null,
              );
            }}
          >
            <option value="all">Все ({report.total})</option>
            <option value="error">Ошибки ({report.counts.error || 0})</option>
            <option value="warning">
              Предупреждения ({report.counts.warning || 0})
            </option>
          </select>
        </label>
      </div>
      {report.unreadable.length > 0 && (
        <p className="warningNote">
          Пропущены нечитаемые файлы · {report.unreadable.join(", ")}
        </p>
      )}
      {!visible.length ? (
        <div className="panel">
          <Empty
            icon="scan"
            title={
              report.total
                ? "Нет находок этого уровня"
                : "По этим правилам замечаний нет"
            }
          >
            Проверяются синтаксис, изменяемые аргументы, повтор ключа, кортеж в
            условии и пустой обработчик исключений.
          </Empty>
        </div>
      ) : (
        <div className="workbench checksBench">
          <div className="candidatePane">
            <div className="paneTitle">
              <Icon name="list" />
              Находки <span className="count">{visible.length}</span>
            </div>
            <div className="checkItems">
              {visible.map((f, i) => (
                <button
                  key={i}
                  className={"finding " + (f === choice ? "chosen" : "")}
                  onClick={() => setChoice(f)}
                >
                  <span className={"ruleTag " + f.level}>{f.rule}</span>
                  <b>{f.title}</b>
                  <small>
                    {f.path} · L{f.line}
                  </small>
                </button>
              ))}
            </div>
          </div>
          <div className="codePane">
            {choice && (
              <>
                <div className="paneTitle">
                  <Icon name="file" />
                  <span className="mono break">{choice.path}</span>
                  <span className="lineTag">L{choice.line}</span>
                </div>
                <div className="findingInfo">
                  <h3>{choice.title}</h3>
                  <p>{choice.message}</p>
                </div>
                {sourceError ? (
                  <ErrorBox text={sourceError} />
                ) : source !== null ? (
                  <CodeView
                    source={source}
                    from={Math.max(1, choice.line - 8)}
                    to={choice.line + 12}
                    mark={[choice.line, choice.line]}
                    focus={choice.line}
                  />
                ) : (
                  <div className="loadingStrip">Открываем исходник…</div>
                )}
              </>
            )}
          </div>
        </div>
      )}
      {report.total > report.findings.length && (
        <p className="muted small">
          Показаны первые {report.findings.length} находок из {report.total}.
        </p>
      )}
    </div>
  );
}
