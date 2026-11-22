import { useEffect, useState } from "react";
import { api, labels, ModelStatus, Recent, Report, Snapshot } from "./data";
import { Empty, ErrorBox, Icon } from "./ui";

export function HistoryView({
  snapshots,
  onOpen,
}: {
  snapshots: Snapshot[];
  onOpen: (id: string) => void;
}) {
  const [rows, setRows] = useState<Recent[] | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    const c = new AbortController();
    api<Recent[]>("/v1/investigations?limit=60", { signal: c.signal })
      .then(setRows)
      .catch((e) => {
        if (!c.signal.aborted) setError(e.message);
      });
    return () => c.abort();
  }, []);
  return (
    <>
      <div className="pageHeading">
        <div>
          <h1>История</h1>
          <p>
            Сохранённые запросы и выдача с теми версиями кода, на которых они
            запускались.
          </p>
        </div>
      </div>
      {error ? (
        <ErrorBox text={error} />
      ) : !rows ? (
        <div className="loadingStrip">Загружаем историю…</div>
      ) : !rows.length ? (
        <div className="panel">
          <Empty icon="history" title="Пока нет расследований">
            Запустите поиск по симптому. Результат появится здесь.
          </Empty>
        </div>
      ) : (
        <div className="panel history">
          {rows.map((row) => (
            <button key={row.id} onClick={() => onOpen(row.id)}>
              <span className="listIcon">
                <Icon name="history" />
              </span>
              <div>
                <b>{row.title}</b>
                <small>
                  {snapshots.find((s) => s.id === row.snapshot_id)
                    ?.repository || row.snapshot_id}{" "}
                  · {labels[row.mode]} · {labels[row.status] || row.status}
                </small>
              </div>
              <time dateTime={new Date(row.created_at * 1000).toISOString()}>
                {new Date(row.created_at * 1000).toLocaleString("ru", {
                  day: "numeric",
                  month: "short",
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </time>
              <Icon name="chevron" size={15} />
            </button>
          ))}
        </div>
      )}
    </>
  );
}

export function LabView({ models }: { models: ModelStatus | null }) {
  const [reports, setReports] = useState<Report[] | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    const c = new AbortController();
    api<Report[]>("/v1/reports?compact=true", { signal: c.signal })
      .then(setReports)
      .catch((e) => {
        if (!c.signal.aborted) setError(e.message);
      });
    return () => c.abort();
  }, []);
  const components = [
    [
      "Энкодер",
      models?.encoder_name || "—",
      models?.encoder_loaded ? "Подключён" : "Не загружен",
      models?.encoder_name === "Compact-64"
        ? "Компактная модель поиска. Веса входят в релиз."
        : "Сопоставляет описание бага с кодом функции.",
    ],
    [
      "Ранжировщик",
      "CodeBERT",
      models?.reranker_loaded ? "Подключён" : "Не подключён",
      "После обучения перечитывает запрос вместе с кодом.",
    ],
    [
      "Калибратор",
      "Logistic regression",
      models?.calibrator_loaded ? "Подключён" : "Не подключён",
      "Оценивает вероятность попадания первого места в патч.",
    ],
  ];
  return (
    <>
      <div className="pageHeading">
        <div>
          <h1>Лаборатория</h1>
          <p>Загруженные модели и измерения на исторических исправлениях.</p>
        </div>
      </div>
      <div className="modelGrid">
        {components.map(([label, title, status, desc]) => (
          <section className="panel model" key={label}>
            <span className="modelLabel">{label}</span>
            <h2>{title}</h2>
            <span
              className={"stateChip " + (status === "Подключён" ? "on" : "")}
            >
              {status}
            </span>
            <p>{desc}</p>
          </section>
        ))}
      </div>
      <div className="sectionHeading">
        <div>
          <h2>Результаты экспериментов</h2>
          <p>
            Hit@5 — доля запросов, где хотя бы одна функция из исправления
            попала в первую пятёрку.
          </p>
        </div>
      </div>
      {error && <ErrorBox text={error} />}
      {reports === null && !error && (
        <div className="loadingStrip" role="status">
          Загрузка отчётов…
        </div>
      )}
      {reports
        ?.filter((r) => r.summaries)
        .map((r) => (
          <section key={r.file} className="panel report">
            <div className="reportHead">
              <Icon name="lab" />
              <div>
                <h3>
                  {r.file === "compact_evaluation.json"
                    ? "Баги четырёх Python-проектов"
                    : r.file === "demo_evaluation.json"
                      ? "Демонстрационные кейсы Requests"
                      : r.file}
                </h3>
                <small>
                  {r.file === "compact_evaluation.json"
                    ? "Демо исключены · Compact-64"
                    : r.file === "demo_evaluation.json"
                      ? "Три демонстрации · UniXcoder"
                      : r.splits?.join(", ")}
                </small>
              </div>
            </div>
            <div className="tableScroll">
              <table>
                <thead>
                  <tr>
                    {[
                      "Метод",
                      "Кейсы",
                      "Hit@1",
                      "Hit@5",
                      "Recall@50",
                      "p95",
                    ].map((t) => (
                      <th key={t}>{t}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(r.summaries || {}).map(([mode, v]) => (
                    <tr key={mode}>
                      <td>{labels[mode] || mode}</td>
                      <td>{v.count}</td>
                      {["hit@1", "hit@5", "recall@50"].map((m) => (
                        <td key={m}>
                          {v[m] == null ? "—" : Math.round(v[m]! * 100) + "%"}
                        </td>
                      ))}
                      <td>
                        {v.latency_p95_ms == null
                          ? "—"
                          : (v.latency_p95_ms / 1000).toFixed(2) + " с"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        ))}
      <section className="panel setup">
        <Icon name="terminal" />
        <div>
          <h3>Запуск с UniXcoder</h3>
          <p>
            Первый запуск скачает дополнительные зависимости и веса. Обучение
            ранжировщика и воспроизведение отчётов описаны в README.
          </p>
          <code>python run.py --neural</code>
        </div>
      </section>
    </>
  );
}
