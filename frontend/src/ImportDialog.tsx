import { useEffect, useRef, useState } from "react";
import packetExample from "../../examples/packet_queue.py?raw";
import { api, Snapshot } from "./data";
import { Dialog, ErrorBox, Icon } from "./ui";

export function ImportDialog({
  onClose,
  onImport,
}: {
  onClose: () => void;
  onImport: (s: Snapshot) => void;
}) {
  const [tab, setTab] = useState<"file" | "paste">("file");
  const [file, setFile] = useState<File | null>(null),
    [code, setCode] = useState(""),
    [name, setName] = useState("");
  const [over, setOver] = useState(false),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const input = useRef<HTMLInputElement>(null),
    controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  function select(next: File | undefined) {
    if (!next) return;
    setError("");
    setFile(null);
    if (!/\.(py|zip)$/i.test(next.name)) {
      setError("Выберите файл .py или ZIP с Python-проектом.");
      return;
    }
    const limit = next.name.toLowerCase().endsWith(".zip") ? 16 : 1;
    if (next.size > limit * 1024 * 1024) {
      setError(`Для этого файла лимит составляет ${limit} МиБ.`);
      return;
    }
    setFile(next);
  }
  async function submit() {
    if (busy) return;
    const source =
      tab === "file"
        ? file
        : new File([code], "snippet.py", { type: "text/plain" });
    if (!source) return;
    const params = new URLSearchParams({ filename: source.name });
    if (name.trim()) params.set("name", name.trim());
    const abort = new AbortController();
    controller.current = abort;
    setBusy(true);
    setError("");
    try {
      const result = await api<Snapshot>("/v1/snapshots/import?" + params, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: source,
        signal: abort.signal,
      });
      if (!abort.signal.aborted) onImport(result);
    } catch (e) {
      if (!abort.signal.aborted) setError((e as Error).message);
    } finally {
      if (!abort.signal.aborted) setBusy(false);
    }
  }
  return (
    <Dialog title="Подключить свой код" onClose={onClose}>
      <form
        className="dialogBody"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <div className="segmented">
          {(
            [
              ["file", "Файл или ZIP"],
              ["paste", "Вставить код"],
            ] as const
          ).map(([id, label]) => (
            <button
              type="button"
              key={id}
              aria-pressed={id === tab}
              disabled={busy}
              onClick={() => {
                setTab(id);
                setError("");
              }}
            >
              {label}
            </button>
          ))}
        </div>
        {tab === "file" ? (
          <>
            <input
              className="visuallyHidden"
              ref={input}
              type="file"
              aria-label="Python-файл или ZIP"
              accept=".py,.zip"
              onChange={(e) => select(e.target.files?.[0])}
            />
            <button
              type="button"
              disabled={busy}
              className={"dropzone " + (over ? "over" : "")}
              onClick={() => input.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setOver(true);
              }}
              onDragLeave={() => setOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setOver(false);
                if (!busy) select(e.dataTransfer.files[0]);
              }}
            >
              <span className="dropIcon">
                <Icon name={file ? "file" : "upload"} size={24} />
              </span>
              <b>{file ? file.name : "Выбрать или перетащить файл"}</b>
              <small>
                {file
                  ? `${Math.max(1, Math.round(file.size / 1024))} КиБ · нажмите, чтобы заменить`
                  : "Python до 1 МиБ · ZIP до 16 МиБ"}
              </small>
            </button>
          </>
        ) : (
          <>
            <label className="field">
              Код Python
              <textarea
                className="pasteCode"
                aria-label="Код Python"
                value={code}
                onChange={(e) => setCode(e.target.value)}
                placeholder={
                  "def collect(item, items=[]):\n    items.append(item)\n    return items"
                }
                spellCheck={false}
                required
              />
            </label>
            <button
              className="link"
              type="button"
              disabled={busy || !!code.trim()}
              onClick={() => {
                setCode(packetExample);
                if (!name.trim()) setName("packet-queue");
              }}
            >
              <Icon name="code" size={15} /> Вставить учебный пример
            </button>
          </>
        )}
        <label className="field">
          Название проекта <small>необязательно</small>
          <input
            placeholder={
              tab === "file"
                ? file?.name.replace(/\.[^.]+$/, "") || "Из имени файла"
                : "snippet"
            }
            maxLength={100}
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        <p className="muted small">
          <Icon name="laptop" size={14} /> Исходники остаются на машине, где
          запущен сервис. Импорт читает код без его выполнения.
        </p>
        {error && <ErrorBox text={error} />}
        <div className="actions end">
          <button type="button" className="secondary" onClick={onClose}>
            Закрыть
          </button>
          <button
            className="primary"
            disabled={busy || (tab === "file" ? !file : !code.trim())}
          >
            {busy ? (
              <>
                <i className="spinner" /> Индексируем…
              </>
            ) : (
              <>
                Подключить проект <Icon name="arrow" size={16} />
              </>
            )}
          </button>
        </div>
      </form>
    </Dialog>
  );
}
