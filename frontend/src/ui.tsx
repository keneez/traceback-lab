import { ReactNode, useEffect, useId, useRef, useState } from "react";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
hljs.registerLanguage("python", python);

const paths = {
  trace: "M3 17h5l4-11 4 11h5",
  layers: "m12 3 9 5-9 5-9-5 9-5ZM3 12l9 5 9-5M3 16l9 5 9-5",
  search: "M20 20l-5-5m2-5a7 7 0 1 1-14 0 7 7 0 0 1 14 0Z",
  history: "M3 11a9 9 0 1 1 2 7m-2-14v6h6m3-3v5l3 2",
  lab: "M9 3h6m-5 0v6L4 19a1 1 0 0 0 1 2h14a1 1 0 0 0 1-2L14 9V3m-7 12h10",
  brush:
    "M12 3a9 9 0 1 0 0 18h1.5a2 2 0 0 0 1-3.6c-.8-.6-.3-2.4.9-2.4H18a3 3 0 0 0 3-3 9 9 0 0 0-9-9ZM7 10h.02M10 7h.02M15 7.5h.02M17 11h.02",
  plus: "M12 5v14M5 12h14",
  arrow: "M5 12h14m-6-6 6 6-6 6",
  chevron: "m9 6 6 6-6 6",
  down: "m6 9 6 6 6-6",
  code: "m8 8-4 4 4 4m8-8 4 4-4 4m-3-12-2 16",
  file: "M6 3h8l4 4v14H6V3Zm8 0v5h4",
  branch:
    "M6 7v10m0-2c8 0 12-2 12-7M8 5a2 2 0 1 1-4 0 2 2 0 0 1 4 0Zm0 14a2 2 0 1 1-4 0 2 2 0 0 1 4 0Zm12-14a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z",
  folder: "M3 6h6l2 2h10v11H3V6Z",
  check: "m5 12 4 4L19 6",
  upload: "M12 15V3m-5 6 5-6 5 6M4 15v5h16v-5",
  close: "m6 6 12 12M18 6 6 18",
  issue: "M4 4h16v13H9l-5 4V4Zm4 5h8m-8 4h5",
  settings: "M3 7h5m4 0h9M3 17h10m4 0h4M8 4v6m8 4v6",
  compare: "M4 6h14m-3-3 3 3-3 3M20 18H6m3-3-3 3 3 3",
  copy: "M8 8h12v12H8V8ZM4 15V4h11",
  patch: "M4 3h10l5 5v13H4V3Zm10 0v5h5M7 12h5m-2.5-2.5v5M7 17h6",
  eye: "M2 12c5-8 15-8 20 0-5 8-15 8-20 0Zm13 0a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z",
  clock: "M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-5v6l4 2",
  cpu: "M7 7h10v10H7zM4 9H2m2 6H2m20-6h-2m2 6h-2M9 4V2m6 2V2M9 22v-2m6 2v-2",
  terminal: "M3 5h18v14H3V5Zm4 4 3 3-3 3m6 0h4",
  moon: "M20 15A8 8 0 0 1 9 4a9 9 0 1 0 11 11Z",
  sun: "M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0ZM12 2v2m0 16v2M2 12h2m16 0h2M5 5l2 2m10 10 2 2M5 19l2-2M17 7l2-2",
  laptop: "M5 4h14v12H5V4Zm-3 15h20",
  warn: "m12 3 10 17H2L12 3Zm0 6v5m0 3h.01",
  list: "M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01",
  download: "M12 3v12m-5-5 5 5 5-5M4 16v4h16v-4",
  book: "M3 5c3-1 6-1 9 1 3-2 6-2 9-1v14c-3-1-6-1-9 1-3-2-6-2-9-1V5Zm9 1v14",
  scan: "M3 9V4h5m8 0h5v5m0 6v5h-5M8 20H3v-5m3-3h12",
  exit: "m14 4 6 0v6m0-6-9 9M10 5H4v15h15v-7",
} as const;
export type IconName = keyof typeof paths;
export function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  return (
    <svg
      aria-hidden="true"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.65"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d={paths[name]} />
    </svg>
  );
}
export function Dialog({
  title,
  onClose,
  children,
  wide = false,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const label = useId();
  useEffect(() => {
    const dialog = ref.current!;
    const previous = document.activeElement as HTMLElement | null;
    dialog.showModal();
    return () => {
      dialog.close();
      previous?.focus();
    };
  }, []);
  return (
    <dialog
      className={wide ? "dialog wide" : "dialog"}
      ref={ref}
      aria-labelledby={label}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      onClick={(e) => {
        if (e.target === ref.current) {
          const b = ref.current!.getBoundingClientRect();
          if (
            e.clientX < b.left ||
            e.clientX > b.right ||
            e.clientY < b.top ||
            e.clientY > b.bottom
          )
            onClose();
        }
      }}
    >
      <div className="dialogHead">
        <h2 id={label}>{title}</h2>
        <button
          className="iconButton"
          onClick={onClose}
          aria-label="Закрыть окно"
        >
          <Icon name="close" />
        </button>
      </div>
      {children}
    </dialog>
  );
}
type Appearance = {
  theme: "dark" | "light" | "system";
  accent: "cobalt" | "violet" | "amber";
};
const defaults: Appearance = { theme: "dark", accent: "cobalt" };
export function applyAppearance(p: Appearance) {
  const light =
    p.theme === "light" ||
    (p.theme === "system" &&
      window.matchMedia("(prefers-color-scheme: light)").matches);
  const root = document.documentElement;
  root.dataset.theme = light ? "light" : "dark";
  root.dataset.accent = p.accent;
  document
    .querySelector('meta[name="theme-color"]')
    ?.setAttribute("content", light ? "#f5f6fa" : "#14151b");
}
export function readAppearance(): Appearance {
  try {
    const p = JSON.parse(
      localStorage.getItem("traceback.appearance.v2") || "{}",
    );
    return {
      theme: ["dark", "light", "system"].includes(p.theme) ? p.theme : "dark",
      accent: ["cobalt", "violet", "amber"].includes(p.accent)
        ? p.accent
        : "cobalt",
    };
  } catch {
    return defaults;
  }
}
applyAppearance(readAppearance());
export function ThemeButton() {
  const [prefs, setPrefs] = useState(readAppearance),
    [open, setOpen] = useState(false);
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: light)");
    const apply = () => applyAppearance(prefs);
    apply();
    media.addEventListener("change", apply);
    try {
      localStorage.setItem("traceback.appearance.v2", JSON.stringify(prefs));
    } catch {}
    return () => media.removeEventListener("change", apply);
  }, [prefs]);
  return (
    <>
      <button
        className="themeButton"
        onClick={() => setOpen(true)}
        title="Оформление"
        aria-label="Оформление"
      >
        <Icon name="brush" />
        <span>Оформление</span>
      </button>
      {open && (
        <Dialog title="Оформление" onClose={() => setOpen(false)}>
          <div className="dialogBody">
            <fieldset>
              <legend>Тема</legend>
              <div className="themeOptions">
                {(
                  [
                    ["dark", "Графит", "moon"],
                    ["light", "Бумага", "sun"],
                    ["system", "Системная", "laptop"],
                  ] as const
                ).map(([id, label, icon]) => (
                  <button
                    key={id}
                    className="themeOption"
                    aria-pressed={prefs.theme === id}
                    onClick={() => setPrefs({ ...prefs, theme: id })}
                  >
                    <Icon name={icon} />
                    {label}
                  </button>
                ))}
              </div>
            </fieldset>
            <fieldset>
              <legend>Акцентный цвет</legend>
              <div className="themeOptions">
                {(
                  [
                    ["cobalt", "Кобальт"],
                    ["violet", "Аметист"],
                    ["amber", "Янтарь"],
                  ] as const
                ).map(([id, label]) => (
                  <button
                    key={id}
                    className="swatchButton"
                    aria-pressed={prefs.accent === id}
                    onClick={() => setPrefs({ ...prefs, accent: id })}
                  >
                    <i className={"swatch " + id} />
                    {label}
                  </button>
                ))}
              </div>
            </fieldset>
            <p className="muted small">Выбор сохраняется в этом браузере.</p>
          </div>
        </Dialog>
      )}
    </>
  );
}
export function CopyButton({ text, label }: { text: string; label: string }) {
  const [state, set] = useState("");
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(timer.current), []);
  return (
    <button
      className="iconButton"
      title={state || label}
      aria-label={state || label}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          set("Скопировано");
        } catch {
          set("Копирование недоступно");
        }
        clearTimeout(timer.current);
        timer.current = setTimeout(() => set(""), 1800);
      }}
    >
      <Icon name={state === "Скопировано" ? "check" : "copy"} size={15} />
      {state && (
        <span role="status" className="copyFeedback">
          {state}
        </span>
      )}
    </button>
  );
}
export function CodeView({
  source,
  from,
  to,
  mark,
  focus,
}: {
  source: string;
  from: number;
  to: number;
  mark?: [number, number];
  focus?: number;
}) {
  const holder = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const box = holder.current;
    if (!box) return;
    const line = focus
      ? box.querySelector<HTMLElement>(`[data-line="${focus}"]`)
      : null;
    box.scrollTop = line
      ? Math.max(0, line.offsetTop - box.clientHeight / 3)
      : 0;
    box.scrollLeft = 0;
  }, [source, from, focus]);
  return (
    <div
      className="codeScroll"
      role="region"
      aria-label="Исходный код"
      tabIndex={0}
      ref={holder}
    >
      <pre>
        {source
          .split("\n")
          .slice(from - 1, to)
          .map((line, i) => {
            const n = from + i;
            return (
              <div
                key={n}
                data-line={n}
                className={
                  "codeLine " +
                  (mark && n >= mark[0] && n <= mark[1] ? "marked" : "")
                }
              >
                <span className="lineNumber" aria-hidden="true">
                  {n}
                </span>
                <code
                  dangerouslySetInnerHTML={{
                    __html: hljs.highlight(line || " ", { language: "python" })
                      .value,
                  }}
                />
              </div>
            );
          })}
      </pre>
    </div>
  );
}
export function Empty({
  icon,
  title,
  children,
}: {
  icon: IconName;
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="emptyState">
      <span className="emptyIcon">
        <Icon name={icon} size={23} />
      </span>
      <h3>{title}</h3>
      {children && <p>{children}</p>}
    </div>
  );
}
export function ErrorBox({ text }: { text: string }) {
  return (
    <div className="errorBox" role="alert">
      <Icon name="warn" size={17} />
      <span>{text}</span>
    </div>
  );
}
