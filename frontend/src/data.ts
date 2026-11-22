export type Snapshot = {
  id: string;
  repository: string;
  revision: string;
  symbol_count: number;
  edge_count: number;
  file_count: number;
};
export type Case = {
  id: string;
  snapshot_id: string;
  repository: string;
  title: string;
  description: string;
  traceback: string;
  trace_revision: string | null;
  historical_text: string;
};
export type Frame = {
  path: string;
  line: number;
  name: string;
  symbol_id: string | null;
  status: string;
};
export type Candidate = {
  id: string;
  rank: number;
  path: string;
  qualname: string;
  start_line: number;
  end_line: number;
  terms: string[];
  origin: string;
  ranks: Record<string, number>;
  raw_scores: Record<string, number>;
  rerank_score: number | null;
  trace: Frame[];
  graph: null | {
    seed: string;
    edge: { path: string; line: number; expression: string };
    direction: string;
  };
  window: null | { start_line: number; end_line: number };
};
export type Result = {
  snapshot_id: string;
  repository: string;
  revision: string;
  config: { mode: string; budget: number };
  candidates: Candidate[];
  frames: Frame[];
  methods: string[];
  warnings: string[];
  models?: { encoder: string | null };
  reranked: boolean;
  timings: { total_ms: number };
  graph_enabled: boolean;
  confidence: { status: string; probability: number | null; accepted: boolean };
};
export type CodeSymbol = {
  id: string;
  path: string;
  qualname: string;
  code: string;
  file_source: string;
  start_line: number;
  end_line: number;
};
export type Gold = {
  patch: string;
  fix_url: string;
  positive_ids: string[];
  positive_symbols: { id: string; path: string; qualname: string }[];
  mapping_coverage: number;
  dynamic_reproduction: string;
};
export type ModelStatus = {
  encoder_name: string | null;
  encoder_loaded: boolean;
  reranker_loaded: boolean;
  calibrator_loaded: boolean;
  identities: Record<string, string | null>;
};
export type Job = {
  id: string;
  status: string;
  error: string | null;
  result: Result | null;
  cached: boolean;
  request: Investigation;
  created_at: number;
};
export type Investigation = {
  snapshot_id: string;
  title: string;
  description: string;
  traceback: string;
  trace_revision: string | null;
  mode: string;
  profile: string;
  graph: boolean;
  rerank: boolean;
};
export type Recent = {
  id: string;
  created_at: number;
  snapshot_id: string;
  status: string;
  title: string;
  mode: string;
  count: number;
};
export type Finding = {
  rule: string;
  level: string;
  path: string;
  line: number;
  title: string;
  message: string;
};
export type Checks = {
  findings: Finding[];
  total: number;
  counts: Record<string, number>;
  rule_count: number;
  files_scanned: number;
  unreadable: string[];
};
export type Report = {
  file: string;
  splits?: string[];
  summaries?: Record<string, Record<string, number | null>>;
  encoder_name?: string;
};
export async function api<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(
      typeof error.detail === "string"
        ? error.detail
        : `Ошибка запроса (${response.status})`,
    );
  }
  return response.json() as Promise<T>;
}
export const labels: Record<string, string> = {
  context: "BM25 + контекст",
  hybrid: "Гибридный",
  bm25: "BM25",
  dense: "Нейросеть",
  traceback: "Стек",
  queued: "В очереди",
  parsing: "Разбор запроса",
  indexing: "Расчёт векторов функций",
  retrieving: "Поиск функций",
  reranking: "Ранжирование",
  completed: "Завершено",
  failed: "Ошибка",
  cancelled: "Отменено",
  retrieval: "Поиск",
  graph: "Граф",
  external: "внешний файл",
  ambiguous: "несколько файлов",
  unresolved: "не привязан",
  approximate: "приблизительно",
  confirmed: "подтверждено",
};
const demoTitles: Record<string, string> = {
  "psf__requests-1963": "POST после перенаправления",
  "psf__requests-2674": "Исключения urllib3",
  "psf__requests-3362": "Unicode в потоковом ответе",
};
export const demoTitle = (item: Case) => demoTitles[item.id] || item.title;
export const shortRev = (s: Snapshot) =>
  s.revision === "upload" ? s.id.slice(0, 8) : s.revision.slice(0, 8);
export const terminal = (status: string) =>
  ["completed", "failed", "cancelled"].includes(status);
export const emptyQuery = (id = ""): Investigation => ({
  snapshot_id: id,
  title: "",
  description: "",
  traceback: "",
  trace_revision: null,
  mode: "hybrid",
  profile: "quality",
  graph: true,
  rerank: true,
});
export function caseQuery(item: Case, options: Investigation): Investigation {
  return {
    ...options,
    snapshot_id: item.snapshot_id,
    title: item.title,
    description: item.description,
    traceback: item.traceback || "",
    trace_revision: item.trace_revision,
  };
}
