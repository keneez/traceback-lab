"""One local service for immutable snapshots, jobs, and the compiled React UI."""
from asyncio import Semaphore
from collections import OrderedDict
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from threading import Event, Lock
import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .diagnostics import inspect_snapshot
from .uploads import MAX_UPLOAD, UploadLimit, import_source
from .cases import CaseStore, atomic_json
from .config import DATA_ROOT, ROOT, STATE_ROOT, SearchConfig, fingerprint
from .retrieval import SearchEngine
from .storage import SnapshotStore

LOGGER = logging.getLogger(__name__)


class InvestigationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_id: str
    title: str = Field(default="", max_length=1000)
    description: str = Field(default="", max_length=60000)
    traceback: str = Field(default="", max_length=60000)
    trace_revision: str | None = None
    profile: str = "quality"
    mode: str = "hybrid"
    graph: bool = True
    rerank: bool = True


class JobManager:
    def __init__(self, engine, directory):
        self.engine = engine
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="traceback-inference")
        self.lock = Lock()
        self.jobs, self.events = {}, {}
        self.cache = OrderedDict()
        for file in self.directory.glob("*.json"):
            try:
                row = json.loads(file.read_text(encoding="utf-8"))
                if row["status"] not in {"completed", "failed", "cancelled"}:
                    row.update(status="failed", error="Процесс завершился до окончания расследования")
                    self._save(row)
                self.jobs[row["id"]] = row
            except (ValueError, KeyError):
                LOGGER.warning("Skipping invalid job record %s", file.name)

    def _save(self, job):
        atomic_json(self.directory / f"{job['id']}.json", job)

    def submit(self, request):
        config = SearchConfig.profile(request.profile, request.mode, request.graph)
        config = replace(config, rerank=request.rerank)
        if not self.engine.store.manifest(request.snapshot_id)["symbol_count"]:
            raise ValueError("В снимке нет функций для поиска. Откройте «Проверки кода».")
        if not (request.title + request.description + request.traceback).strip():
            raise ValueError("Добавьте описание проблемы или traceback")
        key = fingerprint([request.model_dump(), self.engine.identity()])
        with self.lock:
            running = sum(j["status"] not in {"completed", "failed", "cancelled"}
                          for j in self.jobs.values())
            if running >= 32:
                raise OverflowError("Очередь заполнена")
            job_id = uuid.uuid4().hex
            job = {"id": job_id, "status": "queued", "created_at": time.time(),
                   "snapshot_id": request.snapshot_id, "request": request.model_dump(),
                   "result": None, "error": None, "cached": False}
            self.jobs[job_id] = job
            if key in self.cache:
                job.update(status="completed", result=self.cache[key], cached=True)
                self._save(job)
                return dict(job)
            self._save(job)
            self.events[job_id] = Event()
            self.pool.submit(self._run, job_id, request, config, key)
            return dict(job)

    def _run(self, job_id, request, config, key):
        def progress(stage):
            if self.events[job_id].is_set():
                raise CancelledError()
            with self.lock:
                if self.jobs[job_id]["status"] != stage:
                    self.jobs[job_id]["status"] = stage
                    self._save(self.jobs[job_id])
        try:
            result = self.engine.search(
                request.snapshot_id, request.title, request.description, request.traceback,
                config, request.trace_revision, progress)
            with self.lock:
                if self.events[job_id].is_set():
                    raise CancelledError()
                self.jobs[job_id].update(status="completed", result=result)
                self.cache[key] = result
                if len(self.cache) > 128:
                    self.cache.popitem(last=False)
        except CancelledError:
            with self.lock:
                self.jobs[job_id]["status"] = "cancelled"
        except Exception as error:
            LOGGER.exception("Investigation %s failed", job_id)
            with self.lock:
                self.jobs[job_id].update(status="failed", error=str(error))
        finally:
            with self.lock:
                self.events.pop(job_id, None)
                self._save(self.jobs[job_id])

    def recent(self, limit):
        with self.lock:
            rows = sorted(self.jobs.values(), key=lambda row: row["created_at"], reverse=True)[:limit]
            return [{key: row[key] for key in ("id", "created_at", "snapshot_id", "status")} | {
                "title": row["request"]["title"] or row["request"]["description"][:100] or "Traceback",
                "mode": row["request"]["mode"],
                "count": len((row.get("result") or {}).get("candidates", []))
            } for row in rows]

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return dict(self.jobs[job_id])

    def cancel(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            if job_id in self.events:
                self.events[job_id].set()
            return dict(self.jobs[job_id])

    def close(self):
        with self.lock:
            for event in self.events.values():
                event.set()
        self.pool.shutdown(wait=True)


def create_app(data_root=DATA_ROOT, state_root=STATE_ROOT, engine=None):
    data_root, state_root = Path(data_root), Path(state_root)
    store = engine.store if engine else SnapshotStore(data_root / "snapshots")
    uploads = SnapshotStore(state_root / "projects" / "snapshots")
    store.include(uploads)
    if engine is None:
        from .ml.compact import CompactEncoder
        engine = SearchEngine(store, CompactEncoder())
    cases = CaseStore(data_root)
    jobs = JobManager(engine, state_root / "jobs")

    import_slots = Semaphore(2)

    @lru_cache(maxsize=12)
    def diagnostics_for(snapshot_id):
        return inspect_snapshot(store, snapshot_id)

    @asynccontextmanager
    async def lifespan(app):
        yield
        jobs.close()
        diagnostics_for.cache_clear()

    app = FastAPI(title="Traceback Lab", version=__version__, lifespan=lifespan)
    app.state.engine = engine
    app.state.jobs = jobs

    @app.get("/v1/health")
    def health():
        return {"status": "ok", "snapshots": len(store.list())}

    @app.get("/v1/snapshots")
    def snapshots(compact: bool = False):
        rows = store.list()
        return [summary(row) for row in rows] if compact else rows

    def summary(row):
        return {key: row[key] for key in ("id", "repository", "revision", "symbol_count",
                                          "file_count", "edge_count")}

    @app.post("/v1/snapshots/import", status_code=201)
    async def upload(request: Request, filename: str = Query(min_length=1, max_length=255),
                     name: str | None = Query(default=None, max_length=100)):
        if request.headers.get("content-type", "").split(";")[0].lower() != "application/octet-stream":
            raise HTTPException(415, "Ожидается файл в теле запроса (application/octet-stream)")
        async with import_slots:
            payload = bytearray()
            async for chunk in request.stream():
                if len(payload) + len(chunk) > MAX_UPLOAD:
                    raise HTTPException(413, "Размер загрузки превышает 16 МиБ")
                payload.extend(chunk)
            try:
                row = await run_in_threadpool(import_source, uploads, payload, filename, name)
                return summary(row)
            except UploadLimit as error:
                raise HTTPException(413, str(error))
            except ValueError as error:
                raise HTTPException(422, str(error))

    @app.get("/v1/snapshots/{snapshot_id}/checks")
    def checks(snapshot_id: str):
        try:
            return diagnostics_for(snapshot_id)
        except KeyError:
            raise HTTPException(404, "Снимок не найден")

    @app.get("/v1/snapshots/{snapshot_id}/source")
    def source(snapshot_id: str, path: str = Query(min_length=1)):
        try:
            return store.source(snapshot_id, path)
        except KeyError:
            raise HTTPException(404, "Файл не найден в этом снимке")

    @app.get("/v1/cases")
    def list_cases():
        return cases.list()

    @app.get("/v1/models")
    def models():
        return {"identities": engine.identity(), "encoder_loaded": engine.encoder is not None,
                "reranker_loaded": engine.reranker is not None,
                "calibrator_loaded": engine.calibrator is not None,
                "encoder_name": getattr(engine.encoder, "name", None),
                "encoder_path": str(engine.encoder.path.name) if engine.encoder else None,
                "setup": "python run.py --neural"}

    @app.post("/v1/investigations", status_code=202)
    def investigate(request: InvestigationRequest):
        try:
            return jobs.submit(request)
        except KeyError:
            raise HTTPException(404, "Снимок не найден")
        except ValueError as error:
            raise HTTPException(422, str(error))
        except OverflowError as error:
            raise HTTPException(503, str(error))

    @app.get("/v1/investigations")
    def recent(limit: int = Query(default=40, ge=1, le=100)):
        return jobs.recent(limit)

    @app.get("/v1/investigations/{job_id}/baseline")
    def baseline(job_id: str):
        try:
            job = jobs.get(job_id)
        except KeyError:
            raise HTTPException(404, "Расследование не найдено")
        if job["status"] != "completed":
            raise HTTPException(409, "Дождитесь завершения расследования")
        req = InvestigationRequest(**job["request"])
        config = replace(SearchConfig.profile(req.profile, "bm25", False), rerank=False)
        return SearchEngine(store).search(req.snapshot_id, req.title, req.description,
                                          req.traceback, config, req.trace_revision)

    @app.get("/v1/investigations/{job_id}")
    def investigation(job_id: str):
        try:
            return jobs.get(job_id)
        except KeyError:
            raise HTTPException(404, "Расследование не найдено")

    @app.delete("/v1/investigations/{job_id}")
    def cancel(job_id: str):
        try:
            return jobs.cancel(job_id)
        except KeyError:
            raise HTTPException(404, "Расследование не найдено")

    @app.get("/v1/snapshots/{snapshot_id}/symbols/{symbol_id}")
    def symbol(snapshot_id: str, symbol_id: str):
        try:
            return store.symbol(snapshot_id, symbol_id)
        except KeyError:
            raise HTTPException(404, "Функция не найдена в этом снимке")

    @app.post("/v1/cases/{case_id}/reveal")
    def reveal(case_id: str):
        try:
            return cases.reveal(case_id)
        except KeyError:
            raise HTTPException(404, "Для этого кейса нет сохранённого исправления")

    @app.get("/v1/reports")
    def reports(compact: bool = False):
        output = []
        for path in sorted((ROOT / "reports").glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if compact:
                    value = {k: value[k] for k in ("summaries", "splits", "encoder_name") if k in value}
                output.append({"file": path.name, **value})
            except ValueError:
                LOGGER.warning("Invalid report %s", path.name)
        return output

    web = Path(__file__).parent / "web"
    if (web / "index.html").exists():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app

