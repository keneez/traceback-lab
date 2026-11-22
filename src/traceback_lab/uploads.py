"""Read the user's source as data. Archive members are never extracted or executed."""
from pathlib import PurePosixPath, PureWindowsPath
import io
import stat
import zipfile
import zlib

from .indexing import EXCLUDED, build_snapshot, in_scope

MIB = 1024 * 1024
MAX_UPLOAD = 16 * MIB
MAX_SOURCE = 32 * MIB
MAX_FILE = MIB
MAX_FILES = 1000
MAX_ENTRIES = 10000


class UploadLimit(ValueError):
    pass


def safe_path(name):
    name = name.replace("\\", "/")
    if (not name or name.startswith("/") or PureWindowsPath(name).drive or
            ".." in name.split("/") or any(ord(c) < 32 for c in name)):
        raise ValueError("Недопустимый путь внутри загруженного файла")
    path = PurePosixPath(name)
    if not path.parts:
        raise ValueError("Пустое имя файла")
    return path


def read_zip(payload, stem):
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ENTRIES:
                raise UploadLimit("В ZIP больше 10 000 записей")
            candidates = []
            for entry in entries:
                path = safe_path(entry.filename)
                if entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16):
                    continue
                if path.suffix != ".py" or "__MACOSX" in path.parts:
                    continue
                candidates.append((entry, path))
            # Remove a typical GitHub archive envelope, keeping importable packages intact.
            wrapped = bool(candidates) and all(len(p.parts) > 1 and p.parts[0] == stem
                                               for _, p in candidates)
            wrapped = wrapped and stem not in EXCLUDED and not any(
                p.parts == (stem, "__init__.py") for _, p in candidates)
            files, total = {}, 0
            for entry, path in candidates:
                relative = PurePosixPath(*path.parts[1:]).as_posix() if wrapped else path.as_posix()
                if not in_scope(relative):
                    continue
                if entry.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                    raise ValueError("Пересохраните ZIP с обычным сжатием Deflate или без сжатия")
                if entry.flag_bits & 1:
                    raise ValueError("Для импорта нужен ZIP без пароля")
                if relative in files:
                    raise ValueError(f"Одинаковые пути в архиве: {relative}")
                if entry.file_size > MAX_FILE or total + entry.file_size > MAX_SOURCE:
                    raise UploadLimit("Лимит исходников — 1 МиБ на файл и 32 МиБ на проект")
                if len(files) >= MAX_FILES:
                    raise UploadLimit("В одном импорте поддерживается до 1000 Python-файлов")
                with archive.open(entry) as stream:
                    data = stream.read(MAX_FILE + 1)
                if len(data) != entry.file_size:
                    raise ValueError("Размер файла не совпадает с каталогом ZIP")
                files[relative] = data
                total += len(data)
            return files
    except (zipfile.BadZipFile, EOFError, NotImplementedError, zlib.error, OSError) as exc:
        raise ValueError("Повреждённый или неподдерживаемый ZIP") from exc


def import_source(store, payload, filename, name=None):
    path = safe_path(filename)
    if len(path.parts) != 1:
        raise ValueError("Передайте имя файла без пути")
    if len(payload) > MAX_UPLOAD:
        raise UploadLimit("Размер загрузки превышает 16 МиБ")
    if path.suffix.lower() == ".zip":
        files = read_zip(payload, path.stem)
    elif path.suffix == ".py":
        if len(payload) > MAX_FILE:
            raise UploadLimit("Для одного Python-файла лимит составляет 1 МиБ")
        files = {path.name: payload} if in_scope(path.name) else {}
    else:
        raise ValueError("Выберите Python-файл (.py) или ZIP с исходниками")
    if not files:
        raise ValueError("Нет Python-файлов для индекса. Каталоги тестов и зависимостей пропускаются.")
    project = (name or path.stem).strip() or path.stem
    if len(project) > 100 or any(ord(c) < 32 for c in project):
        raise ValueError("Укажите название проекта длиной до 100 символов")
    return build_snapshot(store, files, project, "upload", allow_empty=True)
