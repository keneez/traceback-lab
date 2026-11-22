from collections import Counter
import re

STOPWORDS = frozenset("""
a an the and or of to in on at for from by with is are was were be been being
it this that these those i we you he she they my our your as when then than
if but so can could should would will do does did has have had not no into
using use used after before also issue problem bug expected actual behavior
""".split())


def terms(text: str) -> list[str]:
    words = re.findall(r"[^\W\d]\w*", text, re.UNICODE)
    output = []
    for word in words:
        full = word.casefold()
        output.append(full)
        split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word)
        for part in re.split(r"[_\s]+", split):
            if part.casefold() != full and len(part) > 1:
                output.append(part.casefold())
    return output


def lexical_text(text: str):
    return " ".join(terms(text))


def query_terms(text: str, limit=60):
    count = Counter(w for w in terms(text) if len(w) > 1 and w not in STOPWORDS)
    return [w for w, _ in count.most_common(limit)]


def query_text(title: str, description: str, traceback: str):
    exception_lines = [s.strip() for s in traceback.splitlines()
                       if re.match(r"^[\w.]+(?:Error|Exception|Warning)\b", s.strip())]
    frames = re.findall(r'\bFile "([^"]+)", line (\d+), in ([^\n]+)', traceback)
    stack_summary = "\n".join(f"{p} {name}" for p, _, name in frames[-10:])
    return "\n".join(s for s in [title.strip(), "\n".join(exception_lines),
                                 stack_summary, description.strip()] if s)

