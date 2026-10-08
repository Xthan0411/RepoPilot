"""Fixed Python helpers executed inside the isolated Docker workspace."""

COPY_PATCH_HELPER = """\
import hashlib
import os
import re
import sys

name = sys.argv[1]
maximum = int(sys.argv[2])
expected_sha256 = sys.argv[3]
if re.fullmatch(r"[0-9a-f-]{36}\\.patch\\.diff", name) is None:
    raise SystemExit(41)
content = sys.stdin.buffer.read(maximum + 1)
if len(content) > maximum or hashlib.sha256(content).hexdigest() != expected_sha256:
    raise SystemExit(42)
root_fd = os.open("/workspace/artifacts", os.O_RDONLY | os.O_DIRECTORY)
try:
    file_fd = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=root_fd,
    )
    with os.fdopen(file_fd, "wb") as target:
        target.write(content)
        target.flush()
        os.fsync(target.fileno())
finally:
    os.close(root_fd)
"""


APPLY_PATCH_HELPER = """\
import hashlib
import json
import os
import re
import stat
import sys
import uuid

MARKER = "repopilot_patch_error:"
DENIED = {".aws", ".codex", ".config", ".docker", ".env", ".git", ".ssh", "artifacts"}
HEADER = re.compile(r"^diff --git a/(\\S+) b/(\\S+)\\n$")
HUNK = re.compile(r"^@@ -(\\d+)(?:,(\\d+))? \\+(\\d+)(?:,(\\d+))? @@(?: [^\\r\\n]*)?\\n$")
UNSUPPORTED = (
    "Binary files ", "GIT binary patch", "copy from ", "copy to ",
    "deleted file mode ", "new file mode ", "old mode ", "new mode ",
    "rename from ", "rename to ", "similarity index ",
)

class PatchFailure(Exception):
    def __init__(self, code):
        self.code = code

def fail(code, status):
    sys.stderr.write(MARKER + code + "\\n")
    raise SystemExit(status)

def valid_path(value):
    if (
        not value or value.startswith("/") or "\\x00" in value or "\\\\" in value
        or any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in value
        )
    ):
        raise PatchFailure("path_denied")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts) or parts[0] in DENIED:
        raise PatchFailure("path_denied")
    if "/".join(parts) != value:
        raise PatchFailure("path_denied")
    return value

def parse_patch(text, max_files, max_hunks, max_changed):
    if "\\x00" in text or "\\r" in text or not text.endswith("\\n"):
        raise PatchFailure("invalid_diff")
    lines = text.splitlines(keepends=True)
    if not lines or any(line.startswith(prefix) for line in lines for prefix in UNSUPPORTED):
        raise PatchFailure("unsupported_patch")
    index = 0
    files = []
    total_hunks = 0
    total_changed = 0
    while index < len(lines):
        match = HEADER.fullmatch(lines[index])
        if match is None or match.group(1) != match.group(2):
            raise PatchFailure("invalid_diff")
        path = valid_path(match.group(1))
        if path in [item[0] for item in files]:
            raise PatchFailure("invalid_diff")
        if len(files) + 1 > max_files:
            raise PatchFailure("too_many_files")
        index += 1
        if index < len(lines) and lines[index].startswith("index "):
            index += 1
        if (
            index + 1 >= len(lines)
            or lines[index] != "--- a/" + path + "\\n"
            or lines[index + 1] != "+++ b/" + path + "\\n"
        ):
            raise PatchFailure("unsupported_patch")
        index += 2
        hunks = []
        previous_end = 0
        previous_new_end = 0
        while index < len(lines) and not lines[index].startswith("diff --git "):
            match = HUNK.fullmatch(lines[index])
            if match is None:
                raise PatchFailure("invalid_diff")
            old_start = int(match.group(1))
            old_count = int(match.group(2) or "1")
            new_start = int(match.group(3))
            new_count = int(match.group(4) or "1")
            if (
                (old_count > 0 and old_start == 0)
                or (new_count > 0 and new_start == 0)
                or old_start < previous_end
                or new_start < previous_new_end
            ):
                raise PatchFailure("invalid_diff")
            previous_end = old_start + old_count
            previous_new_end = new_start + new_count
            total_hunks += 1
            if total_hunks > max_hunks:
                raise PatchFailure("too_many_hunks")
            index += 1
            body = []
            observed_old = 0
            observed_new = 0
            changed = 0
            while (
                index < len(lines)
                and not lines[index].startswith("@@ ")
                and not lines[index].startswith("diff --git ")
            ):
                line = lines[index]
                if line.startswith("\\\\"):
                    raise PatchFailure("unsupported_patch")
                if not line or line[0] not in {" ", "+", "-"}:
                    raise PatchFailure("invalid_diff")
                observed_old += line[0] in {" ", "-"}
                observed_new += line[0] in {" ", "+"}
                changed += line[0] in {"+", "-"}
                body.append(line)
                index += 1
            if observed_old != old_count or observed_new != new_count or changed == 0:
                raise PatchFailure("invalid_diff")
            total_changed += changed
            if total_changed > max_changed:
                raise PatchFailure("too_many_changed_lines")
            hunks.append((old_start, old_count, new_start, new_count, body))
        if not hunks:
            raise PatchFailure("invalid_diff")
        files.append((path, hunks))
    return files

def open_parent(root_fd, path):
    current_fd = os.dup(root_fd)
    try:
        parts = path.split("/")
        for part in parts[:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current_fd,
            )
            os.close(current_fd)
            current_fd = next_fd
        return current_fd, parts[-1]
    except Exception:
        os.close(current_fd)
        raise

def load_file(root_fd, path):
    parent_fd, name = open_parent(root_fd, path)
    try:
        file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        with os.fdopen(file_fd, "rb") as source:
            metadata = os.fstat(source.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError
            content = source.read()
        text = content.decode("utf-8")
        return content, text, metadata.st_mode & 0o777, metadata.st_dev, metadata.st_ino
    finally:
        os.close(parent_fd)

def apply_hunks(source, hunks):
    source_lines = source.splitlines(keepends=True)
    output = []
    cursor = 0
    for old_start, old_count, _new_start, _new_count, body in hunks:
        start = old_start if old_count == 0 else old_start - 1
        if start < cursor or start > len(source_lines):
            raise PatchFailure("apply_conflict")
        output.extend(source_lines[cursor:start])
        position = start
        for line in body:
            value = line[1:]
            if line[0] in {" ", "-"}:
                if position >= len(source_lines) or source_lines[position] != value:
                    raise PatchFailure("apply_conflict")
                position += 1
            if line[0] in {" ", "+"}:
                output.append(value)
        if position - start != old_count:
            raise PatchFailure("apply_conflict")
        cursor = position
    output.extend(source_lines[cursor:])
    return "".join(output).encode("utf-8")

def write_temp(parent_fd, mode, content):
    name = ".repopilot-" + uuid.uuid4().hex + ".tmp"
    file_fd = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=parent_fd,
    )
    try:
        os.fchmod(file_fd, mode)
        with os.fdopen(file_fd, "wb", closefd=False) as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
    finally:
        os.close(file_fd)
    return name

name = sys.argv[1]
expected_size = int(sys.argv[2])
expected_sha = sys.argv[3]
max_files = int(sys.argv[4])
max_hunks = int(sys.argv[5])
max_changed = int(sys.argv[6])
artifact_fd = -1
artifact_root_fd = -1
repository_fd = -1
staged = []
replaced = []
try:
    if re.fullmatch(r"[0-9a-f-]{36}\\.patch\\.diff", name) is None:
        raise PatchFailure("artifact_mismatch")
    artifact_root_fd = os.open("/workspace/artifacts", os.O_RDONLY | os.O_DIRECTORY)
    artifact_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=artifact_root_fd)
    with os.fdopen(artifact_fd, "rb", closefd=False) as source:
        content = source.read(expected_size + 1)
    if len(content) != expected_size or hashlib.sha256(content).hexdigest() != expected_sha:
        raise PatchFailure("artifact_mismatch")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise PatchFailure("invalid_diff") from None
    parsed = parse_patch(text, max_files, max_hunks, max_changed)
    repository_fd = os.open("/workspace/repository", os.O_RDONLY | os.O_DIRECTORY)
    prepared = []
    for path, hunks in parsed:
        original, source_text, mode, device, inode = load_file(repository_fd, path)
        replacement = apply_hunks(source_text, hunks)
        if replacement == original:
            raise PatchFailure("apply_conflict")
        prepared.append((path, original, replacement, mode, device, inode))
    for path, original, replacement, mode, device, inode in prepared:
        parent_fd, target = open_parent(repository_fd, path)
        metadata = os.stat(target, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_dev != device
            or metadata.st_ino != inode
        ):
            os.close(parent_fd)
            raise PatchFailure("apply_conflict")
        temporary = write_temp(parent_fd, mode, replacement)
        staged.append((path, parent_fd, target, temporary, original, mode))
    try:
        for item in staged:
            path, parent_fd, target, temporary, original, mode = item
            os.replace(temporary, target, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            replaced.append(item)
    except OSError:
        rollback_failed = False
        for item in reversed(replaced):
            _path, parent_fd, target, _temporary, original, mode = item
            try:
                rollback = write_temp(parent_fd, mode, original)
                os.replace(rollback, target, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            except OSError:
                rollback_failed = True
        if rollback_failed:
            raise PatchFailure("rollback_failed") from None
        raise PatchFailure("apply_conflict") from None
    sys.stdout.write(json.dumps({
        "files_changed": [path for path, _hunks in parsed],
        "patch_sha256": expected_sha,
    }, separators=(",", ":"), sort_keys=True))
except PatchFailure as error:
    statuses = {
        "artifact_mismatch": 42,
        "path_denied": 43,
        "invalid_diff": 44,
        "unsupported_patch": 44,
        "too_many_files": 44,
        "too_many_hunks": 44,
        "too_many_changed_lines": 44,
        "apply_conflict": 45,
        "rollback_failed": 46,
    }
    fail(error.code, statuses.get(error.code, 44))
except (OSError, UnicodeError, ValueError):
    fail("apply_conflict", 45)
finally:
    for _path, parent_fd, _target, temporary, _original, _mode in staged:
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except OSError:
            pass
        os.close(parent_fd)
    if repository_fd >= 0:
        os.close(repository_fd)
    if artifact_fd >= 0:
        os.close(artifact_fd)
    if artifact_root_fd >= 0:
        os.close(artifact_root_fd)
"""
