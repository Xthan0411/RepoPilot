"""Persistent host-side artifacts copied out of ephemeral task execution."""

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import suppress
from pathlib import Path
from uuid import UUID, uuid4

from repopilot.schemas import ArtifactRef

_ARTIFACT_DIRECTORY_MODE = 0o700
_ARTIFACT_FILE_MODE = 0o600


class SandboxArtifactError(RuntimeError):
    """Safe artifact-store failure that omits the host path."""


class SandboxArtifactStore:
    """Write immutable, hash-addressed metadata beneath one trusted output root."""

    def __init__(self, root: Path) -> None:
        if root.is_symlink():
            raise SandboxArtifactError("sandbox artifact root cannot be a symlink")
        try:
            root.mkdir(parents=True, exist_ok=True, mode=_ARTIFACT_DIRECTORY_MODE)
            resolved = root.resolve(strict=True)
        except OSError as error:
            raise SandboxArtifactError("sandbox artifact root is unavailable") from error
        if not resolved.is_dir():
            raise SandboxArtifactError("sandbox artifact root must be a directory")
        self._root = resolved
        self._sandbox_id: str | None = None
        self._sandbox_directory: Path | None = None
        self._artifacts: dict[str, ArtifactRef] = {}

    @property
    def root(self) -> Path:
        return self._root

    def prepare(self, sandbox_id: str, *, repository_source: Path) -> None:
        """Create exactly one output namespace disjoint from the repository source."""

        if self._sandbox_id is not None:
            raise SandboxArtifactError("sandbox artifact store is already prepared")
        if (
            self._root == repository_source
            or self._root in repository_source.parents
            or repository_source in self._root.parents
        ):
            raise SandboxArtifactError(
                "sandbox artifact root must be separate from the repository source"
            )
        directory = self._root / sandbox_id
        try:
            directory.mkdir(mode=_ARTIFACT_DIRECTORY_MODE)
        except OSError as error:
            raise SandboxArtifactError("sandbox artifact namespace is unavailable") from error
        self._sandbox_id = sandbox_id
        self._sandbox_directory = directory

    def persist(
        self,
        call_id: UUID,
        stream_name: str,
        content: bytes,
    ) -> ArtifactRef:
        """Atomically persist one stdout/stderr/read stream without overwriting."""

        if self._sandbox_id is None or self._sandbox_directory is None:
            raise SandboxArtifactError("sandbox artifact store is not prepared")
        if stream_name not in {"read", "stderr", "stdout"}:
            raise SandboxArtifactError("sandbox artifact stream is invalid")
        filename = f"{call_id}.{stream_name}.txt"
        return self._persist_file(
            filename,
            content,
            media_type="text/plain; charset=utf-8",
        )

    def persist_patch(self, call_id: UUID, content: bytes) -> ArtifactRef:
        """Persist one immutable candidate unified diff for contained application."""

        if self._sandbox_id is None or self._sandbox_directory is None:
            raise SandboxArtifactError("sandbox artifact store is not prepared")
        return self._persist_file(
            f"{call_id}.patch.diff",
            content,
            media_type="text/x-diff; charset=utf-8",
        )

    def read(self, artifact: ArtifactRef) -> bytes:
        """Re-read one registered artifact and verify its complete identity."""

        if self._sandbox_id is None or self._sandbox_directory is None:
            raise SandboxArtifactError("sandbox artifact store is not prepared")
        filename = Path(artifact.relative_path).name
        if self._artifacts.get(filename) != artifact:
            raise SandboxArtifactError("sandbox artifact reference is not registered")
        destination = self._sandbox_directory / filename
        try:
            descriptor = os.open(destination, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as stream:
                metadata = os.fstat(stream.fileno())
                if not stat.S_ISREG(metadata.st_mode):
                    raise SandboxArtifactError("sandbox artifact is not a regular file")
                content = stream.read()
        except OSError as error:
            raise SandboxArtifactError("sandbox artifact read failed") from error
        if (
            len(content) != artifact.size_bytes
            or hashlib.sha256(content).hexdigest() != artifact.sha256
        ):
            raise SandboxArtifactError("sandbox artifact integrity check failed")
        return content

    def _persist_file(
        self,
        filename: str,
        content: bytes,
        *,
        media_type: str,
    ) -> ArtifactRef:
        if self._sandbox_id is None or self._sandbox_directory is None:
            raise SandboxArtifactError("sandbox artifact store is not prepared")
        if filename in self._artifacts:
            raise SandboxArtifactError("sandbox artifact already exists")
        destination = self._sandbox_directory / filename
        temporary = self._sandbox_directory / f".{filename}.{uuid4().hex}.tmp"
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                _ARTIFACT_FILE_MODE,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
            with suppress(OSError):
                temporary.unlink()
        except OSError as error:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            raise SandboxArtifactError("sandbox artifact write failed") from error
        artifact = ArtifactRef(
            relative_path=f"artifacts/{self._sandbox_id}/{filename}",
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            media_type=media_type,
        )
        self._artifacts[filename] = artifact
        return artifact

    def collect(self) -> tuple[ArtifactRef, ...]:
        """Return stable references even after the Docker container is destroyed."""

        if self._sandbox_id is None:
            raise SandboxArtifactError("sandbox artifact store is not prepared")
        return tuple(self._artifacts[name] for name in sorted(self._artifacts))
