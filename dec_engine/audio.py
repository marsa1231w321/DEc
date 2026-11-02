import os
import shutil
import subprocess
import wave


class AudioPlayer:
    def __init__(self):
        self.ffplay = shutil.which("ffplay")
        self.ffprobe = shutil.which("ffprobe")
        self.process = None
        self._path = None
        self._loop = False

    def validate(self, path):
        if not os.path.isfile(path):
            raise ValueError(f"Аудиофайл не найден: {path}")
        extension = os.path.splitext(path)[1].lower()
        if extension == ".wav":
            try:
                with wave.open(path, "rb") as sound:
                    if sound.getframerate() <= 0 or (
                        not self.ffplay and sound.getnchannels() not in (1, 2)
                    ):
                        raise ValueError("WAV должен содержать один или два аудиоканала.")
            except (OSError, EOFError, wave.Error) as error:
                if not (self.ffplay and self.ffprobe):
                    raise ValueError(
                        "Файл с расширением .wav не является корректным RIFF/WAVE."
                    ) from error
                try:
                    self._probe(path)
                except ValueError as probe_error:
                    raise ValueError(
                        "Файл с расширением .wav не является корректным RIFF/WAVE."
                    ) from probe_error
        elif not self.ffplay:
            raise ValueError(
                "Для этого аудиоформата нужен ffplay. Установите FFmpeg и добавьте его в PATH."
            )
        elif self.ffprobe:
            self._probe(path)

    def _probe(self, path):
        if not self.ffprobe:
            return
        try:
            result = subprocess.run(
                [
                    self.ffprobe, "-v", "error", "-select_streams", "a:0",
                    "-show_entries", "stream=codec_name",
                    "-of", "default=noprint_wrappers=1:nokey=1", path,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=8,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ValueError("Проверка аудиофайла превысила лимит 8 секунд.") from error
        if result.returncode or not result.stdout.strip():
            detail = result.stderr.strip().splitlines()
            message = detail[-1] if detail else "Аудиодорожка не найдена."
            raise ValueError(f"Файл не удалось декодировать: {message}")

    def play(self, path, loop=False):
        self.validate(path)
        self.stop()
        if not self.ffplay:
            import winsound

            flags = winsound.SND_FILENAME | winsound.SND_ASYNC
            if loop:
                flags |= winsound.SND_LOOP
            winsound.PlaySound(path, flags)
        else:
            self.process = subprocess.Popen(
                [self.ffplay, "-nodisp", "-autoexit", "-loglevel", "error", path],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        self._path = path
        self._loop = loop

    def update(self):
        if self.process is not None and self.process.poll() is not None:
            return_code = self.process.returncode
            self.process = None
            if return_code:
                self._path = None
                self._loop = False
                raise RuntimeError("FFmpeg не смог декодировать или воспроизвести аудиофайл.")
            if self._loop and self._path:
                path = self._path
                self.play(path, loop=True)

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
        elif not self.ffplay:
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        self._path = None
        self._loop = False
