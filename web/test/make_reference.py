"""Reference stems for the golden test, made by UtauCover's own separator (python-audio-separator).
Needs UtauCover's environment. Run once per quality, each in its own process (memory):

    python web/test/make_reference.py excerpt
    python web/test/make_reference.py fast
    python web/test/make_reference.py best
"""
import shutil, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent / "golden"
EXCERPT = HERE / "excerpt.wav"
SONG = Path.home() / "Downloads/Here Come The Cats.mp3"

what = sys.argv[1]
if what == "excerpt":
    import librosa, soundfile as sf
    y, _ = librosa.load(str(SONG), sr=44100, mono=False, offset=50, duration=30)
    p = np.abs(y).max()
    if p > 0.98:
        y = y * (0.98 / p)
    sf.write(str(EXCERPT), y.T, 44100, subtype="PCM_16")
    print("excerpt peak", p, "->", np.abs(y).max())
else:
    from utaucover.separate import StemSeparator
    out = HERE / what
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        sep = StemSeparator(tmp, quality=what, progress=print)
        stems = sep.separate(EXCERPT, backing_split=True)
        for name, path in stems.items():
            shutil.copy2(path, out / f"{name}.wav")
        sep.close()
    print("reference stems:", sorted(p.name for p in out.iterdir()))
