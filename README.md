# PyMorse

> **A minimal pure-Python Morse-code encoder/decoder**
> *Re-implementation of the excellent [`cduck/morse`](https://github.com/cduck/morse).*
> MIT-licensed.

---

## Features

* **Encode** ASCII text → CW audio (NumPy buffer or WAV file).
* **Decode** CW audio (WAV or NumPy array) → uppercase text.
* Optional real-time playback with `sounddevice`.
* Single public class: `MorseCode` – no CLI bundled.

---

## Requirements

|   Purpose    |      Package      |         Version        |
|--------------|-------------------|------------------------|
|   **Core**   | `numpy`           | ≥ 2.0.0                |
|              | `scipy`           | ≥ 1.10                 |
| **Optional** | `sounddevice`     | ≥ 0.4 &nbsp;*(audio playback)* |

Python **3.9 +** is required (uses `importlib.metadata`, included in the std-lib).

---

## Installation

### Directly from GitHub

```bash
# core only
pip install git+https://github.com/0x-FFFFFF/PyMorse.git#egg=PyMorse

# with optional audio playback support
pip install "git+https://github.com/0x-FFFFFF/PyMorse.git#egg=PyMorse[audio]"
```

---

## Quick Start

```py
from pymorse import MorseCode

mc = MorseCode(wpm=60, hz=750, fs=20)

# text to morse wav
mc.to_wav("output.wav", "0xFFFFFF")

# morse wav to text
print(mc.from_wav("output.wav"))
```

## Credits

This project stands on the shoulders of [`cduck/morse`](https://github.com/cduck/morse)’s original morse implementation.
