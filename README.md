# 👁️ OpenCV Projects

A collection of real-time, webcam-based computer vision projects built with Python, OpenCV, and MediaPipe. Each project is a standalone script that demonstrates a different aspect of hand gesture recognition and interactive control.

---

## 🗂️ Projects

| Project | Description | Key Libraries |
|---|---|---|
| [Virtual Painter](./Virtual%20painter.py) | Draw on a live webcam feed using hand gestures. Raise one finger to draw, two fingers to hover and switch colors or erase from the sidebar. Press `c` to clear, `q` to quit. | `opencv-python`, `mediapipe`, `numpy` |
| [Virtual Object Gesture Control](./Virtual%20object%20gesture%20control.py) | Move and resize a virtual object overlaid on the webcam feed. Pinch with one hand to drag the object; pinch with both hands simultaneously to scale it. | `opencv-python`, `mediapipe` |
| [Volume Control using Hand](./Volume%20Control%20using%20hand.py) | Control your system's master volume by adjusting the distance between your thumb and index finger in front of the camera. A live volume bar is rendered on screen. *(Windows only)* | `opencv-python`, `mediapipe`, `pycaw` |

> **Adding a new project?** Just drop your `.py` file into the repo and add a row to the table above.

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **[OpenCV](https://opencv.org/)** — webcam capture and real-time video rendering
- **[MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)** (Tasks API) — hand landmark detection
- **[NumPy](https://numpy.org/)** — numerical operations and array handling
- **[pycaw](https://github.com/AndreMiras/pycaw)** — Windows Core Audio API wrapper (Volume Control project only)

> All projects that use MediaPipe's Hand Landmarker will **automatically download** the `hand_landmarker.task` model file on first run if it isn't already present.

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/ShreyanMehta/OpenCV-projects.git
cd OpenCV-projects
```

### 2. Install dependencies

Install the common dependencies:

```bash
pip install opencv-python mediapipe numpy
```

For the **Volume Control** project (Windows only):

```bash
pip install pycaw
```

### 3. Run a project

Each project is a self-contained script. Run whichever you want:

```bash
python "Virtual painter.py"
python "Virtual object gesture control.py"
python "Volume Control using hand.py"
```

---

## 📋 Requirements

| Requirement | Notes |
|---|---|
| Webcam | A standard USB or built-in webcam is required by all projects |
| Python 3.8+ | Tested with 3.8 and above |
| Windows | Required only for the Volume Control project (`pycaw` dependency) |
| Internet (first run) | Needed to auto-download the MediaPipe `hand_landmarker.task` model |

---

## 📁 Repository Structure

```
OpenCV-projects/
│
├── Virtual painter.py
├── Virtual object gesture control.py
├── Volume Control using hand.py
│
└── hand_landmarker.task       # Auto-downloaded on first run (not committed)
```

---

## 📄 License

This repository is open for personal and educational use.
