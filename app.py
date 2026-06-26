# Run with: python app.py

import threading
import time
from collections import Counter
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

import config
from detector import ObjectDetector
from display import FrameRenderer
from video_source import VideoSource

_PROJECT_DIR = Path(__file__).resolve().parent
_DISPLAY_SIZE = (800, 450)
_JPEG_QUALITY = 70

app = Flask(__name__)

_state = {
    "running": False,
    "stop_requested": False,
    "total_frames": 0,
    "fps_samples": [],
    "last_detections": [],
    "lock": threading.Lock(),
}
_stream_lock = threading.Lock()

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>YOLOv8 Object Detection System</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #1a1a2e;
      --bg-panel: #16213e;
      --bg-card: #0f3460;
      --accent: #00d4ff;
      --accent-dim: rgba(0, 212, 255, 0.15);
      --text: #e8e8e8;
      --text-muted: #a0a0b8;
      --green: #00c853;
      --red: #ff5252;
      --border: rgba(0, 212, 255, 0.25);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }
    .layout {
      display: flex;
      min-height: 100vh;
    }
    .sidebar {
      width: 300px;
      min-width: 300px;
      background: var(--bg-panel);
      border-right: 1px solid var(--border);
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.1rem;
    }
    .sidebar h1 {
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--accent);
      line-height: 1.35;
      letter-spacing: 0.02em;
    }
    .field label {
      display: block;
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 0.45rem;
    }
    .radio-group {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }
    .radio-group label {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.9rem;
      font-weight: 400;
      text-transform: none;
      letter-spacing: 0;
      color: var(--text);
      cursor: pointer;
    }
    .radio-group input { accent-color: var(--accent); }
    input[type="text"] {
      width: 100%;
      padding: 0.55rem 0.75rem;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      color: var(--text);
      font-family: inherit;
      font-size: 0.9rem;
    }
    input[type="text"]:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 2px var(--accent-dim);
    }
    .slider-row {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 0.35rem;
    }
    .slider-value {
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--accent);
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    .btn-row {
      display: flex;
      gap: 0.6rem;
    }
    button {
      flex: 1;
      padding: 0.65rem 0.75rem;
      border: none;
      border-radius: 6px;
      font-family: inherit;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.15s, transform 0.1s;
    }
    button:hover { opacity: 0.9; transform: translateY(-1px); }
    button:active { transform: translateY(0); }
    .btn-start { background: var(--green); color: #0a1a0a; }
    .btn-stop { background: var(--red); color: #fff; }
    .stats {
      margin-top: auto;
      padding-top: 1rem;
      border-top: 1px solid var(--border);
    }
    .stat-item {
      display: flex;
      justify-content: space-between;
      padding: 0.45rem 0;
      font-size: 0.88rem;
    }
    .stat-item span:first-child { color: var(--text-muted); }
    .stat-item span:last-child { font-weight: 600; color: var(--accent); }
    .main {
      flex: 1;
      padding: 1.5rem 2rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      overflow-y: auto;
    }
    .video-wrap {
      background: var(--bg-panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 450px;
    }
    #videoFeed {
      width: 800px;
      max-width: 100%;
      height: 450px;
      object-fit: contain;
      background: #0d0d1a;
      display: none;
    }
    .video-placeholder {
      color: var(--text-muted);
      font-size: 0.95rem;
      padding: 2rem;
      text-align: center;
    }
    .chart-card, .table-card {
      background: var(--bg-panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.25rem;
    }
    .chart-card h2, .table-card h2 {
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 1rem;
    }
    .chart-container { height: 220px; position: relative; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }
    th {
      text-align: left;
      padding: 0.6rem 0.75rem;
      color: var(--accent);
      font-weight: 600;
      border-bottom: 1px solid var(--border);
    }
    td {
      padding: 0.55rem 0.75rem;
      border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    tr:hover td { background: var(--accent-dim); }
    .empty-row td {
      text-align: center;
      color: var(--text-muted);
      font-style: italic;
    }
    .error-toast {
      display: none;
      background: rgba(255, 82, 82, 0.15);
      border: 1px solid var(--red);
      color: #ffb4b4;
      padding: 0.65rem 0.85rem;
      border-radius: 6px;
      font-size: 0.85rem;
    }
    .error-toast.visible { display: block; }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1>YOLOv8 Object Detection System</h1>

      <div class="field">
        <label>Input source</label>
        <div class="radio-group">
          <label><input type="radio" name="source" value="webcam"> Webcam</label>
          <label><input type="radio" name="source" value="file" checked> Video File</label>
        </div>
      </div>

      <div class="field" id="videoFileField">
        <label for="videoFilename">Video filename</label>
        <input type="text" id="videoFilename" value="Test_Video.mp4">
      </div>

      <div class="field">
        <div class="slider-row">
          <label for="confidence">Confidence Threshold</label>
          <span class="slider-value" id="confidenceValue">0.35</span>
        </div>
        <input type="range" id="confidence" min="0.10" max="0.90" step="0.05" value="0.35">
      </div>

      <div class="field">
        <div class="slider-row">
          <label for="iou">IOU Threshold</label>
          <span class="slider-value" id="iouValue">0.50</span>
        </div>
        <input type="range" id="iou" min="0.10" max="0.90" step="0.05" value="0.50">
      </div>

      <div id="errorToast" class="error-toast"></div>

      <div class="btn-row">
        <button type="button" class="btn-start" id="btnStart">Start Detection</button>
        <button type="button" class="btn-stop" id="btnStop">Stop Detection</button>
      </div>

      <div class="stats">
        <div class="stat-item">
          <span>Total Frames Processed</span>
          <span id="statFrames">0</span>
        </div>
        <div class="stat-item">
          <span>Average FPS</span>
          <span id="statFps">0.0</span>
        </div>
      </div>
    </aside>

    <main class="main">
      <div class="video-wrap">
        <p class="video-placeholder" id="videoPlaceholder">Click <strong>Start Detection</strong> in the sidebar to begin.</p>
        <img id="videoFeed" alt="Live detection feed">
      </div>

      <div class="chart-card">
        <h2>Detected classes (live)</h2>
        <div class="chart-container">
          <canvas id="classChart"></canvas>
        </div>
      </div>

      <div class="table-card">
        <h2>Current frame detections</h2>
        <table>
          <thead>
            <tr>
              <th>Class</th>
              <th>Count</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody id="detectionTable">
            <tr class="empty-row"><td colspan="3">No detections yet</td></tr>
          </tbody>
        </table>
      </div>
    </main>
  </div>

  <script>
    const confidenceSlider = document.getElementById('confidence');
    const iouSlider = document.getElementById('iou');
    const confidenceValue = document.getElementById('confidenceValue');
    const iouValue = document.getElementById('iouValue');
    const videoFeed = document.getElementById('videoFeed');
    const videoPlaceholder = document.getElementById('videoPlaceholder');
    const errorToast = document.getElementById('errorToast');
    const videoFileField = document.getElementById('videoFileField');
    const statFrames = document.getElementById('statFrames');
    const statFps = document.getElementById('statFps');
    const detectionTable = document.getElementById('detectionTable');

    let statsInterval = null;
    let detectionActive = false;

    const chartCtx = document.getElementById('classChart').getContext('2d');
    const classChart = new Chart(chartCtx, {
      type: 'bar',
      data: {
        labels: [],
        datasets: [{
          label: 'Count',
          data: [],
          backgroundColor: 'rgba(0, 212, 255, 0.65)',
          borderColor: '#00d4ff',
          borderWidth: 1,
          borderRadius: 4,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
        },
        scales: {
          x: {
            ticks: { color: '#a0a0b8', font: { family: 'Inter' } },
            grid: { color: 'rgba(255,255,255,0.06)' },
          },
          y: {
            beginAtZero: true,
            ticks: {
              color: '#a0a0b8',
              font: { family: 'Inter' },
              stepSize: 1,
            },
            grid: { color: 'rgba(255,255,255,0.06)' },
          },
        },
      },
    });

    function showError(msg) {
      errorToast.textContent = msg;
      errorToast.classList.add('visible');
    }

    function hideError() {
      errorToast.classList.remove('visible');
    }

    function updateSliderDisplays() {
      confidenceValue.textContent = parseFloat(confidenceSlider.value).toFixed(2);
      iouValue.textContent = parseFloat(iouSlider.value).toFixed(2);
    }

    confidenceSlider.addEventListener('input', updateSliderDisplays);
    iouSlider.addEventListener('input', updateSliderDisplays);
    updateSliderDisplays();

    document.querySelectorAll('input[name="source"]').forEach((radio) => {
      radio.addEventListener('change', () => {
        videoFileField.style.display =
          document.querySelector('input[name="source"]:checked').value === 'file'
            ? 'block'
            : 'none';
      });
    });

    function getSettingsPayload() {
      const source = document.querySelector('input[name="source"]:checked').value;
      return {
        source,
        video_filename: document.getElementById('videoFilename').value,
        confidence: parseFloat(confidenceSlider.value),
        iou: parseFloat(iouSlider.value),
      };
    }

    function updateChartFromDetections(detections) {
      const counts = {};
      detections.forEach((d) => {
        counts[d.class_name] = (counts[d.class_name] || 0) + 1;
      });
      const labels = Object.keys(counts).sort();
      classChart.data.labels = labels;
      classChart.data.datasets[0].data = labels.map((l) => counts[l]);
      classChart.update('none');
    }

    function updateTable(detections) {
      if (!detections || detections.length === 0) {
        detectionTable.innerHTML =
          '<tr class="empty-row"><td colspan="3">No detections in current frame</td></tr>';
        return;
      }
      const grouped = {};
      detections.forEach((d) => {
        if (!grouped[d.class_name]) {
          grouped[d.class_name] = { count: 0, maxConf: 0 };
        }
        grouped[d.class_name].count += 1;
        grouped[d.class_name].maxConf = Math.max(grouped[d.class_name].maxConf, d.confidence);
      });
      const rows = Object.entries(grouped)
        .sort((a, b) => a[0].localeCompare(b[0]))
        .map(
          ([name, info]) =>
            `<tr><td>${name}</td><td>${info.count}</td><td>${info.maxConf.toFixed(3)}</td></tr>`
        )
        .join('');
      detectionTable.innerHTML = rows;
    }

    async function fetchStats() {
      try {
        const res = await fetch('/stats');
        const data = await res.json();
        statFrames.textContent = data.total_frames;
        statFps.textContent = data.fps.toFixed(1);
        updateChartFromDetections(data.detections || []);
        updateTable(data.detections || []);
      } catch (e) {
        console.warn('Stats fetch failed', e);
      }
    }

    function startStatsPolling() {
      if (statsInterval) clearInterval(statsInterval);
      fetchStats();
      statsInterval = setInterval(fetchStats, 1000);
    }

    function stopStatsPolling() {
      if (statsInterval) {
        clearInterval(statsInterval);
        statsInterval = null;
      }
    }

    document.getElementById('btnStart').addEventListener('click', async () => {
      hideError();
      try {
        const res = await fetch('/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(getSettingsPayload()),
        });
        const data = await res.json();
        if (!res.ok || !data.success) {
          showError(data.error || 'Failed to start detection');
          return;
        }
        detectionActive = true;
        videoPlaceholder.style.display = 'none';
        videoFeed.style.display = 'block';
        videoFeed.src = '/video_feed?' + Date.now();
        startStatsPolling();
      } catch (e) {
        showError('Could not reach server: ' + e.message);
      }
    });

    document.getElementById('btnStop').addEventListener('click', async () => {
      try {
        await fetch('/stop', { method: 'POST' });
      } catch (e) {
        console.warn('Stop request failed', e);
      }
      detectionActive = false;
      videoFeed.src = '';
      videoFeed.style.display = 'none';
      videoPlaceholder.style.display = 'block';
      stopStatsPolling();
    });
  </script>
</body>
</html>
"""


def _resolve_video_path(filename):
    path = Path(filename)
    if path.is_file():
        return str(path.resolve())
    project_path = _PROJECT_DIR / filename
    if project_path.is_file():
        return str(project_path)
    return None


def _apply_config(source_type, video_filename, confidence, iou):
    config.CONFIDENCE_THRESHOLD = confidence
    config.IOU_THRESHOLD = iou
    if source_type == "webcam":
        config.SOURCE = 0
    else:
        path = _resolve_video_path(video_filename)
        if path is None:
            raise FileNotFoundError(
                f"Video file not found: '{video_filename}'. "
                "Place it in the project folder or enter a valid path."
            )
        config.SOURCE = path


def _reset_stats():
    with _state["lock"]:
        _state["total_frames"] = 0
        _state["fps_samples"] = []
        _state["last_detections"] = []


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/settings", methods=["POST"])
def settings_route():
    data = request.get_json(silent=True) or {}
    source_type = data.get("source", "file")
    video_filename = data.get("video_filename", "Test_Video.mp4")
    confidence = float(data.get("confidence", 0.35))
    iou = float(data.get("iou", 0.50))

    with _state["lock"]:
        _state["stop_requested"] = True
        _state["running"] = False

    try:
        _apply_config(source_type, video_filename, confidence, iou)
    except FileNotFoundError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

    _reset_stats()
    with _state["lock"]:
        _state["stop_requested"] = False
        _state["running"] = True
    return jsonify({"success": True})


@app.route("/stop", methods=["POST"])
def stop_route():
    with _state["lock"]:
        _state["stop_requested"] = True
        _state["running"] = False
    return jsonify({"success": True})


@app.route("/stats")
def stats_route():
    with _state["lock"]:
        detections = list(_state["last_detections"])
        total_frames = _state["total_frames"]
        fps_samples = list(_state["fps_samples"])
    fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0.0
    return jsonify({
        "fps": fps,
        "total_frames": total_frames,
        "detections": [
            {"class_name": d["class_name"], "confidence": d["confidence"]}
            for d in detections
        ],
    })


def _generate_frames():
    detector = None
    renderer = None

    try:
        with VideoSource() as video:
            detector = ObjectDetector()
            renderer = FrameRenderer()

            while True:
                with _state["lock"]:
                    if _state["stop_requested"] or not _state["running"]:
                        break

                frame = video.read()
                if frame is None:
                    with _state["lock"]:
                        _state["running"] = False
                    break

                detections = detector.detect(frame)
                annotated = renderer.draw(frame.copy(), detections)
                display_frame = cv2.resize(annotated, _DISPLAY_SIZE)
                ok, jpeg = cv2.imencode(
                    ".jpg",
                    display_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
                )
                if not ok:
                    continue

                with _state["lock"]:
                    _state["last_detections"] = detections
                    _state["total_frames"] += 1
                    _state["fps_samples"].append(renderer.fps)

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                )
    finally:
        with _state["lock"]:
            _state["running"] = False
            _state["stop_requested"] = True


@app.route("/video_feed")
def video_feed():
    with _state["lock"]:
        if not _state["running"]:
            return Response(status=204)

    def stream():
        if not _stream_lock.acquire(timeout=5):
            return
        try:
            yield from _generate_frames()
        finally:
            _stream_lock.release()

    return Response(
        stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
