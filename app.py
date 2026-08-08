"""
app.py — TrafficGuard Flask server
"""
import os, time, cv2
import base64
import io
from flask import Flask, Response, render_template, jsonify, request, send_from_directory, abort
from dotenv import load_dotenv
from config import (
    CAM_LAPTOP_SOURCE, CAM_PI_SOURCE,
    FLASK_HOST, FLASK_PORT, EVIDENCE_DIR
)
from camera import CameraProcessor
from database import get_violations, get_stats, mark_reviewed, init_db, get_comprehensive_analytics
import violations as viol_module
from google import genai
from google.genai import types
import matplotlib.pyplot as plt
import matplotlib
from api_chat_patch import api_chat
from alerts import stream as alert_stream
matplotlib.use('Agg')  # Use non-interactive backend

load_dotenv()  # Load environment variables from .env file
api_key = os.getenv('API_KEY')
app = Flask(__name__)

# ---------------- Gemini ---------------- #

client = genai.Client(api_key=api_key)

# ── Init DB ──────────────────────────────────────────────────────
init_db()

# ── Camera processors ────────────────────────────────────────────
cameras: dict[str, CameraProcessor] = {}

cam_laptop = CameraProcessor(
    cam_id='laptop',
    name='Laptop Camera',
    location='Main Entrance',
    source=CAM_LAPTOP_SOURCE,
    backend=cv2.CAP_ANY,    # macOS — remove or change to cv2.CAP_ANY on Linux/Windows
)
cam_laptop.start()
cameras['laptop'] = cam_laptop

if CAM_PI_SOURCE is not None:
    cam_pi = CameraProcessor(
        cam_id='pi',
        name='Pi Camera',
        location='Exit Gate',
        source=CAM_PI_SOURCE,
        backend=cv2.CAP_ANY,    # macOS — remove or change to cv2.CAP_ANY on Linux/Windows
    )
    cam_pi.start()
    cameras['pi'] = cam_pi


# ── MJPEG stream generator ───────────────────────────────────────
def _gen_frames(cam: CameraProcessor):
    while True:
        frame = cam.get_jpeg()
        if frame:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.05)


# ── Routes ───────────────────────────────────────────────────────
@app.route('/')
def index():
    cam_list = [
        dict(id=cid, name=c.name, location=c.location, online=c.online)
        for cid, c in cameras.items()
    ]
    return render_template('index.html', cameras=cam_list)


@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):
    if cam_id not in cameras:
        abort(404)
    return Response(_gen_frames(cameras[cam_id]),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/violations')
def api_violations():
    limit  = int(request.args.get('limit', 60))
    cam_id = request.args.get('cam_id', None)
    vtype  = request.args.get('vtype', None)
    return jsonify(get_violations(limit=limit, cam_id=cam_id, vtype=vtype))


@app.route('/api/stats')
def api_stats():
    stats = get_stats()
    stats['cameras'] = [
        dict(
            id=cid,
            name=c.name,
            location=c.location,
            online=c.online,
            vehicles=c.vehicle_count,
            fps=round(c.fps, 1),
        )
        for cid, c in cameras.items()
    ]
    stats['light_green'] = viol_module.get_traffic_light()
    return jsonify(stats)


@app.route('/api/violations/<int:vid>/review', methods=['POST'])
def api_review(vid):
    mark_reviewed(vid)
    return jsonify({'ok': True})


@app.route('/api/set_light', methods=['POST'])
def api_set_light():
    data     = request.get_json(force=True, silent=True) or {}
    is_green = bool(data.get('green', False))
    viol_module.set_traffic_light(is_green)
    return jsonify({'ok': True, 'green': is_green})


@app.route('/api/light_state')
def api_light_state():
    return jsonify({'green': viol_module.get_traffic_light()})


@app.route('/evidence/<path:filepath>')
def serve_evidence(filepath):
    return send_from_directory(EVIDENCE_DIR, filepath)


# ── Chart Generation Functions ───────────────────────────────────────

def generate_chart_base64(figure):
    """Convert matplotlib figure to base64 encoded string."""
    img_buffer = io.BytesIO()
    figure.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode()
    plt.close(figure)
    return img_str


def create_hourly_chart(hourly_data, title="Hourly Violation Distribution"):
    """Create a bar chart for hourly violation data."""
    if not hourly_data:
        return None
    
    hours = [int(h['hour']) for h in hourly_data]
    counts = [h['count'] for h in hourly_data]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(hours, counts, color='steelblue', alpha=0.7)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Number of Violations')
    ax.set_title(title)
    ax.set_xticks(range(24))
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=8)
    
    return fig


def create_violation_type_chart(type_data, title="Violations by Type"):
    """Create a pie chart for violation type distribution."""

    if not type_data:
        return None

    types = [t['type'] for t in type_data]
    counts = [t['count'] for t in type_data]

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = [
        '#00e5b8',
        '#ff6b6b',
        '#ffd166',
        '#4dabf7',
        '#c77dff'
    ]

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=types,
        autopct='%1.1f%%',
        colors=colors[:len(types)],
        startangle=90
    )

    ax.set_title(title)

    return fig


def create_daily_trend_chart(daily_data, title="Daily Violation Trend"):
    """Create a line chart for daily violation trends."""
    if not daily_data:
        return None
    
    dates = [d['date'] for d in daily_data]
    counts = [d['count'] for d in daily_data]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, counts, marker='o', linewidth=2, markersize=6, color='darkblue')
    ax.fill_between(dates, counts, alpha=0.3, color='steelblue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Violations')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    return fig


def create_camera_comparison_chart(camera_data, title="Violations by Camera"):
    """Create a bar chart comparing violations across cameras."""
    if not camera_data:
        return None
    
    cameras = [f"{c['cam_id']} ({c['location']})" for c in camera_data]
    counts = [c['count'] for c in camera_data]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(cameras, counts, color='coral', alpha=0.7)
    ax.set_xlabel('Camera')
    ax.set_ylabel('Number of Violations')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)
    
    return fig


def create_pattern_chart(pattern_data, pattern_type="hourly", title="Violation Patterns"):
    """Create charts for time-based patterns."""
    if not pattern_data:
        return None
    
    if pattern_type == "hourly":
        labels = [int(p['hour']) for p in pattern_data]
        values = [p['count'] for p in pattern_data]
        xlabel = "Hour of Day"
    elif pattern_type == "daily":
        labels = [p['day'] for p in pattern_data]
        values = [p['count'] for p in pattern_data]
        xlabel = "Day of Week"
    else:  # monthly
        labels = [p['month'] for p in pattern_data]
        values = [p['count'] for p in pattern_data]
        xlabel = "Month"
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, values, color='mediumseagreen', alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Number of Violations')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    
    if pattern_type == "monthly":
        plt.xticks(rotation=45, ha='right')
    
    return fig


@app.route('/api/analytics/charts', methods=["GET"])
def api_analytics_charts():
    """Generate and return charts based on analytics data."""
    try:
        analytics = get_comprehensive_analytics()
        charts = {}
        
        # Daily hourly chart
        if analytics['daily']['hourly_breakdown']:
            hourly_fig = create_hourly_chart(
                analytics['daily']['hourly_breakdown'],
                f"Hourly Violations - {analytics['daily']['date']}"
            )
            if hourly_fig:
                charts['daily_hourly'] = generate_chart_base64(hourly_fig)
        
        # Daily violation type chart
        if analytics['daily']['by_type']:
            type_fig = create_violation_type_chart(
                analytics['daily']['by_type'],
                f"Violation Types - {analytics['daily']['date']}"
            )
            if type_fig:
                charts['daily_types'] = generate_chart_base64(type_fig)
        
        # Weekly trend chart
        if analytics['weekly']['daily_breakdown']:
            weekly_fig = create_daily_trend_chart(
                analytics['weekly']['daily_breakdown'],
                f"Weekly Trend: {analytics['weekly']['week_start']} to {analytics['weekly']['week_end']}"
            )
            if weekly_fig:
                charts['weekly_trend'] = generate_chart_base64(weekly_fig)
        
        # Monthly trend chart
        if analytics['monthly']['daily_breakdown']:
            monthly_fig = create_daily_trend_chart(
                analytics['monthly']['daily_breakdown'],
                f"Monthly Trend - {analytics['monthly']['month']}"
            )
            if monthly_fig:
                charts['monthly_trend'] = generate_chart_base64(monthly_fig)
        
        # Camera comparison
        if analytics['daily']['by_camera']:
            camera_fig = create_camera_comparison_chart(
                analytics['daily']['by_camera'],
                "Violations by Camera (Today)"
            )
            if camera_fig:
                charts['camera_comparison'] = generate_chart_base64(camera_fig)
        
        # Time pattern charts
        if analytics['time_patterns']['hourly_patterns']:
            hourly_pattern_fig = create_pattern_chart(
                analytics['time_patterns']['hourly_patterns'],
                "hourly",
                "All-Time Hourly Patterns"
            )
            if hourly_pattern_fig:
                charts['hourly_patterns'] = generate_chart_base64(hourly_pattern_fig)
        
        if analytics['time_patterns']['day_patterns']:
            day_pattern_fig = create_pattern_chart(
                analytics['time_patterns']['day_patterns'],
                "daily",
                "All-Time Day of Week Patterns"
            )
            if day_pattern_fig:
                charts['day_patterns'] = generate_chart_base64(day_pattern_fig)
        
        return jsonify({
            "charts": charts,
            "analytics_summary": {
                "daily_total": analytics['daily']['total_violations'],
                "weekly_total": analytics['weekly']['total_violations'],
                "monthly_total": analytics['monthly']['total_violations'],
                "peak_hour": analytics['daily']['peak_hour'],
                "peak_day": analytics['weekly']['peak_day']
            }
        })
        
    except Exception as e:
        print(f"Chart generation error: {e}")
        return jsonify({
            "error": str(e),
            "charts": {}
        }), 500

@app.route("/api/chat", methods=["POST"])
def chat_route():
    return api_chat(client, cameras)

@app.route('/api/alerts/stream')
def api_alerts_stream():
    return Response(alert_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
