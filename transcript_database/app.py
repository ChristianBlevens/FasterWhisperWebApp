"""
Transcript Database Service
Simple Flask API for storing and retrieving video transcripts
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import json
import hashlib
from pathlib import Path
from datetime import datetime

app = Flask(__name__)
CORS(app)

DB_PATH = Path(__file__).parent / "data" / "transcripts.db"


@app.before_request
def log_request():
    """Log every incoming request"""
    print(f"[REQUEST] {request.method} {request.path} from {request.remote_addr}", flush=True)


def get_db():
    """Get database connection"""
    # Ensure parent directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database schema"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT UNIQUE NOT NULL,
            video_url TEXT NOT NULL,
            video_title TEXT,
            transcript_data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_video_id ON transcripts(video_id)
    """)

    conn.commit()
    conn.close()


def extract_video_id(video_url: str) -> str:
    """Extract YouTube video ID from URL"""
    if "youtube.com/watch?v=" in video_url:
        return video_url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in video_url:
        return video_url.split("youtu.be/")[1].split("?")[0]
    else:
        return hashlib.md5(video_url.encode()).hexdigest()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    print("[HEALTH] Health check requested", flush=True)
    return jsonify({'status': 'ok', 'service': 'transcript-database'})


@app.route('/transcript', methods=['GET'])
def get_transcript():
    """
    Get transcript for a video
    Query params: video_url or video_id
    """
    video_url = request.args.get('video_url')
    video_id = request.args.get('video_id')

    if not video_url and not video_id:
        return jsonify({'error': 'video_url or video_id required'}), 400

    if video_url and not video_id:
        video_id = extract_video_id(video_url)

    print(f"[GET] Looking up transcript for video_id: {video_id}", flush=True)

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM transcripts WHERE video_id = ?", (video_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        print(f"[GET] Found transcript: {row['video_title']}", flush=True)
        return jsonify({
            'found': True,
            'video_id': row['video_id'],
            'video_url': row['video_url'],
            'video_title': row['video_title'],
            'transcript_data': json.loads(row['transcript_data']),
            'created_at': row['created_at']
        })
    else:
        print(f"[GET] Transcript not found for video_id: {video_id}", flush=True)
        return jsonify({'found': False})


@app.route('/transcript', methods=['POST'])
def save_transcript():
    """
    Save transcript for a video
    Body: {video_url, video_title, transcript_data}
    """
    data = request.json

    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    video_url = data.get('video_url')
    video_title = data.get('video_title', 'Unknown')
    transcript_data = data.get('transcript_data')

    if not video_url or not transcript_data:
        return jsonify({'error': 'video_url and transcript_data required'}), 400

    video_id = extract_video_id(video_url)
    segment_count = len(transcript_data) if isinstance(transcript_data, list) else 0

    print(f"[POST] Saving transcript: {video_title} (video_id: {video_id}, {segment_count} segments)", flush=True)

    conn = get_db()
    cursor = conn.cursor()

    try:
        # Check if already exists
        cursor.execute("SELECT video_id FROM transcripts WHERE video_id = ?", (video_id,))
        existing = cursor.fetchone()
        is_update = existing is not None

        cursor.execute("""
            INSERT INTO transcripts (video_id, video_url, video_title, transcript_data)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                video_url = excluded.video_url,
                video_title = excluded.video_title,
                transcript_data = excluded.transcript_data,
                updated_at = CURRENT_TIMESTAMP
        """, (video_id, video_url, video_title, json.dumps(transcript_data)))

        conn.commit()
        conn.close()

        action = "Updated" if is_update else "Saved"
        print(f"[POST] {action} transcript successfully: {video_title}", flush=True)

        return jsonify({
            'success': True,
            'video_id': video_id,
            'message': 'Transcript saved successfully'
        })

    except Exception as e:
        conn.close()
        print(f"[POST] Error saving transcript: {str(e)}", flush=True)
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Get database statistics"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as count FROM transcripts")
    count = cursor.fetchone()['count']

    cursor.execute("SELECT created_at FROM transcripts ORDER BY created_at DESC LIMIT 1")
    latest = cursor.fetchone()
    latest_at = latest['created_at'] if latest else None

    conn.close()

    return jsonify({
        'total_transcripts': count,
        'latest_added': latest_at
    })


if __name__ == '__main__':
    print("\n" + "=" * 60, flush=True)
    print("TRANSCRIPT DATABASE SERVICE", flush=True)
    print("=" * 60, flush=True)
    init_db()
    print("Ready to accept requests on port 5001", flush=True)
    print("=" * 60 + "\n", flush=True)
    app.run(host='0.0.0.0', port=5001, debug=False)
