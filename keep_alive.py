"""
Enhanced Flask Server for Render Deployment
Provides health checks, statistics, and manual scan endpoints.
"""

from flask import Flask, jsonify, request
from threading import Thread
import os
from datetime import datetime
import json

app = Flask(__name__)

# Store bot status
bot_status = {
    "started_at": datetime.now().isoformat(),
    "last_scan": None,
    "total_scans": 0,
    "total_signals": 0,
    "open_positions": 0,
    "status": "running"
}


@app.route('/')
def home():
    """Basic health check endpoint."""
    return jsonify({
        "status": "alive",
        "service": "Gold Breakout Bot",
        "uptime": _calculate_uptime(),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/health')
def health():
    """Detailed health check for monitoring."""
    try:
        # Check if database exists
        db_exists = os.path.exists("trade_history.db")
        
        # Check if config exists
        config_exists = os.path.exists("config.json")
        
        health_status = {
            "status": "healthy" if db_exists and config_exists else "degraded",
            "checks": {
                "database": "ok" if db_exists else "missing",
                "config": "ok" if config_exists else "missing",
                "uptime": _calculate_uptime()
            },
            "bot_status": bot_status,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(health_status), 200 if health_status["status"] == "healthy" else 503
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/stats')
def stats():
    """Get bot statistics and performance metrics."""
    try:
        from position_manager import position_manager
        from performance_analytics import analytics
        
        # Get open positions
        open_positions = position_manager.get_open_positions()
        
        # Get performance metrics (last 7 days)
        try:
            metrics = analytics.get_comprehensive_metrics(days=7)
        except:
            metrics = {"error": "No trade data available"}
        
        # Get recent trade log stats
        from logger import get_trade_statistics
        trade_stats = get_trade_statistics(days=7)
        
        return jsonify({
            "bot_status": bot_status,
            "open_positions_count": len(open_positions),
            "open_positions": open_positions[:5],  # Limit to 5 for response size
            "performance_7d": metrics,
            "signal_stats_7d": trade_stats,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/scan', methods=['POST'])
def manual_scan():
    """Trigger a manual market scan (requires authentication)."""
    try:
        # Simple authentication via header or query param
        auth_token = request.headers.get('X-Auth-Token') or request.args.get('token')
        expected_token = os.getenv('MANUAL_SCAN_TOKEN', 'change-me-in-production')
        
        if auth_token != expected_token:
            return jsonify({
                "error": "Unauthorized",
                "message": "Invalid or missing authentication token"
            }), 401
        
        # Trigger scan (this would need to be implemented in main.py)
        return jsonify({
            "status": "scan_triggered",
            "message": "Manual scan initiated",
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/config')
def get_config():
    """Get current bot configuration (sanitized)."""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Remove sensitive data if any
        sanitized_config = {k: v for k, v in config.items() 
                           if k not in ['api_key', 'secret', 'password']}
        
        return jsonify({
            "config": sanitized_config,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


def _calculate_uptime():
    """Calculate bot uptime."""
    try:
        start_time = datetime.fromisoformat(bot_status["started_at"])
        uptime = datetime.now() - start_time
        
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return f"{days}d {hours}h {minutes}m {seconds}s"
    except:
        return "unknown"


def update_bot_status(key, value):
    """Update bot status from main.py."""
    global bot_status
    if key == "total_scans" and value is None:
        # Increment counter
        bot_status["total_scans"] = bot_status.get("total_scans", 0) + 1
    else:
        bot_status[key] = value


def run():
    """Run Flask server."""
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


def keep_alive():
    """Start Flask server in background thread."""
    print(f"🌐 Starting web server on port {os.getenv('PORT', 5000)}...")
    t = Thread(target=run, daemon=True)
    t.start()
    print("✅ Web server started")


if __name__ == '__main__':
    # For testing
    keep_alive()
    import time
    while True:
        time.sleep(1)
