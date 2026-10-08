from flask import Blueprint, jsonify, request, send_file
import sqlite3
import os
import csv
import io
from app.core.container import Container

bp = Blueprint('inspection_v1', __name__, url_prefix='/api/v1/inspection')

def get_db_path():
    container = Container.get_instance()
    if container and container.config:
        data_dir = container.config.get("data_dir", "data")
    else:
        data_dir = "data"
    return os.path.join(data_dir, "inspections.db")

@bp.route('/history', methods=['GET'])
def get_history():
    """Get recent inspection history from SQLite."""
    limit = int(request.args.get('limit', 50))
    module_id = request.args.get('module')
    
    db_path = get_db_path()
    if not os.path.exists(db_path):
        return jsonify({"items": []})
        
    query = "SELECT id, ts, trigger_id, module_id, judgment, code, reason, values_json, raw_path FROM inspections"
    params = []
    
    if module_id:
        query += " WHERE module_id = ?"
        params.append(module_id)
        
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            items = []
            for row in rows:
                items.append({
                    "id": row["id"],
                    "timestamp": row["ts"],
                    "trigger_id": row["trigger_id"],
                    "module_id": row["module_id"],
                    "judgment": row["judgment"],
                    "code": row["code"],
                    "reason": row["reason"],
                    "values": row["values_json"],
                    "has_image": bool(row["raw_path"])
                })
            return jsonify({"items": items})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route('/export.csv', methods=['GET'])
def export_csv():
    """Export the entire inspection history as CSV."""
    db_path = get_db_path()
    if not os.path.exists(db_path):
        return "No data", 404
        
    try:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID", "Timestamp", "Trigger_ID", "Module_ID", "Judgment", "Code", "Reason", "Values"])
        
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM inspections ORDER BY id ASC")
            for row in cursor.fetchall():
                writer.writerow([
                    row["id"], row["ts"], row["trigger_id"], row["module_id"],
                    row["judgment"], row["code"], row["reason"], row["values_json"]
                ])
                
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='inspections_export.csv'
        )
    except Exception as e:
        return str(e), 500

@bp.route('/image/<int:row_id>', methods=['GET'])
def get_image(row_id):
    """Serve the raw NG image for a specific inspection row."""
    db_path = get_db_path()
    if not os.path.exists(db_path):
        return "Not found", 404
        
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT raw_path FROM inspections WHERE id = ?", (row_id,))
            row = cursor.fetchone()
            if row and row[0] and os.path.exists(row[0]):
                return send_file(row[0], mimetype='image/jpeg')
        return "Image not found", 404
    except Exception as e:
        return str(e), 500
