import os
from flask import Blueprint, jsonify, request, send_file
from app.core.container import Container

bp = Blueprint('gallery_v1', __name__, url_prefix='/api/v1/gallery')

@bp.route('/files', methods=['GET'])
def get_gallery_files():
    container = Container.get_instance()
    config = container.config
    
    files = []
    # Directories to scan based on the new architecture paths
    session_dir = config.get("session_dir", os.path.join(os.getcwd(), "sessions", "default"))
    captures_dir = os.path.join(session_dir, "captures")
    exports_dir = os.path.join(session_dir, "exports")
    
    dirs = [
        (captures_dir, 'image', ['.png', '.jpg', '.jpeg', '.webp']),
        (captures_dir, 'video', ['.mp4', '.avi']),
        (exports_dir, 'document', ['.csv', '.json', '.txt', '.pdf', '.xlsx'])
    ]
    
    for d, cat, exts in dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if any(f.lower().endswith(ext) for ext in exts):
                    path = os.path.join(d, f)
                    if os.path.isfile(path):
                        st = os.stat(path)
                        files.append({
                            'name': f,
                            'category': cat,
                            'size': st.st_size,
                            'modified': st.st_mtime,
                            'path': f"/api/v1/gallery/download?cat={cat}&file={f}"
                        })
                        
    files.sort(key=lambda x: x['modified'], reverse=True)
    return jsonify({'files': files})
    
@bp.route('/download', methods=['GET'])
def download_gallery_file():
    cat = request.args.get('cat')
    f = request.args.get('file')
    if not f or '..' in f or '/' in f or '\\' in f:
        return "Invalid file", 400
        
    container = Container.get_instance()
    config = container.config
    session_dir = config.get("session_dir", os.path.join(os.getcwd(), "sessions", "default"))
    
    if cat in ['image', 'video']:
        d = os.path.join(session_dir, "captures")
    elif cat == 'document':
        d = os.path.join(session_dir, "exports")
    else:
        return "Invalid category", 400
    
    path = os.path.join(d, f)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "File not found", 404
    
@bp.route('/delete', methods=['POST'])
def delete_gallery_file():
    data = request.get_json()
    cat = data.get('cat')
    f = data.get('file')
    if not f or '..' in f or '/' in f or '\\' in f:
        return jsonify({'error': 'Invalid file'}), 400
        
    container = Container.get_instance()
    config = container.config
    session_dir = config.get("session_dir", os.path.join(os.getcwd(), "sessions", "default"))
    
    if cat in ['image', 'video']:
        d = os.path.join(session_dir, "captures")
    elif cat == 'document':
        d = os.path.join(session_dir, "exports")
    else:
        return jsonify({'error': 'Invalid category'}), 400
    
    path = os.path.join(d, f)
    if os.path.exists(path):
        try:
            os.remove(path)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'File not found'}), 404
