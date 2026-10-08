with open('app/api/v1/ai_identify.py', 'r') as f:
    code = f.read()

impl = '''
@bp.route('/references/<kind>/<int:index>', methods=['GET'])
def get_reference_image(kind, index):
    import cv2
    from app.core.container import Container
    controller = Container.get_instance().module_controller
    if not controller.active_metadata or controller.active_metadata.id != 'core-ai-identify':
        return jsonify({"error": "AI Identify is not active"}), 404
    
    service = controller.get_active_service()
    if not service:
        return jsonify({"error": "Service not found"}), 404
        
    img = None
    with service._lock:
        if kind == 'good' and 0 <= index < len(service.good_references):
            img = service.good_references[index]['image']
        elif kind == 'bad' and 0 <= index < len(service.bad_references):
            img = service.bad_references[index]['image']
            
    if img is None:
        return jsonify({"error": "Reference not found"}), 404
        
    success, encoded = cv2.imencode('.png', img)
    if success:
        return Response(encoded.tobytes(), mimetype='image/png')
    return jsonify({"error": "Failed to encode image"}), 500
'''

import re
code = re.sub(r'@bp\.route\(''/references/<kind>/<int:index>.*?404\n', impl, code, flags=re.DOTALL)
with open('app/api/v1/ai_identify.py', 'w') as f:
    f.write(code)
