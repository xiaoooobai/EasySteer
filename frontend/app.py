from flask import Flask, jsonify
from flask_cors import CORS
import logging

# Import separated API modules
from training_api import training_bp
from inference_api import inference_bp
from extraction_api import extraction_bp

app = Flask(__name__)
CORS(app)  # Enable CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register blueprints
app.register_blueprint(training_bp)
app.register_blueprint(inference_bp)
app.register_blueprint(extraction_bp)

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'EasySteer Backend is running',
        'status': 'ok',
        'modules': ['inference', 'training']
    }), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Get status from various modules
    from inference_api import active_steer_vectors, llm_instances
    
    return jsonify({
        'status': 'healthy',
        'active_steer_vectors': len(active_steer_vectors),
        'loaded_models': len(llm_instances),
        'available_endpoints': [
            # Inference related
            'POST /api/generate',
            'GET /api/config/<config_name>',
            'GET /api/configs',
            'POST /api/steer-vector',
            'GET /api/steer-vector/<id>',
            'GET /api/steer-vectors',
            'DELETE /api/steer-vector/<id>',
            'POST /api/restart',
            # Training related
            'POST /api/train',
            'GET /api/train-configs',
            'GET /api/train-config/<config_name>',
            'GET /api/train-status',
            'POST /api/train-restart',
            # Extraction related
            'POST /api/extract',
            'GET /api/extract-status',
            'GET /api/extract-configs',
            'GET /api/extract-config/<config_name>',
            'POST /api/extract-restart'
        ]
    }), 200

if __name__ == '__main__':
    print("🚀 Starting EasySteer Backend Server...")
    print("📍 Server URL: http://localhost:5000")
    print("🔍 Health Check: http://localhost:5000/api/health")
    print("🧠 Inference APIs: /api/generate, /api/configs")
    print("🎓 Training APIs: /api/train, /api/train-configs")
    print("=" * 60)
    
    try:
        # Development mode configuration
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc() 