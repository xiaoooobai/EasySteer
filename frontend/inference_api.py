from flask import Blueprint, request, jsonify
import os
import sys
import logging
import json

# Import vllm related modules (using pip-installed vllm)
from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest

# Create a blueprint for inference-related endpoints
inference_bp = Blueprint('inference', __name__)

# Configure logging
logger = logging.getLogger(__name__)

# Store active steer vector configurations
active_steer_vectors = {}

# Store LLM instances (to avoid reloading)
llm_instances = {}

def get_message(key, lang='zh', **kwargs):
    """Get a message in the specified language"""
    error_messages = {
        'zh': {
            'missing_field': '缺少必填字段: {field}',
            'file_not_found': '文件不存在: {path}',
            'server_error': '服务器错误: {error}',
            'not_found': 'Steer Vector ID {id} 不存在',
            'deleted': 'Steer Vector {name} 已删除',
            'created': 'Steer Vector配置创建成功',
            'generation_error': '生成失败: {error}',
            'model_loading_error': '模型加载失败: {error}'
        },
        'en': {
            'missing_field': 'Missing required field: {field}',
            'file_not_found': 'File not found: {path}',
            'server_error': 'Server error: {error}',
            'not_found': 'Steer Vector ID {id} does not exist',
            'deleted': 'Steer Vector {name} has been deleted',
            'created': 'Steer Vector configuration created successfully',
            'generation_error': 'Generation failed: {error}',
            'model_loading_error': 'Model loading failed: {error}'
        }
    }
    messages = error_messages.get(lang, error_messages['zh'])
    message = messages.get(key, key)
    return message.format(**kwargs)

def get_or_create_llm(model_path, gpu_devices):
    """Get or create an LLM instance"""
    # Create a unique key
    key = f"{model_path}_{gpu_devices}"
    
    if key not in llm_instances:
        try:
            # Set environment variables - ensure V0 is used to support steer vectors
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
            os.environ["VLLM_USE_V1"] = "0"
            
            # Calculate tensor_parallel_size
            gpu_count = len(gpu_devices.split(','))
            
            # Create LLM instance
            llm_instances[key] = LLM(
                model=model_path,
                enable_steer_vector=True,
                enforce_eager=True,
                tensor_parallel_size=gpu_count
            )
            logger.info(f"Created LLM instance for model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to create LLM instance: {str(e)}")
            raise e
    
    return llm_instances[key]

@inference_bp.route('/api/generate', methods=['POST'])
def generate():
    """Generate text using a Steer Vector"""
    try:
        data = request.json
        lang = request.headers.get('Accept-Language', 'zh').split(',')[0].split('-')[0]
        
        # Validate required fields
        required_fields = ['model_path', 'instruction', 'steer_vector_name', 'steer_vector_id', 'steer_vector_local_path']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': get_message('missing_field', lang, field=field)}), 400
        
        # Get or create LLM instance
        try:
            llm = get_or_create_llm(
                data['model_path'],
                data.get('gpu_devices', '0')
            )
        except Exception as e:
            return jsonify({'error': get_message('model_loading_error', lang, error=str(e))}), 500
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=data.get('sampling_params', {}).get('temperature', 0.0),
            max_tokens=data.get('sampling_params', {}).get('max_tokens', 128),
            repetition_penalty=data.get('sampling_params', {}).get('repetition_penalty', 1.1)
        )
        
        # Create SteerVectorRequest object
        steer_vector_request = SteerVectorRequest(
            steer_vector_name=data['steer_vector_name'],
            steer_vector_id=data['steer_vector_id'],
            steer_vector_local_path=data['steer_vector_local_path'],
            scale=data.get('scale', 1.0),
            target_layers=data.get('target_layers'),
            prefill_trigger_tokens=data.get('prefill_trigger_tokens'),
            prefill_trigger_positions=data.get('prefill_trigger_positions'),
            generate_trigger_tokens=data.get('generate_trigger_tokens'),
            debug=data.get('debug', False),
            algorithm=data.get('algorithm', 'direct')
        )
        
        # Format input (assuming Qwen model format)
        prompt_template = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
        prompt = prompt_template % data['instruction']
        
        # Generate text
        try:
            output = llm.generate(
                prompt,
                sampling_params,
                steer_vector_request=steer_vector_request
            )
            
            # Extract generated text
            generated_text = output[0].outputs[0].text
            
            # Return success response
            response = {
                'success': True,
                'generated_text': generated_text,
                'prompt': prompt,
                'config': {
                    'model_path': data['model_path'],
                    'steer_vector_name': steer_vector_request.steer_vector_name,
                    'algorithm': steer_vector_request.algorithm,
                    'scale': steer_vector_request.scale,
                    'target_layers': steer_vector_request.target_layers
                }
            }
            
            logger.info(f"Generated text with steer vector: {steer_vector_request.steer_vector_name}")
            
            return jsonify(response), 200
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return jsonify({'error': get_message('generation_error', lang, error=str(e))}), 500
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        lang = request.headers.get('Accept-Language', 'zh').split(',')[0].split('-')[0]
        return jsonify({'error': get_message('server_error', lang, error=str(e))}), 500

@inference_bp.route('/api/config/<config_name>', methods=['GET'])
def get_config(config_name):
    """Get a configuration file"""
    try:
        # Validate config name
        allowed_configs = ['emoji_loreft', 'emotion_direct']
        if config_name not in allowed_configs:
            return jsonify({"error": f"Config {config_name} not found"}), 404
        
        # Read config file
        config_path = os.path.join(os.path.dirname(__file__), 'configs', 'inference', f'{config_name}.json')
        if not os.path.exists(config_path):
            return jsonify({"error": f"Config file {config_path} not found"}), 404
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return jsonify(config)
    
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        return jsonify({"error": f"Failed to get config: {str(e)}"}), 500

@inference_bp.route('/api/configs', methods=['GET'])
def list_configs():
    """List all available configuration files"""
    try:
        configs_dir = os.path.join(os.path.dirname(__file__), 'configs', 'inference')
        if not os.path.exists(configs_dir):
            return jsonify({"configs": []})
        
        # Define friendly names for config files
        config_display_names = {
            'emoji_loreft': 'Emoji LoReft Configuration',
            'emotion_direct': 'Emotion Direct Configuration'
        }
        
        config_files = []
        for filename in os.listdir(configs_dir):
            if filename.endswith('.json'):
                config_name = filename[:-5]  # Remove .json extension
                display_name = config_display_names.get(config_name, config_name.replace('_', ' ').title())
                config_files.append({
                    "name": config_name,
                    "display_name": display_name
                })
        
        return jsonify({"configs": config_files})
    
    except Exception as e:
        logger.error(f"Failed to list configs: {e}")
        return jsonify({"error": f"Failed to list configs: {str(e)}"}), 500

@inference_bp.route('/api/steer-vector', methods=['POST'])
def create_steer_vector():
    """Create or update a Steer Vector configuration (kept for config management)"""
    try:
        data = request.json
        lang = request.headers.get('Accept-Language', 'zh').split(',')[0].split('-')[0]
        
        # Validate required fields
        required_fields = ['steer_vector_name', 'steer_vector_id', 'steer_vector_local_path']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': get_message('missing_field', lang, field=field)}), 400
        
        # Create SteerVectorRequest object
        steer_vector_request = SteerVectorRequest(
            steer_vector_name=data['steer_vector_name'],
            steer_vector_id=data['steer_vector_id'],
            steer_vector_local_path=data['steer_vector_local_path'],
            scale=data.get('scale', 1.0),
            target_layers=data.get('target_layers'),
            prefill_trigger_tokens=data.get('prefill_trigger_tokens'),
            prefill_trigger_positions=data.get('prefill_trigger_positions'),
            generate_trigger_tokens=data.get('generate_trigger_tokens'),
            debug=data.get('debug', False),
            algorithm=data.get('algorithm', 'direct')
        )
        
        # Validate if file exists
        if not os.path.exists(steer_vector_request.steer_vector_local_path):
            return jsonify({'error': get_message('file_not_found', lang, path=steer_vector_request.steer_vector_local_path)}), 400
        
        # Store configuration
        active_steer_vectors[steer_vector_request.steer_vector_id] = steer_vector_request
        
        # Return success response
        response = {
            'success': True,
            'message': get_message('created', lang),
            'steer_vector_id': steer_vector_request.steer_vector_id,
            'config': {
                'name': steer_vector_request.steer_vector_name,
                'id': steer_vector_request.steer_vector_id,
                'path': steer_vector_request.steer_vector_local_path,
                'scale': steer_vector_request.scale,
                'algorithm': steer_vector_request.algorithm,
                'target_layers': steer_vector_request.target_layers,
                'prefill_trigger_tokens': steer_vector_request.prefill_trigger_tokens,
                'prefill_trigger_positions': steer_vector_request.prefill_trigger_positions,
                'generate_trigger_tokens': steer_vector_request.generate_trigger_tokens,
                'debug': steer_vector_request.debug
            }
        }
        
        logger.info(f"Created steer vector: {steer_vector_request.steer_vector_name} (ID: {steer_vector_request.steer_vector_id})")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error creating steer vector: {str(e)}")
        lang = request.headers.get('Accept-Language', 'zh').split(',')[0].split('-')[0]
        return jsonify({'error': get_message('server_error', lang, error=str(e))}), 500

@inference_bp.route('/api/steer-vector/<int:steer_vector_id>', methods=['GET'])
def get_steer_vector(steer_vector_id):
    """Get a specific Steer Vector configuration"""
    lang = request.headers.get('Accept-Language', 'zh').split(',')[0].split('-')[0]
    
    if steer_vector_id in active_steer_vectors:
        sv = active_steer_vectors[steer_vector_id]
        return jsonify({
            'success': True,
            'config': {
                'name': sv.steer_vector_name,
                'id': sv.steer_vector_id,
                'path': sv.steer_vector_local_path,
                'scale': sv.scale,
                'algorithm': sv.algorithm,
                'target_layers': sv.target_layers,
                'prefill_trigger_tokens': sv.prefill_trigger_tokens,
                'prefill_trigger_positions': sv.prefill_trigger_positions,
                'generate_trigger_tokens': sv.generate_trigger_tokens,
                'debug': sv.debug
            }
        }), 200
    else:
        return jsonify({'error': get_message('not_found', lang, id=steer_vector_id)}), 404

@inference_bp.route('/api/steer-vectors', methods=['GET'])
def list_steer_vectors():
    """List all active Steer Vector configurations"""
    vectors = []
    for sv_id, sv in active_steer_vectors.items():
        vectors.append({
            'id': sv.steer_vector_id,
            'name': sv.steer_vector_name,
            'algorithm': sv.algorithm,
            'scale': sv.scale
        })
    
    return jsonify({
        'success': True,
        'count': len(vectors),
        'steer_vectors': vectors
    }), 200

@inference_bp.route('/api/steer-vector/<int:steer_vector_id>', methods=['DELETE'])
def delete_steer_vector(steer_vector_id):
    """Delete a Steer Vector configuration"""
    lang = request.headers.get('Accept-Language', 'zh').split(',')[0].split('-')[0]
    
    if steer_vector_id in active_steer_vectors:
        sv_name = active_steer_vectors[steer_vector_id].steer_vector_name
        del active_steer_vectors[steer_vector_id]
        logger.info(f"Deleted steer vector: {sv_name} (ID: {steer_vector_id})")
        return jsonify({
            'success': True,
            'message': get_message('deleted', lang, name=sv_name)
        }), 200
    else:
        return jsonify({'error': get_message('not_found', lang, id=steer_vector_id)}), 404

@inference_bp.route('/api/restart', methods=['POST'])
def restart_backend():
    """Fully restart the backend process"""
    try:
        import sys
        import threading
        import time
        
        logger.info("Preparing to fully restart the backend process...")
        
        def delayed_restart():
            """Delayed restart to allow response to be sent"""
            time.sleep(1)  # Wait 1 second for the response to be sent
            logger.info("Restarting backend process...")
            
            # Get current Python executable and script arguments
            python_executable = sys.executable
            script_args = sys.argv
            
            # Use os.execv to restart the process
            import os
            os.execv(python_executable, [python_executable] + script_args)
        
        # Execute restart in a new thread to avoid blocking the response
        restart_thread = threading.Thread(target=delayed_restart)
        restart_thread.daemon = True
        restart_thread.start()
        
        return jsonify({
            "success": True,
            "message": "Backend is restarting, please try again in a few seconds..."
        })
    
    except Exception as e:
        logger.error(f"Failed to restart backend: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Failed to restart backend: {str(e)}"
        }), 500 