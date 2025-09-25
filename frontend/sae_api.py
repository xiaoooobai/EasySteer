from flask import Blueprint, request, jsonify
import os
import sys
import logging
import json
import torch
import numpy as np

# Temporarily add project root to Python path for local modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import local modules
try:
    from easysteer.steer.sae import (
        search_sae_features, 
        get_sae_feature_explanation, 
        extract_sae_decoder_vector
    )
except ImportError:
    def search_sae_features(*args, **kwargs):
        return []
    
    def get_sae_feature_explanation(*args, **kwargs):
        return {}
    
    def extract_sae_decoder_vector(*args, **kwargs):
        return None

# Create blueprint for SAE-related endpoints
sae_bp = Blueprint('sae', __name__)

# Configure logging
logger = logging.getLogger(__name__)

@sae_bp.route('/api/sae/search', methods=['POST'])
def search_features():
    """API endpoint to search SAE features"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['model_id', 'sae_id', 'query']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Search for SAE features
        results = search_sae_features(
            model_id=data['model_id'],
            sae_id=data['sae_id'],
            query=data['query'],
            api_key=data.get('api_key')
        )
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in search_features endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@sae_bp.route('/api/sae/feature/<model_id>/<sae_id>/<int:feature_index>', methods=['GET'])
def get_feature_details(model_id, sae_id, feature_index):
    """API endpoint to get details for a specific SAE feature"""
    try:
        # Get the API key from query parameter or default to None
        api_key = request.args.get('api_key')
        
        # Get feature explanation
        result = get_sae_feature_explanation(
            model_id=model_id,
            sae_id=sae_id,
            feature_index=feature_index,
            api_key=api_key
        )
        
        return jsonify({
            'success': True,
            'feature': result
        })
        
    except Exception as e:
        logger.error(f"Error in get_feature_details endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@sae_bp.route('/api/sae/extract-vector', methods=['POST'])
def extract_sae_vector():
    """API endpoint to extract SAE feature vector and save as a steering vector"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['feature_index', 'vector_name']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        feature_index = data['feature_index']
        vector_name = data['vector_name']
        scale = data.get('scale', 1.0)
        api_key = data.get('api_key')
        
        # Create vectors directory if it doesn't exist
        vectors_dir = os.path.join(project_root, 'vectors')
        if not os.path.exists(vectors_dir):
            os.makedirs(vectors_dir)
        
        # Create output filename using feature ID
        vector_filename = f"{feature_index}.pt"
        output_path = os.path.join(vectors_dir, vector_filename)
        
        # Use fixed model file path
        model_file = "/home/shenyl/hf/model/google/gemma-scope-9b-it-res/layer_31/width_16k/average_l0_76/params.npz"
        
        # Try to extract the decoder vector
        logger.info(f"Extracting SAE vector for feature {feature_index} from {model_file}")
        vector = extract_sae_decoder_vector(model_file, feature_index, output_path)
        
        if vector is None:
            return jsonify({
                'success': False,
                'error': 'Failed to extract vector, check server logs for details'
            }), 500
        
        # Return success response with vector info
        return jsonify({
            'success': True,
            'vector': {
                'name': vector_name,
                'feature_index': feature_index,
                'file_path': output_path,
                'scale': scale
            }
        })
        
    except Exception as e:
        logger.error(f"Error in extract_sae_vector endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500 