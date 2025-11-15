# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for model and data
model = None
df_recipes = None

def load_model_and_data():
    global model, df_recipes
    
    try:
        # Load the model
        with open('recipe_model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.info("✅ Model loaded successfully")
        
        # Load the recipe data
        df_recipes = pd.read_csv('recipes.csv')
        logger.info(f"✅ CSV loaded: {len(df_recipes)} recipes")
            
    except Exception as e:
        logger.error(f"❌ Error loading model or data: {str(e)}")
        raise

def preprocess_ingredients(ingredients):
    """Preprocess ingredients for matching"""
    if isinstance(ingredients, list):
        ingredients_text = ' '.join(ingredients)
    else:
        ingredients_text = str(ingredients)
    
    # Clean the text
    ingredients_text = ingredients_text.lower()
    ingredients_text = re.sub(r'[^\w\s]', '', ingredients_text)
    
    return ingredients_text

def find_similar_recipes(ingredients, top_n=5):
    """Find similar recipes based on ingredients"""
    if df_recipes is None:
        return []
    
    # Convert ingredients to lowercase for matching
    ingredients_lower = [ingredient.lower().strip() for ingredient in ingredients]
    
    matches = []
    for idx, row in df_recipes.iterrows():
        recipe_ingredients = str(row['ingredients']).lower()
        
        # Count how many ingredients match
        match_count = 0
        matched_ingredients = []
        
        for user_ing in ingredients_lower:
            if user_ing and user_ing in recipe_ingredients:
                match_count += 1
                matched_ingredients.append(user_ing)
        
        if match_count > 0:
            matches.append({
                'index': idx,
                'match_count': match_count,
                'matched_ingredients': matched_ingredients,
                'recipe': {
                    'name': row.get('name', 'Unknown Recipe'),
                    'cuisine': row.get('cuisine', 'Unknown'),
                    'ingredients': row.get('ingredients', ''),
                    'instructions': row.get('instructions', 'No instructions available'),
                    'cooking_time': row.get('cooking_time', 'Not specified'),
                    'difficulty': row.get('difficulty', 'Not specified')
                }
            })
    
    # Sort by number of matches and get top N
    matches.sort(key=lambda x: x['match_count'], reverse=True)
    return [match['recipe'] for match in matches[:top_n]]

# Health check endpoint
@app.route('/health', methods=['GET', 'HEAD'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "recipes_loaded": len(df_recipes) if df_recipes is not None else 0,
        "model_loaded": model is not None
    })

# Root endpoint
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Recipe Generator API",
        "status": "running",
        "endpoints": {
            "/health": "Health check (GET/HEAD)",
            "/generate": "Generate recipes (POST)"
        },
        "usage": {
            "generate_endpoint": {
                "method": "POST",
                "body": {"ingredients": ["list", "of", "ingredients"]}
            }
        }
    })

# Recipe generation endpoint
@app.route('/generate', methods=['POST', 'OPTIONS'])
def generate_recipe():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json()
        
        if not data or 'ingredients' not in data:
            return jsonify({
                "success": False, 
                "error": "Please provide 'ingredients' in the request body"
            }), 400
        
        ingredients = data.get('ingredients', [])
        
        if not ingredients:
            return jsonify({
                "success": False,
                "error": "Ingredients list cannot be empty"
            }), 400
        
        # Validate that ingredients is a list
        if not isinstance(ingredients, list):
            return jsonify({
                "success": False,
                "error": "Ingredients must be provided as a list"
            }), 400
        
        logger.info(f"Received request for ingredients: {ingredients}")
        
        # Find similar recipes based on ingredient matching
        similar_recipes = find_similar_recipes(ingredients, top_n=5)
        
        if not similar_recipes:
            return jsonify({
                "success": True,
                "message": "No recipes found with these ingredients. Try different ingredients.",
                "ingredients": ingredients,
                "recipes": []
            })
        
        return jsonify({
            "success": True,
            "ingredients": ingredients,
            "recipes_found": len(similar_recipes),
            "recipes": similar_recipes
        })
        
    except Exception as e:
        logger.error(f"Error generating recipes: {str(e)}")
        return jsonify({
            "success": False, 
            "error": f"Internal server error: {str(e)}"
        }), 500

# Handle 405 errors gracefully
@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        "success": False,
        "error": "Method not allowed",
        "message": "This endpoint does not support the requested method"
    }), 405

# Handle 404 errors
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist"
    }), 404

# Handle preflight requests
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    logger.info("🚀 Starting Recipe Generator API...")
    load_model_and_data()
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"✅ Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
