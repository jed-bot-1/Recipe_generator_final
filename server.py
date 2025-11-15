# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
df_recipes = None

def load_data():
    global df_recipes
    
    try:
        # Load the recipe data
        df_recipes = pd.read_csv('Recipes.csv')
        logger.info(f"✅ CSV loaded: {len(df_recipes)} recipes")
        
        # Try to load model but don't fail if it doesn't work
        try:
            with open('recipe_model.pkl', 'rb') as f:
                model = pickle.load(f)
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Model loading failed: {e}. Using simple matching.")
            
    except Exception as e:
        logger.error(f"❌ Error loading data: {str(e)}")
        raise

def find_similar_recipes(ingredients, top_n=5):
    """Find similar recipes based on ingredients"""
    if df_recipes is None:
        return []
    
    ingredients_lower = [ingredient.lower().strip() for ingredient in ingredients]
    
    matches = []
    for idx, row in df_recipes.iterrows():
        recipe_ingredients = str(row['ingredients']).lower()
        
        match_count = 0
        matched_ingredients = []
        
        for user_ing in ingredients_lower:
            if user_ing and user_ing in recipe_ingredients:
                match_count += 1
                matched_ingredients.append(user_ing)
        
        if match_count > 0:
            matches.append({
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
    
    matches.sort(key=lambda x: x['match_count'], reverse=True)
    return [match['recipe'] for match in matches[:top_n]]

# Health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "recipes_loaded": len(df_recipes) if df_recipes is not None else 0
    })

# Home
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Recipe Generator API",
        "status": "running"
    })

# Generate recipes
@app.route('/generate', methods=['POST'])
def generate_recipe():
    try:
        data = request.get_json()
        
        if not data or 'ingredients' not in data:
            return jsonify({"success": False, "error": "Please provide 'ingredients'"}), 400
        
        ingredients = data.get('ingredients', [])
        
        if not ingredients or not isinstance(ingredients, list):
            return jsonify({"success": False, "error": "Ingredients must be a non-empty list"}), 400
        
        logger.info(f"Received request for ingredients: {ingredients}")
        
        similar_recipes = find_similar_recipes(ingredients, top_n=5)
        
        return jsonify({
            "success": True,
            "ingredients": ingredients,
            "recipes_found": len(similar_recipes),
            "recipes": similar_recipes
        })
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Recipe Generator API...")
    load_data()
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"✅ Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
