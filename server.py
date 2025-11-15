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
        df_recipes = pd.read_csv('Recipe.csv')
        logger.info(f"✅ CSV loaded: {len(df_recipes)} recipes")
        
        # Debug: Show column names and sample data
        logger.info(f"CSV columns: {df_recipes.columns.tolist()}")
        if len(df_recipes) > 0:
            logger.info("Sample of first recipe:")
            logger.info(f"  Name: {df_recipes.iloc[0].get('recipe_name', 'N/A')}")
            logger.info(f"  Ingredients: {str(df_recipes.iloc[0].get('ingredients', 'N/A'))[:100]}...")
            logger.info(f"  Steps: {str(df_recipes.iloc[0].get('steps', 'N/A'))[:100]}...")
        
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
    logger.info(f"🔍 Searching for recipes with ingredients: {ingredients_lower}")
    
    matches = []
    for idx, row in df_recipes.iterrows():
        # Use the correct column names from your CSV
        recipe_ingredients = str(row.get('ingredients', '')).lower()
        recipe_name = str(row.get('recipe_name', 'Unknown Recipe'))
        
        match_count = 0
        matched_ingredients = []
        
        for user_ing in ingredients_lower:
            # More flexible matching - check if ingredient appears in the recipe ingredients
            if user_ing and user_ing in recipe_ingredients:
                match_count += 1
                matched_ingredients.append(user_ing)
        
        if match_count > 0:
            matches.append({
                'match_count': match_count,
                'matched_ingredients': matched_ingredients,
                'recipe': {
                    'name': recipe_name,
                    'ingredients': recipe_ingredients,
                    'ingredient_quantities': row.get('ingredient_quantities', 'Not specified'),
                    'steps': row.get('steps', 'No instructions available'),
                    # Note: 'cuisine', 'cooking_time', 'difficulty' columns don't exist in your CSV
                }
            })
            logger.info(f"  ✅ Found match: {recipe_name} ({match_count} ingredients matched)")
    
    # Sort by number of matches
    matches.sort(key=lambda x: x['match_count'], reverse=True)
    
    logger.info(f"📊 Total matches found: {len(matches)}")
    
    return [match['recipe'] for match in matches[:top_n]]

# Health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "recipes_loaded": len(df_recipes) if df_recipes is not None else 0,
        "columns": df_recipes.columns.tolist() if df_recipes is not None else []
    })

# Home
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Recipe Generator API",
        "status": "running",
        "endpoints": {
            "/health": "Health check",
            "/generate": "Generate recipes (POST)"
        }
    })

# Generate recipes
@app.route('/generate', methods=['POST'])
def generate_recipe():
    try:
        data = request.get_json()
        
        if not data or 'ingredients' not in data:
            return jsonify({"success": False, "error": "Please provide 'ingredients'"}), 400
        
        ingredients = data.get('ingredients', [])
        
        # Handle both string and list input
        if isinstance(ingredients, str):
            # Split string by commas and clean up
            ingredients = [ing.strip() for ing in ingredients.split(',')]
            ingredients = [ing for ing in ingredients if ing]  # Remove empty strings
        elif not isinstance(ingredients, list):
            return jsonify({
                "success": False, 
                "error": "Ingredients must be a list or comma-separated string"
            }), 400
        
        if not ingredients:
            return jsonify({
                "success": False,
                "error": "Ingredients list cannot be empty"
            }), 400
        
        logger.info(f"🎯 Received request for ingredients: {ingredients}")
        
        similar_recipes = find_similar_recipes(ingredients, top_n=5)
        
        if not similar_recipes:
            return jsonify({
                "success": True,
                "message": "No recipes found with these exact ingredients. Try more common ingredients or check spelling.",
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
        logger.error(f"❌ Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Recipe Generator API...")
    load_data()
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"✅ Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
