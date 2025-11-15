from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Global variables for model
model_data = None
df = None
vectorizer = None
label_encoder = None
ingredient_vectors = None

def preprocess_ingredients(ingredients_text: str) -> str:
    """Clean and preprocess ingredients"""
    if pd.isna(ingredients_text):
        return ""
    
    ingredients = ingredients_text.split(',')
    cleaned_ingredients = []
    
    for ingredient in ingredients:
        ingredient_clean = re.sub(r'\d+\.?\d*\s*(g|kg|cup|cups|tsp|tbsp|pcs|cloves|bundle|thumb|pinch|\/| )', '', ingredient)
        ingredient_clean = ingredient_clean.strip()
        if ingredient_clean:
            cleaned_ingredients.append(ingredient_clean.lower())
    
    return ' '.join(cleaned_ingredients)

def load_model():
    """Load the pre-trained model and data"""
    global model_data, df, vectorizer, label_encoder, ingredient_vectors
    
    try:
        # Load model
        model_data = joblib.load('recipe_model.joblib')
        vectorizer = model_data['vectorizer']
        label_encoder = model_data['label_encoder']
        ingredient_vectors = model_data['ingredient_vectors']
        df = model_data['df']
        
        # Verify CSV exists
        csv_df = pd.read_csv('Recipe.csv')
        print(f"✅ Model loaded: {len(df)} recipes")
        print(f"✅ CSV verified: {len(csv_df)} recipes")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Recipe Generator API...")
    if not load_model():
        raise Exception("Failed to load model on startup")
    yield
    # Shutdown
    print("🛑 Shutting down Recipe Generator API...")

# FastAPI app with lifespan
app = FastAPI(
    title="Recipe Generator API",
    description="AI-powered recipe generator based on ingredients",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class AlternativeRecipe(BaseModel):
    recipe_name: str
    similarity_score: str
    matched_ingredients: List[str]
    missing_ingredients: List[str]
    ingredients_with_quantities: str
    steps: str

class RecipeSuggestion(BaseModel):
    recipe_name: str
    similarity_score: str
    matched_ingredients: List[str]
    missing_ingredients: List[str]
    ingredients_with_quantities: str
    steps: str

class RecipeResponse(BaseModel):
    success: bool
    suggestion: Optional[RecipeSuggestion] = None
    alternatives: List[AlternativeRecipe] = []
    message: Optional[str] = None

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    total_recipes: int
    total_ingredients: int

class TestInput(BaseModel):
    ingredients: str

def find_similar_recipes(user_ingredients: str, top_n: int = 5) -> pd.DataFrame:
    """Find recipes similar to user's ingredients"""
    user_cleaned = preprocess_ingredients(user_ingredients)
    
    if not user_cleaned:
        return pd.DataFrame()
    
    user_vector = vectorizer.transform([user_cleaned])
    similarities = cosine_similarity(user_vector, ingredient_vectors)
    top_indices = similarities.argsort()[0][-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        similarity_score = similarities[0][idx]
        recipe = df.iloc[idx].copy()
        recipe['similarity_score'] = similarity_score
        
        # Get matched ingredients
        user_ing_list = user_cleaned.split()
        recipe_ing_list = recipe['ingredient_list']
        matched = list(set(user_ing_list) & set(recipe_ing_list))
        recipe['matched_ingredients'] = matched
        
        results.append(recipe)
    
    return pd.DataFrame(results)

# Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Recipe Generator API", 
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        model_loaded=model_data is not None,
        total_recipes=len(df) if df is not None else 0,
        total_ingredients=len(vectorizer.get_feature_names_out()) if vectorizer is not None else 0
    )

@app.get("/ingredients", response_model=Dict[str, Any])
async def get_ingredients():
    """Get list of all available ingredients"""
    if df is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    all_ingredients = set()
    for ingredients in df['ingredient_list']:
        all_ingredients.update(ingredients)
    
    return {
        "total_ingredients": len(all_ingredients),
        "ingredients": sorted(list(all_ingredients))
    }

@app.get("/recipes", response_model=Dict[str, Any])
async def get_recipes(limit: int = Query(10, ge=1, le=50)):
    """Get list of all recipes"""
    if df is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    recipes = df.head(limit)[['recipe_name', 'ingredients']].to_dict('records')
    
    return {
        "total_recipes": len(df),
        "recipes": recipes
    }

@app.get("/suggest", response_model=RecipeResponse)
async def suggest_recipe(
    ingredients: str = Query(..., description="Comma-separated list of ingredients"),
    top_alternatives: int = Query(3, ge=1, le=10, description="Number of alternative recipes to return")
):
    """Get recipe suggestions based on ingredients"""
    if df is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    similar_recipes = find_similar_recipes(ingredients, top_n=top_alternatives + 1)
    
    if similar_recipes.empty:
        return RecipeResponse(
            success=False,
            message="No recipes found with those ingredients. Try different ingredients!"
        )
    
    # Main suggestion
    best_recipe = similar_recipes.iloc[0]
    
    # Get missing ingredients
    user_cleaned = preprocess_ingredients(ingredients)
    user_ing_list = user_cleaned.split()
    missing_ingredients = list(set(best_recipe['ingredient_list']) - set(user_ing_list))
    
    suggestion = RecipeSuggestion(
        recipe_name=best_recipe['recipe_name'],
        similarity_score=f"{best_recipe['similarity_score']:.2%}",
        matched_ingredients=best_recipe['matched_ingredients'],
        missing_ingredients=missing_ingredients,
        ingredients_with_quantities=best_recipe['ingredient_quantities'],
        steps=best_recipe['steps']
    )
    
    # Alternatives with full details
    alternatives = []
    for i in range(1, min(len(similar_recipes), top_alternatives + 1)):
        alt_recipe = similar_recipes.iloc[i]
        
        # Get missing ingredients for alternative
        alt_missing_ingredients = list(set(alt_recipe['ingredient_list']) - set(user_ing_list))
        
        alternative = AlternativeRecipe(
            recipe_name=alt_recipe['recipe_name'],
            similarity_score=f"{alt_recipe['similarity_score']:.2%}",
            matched_ingredients=alt_recipe['matched_ingredients'],
            missing_ingredients=alt_missing_ingredients,
            ingredients_with_quantities=alt_recipe['ingredient_quantities'],
            steps=alt_recipe['steps']
        )
        alternatives.append(alternative)
    
    return RecipeResponse(
        success=True,
        suggestion=suggestion,
        alternatives=alternatives,
        message="Recipe suggestion generated successfully"
    )

@app.post("/suggest", response_model=RecipeResponse)
async def suggest_recipe_post(test_input: TestInput):
    """Get recipe suggestions based on ingredients (POST endpoint)"""
    # Use a fixed number of alternatives for POST requests
    top_alternatives = 3
    
    if df is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    similar_recipes = find_similar_recipes(test_input.ingredients, top_n=top_alternatives + 1)
    
    if similar_recipes.empty:
        return RecipeResponse(
            success=False,
            message="No recipes found with those ingredients. Try different ingredients!"
        )
    
    # Main suggestion
    best_recipe = similar_recipes.iloc[0]
    
    # Get missing ingredients
    user_cleaned = preprocess_ingredients(test_input.ingredients)
    user_ing_list = user_cleaned.split()
    missing_ingredients = list(set(best_recipe['ingredient_list']) - set(user_ing_list))
    
    suggestion = RecipeSuggestion(
        recipe_name=best_recipe['recipe_name'],
        similarity_score=f"{best_recipe['similarity_score']:.2%}",
        matched_ingredients=best_recipe['matched_ingredients'],
        missing_ingredients=missing_ingredients,
        ingredients_with_quantities=best_recipe['ingredient_quantities'],
        steps=best_recipe['steps']
    )
    
    # Alternatives with full details
    alternatives = []
    for i in range(1, min(len(similar_recipes), top_alternatives + 1)):
        alt_recipe = similar_recipes.iloc[i]
        
        # Get missing ingredients for alternative
        alt_missing_ingredients = list(set(alt_recipe['ingredient_list']) - set(user_ing_list))
        
        alternative = AlternativeRecipe(
            recipe_name=alt_recipe['recipe_name'],
            similarity_score=f"{alt_recipe['similarity_score']:.2%}",
            matched_ingredients=alt_recipe['matched_ingredients'],
            missing_ingredients=alt_missing_ingredients,
            ingredients_with_quantities=alt_recipe['ingredient_quantities'],
            steps=alt_recipe['steps']
        )
        alternatives.append(alternative)
    
    return RecipeResponse(
        success=True,
        suggestion=suggestion,
        alternatives=alternatives,
        message="Recipe suggestion generated successfully"
    )

@app.get("/recipe/{recipe_name}", response_model=Dict[str, Any])
async def get_recipe_details(recipe_name: str):
    """Get detailed information about a specific recipe"""
    if df is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    recipe = df[df['recipe_name'] == recipe_name]
    
    if recipe.empty:
        raise HTTPException(status_code=404, detail="Recipe not found")
    
    recipe_data = recipe.iloc[0]
    
    return {
        "recipe_name": recipe_data['recipe_name'],
        "ingredients": recipe_data['ingredients'],
        "ingredient_quantities": recipe_data['ingredient_quantities'],
        "steps": recipe_data['steps'],
        "cleaned_ingredients": recipe_data['cleaned_ingredients']
    }

@app.get("/search")
async def search_recipes(
    query: str = Query(..., description="Search query for recipe names"),
    limit: int = Query(10, ge=1, le=50)
):
    """Search recipes by name"""
    if df is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    matching_recipes = df[df['recipe_name'].str.contains(query, case=False, na=False)]
    results = matching_recipes.head(limit)[['recipe_name', 'ingredients']].to_dict('records')
    
    return {
        "query": query,
        "total_found": len(matching_recipes),
        "results": results
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )