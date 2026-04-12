from __future__ import annotations

import pytest

from backend.recipe_generator import RecipeGenerator, RecipePreferences, validate_diet_request


@pytest.mark.unit
def test_query_parsing_logic_extracts_dish_and_ingredient(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RecipeGenerator, "_load_dataset_rows", lambda self: [])
    generator = RecipeGenerator()

    cleaned = generator._clean_user_query_text("I want to make tomato bath")
    ingredient, dish = generator._extract_intent_parts(cleaned)

    assert cleaned == "tomato bath"
    assert ingredient == "tomato"
    assert dish == "bath"


@pytest.mark.unit
def test_recipe_dataset_matching_prefers_spinach_rice_rejects_sizzler(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "TranslatedRecipeName": "Spinach Rice",
            "Cleaned-Ingredients": "spinach, rice, salt, oil",
            "TranslatedIngredients": "spinach, rice, salt, oil",
            "TranslatedInstructions": "Cook rice. Add spinach.",
            "TotalTimeInMins": "25",
        },
        {
            "TranslatedRecipeName": "Vegetable Sizzler with Corn",
            "Cleaned-Ingredients": "corn, carrot, beans, sauce",
            "TranslatedIngredients": "corn, carrot, beans, sauce",
            "TranslatedInstructions": "Mix and grill.",
            "TotalTimeInMins": "20",
        },
    ]
    monkeypatch.setattr(RecipeGenerator, "_load_dataset_rows", lambda self: rows)
    generator = RecipeGenerator()

    matched, source = generator._find_dataset_row("spinach rice", ["spinach", "rice"])

    assert matched is not None
    assert "spinach" in matched.get("TranslatedRecipeName", "").lower()
    assert source in {"dataset_exact", "dataset_similarity"}


@pytest.mark.unit
def test_extra_ingredient_detection_contains_salt_and_oil(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RecipeGenerator, "_load_dataset_rows", lambda self: [])
    generator = RecipeGenerator()

    prefs = RecipePreferences(meal_type="lunch", health_goals=[])
    missing = generator._get_missing_essentials(["tomato", "rice"], prefs)

    assert "salt" in missing
    assert "oil" in missing


@pytest.mark.unit
def test_health_goal_logic_diabetic_removes_sugar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RecipeGenerator, "_load_dataset_rows", lambda self: [])
    generator = RecipeGenerator()

    rows = [
        {"ingredient": "sugar", "measurement": "2 tbsp"},
        {"ingredient": "rice", "measurement": "1 cup"},
    ]
    prefs = RecipePreferences(health_goals=["diabetic friendly"])

    adjusted_rows, adjusted_full, health_note = generator._apply_health_goals_to_rows(
        rows,
        ["sugar", "rice"],
        prefs,
    )

    names = [item["ingredient"] for item in adjusted_rows]
    assert "sugar" not in names
    assert "sugar" not in adjusted_full
    assert "diabetic friendly" in health_note


@pytest.mark.unit
def test_nutrition_calculation_has_positive_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RecipeGenerator, "_load_dataset_rows", lambda self: [])
    generator = RecipeGenerator()

    rows = [
        {"ingredient": "tomato", "measurement": "2 medium"},
        {"ingredient": "rice", "measurement": "1 cup"},
    ]
    nutrition = generator._normalize_nutrition({}, rows, servings=2)

    assert "protein" in nutrition
    calories = int(str(nutrition["calories"]).split()[0])
    assert calories > 0


@pytest.mark.unit
def test_diet_validation_blocks_non_veg_and_vegan_dishes() -> None:
    veg_error = validate_diet_request("veg", ["chicken"], "Chicken kabab")
    vegan_error = validate_diet_request("vegan", ["paneer"], "Paneer curry")

    assert veg_error is not None
    assert "Non-Veg ingredients" in veg_error
    assert vegan_error is not None
    assert "dairy or animal products" in vegan_error


@pytest.mark.unit
def test_model_payload_uses_available_ingredients_for_missing_items(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RecipeGenerator, "_load_dataset_rows", lambda self: [])
    generator = RecipeGenerator()

    prefs = RecipePreferences(dish_name="chicken kabab", diet="non-veg", meal_type="dinner")
    payload = {
        "recipe_name": "Chicken Kabab",
        "ingredients": ["chicken", "salt", "oil", "onion", "tomato", "turmeric", "red chilli powder"],
        "ingredients_with_measurements": [
            {"ingredient": "chicken", "measurement": "250 grams"},
            {"ingredient": "salt", "measurement": "to taste"},
            {"ingredient": "oil", "measurement": "2 tbsp"},
            {"ingredient": "onion", "measurement": "1 medium"},
            {"ingredient": "tomato", "measurement": "2 medium"},
            {"ingredient": "turmeric", "measurement": "1/4 tsp"},
            {"ingredient": "red chilli powder", "measurement": "1/2 tsp"},
        ],
        "steps": ["Mix ingredients."],
    }

    processed = generator._post_process(payload, ["chicken"], ["salt", "oil"], prefs)

    assert processed["matched_ingredients"] == ["chicken"]
    assert "salt" in processed["missing_ingredients"]
    assert "oil" in processed["missing_ingredients"]
    assert "onion" in processed["missing_ingredients"]
    assert "extra_ingredients" in processed
    assert processed["extra_ingredients"]
