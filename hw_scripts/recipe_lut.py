import pickle

def get_recipe(ingredients:list):
    '''
    Given a list of ingredients, searches the recipe dictionary for the top recipe that uses the available ingredients
    Dictionary of valid recipes should be stored as reduced_recipe_cache.pkl in the current dir
    '''
    with open("reduced_recipe_cache.pkl", "rb") as f:
        ds = pickle.load(f)

    ingredients = set(ingredients)

    winner = None
    best_score = -1

    #iterate through recipes, rank by how many of the available ingredients are used
    for row in ds:
        score = sum(1 for ing in row["NER"] if ing in ingredients)

        if score > best_score:
            best_score = score
            winner = row["title"]
    
    return winner

available_ingredients = ['strawberries']
print(get_recipe(available_ingredients))