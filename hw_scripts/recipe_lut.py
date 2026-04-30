import pickle

def get_recipe(ingredients:list):
    '''
    Given a list of ingredients, searches the recipe dictionary for the top recipe that uses the available ingredients
    Dictionary of valid recipes should be stored as reduced_recipe_cache.pkl in the current dir
    '''
    with open("reduced_recipe_cache.pkl", "rb") as f:
        ds = pickle.load(f)
    
    recipe_rankings = {} #dictionary of recipes ranked by how many of the given ingredients are used

    #iterate through recipes, rank by how many of the available ingredients are used
    for row in ds:
        ranking = 0
        for ing in row["NER"]:
            if ing in ingredients:
                ranking+=1
        recipe_rankings[row["title"]] = ranking

    winner = max(recipe_rankings, key=recipe_rankings.get) #save best recipe

    print(winner)
    
    return winner

available_ingredients = ['blueberries']
get_recipe(available_ingredients)