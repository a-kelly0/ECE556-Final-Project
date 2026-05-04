import kagglehub
import pandas as pd
import ast
import pickle
import os

def download_dataset():
    '''
    Downloads the recipeNGL dataset from huggingface
    '''
    # Get recepies data
    path = kagglehub.dataset_download("paultimothymooney/recipenlg")

    print("Path to dataset files:", path)

def read_csv(data_path:str, output_file:str):
    '''
    Reads the full recipe dataset csv, removes unnecessary information, and converts it to a dictionary.
    valid_recipe_dataset.csv should be stored in ../dataset
    Cleaned dictionary is stored as full_recipe_cache.pkl in the current working dir.
    '''
    df = pd.read_csv(data_path) #convert csv to dataframe #TODO change path back
    df = df[["title", "NER", "link"]] #extract title, simplified ingredients, and link
    df["NER"] = df["NER"].apply(ast.literal_eval) #convert NER ingredients list to readable strings
    df = df[df["NER"].apply(lambda x: isinstance(x, list) and len(x) > 0)] #remove all recipes with empty ingredients lists

    ds = df.to_dict(orient="records") #convert to dictionary
    with open(output_file, "wb") as f: #save full recipe dictionary
        pickle.dump(ds, f)

def reduce_ds(ingredients:list):
    '''
    Takes a list of ingredients and reduces the recipe dictionary to those that only use the valid ingredients.
    Recipe dictionary should be stored in the current dir as full_recipe_cache.pkl.
    Stores the reduced recipe dictionary as reduced_recipe_cache.pkl.
    '''
    with open("full_recipe_cache.pkl", "rb") as f:
        ds = pickle.load(f)

    #pair down dataset to only include recipes with the supported ingredients
    limited_ds = []
    for row in ds:
        add = True
        for ing in row["NER"]:
            if not(ing in ingredients): #do not add recepies that have an ingredient that is not in the acceptable list
                add = False

        if (add):
            limited_ds.append(row)

    #save reduced recipe set
    with open("reduced_recipe_cache.pkl", "wb") as f: #save full recipe dictionary #TODO change name back
        pickle.dump(limited_ds, f)

    print(limited_ds)
    return limited_ds

#read and parse full csv of recipes
read_csv(data_path="../dataset/valid_recipe_dataset.csv", output_file="full_recipe_cache.pkl")

# list of valid ingredients
ingredients = ["apple", "blueberries", "carrot", "strawberries", "broccoli", "avocado", 
               "banana", "bell pepper", "blackberry", "cantaloupe", "cherry", "cucumber",
               "dates","ginger", "grape", "lemon", "red onion", "white onion", "orange",
               "peach", "pear", "plum", "raspberry", "strawberry", "tomato", "all purpose flour", 
               "beef", "chicken", "salmon", "spinich", "white sugar"]

# reduce full dataset to only recipes with the valid ingredients
reduce_ds(ingredients)

# read_csv("../dataset/mini_recipe_dataset.csv", "mini_recipe_cache.pkl")

