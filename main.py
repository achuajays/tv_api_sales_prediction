from fastapi import FastAPI
import pandas as pd
import pickle
import numpy as np
import json
pipe = pickle.load(open('Model.pkl','rb'))

app = FastAPI()

@app.get('/{item_id}')
async def p(item_id = float):
    item_id_array = np.array([[item_id]])
    # Create a DataFrame
    input_df = pd.DataFrame(item_id_array, columns=['TV'])
    prediction = pipe.predict(input_df)
    print(prediction)
    nested_list = prediction.tolist()


    # Convert nested list to JSON-formatted string
    json_string = json.dumps(nested_list)

    # Convert JSON string to Python dictionary
    dictionary = json.loads(json_string)
    print(dictionary)
    dictionary = {'value' : dictionary[0]}
    return dictionary
