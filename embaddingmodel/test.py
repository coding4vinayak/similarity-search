
data = 'embaddingmodel/data/embeddings.json'




if isinstance(data, list) and all(isinstance(item, list) for item in data):
    print("Data is in list of lists format.")
else:
    print("Data is not in the expected format.")