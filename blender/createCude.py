import bpy

def collections(collection, col_list):
    col_list.append(collection)
    for sub_collection in collection.children:
        collections(sub_collection, col_list)
    rez_list = []
    collections(bpy.context.collection, rez_list)
    print(rez_list)
print(123)
