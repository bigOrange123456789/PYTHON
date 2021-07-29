import meshio
import os
for i in os.listdir("input"):
    name=i.split(".")[0]
    mesh = meshio.read("./input/"+name+".ply")
    mesh.write("./output/"+name+".obj", file_format="obj")
