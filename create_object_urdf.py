import os
from object2urdf import ObjectUrdfBuilder

# Build single URDFs
object_folder = "meshes/brick"
# builder = ObjectUrdfBuilder(object_folder, urdf_prototype='_prototype_brick.urdf')
# # builder.build_urdf(filename="meshes/brick/30mm_brick.stl", force_overwrite=True, decompose_concave=True, force_decompose=False, center = 'mass')
# builder.build_urdf(filename="meshes/brick/30mm_brick.stl", decompose_concave=True, center = 'mass')
filename="meshes/brick/30mm_brick.stl"
filename = os.path.abspath(filename)
common = os.path.commonprefix([object_folder,filename])
rel = os.path.join(filename.replace(common,''))
if rel[0]==os.path.sep:
    rel = rel[1:] 
name= rel.split(os.path.sep)[0]
rel = rel.replace(os.path.sep,'/')

file_name_raw, file_extension = os.path.splitext(filename)
print(rel)