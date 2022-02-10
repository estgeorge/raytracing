import numpy as np
import matplotlib.pyplot as plt
import time
from modelos import *


def nearest_intersected_object(objects, ray_origin, ray_direction):
    """
    Determina qual o objeto intersectado pelo raio que
    é mais próximo da origem do raio.  
    """
    nearest_object = None
    min_distance = np.inf
    nearest_location = None
    nearest_index_tri = None
    for obj in objects:
        location, index_tri = obj.intersect(ray_origin, ray_direction)
        if location is not None:
            distance = np.linalg.norm(location - ray_origin)
            if distance < min_distance:
                min_distance = distance
                nearest_object = obj
                nearest_location = location
                nearest_index_tri = index_tri
    return nearest_object, nearest_index_tri, nearest_location


def is_shadowed(objects, point, toLight, light):
    """
    Verifica se o point esta em uma sombra
    """
    _, _, intersection = nearest_intersected_object(objects, point, toLight)
    if intersection is not None:
        distance = np.linalg.norm(light.position - intersection)
        distance_to_light = np.linalg.norm(light.position - point)
        return distance < distance_to_light
    else:
        return False


start_time = time.time()
camera = np.array([0, 0, 1])  # posição da camera
width = 600
height = 400
max_depth = 3  # número máximo de reflexões do raio


# Lista de fontes de luz
lights = [
    Light(
        position=np.array([-8, 8, 5]),
        ambient=np.array([1, 1, 1]),
        diffuse=np.array([0.8, 0.8, 0.8]),
        specular=np.array([0.8, 0.8, 0.8])
    ),
    Light(
        position=np.array([10, 4, 1]),
        ambient=np.array([1, 1, 1]),
        diffuse=np.array([0.5, 0.5, 0.5]),
        specular=np.array([0.5, 0.5, 0.5])
    )
]

# Lista de objetos da cena
objects = [
    Mesh(mesh=teapot(),
         material=Material(diffuse=np.array([0.462, 0.298, 0.039]))),
    Sphere(center=np.array([-0.5, 0, -1]),
           radius=0.7,
           material=Material(diffuse=np.array([0.039, 0.113, 0.462]),
                             reflectivity=0.8)),
    Sphere(center=np.array([1.3, -0.1, -1.6]),
           radius=0.6,
           material=Material(diffuse=np.array([0.7, 0, 0.7]))),
    Plane(point=np.array([0, -0.7, 0]),
          normal=np.array([0, 1, 0]),
          material=Material(diffuse=np.array([0.5, 0.8, 0.8]),
                            specular=np.array([0.5, 0.5, 0.5]),
                            reflectivity=0))
]

ratio = float(width) / height
# esquerda, superior, direita, inferior
screen = (-1, 1 / ratio, 1, -1 / ratio)

image = np.zeros((height, width, 3))
for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):

        pixel = np.array([x, y, 0])
        origin = camera
        direction = normalize(pixel - origin)

        color = np.zeros((3))
        reflection = 1

        for k in range(max_depth):

            # Busca por intersecções
            nearest_object, trig_id, intersection = nearest_intersected_object(
                objects, origin, direction)
            if nearest_object is None:
                break

            normal = nearest_object.getNormal(intersection, trig_id)

            # Desloca um pouco o ponto de intersecção para evitar que
            # ocorra intersecção com a própria superfície em que o ponto
            # se encontra.
            shifted_point = intersection + 1e-5 * normal
            toCamera = normalize(origin - intersection)

            # Verifica se o ponto está sob sombras
            for light in lights:
                toLight = normalize(light.position - shifted_point)
                light.visible = not is_shadowed(
                    objects, shifted_point, toLight, light)

            illumination = nearest_object.getColor(
                intersection, normal, toCamera, lights)

            # Computa a contribuição da reflexão
            color += reflection * illumination
            reflection *= nearest_object.getReflectivity()

            origin = shifted_point
            direction = reflected(direction, normal)

        image[i, j] = np.clip(color, 0, 1)
    print("%d/%d" % (i + 1, height))

plt.imsave('image.png', image)


print(f'Elapsed time = {convert(time.time() - start_time)}')
