import numpy as np
import trimesh


def normalize(vector):
    """
    Normaliza um vetor para ter comprimento um
    """
    return vector / np.linalg.norm(vector)


def reflected(vector, axis):
    """
    Reflexao de um vetor com relação a direção axis
    """
    return vector - 2 * np.dot(vector, axis) * axis


def convert(seconds):
    """
    Formata o tempo dados em segundos
    """
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


class Material:
    """
    Objeto com informações sobre o material
    """

    def __init__(self, diffuse, specular=None, ambient=None, shininess=25, reflectivity=0.5):
        self.diffuse = diffuse
        self.shininess = shininess
        self.reflectivity = reflectivity
        self.ambient = ambient
        self.specular = specular
        if (self.ambient is None):
            self.ambient = 0.1*diffuse
        if (self.specular is None):
            self.specular = np.array([1, 1, 1])

    def getColor(self, point, normal, toCamera, lights):
        """
        Determina a cor do ponto usando Blinn-Phong Shading 

        Parameters
        ----------
        normal : Direção normal
        toLight : Direção da luz
        toCamera : Direção da camera
        lights : lista de objetos fontes de luz

        Returns
        ------
        illumination : Cor do ponto
        """

        illumination = np.zeros((3))

        # ambiente
        illumination += self.ambient * lights[0].ambient

        for light in lights:
            if light.visible:

                toLight = normalize(light.position - point)

                # difusa
                illumination += self.diffuse * light.diffuse * \
                    np.dot(toLight, normal)

                # especular
                H = normalize(toLight + toCamera)
                illumination += self.specular * light.specular * np.dot(
                    normal, H) ** self.shininess

        return illumination


class Light:
    """   
    Objetos do tipo fonte de luz
    """

    def __init__(self, position, ambient, diffuse, specular):
        self.position = position
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.visible = True


class GraphicObject:
    """
    Classe base para todos os objetos da cena
    """

    def intersect(self, origin, direction):
        pass

    def getNormal(self, point):
        pass

    def getColor(self, point, normal, toCamera, lights):
        return self.material.getColor(point, normal, toCamera, lights)

    def getReflectivity(self):
        return self.material.reflectivity


class Mesh(GraphicObject):
    """
    Objetos do tipo mesh
    """

    def __init__(self, mesh, material):
        super().__init__()
        self.mesh = mesh
        self.material = material

    def intersect(self, origin, direction):
        index_tri, _, location = self.mesh.ray.intersects_id(
            ray_origins=[origin],
            ray_directions=[direction],
            return_locations=True, multiple_hits=False)

        if len(index_tri) > 0:
            return location[0], index_tri[0]
        else:
            return None, None

    def getNormal(self, point, index_tri):
        # calcula as coordenadas baricentricas
        bary = trimesh.triangles.points_to_barycentric(
            triangles=self.mesh.triangles[[index_tri]],
            points=[point])
        # interpola as normais dos vertices a partir das coordenadas
        # baricentricas
        interp = (self.mesh.vertex_normals[self.mesh.faces[[index_tri]]] *
                  trimesh.unitize(bary).reshape((-1, 3, 1))).sum(axis=1)
        return trimesh.unitize(interp[0])


class Sphere(GraphicObject):
    """
    Objetos do tipo esfera
    """

    def __init__(self, center, radius, material):
        super().__init__()
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, origin, direction):
        b = 2 * np.dot(direction, origin - self.center)
        c = np.linalg.norm(origin - self.center) ** 2 - self.radius ** 2
        delta = b ** 2 - 4 * c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            if t1 > 0 and t2 > 0:
                t = min(t1, t2)
                location = origin + t*direction
                return location, None
        return None, None

    def getNormal(self, point, _):
        return normalize(point - self.center)


class Plane(GraphicObject):
    """
    Objetos do tipo plano
    """

    def __init__(self, point, normal, material):
        super().__init__()
        self.point = point
        self.normal = normal
        self.material = material

    def intersect(self, origin, direction):
        if (abs(np.dot(self.normal, direction)) < 1e-6):
            return None
        t = np.dot(self.normal, self.point - origin) / \
            np.dot(self.normal, direction)
        if (t > 1e-4):
            location = origin + t*direction
            return location, None
        return None, None

    def getNormal(self, point, _):
        return self.normal


# Objetos diversos

def teapot():
    teapot = trimesh.load_mesh('Utah_teapot_(solid).stl')

    M1 = np.array([[1.,  0.,  0., 0.],
                   [0.,  0.,  1., 0.],
                   [0.,  1.,  0., 0.],
                   [0.,  0.,  0.,  1.]])

    M2 = np.array([[0.05,  0.,  0.,  0.4],
                   [0.,  0.05,  0.,  -0.65],
                   [0.,   0.,  0.05,  -0.3],
                   [0.,   0.,  0.,   1.]])

    teapot.apply_transform(np.matmul(M2, M1))
    return teapot
