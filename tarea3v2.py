# coding=utf-8
"""Tarea 3"""

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.scene_graph as sg
import grafica.easy_shaders as es
import grafica.lighting_shaders as ls
import grafica.performance_monitor as pm
from grafica.assets_path import getAssetPath
from operator import add

__author__ = "Patricio A. Viñals M."
__license__ = "MIT"

# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.showAxis = True
        self.cameraTheta = np.pi/4
        self.cameraRadius = 20
        self.cameraPos = np.array([2,0.8,8.3])   #initial position from origin
        self.viewPos = self.cameraPos   #the same
        self.at = np.array([self.viewPos[0] + np.cos(self.cameraTheta) * self.cameraRadius,
                        self.viewPos[1], 
                        self.viewPos[2] + np.sin(self.cameraTheta) * self.cameraRadius])
        self.camUp = np.array([0, 1, 0])
        self.distance = 20
        self.position = 0   #camera and car initial position
        self.speed = 0  #camera and car initial speed
        self.acc = 0.00001   #camera and car acceleration and deceleration


controller = Controller()

def setPlot(texPipeline, axisPipeline, lightPipeline):
    projection = tr.perspective(45, float(width)/float(height), 0.1, 100)

    glUseProgram(axisPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUseProgram(texPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUseProgram(lightPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(lightPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
    
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "lightPosition"), 5, 5, 5)
    
    glUniform1ui(glGetUniformLocation(lightPipeline.shaderProgram, "shininess"), 1000)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "constantAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "linearAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "quadraticAttenuation"), 0.01)

def setView(texPipeline, axisPipeline, lightPipeline):
    view = tr.lookAt(
            controller.viewPos,
            controller.at,
            controller.camUp
        )

    glUseProgram(axisPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(texPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(lightPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(lightPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "viewPosition"), controller.viewPos[0], controller.viewPos[1], controller.viewPos[2])
    

def on_key(window, key, scancode, action, mods):

    if action != glfw.PRESS:
        return
    
    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)


def createOFFShape(pipeline, filename, r,g, b):
    shape = readOFF(getAssetPath(filename), (r, g, b))
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

    return gpuShape

def readOFF(filename, color):
    vertices = []
    normals= []
    faces = []

    with open(filename, 'r') as file:
        line = file.readline().strip()
        assert line=="OFF"

        line = file.readline().strip()
        aux = line.split(' ')

        numVertices = int(aux[0])
        numFaces = int(aux[1])

        for i in range(numVertices):
            aux = file.readline().strip().split(' ')
            vertices += [float(coord) for coord in aux[0:]]
        
        vertices = np.asarray(vertices)
        vertices = np.reshape(vertices, (numVertices, 3))
        print(f'Vertices shape: {vertices.shape}')

        normals = np.zeros((numVertices,3), dtype=np.float32)
        print(f'Normals shape: {normals.shape}')

        for i in range(numFaces):
            aux = file.readline().strip().split(' ')
            aux = [int(index) for index in aux[0:]]
            faces += [aux[1:]]
            
            vecA = [vertices[aux[2]][0] - vertices[aux[1]][0], vertices[aux[2]][1] - vertices[aux[1]][1], vertices[aux[2]][2] - vertices[aux[1]][2]]
            vecB = [vertices[aux[3]][0] - vertices[aux[2]][0], vertices[aux[3]][1] - vertices[aux[2]][1], vertices[aux[3]][2] - vertices[aux[2]][2]]

            res = np.cross(vecA, vecB)
            normals[aux[1]][0] += res[0]  
            normals[aux[1]][1] += res[1]  
            normals[aux[1]][2] += res[2]  

            normals[aux[2]][0] += res[0]  
            normals[aux[2]][1] += res[1]  
            normals[aux[2]][2] += res[2]  

            normals[aux[3]][0] += res[0]  
            normals[aux[3]][1] += res[1]  
            normals[aux[3]][2] += res[2]  
        #print(faces)
        norms = np.linalg.norm(normals,axis=1)
        normals = normals/norms[:,None]

        color = np.asarray(color)
        color = np.tile(color, (numVertices, 1))

        vertexData = np.concatenate((vertices, color), axis=1)
        vertexData = np.concatenate((vertexData, normals), axis=1)

        print(vertexData.shape)

        indices = []
        vertexDataF = []
        index = 0

        for face in faces:
            vertex = vertexData[face[0],:]
            vertexDataF += vertex.tolist()
            vertex = vertexData[face[1],:]
            vertexDataF += vertex.tolist()
            vertex = vertexData[face[2],:]
            vertexDataF += vertex.tolist()
            
            indices += [index, index + 1, index + 2]
            index += 3        



        return bs.Shape(vertexDataF, indices)

def createGPUShape(pipeline, shape):
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

    return gpuShape

def createTexturedArc(d):
    vertices = [d, 0.0, 0.0, 0.0, 0.0,
                d+1.0, 0.0, 0.0, 1.0, 0.0]
    
    currentIndex1 = 0
    currentIndex2 = 1

    indices = []

    cont = 1
    cont2 = 1

    for angle in range(4, 185, 5):
        angle = np.radians(angle)
        rot = tr.rotationY(angle)
        p1 = rot.dot(np.array([[d],[0],[0],[1]]))
        p2 = rot.dot(np.array([[d+1],[0],[0],[1]]))

        p1 = np.squeeze(p1)
        p2 = np.squeeze(p2)
        
        vertices.extend([p2[0], p2[1], p2[2], 1.0, cont/4])
        vertices.extend([p1[0], p1[1], p1[2], 0.0, cont/4])
        
        indices.extend([currentIndex1, currentIndex2, currentIndex2+1])
        indices.extend([currentIndex2+1, currentIndex2+2, currentIndex1])

        if cont > 4:
            cont = 0


        vertices.extend([p1[0], p1[1], p1[2], 0.0, cont/4])
        vertices.extend([p2[0], p2[1], p2[2], 1.0, cont/4])

        currentIndex1 = currentIndex1 + 4
        currentIndex2 = currentIndex2 + 4
        cont2 = cont2 + 1
        cont = cont + 1

    return bs.Shape(vertices, indices)

def createTiledFloor(dim):
    vert = np.array([[-0.5,0.5,0.5,-0.5],[-0.5,-0.5,0.5,0.5],[0.0,0.0,0.0,0.0],[1.0,1.0,1.0,1.0]], np.float32)
    rot = tr.rotationX(-np.pi/2)
    vert = rot.dot(vert)

    indices = [
         0, 1, 2,
         2, 3, 0]

    vertFinal = []
    indexFinal = []
    cont = 0

    for i in range(-dim,dim,1):
        for j in range(-dim,dim,1):
            tra = tr.translate(i,0.0,j)
            newVert = tra.dot(vert)

            v = newVert[:,0][:-1]
            vertFinal.extend([v[0], v[1], v[2], 0, 1])
            v = newVert[:,1][:-1]
            vertFinal.extend([v[0], v[1], v[2], 1, 1])
            v = newVert[:,2][:-1]
            vertFinal.extend([v[0], v[1], v[2], 1, 0])
            v = newVert[:,3][:-1]
            vertFinal.extend([v[0], v[1], v[2], 0, 0])
            
            ind = [elem + cont for elem in indices]
            indexFinal.extend(ind)
            cont = cont + 4

    return bs.Shape(vertFinal, indexFinal)


#Create an object that represents 8 houses, and return a scene graph node that represents 
#all geometry and textures
#Receives the texPipeline as parameter 
def createHouse(pipeline):
    #create a quad that represents a roof texture
    quadTechoShape = createGPUShape(pipeline, bs.createTextureQuad(1.0, 1.0))
    quadTechoShape.texture = es.textureSimpleSetup(
        getAssetPath("roof4.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    #create a part of a roof
    tejado1Node = sg.SceneGraphNode('tejado1')
    tejado1Node.transform = tr.matmul([tr.translate(0, 1.05, -0.45),
                                        tr.rotationX(-5*np.pi/8)])
    tejado1Node.childs += [quadTechoShape]

    #create a part of a roof
    tejado2Node = sg.SceneGraphNode('tejado2')
    tejado2Node.transform = tr.matmul([tr.translate(0, 1.05, 0.45),
                                        tr.rotationX(5*np.pi/8)])
    tejado2Node.childs += [quadTechoShape]

    #create the entire roof
    techoNode = sg.SceneGraphNode('techo')
    techoNode.transform = tr.identity()
    techoNode.childs += [tejado1Node,
                        tejado2Node]

    #create a quad that represents a wall texture
    quadParedShape = createGPUShape(pipeline, bs.createTextureQuad(1.0, 1.0))
    quadParedShape.texture = es.textureSimpleSetup(
        getAssetPath("wall3.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    #create a single wall
    pared1Node = sg.SceneGraphNode('pared1')
    pared1Node.transform = tr.matmul([tr.translate(0, 0.5, 0.5),
                                    ])
    pared1Node.childs += [quadParedShape]

    #create a single wall
    pared2Node = sg.SceneGraphNode('pared2')
    pared2Node.transform = tr.matmul([tr.translate(0, 0.5, -0.5),
                                    ])
    pared2Node.childs += [quadParedShape]

    #create a single wall
    pared3Node = sg.SceneGraphNode('pared3')
    pared3Node.transform = tr.matmul([tr.translate(0.5, 0.5, 0),
                                    tr.rotationY(np.pi/2),
                                    ])
    pared3Node.childs += [quadParedShape]

    #create a single wall
    pared4Node = sg.SceneGraphNode('pared4')
    pared4Node.transform = tr.matmul([tr.translate(-0.5, 0.5, 0),
                                    tr.rotationY(np.pi/2),
                                    ])
    pared4Node.childs += [quadParedShape]

    #create a single wall
    pared5Node = sg.SceneGraphNode('pared5')
    pared5Node.transform = tr.matmul([tr.translate(0,1,0),
                                        tr.rotationX(np.pi/2)])
    pared5Node.childs += [quadParedShape]

    #node that agroups the 5 walls
    cuerpoNode = sg.SceneGraphNode('cuerpo')
    cuerpoNode.transform = tr.identity()
    cuerpoNode.childs += [pared1Node,
                            pared2Node,
                            pared3Node,
                            pared4Node,
                            pared5Node]

    #create an entire house with walls and roof
    casaNode = sg.SceneGraphNode('casa')
    casaNode.transform = tr.matmul([tr.uniformScale(0.8)])
    casaNode.childs += [techoNode,
                        cuerpoNode]

    #node with one house in a specific position
    casa1Node = sg.SceneGraphNode('casa1')
    casa1Node.transform = tr.matmul([tr.translate(3.5, 0, 0)])
    casa1Node.childs += [casaNode]

    #node with one house in a specific position
    casa2Node = sg.SceneGraphNode('casa2')
    casa2Node.transform = tr.matmul([tr.translate(3.5, 0, -4)])
    casa2Node.childs += [casaNode]

    #node with one house in a specific position
    casa3Node = sg.SceneGraphNode('casa3')
    casa3Node.transform = tr.matmul([tr.translate(3.5, 0, 4)])
    casa3Node.childs += [casaNode]

    #node with one house in a specific position
    casa4Node = sg.SceneGraphNode('casa4')
    casa4Node.transform = tr.matmul([tr.translate(-3.5, 0, 0)])
    casa4Node.childs += [casaNode]

    #node with one house in a specific position
    casa5Node = sg.SceneGraphNode('casa5')
    casa5Node.transform = tr.matmul([tr.translate(-3.5, 0, -4)])
    casa5Node.childs += [casaNode]

    #node with one house in a specific position
    casa6Node = sg.SceneGraphNode('casa6')
    casa6Node.transform = tr.matmul([tr.translate(-3.5, 0, 4)])
    casa6Node.childs += [casaNode]

    #node with one house in a specific position
    casa7Node = sg.SceneGraphNode('casa7')
    casa7Node.transform = tr.matmul([tr.translate(0, 0, 9),
                                    tr.rotationY(np.pi/2)])
    casa7Node.childs += [casaNode]

    #node with one house in a specific position
    casa8Node = sg.SceneGraphNode('casa8')
    casa8Node.transform = tr.matmul([tr.translate(0, 0, -8),
                                    tr.rotationY(np.pi/2)])
    casa8Node.childs += [casaNode]

    #node that agroups the 8 houses
    casasNode = sg.SceneGraphNode('casas')
    casasNode.transform = tr.identity()
    casasNode.childs += [casa1Node,
                        casa2Node,
                        casa3Node,
                        casa4Node,
                        casa5Node,
                        casa6Node,
                        casa7Node,
                        casa8Node]

    return casasNode   #return the node with the 8 houses


#Create an object that represents 4 containment walls, and return a scene graph node that represents 
#all geometry and textures
#Receives the texPipeline as parameter 
def createWall(pipeline):
    #create a quad that represents a textured wall
    quadMurallaShape = createGPUShape(pipeline, bs.createTextureQuad(10, 1))
    quadMurallaShape.texture = es.textureSimpleSetup(
        getAssetPath("wall1.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)  

    #create a quad that represents a textured wall 
    quadMuralla2Shape = createGPUShape(pipeline, bs.createTextureQuad(15, 0.6))
    quadMuralla2Shape.texture = es.textureSimpleSetup(
        getAssetPath("wall1.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)  

    #create a quad that represent a textured wall 
    quadMuralla3Shape = createGPUShape(pipeline, bs.createTextureQuad(0.5, 1))
    quadMuralla3Shape.texture = es.textureSimpleSetup(
        getAssetPath("wall1.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)  

    #create one wall face
    cara1Node = sg.SceneGraphNode('cara1Node')
    cara1Node.transform = tr.matmul([tr.translate(0, 1, -0.5)])
    cara1Node.childs += [quadMuralla3Shape]

    #create one wall face
    cara2Node = sg.SceneGraphNode('cara2Node')
    cara2Node.transform = tr.matmul([tr.translate(0, 1, 0.5)])
    cara2Node.childs += [quadMuralla3Shape]

    #create one wall face
    cara3Node = sg.SceneGraphNode('cara3Node')
    cara3Node.transform = tr.matmul([tr.translate(0.5, 1, 0),
                                    tr.rotationY(np.pi/2)])
    cara3Node.childs += [quadMurallaShape]

    #create one wall face
    cara4Node = sg.SceneGraphNode('cara4Node')
    cara4Node.transform = tr.matmul([tr.translate(-0.5, 1, 0),
                                    tr.rotationY(np.pi/2)])
    cara4Node.childs += [quadMurallaShape]

    #create one wall face
    cara5Node = sg.SceneGraphNode('cara5Node')
    cara5Node.transform = tr.matmul([tr.translate(0, 1.5, 0),
                                    tr.rotationX(np.pi/2)])
    cara5Node.childs += [quadMuralla2Shape]

    #node that agroups 5 walls and represent an entire containment wall
    muro1Node = sg.SceneGraphNode('muro1Node')
    muro1Node.transform = tr.matmul([tr.translate(-2.6, -0.2, 0.5),
                                    tr.scale(0.1, 0.25, 10)])
    muro1Node.childs += [cara1Node,
                        cara2Node,
                        cara3Node,
                        cara4Node,
                        cara5Node]

    #node that agroups 5 walls and represent an entire containment wall
    muro2Node = sg.SceneGraphNode('muro2Node')
    muro2Node.transform = tr.matmul([tr.translate(-1.4, -0.2, 0.5),
                                    tr.scale(0.1, 0.25, 10)])
    muro2Node.childs += [cara1Node,
                        cara2Node,
                        cara3Node,
                        cara4Node,
                        cara5Node]

    #node that agroups 5 walls and represent an entire containment wall
    muro3Node = sg.SceneGraphNode('muro3Node')
    muro3Node.transform = tr.matmul([tr.translate(1.4, -0.2, 0.5),
                                    tr.scale(0.1, 0.25, 10)])
    muro3Node.childs += [cara1Node,
                        cara2Node,
                        cara3Node,
                        cara4Node,
                        cara5Node]

    #node that agroups 5 walls and represent an entire containment wall
    muro4Node = sg.SceneGraphNode('muro4Node')
    muro4Node.transform = tr.matmul([tr.translate(2.6, -0.2, 0.5),
                                   tr.scale(0.1, 0.25, 10)])
    muro4Node.childs += [cara1Node,
                        cara2Node,
                        cara3Node,
                        cara4Node,
                        cara5Node]
    
    #node that agroups the 4 containment walls
    murosNode = sg.SceneGraphNode('murosNode')
    murosNode.transform = tr.identity()
    murosNode.childs += [muro1Node,
                        muro2Node,
                        muro3Node,
                        muro4Node]
    return murosNode   #return the node with the 4 containment walls

#create a scene graph por an entire car
def createCarScene(pipeline):
    chasis = createOFFShape(pipeline, 'alfa2.off', 1.0, 0.0, 0.0)
    wheel = createOFFShape(pipeline, 'wheel.off', 0.0, 0.0, 0.0)

    scale = 2.0
    rotatingWheelNode = sg.SceneGraphNode('rotatingWheel')
    rotatingWheelNode.childs += [wheel]

    chasisNode = sg.SceneGraphNode('chasis')
    chasisNode.transform = tr.uniformScale(scale)
    chasisNode.childs += [chasis]

    wheel1Node = sg.SceneGraphNode('wheel1')
    wheel1Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(0.056390,0.037409,0.091705)])
    wheel1Node.childs += [rotatingWheelNode]

    wheel2Node = sg.SceneGraphNode('wheel2')
    wheel2Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(-0.060390,0.037409,-0.091705)])
    wheel2Node.childs += [rotatingWheelNode]

    wheel3Node = sg.SceneGraphNode('wheel3')
    wheel3Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(-0.056390,0.037409,0.091705)])
    wheel3Node.childs += [rotatingWheelNode]

    wheel4Node = sg.SceneGraphNode('wheel4')
    wheel4Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(0.066090,0.037409,-0.091705)])
    wheel4Node.childs += [rotatingWheelNode]

    car1 = sg.SceneGraphNode('car1')
    car1.transform = tr.matmul([tr.translate(2.0, -0.037409, 5.0), tr.rotationY(np.pi)])
    car1.childs += [chasisNode]
    car1.childs += [wheel1Node]
    car1.childs += [wheel2Node]
    car1.childs += [wheel3Node]
    car1.childs += [wheel4Node]

    scene = sg.SceneGraphNode('system')
    scene.childs += [car1]

    return scene


#create the textured static scene for the track and the ground
def createStaticScene(pipeline):

    roadBaseShape = createGPUShape(pipeline, bs.createTextureQuad(1.0, 1.0))
    roadBaseShape.texture = es.textureSimpleSetup(
        getAssetPath("Road_001_basecolor.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    sandBaseShape = createGPUShape(pipeline, createTiledFloor(50))
    sandBaseShape.texture = es.textureSimpleSetup(
        getAssetPath("Sand 002_COLOR.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    arcShape = createGPUShape(pipeline, createTexturedArc(1.5))
    arcShape.texture = roadBaseShape.texture
    
    roadBaseNode = sg.SceneGraphNode('plane')
    roadBaseNode.transform = tr.rotationX(-np.pi/2)
    roadBaseNode.childs += [roadBaseShape]

    arcNode = sg.SceneGraphNode('arc')
    arcNode.childs += [arcShape]

    sandNode = sg.SceneGraphNode('sand')
    sandNode.transform = tr.translate(0.0,-0.1,0.0)
    sandNode.childs += [sandBaseShape]

    linearSector = sg.SceneGraphNode('linearSector')
        
    for i in range(10):
        node = sg.SceneGraphNode('road'+str(i)+'_ls')
        node.transform = tr.translate(0.0,0.0,-1.0*i)
        node.childs += [roadBaseNode]
        linearSector.childs += [node]

    linearSectorLeft = sg.SceneGraphNode('lsLeft')
    linearSectorLeft.transform = tr.translate(-2.0, 0.0, 5.0)
    linearSectorLeft.childs += [linearSector]

    linearSectorRight = sg.SceneGraphNode('lsRight')
    linearSectorRight.transform = tr.translate(2.0, 0.0, 5.0)
    linearSectorRight.childs += [linearSector]

    arcTop = sg.SceneGraphNode('arcTop')
    arcTop.transform = tr.translate(0.0,0.0,-4.5)
    arcTop.childs += [arcNode]

    arcBottom = sg.SceneGraphNode('arcBottom')
    arcBottom.transform = tr.matmul([tr.translate(0.0,0.0,5.5), tr.rotationY(np.pi)])
    arcBottom.childs += [arcNode]
    
    scene = sg.SceneGraphNode('system')
    scene.childs += [linearSectorLeft]
    scene.childs += [linearSectorRight]
    scene.childs += [arcTop]
    scene.childs += [arcBottom]
    scene.childs += [sandNode]
    
    return scene

if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)

    width = 800
    height = 800
    title = "Tarea 3"
    window = glfw.create_window(width, height, title, None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program (pipeline) with both shaders
    axisPipeline = es.SimpleModelViewProjectionShaderProgram()
    texPipeline = es.SimpleTextureModelViewProjectionShaderProgram()
    lightPipeline = ls.SimpleGouraudShaderProgram()
    
    # Telling OpenGL to use our shader program
    glUseProgram(axisPipeline.shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.85, 0.85, 0.85, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    cpuAxis = bs.createAxis(7)
    gpuAxis = es.GPUShape().initBuffers()
    axisPipeline.setupVAO(gpuAxis)
    gpuAxis.fillBuffers(cpuAxis.vertices, cpuAxis.indices, GL_STATIC_DRAW)

    # Here we create an object with the scene
    dibujo = createStaticScene(texPipeline)
    car = createCarScene(lightPipeline)
    houses = createHouse(texPipeline)
    walls = createWall(texPipeline)

    setPlot(texPipeline, axisPipeline, lightPipeline)

    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)

    # glfw will swap buffers as soon as possible
    glfw.swap_interval(0)

    t0 = glfw.get_time()   #variable that will be used for camera rotation

    projection = tr.perspective(60, float(width)/float(height), 0.1, 100)

    while not glfw.window_should_close(window):

        # Using GLFW to check for input events
        glfw.poll_events() 

        # Measuring performance
        t1 = glfw.get_time()
        dt = t1 - t0   #a very short period of time
        t0 = t1

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Control the car and camera to move forward
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            controller.speed += controller.acc
            controller.viewPos[0] += np.cos(controller.cameraTheta) * controller.speed 
            controller.viewPos[2] += np.sin(controller.cameraTheta) * controller.speed
        else:
            if(controller.speed > 0):
                controller.speed -= controller.acc
                controller.viewPos[0] += np.cos(controller.cameraTheta) * controller.speed
                controller.viewPos[2] += np.sin(controller.cameraTheta) * controller.speed
                
        # Control the car an camera to move back
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            controller.speed -= controller.acc
            controller.viewPos[0] += np.cos(controller.cameraTheta) * controller.speed 
            controller.viewPos[2] += np.sin(controller.cameraTheta) * controller.speed
        else:
            if(controller.speed < 0):
                controller.speed += controller.acc
                controller.viewPos[0] += np.cos(controller.cameraTheta) * controller.speed
                controller.viewPos[2] += np.sin(controller.cameraTheta) * controller.speed

        # Control for turn right the camera 
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            controller.cameraTheta += 2* dt

        # Control for turn left the camera
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            controller.cameraTheta -= 2* dt

        controller.position += controller.speed

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        setView(texPipeline, axisPipeline, lightPipeline)

        # Car movement transformation
        model = tr.matmul([tr.translate(controller.viewPos[0]-2,0,controller.viewPos[2]-8)])

        if controller.showAxis:
            glUseProgram(axisPipeline.shaderProgram)
            glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
            axisPipeline.drawCall(gpuAxis, GL_LINES)

        # Update the at parameter from setView
        controller.at = np.array([controller.viewPos[0] + np.cos(controller.cameraTheta) * controller.cameraRadius,
                        controller.viewPos[1], 
                        controller.viewPos[2] + np.sin(controller.cameraTheta) * controller.cameraRadius])

        # Here we draw the object scene
        glUseProgram(texPipeline.shaderProgram)
        sg.drawSceneGraphNode(dibujo, texPipeline, "model")
        sg.drawSceneGraphNode(houses, texPipeline, "model") 
        sg.drawSceneGraphNode(walls, texPipeline, "model") 

        glUseProgram(lightPipeline.shaderProgram)
        sg.drawSceneGraphNode(car, lightPipeline, "model", model)
        
        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    # freeing GPU memory
    gpuAxis.clear()
    dibujo.clear()
    houses.clear()
    walls.clear()
    
    glfw.terminate()