import csv
import cv2
import math
import pickle
import psutil
import shutil
import time
import re
import os
import gc
import numpy as np
import random as r
import matplotlib.pyplot as plt
from numba import cuda, njit, stencil
from abc import ABC, abstractmethod

from pythonabm.backend import record_time, check_direct, template_params, check_existing, get_end_step, Graph, \
    progress_bar, starting_params, check_output_dir, assign_bins_jit, get_neighbors_cpu, get_neighbors_gpu
    
# import the Simulation class and record_time decorator from the PythonABM library
from pythonabm import Simulation, record_time

class TestSimulation(Simulation):
    """ This class inherits the Simulation class allowing it to run a
        simulation with the proper functionality.
    """
    def __init__(self):
        # hold simulation name (will be overridden)
        self.name = "Trial"

        # hold the current number of agents and the step to begin at (updated by continuation mode)
        self.number_agents = 0
        self.current_step = 0

        # hold the real time start of the step in seconds and the total time for select methods
        self.step_start = 0
        self.method_times = dict()

        # hold the names of the agent arrays and the names of any graphs (each agent is a node)
        self.array_names = list()
        self.graph_names = list()

        # hold bounds for specifying types of agents that vary in initial values
        self.agent_types = dict()

        # default values which can be updated in the subclass
        self.num_to_start = 1000
        self.cuda = False
        self.end_step = 10
        self.size = [1000, 1000, 0]
        self.output_values = True
        self.output_images = True
        self.image_quality = 2000
        self.video_quality = 1000
        self.fps = 10
        self.tpb = 4    # blocks per grid for CUDA neighbor search, use higher number if performance is slow

        # read parameters from YAML file and add them as instance variables
        self.yaml_parameters("general.yaml")

        # define instance variables outside of the YAML file
        self.move_step = 2

    def setup(self):
        """ Overrides the setup() method from the Simulation class.
        """
        # add agents to the simulation, indicate agent subtypes
        self.add_agents(self.num_nqo1_agents, agent_type="nqo1pos")
        self.add_agents(self.num_kd_agents, agent_type="nqo1kd")

        # Add patches for concentrations of diffusible substances
        self.diffusivities = np.array(self.diffusivities)
        # Add oxygen with boundary at cell sphere edge
        self.diff_species1 = np.ones((self.size[0],self.size[1]))*.001
        print(self.diff_species1)
        centerx,centery = int(self.size[0]/2), int(self.size[1]/2)
        for i in range(self.diff_species1.shape[0]):
            for j in range(self.diff_species1.shape[1]):
                if np.sqrt((i-centerx)**2+(j-centery)**2)>30:
                    self.diff_species1[i,j] = 500
        # Add peroxide patches which are all 0 at beginning
        self.diff_species2 = np.random.rand(self.size[0],self.size[1])*0.01
        
        # indicate agent arrays for storing agent values
        self.indicate_arrays("locations", "radii", "colors")

        # set initial agent values
        # Check if tumorsphere
        if self.is_sphere:
            # get locations within radius of center
            total_indices_insideR = list()
            scale = 10
            if is_3D:
                centerx, centery, centerz = int(self.size[0]/2)*scale, int(self.size[1]/2)*scale, int(self.size[2]/2)*scale
                    for i in range(self.size[0]*scale):
                        for j in range(self.size[1]*scale):
                            for k in range(self.size[2]*scale):
                                if np.sqrt((i-centerx)**2+(j-centery)**2+(k-centerz)**2)<(self.colony_radius*scale):
                                    total_indices_insideR.append([i,j,k])
            else:
                centerx, centery = int(self.size[0]/2)*scale, int(self.size[1]/2)*scale
                for i in range(self.size[0]*scale):
                    for j in range(self.size[1]*scale):
                        if np.sqrt((i-centerx)**2+(j-centery)**2)<(self.colony_radius*scale):
                            total_indices_insideR.append([i,j,0])
            self.locations = np.array(r.sample(total_indices_insideR,self.number_agents))/scale
        else:
            # randomly assign locations of agents
            self.locations = np.random.rand(self.number_agents, 3) * self.size
        
        self.radii = self.agent_array(initial=lambda: r.uniform(1, 2))
        self.colors = self.agent_array(vector=3, initial={"green": (0, 255, 0), "red": (255, 0, 0)}, dtype=int)

        # indicate agent graphs and create a graph for holding agent neighbors
        self.indicate_graphs("neighbor_graph")
        self.neighbor_graph = self.agent_graph()

        # record initial values
        self.step_values()
        self.step_image()

    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # get all neighbors within radius of 5, updating the graph object
        self.get_neighbors(self.neighbor_graph, 5)
        
        # Update patches with diffusion module
        self.diffusion()
        
        # call the following methods that update agent values
        # Die based on concentrations of peroxide
        self.die()
        
        # No reproduction due to time scale
        #self.reproduce()
        
        # No movement due to time scale
        #self.move()

        # add/remove agents from the simulation
        self.update_populations()

        # save data from the simulation
        self.step_values()
        self.step_image()
        self.temp()
        self.data()
        
            
    @record_time
    def step_image(self, background=(0, 0, 0), origin_bottom=True):
        """ Creates an image of the simulation space.

            :param background: The 0-255 RGB color of the image background.
            :param origin_bottom: If true, the origin will be on the bottom, left of the image.
            :type background: tuple
            :type origin_bottom: bool
        """
        # only continue if outputting images
        if self.output_images:
            # get path and make sure directory exists
            check_direct(self.images_path)

            # get the size of the array used for imaging in addition to the scaling factor
            x_size = self.image_quality
            scale = x_size / self.size[0]
            y_size = math.ceil(scale * self.size[1])

            # create the agent space background image and apply background color
            image = np.zeros((y_size, x_size, 3), dtype=np.uint8)
            
            background = (background[2], background[1], background[0])
            minimum = np.amin(self.diff_species1)
            maximum = np.amax(self.diff_species1)
            print("min is " + repr(minimum))
            print("max is " + repr(maximum))

            # normalize image, change data type, and
            flat_image = np.floor(255 * (self.diff_species1 - minimum) / (maximum - minimum)).astype(np.uint8)
            image[:] = np.dstack((flat_image,flat_image,flat_image))
            # go through all of the agents
            for index in range(self.number_agents):
                # get xy coordinates, the axis lengths, and color of agent
                x, y = int(scale * self.locations[index][0]), int(scale * self.locations[index][1])
                major, minor = int(scale * self.radii[index]), int(scale * self.radii[index])
                color = (int(self.colors[index][2]), int(self.colors[index][1]), int(self.colors[index][0]))

                # draw the agent and a black outline to distinguish overlapping agents
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, color, -1)
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, (0, 0, 0), 1)

            # if the origin should be bottom-left flip it, otherwise it will be top-left
            if origin_bottom:
                image = cv2.flip(image, 0)

            # save the image as a PNG
            image_compression = 4  # image compression of png (0: no compression, ..., 9: max compression)
            file_name = f"{self.name}_image_{self.current_step}.png"
            cv2.imwrite(self.images_path + file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, image_compression])
    
    @record_time
    def step_image_3d(self):
        """ Creates an image of the 3D simulation space.
        """
        # only continue if outputting images
        if self.output_images:
            # get path and make sure directory exists
            check_direct(self.images_path)

            # use dark_background theme
            plt.style.use('dark_background')

            # dots per inch for plot resolution, self.image_quality will specify image size
            dpi = 300

            # create a new figure and add an axe subplot to the figure
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            # get x,y,z values of agents and map colors from 0-255 to 0-1
            x, y, z = self.locations[:, 0], self.locations[:, 1], self.locations[:, 2]
            colors = self.colors / 255

            # create scatter plot
            ax.scatter(x, y, z, c=colors, marker="o", alpha=1)

            # turn off gridlines and ticks
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            # set bounds of figure and aspect ratio
            ax.set_xlim([0, self.size[0]])
            ax.set_ylim([0, self.size[1]])
            ax.set_zlim([0, self.size[2]])
            ax.set_box_aspect((self.size[0], self.size[1], self.size[2]))

            # reduce margins around figure
            fig.tight_layout()

            # calculate size of figure in inches to match self.image_quality size
            inches = self.image_quality / dpi
            fig.set_size_inches(inches, inches)

            # get file name and save to image directory
            file_name = f"{self.name}_image_{self.current_step}.png"
            fig.savefig(self.images_path + file_name, dpi=dpi)

            # close figure and garbage collect to prevent memory leak
            fig.clf()
            plt.close("all")
            gc.collect(2)
            
    def end(self):
        """ Overrides the end() method from the Simulation class.
        """
        self.create_video()

    @record_time
    def die(self):
        """ Determine which agents will die during this step.
        """
        for index in range(self.number_agents):
            if r.random() < 0.05:
                self.mark_to_remove(index)
            elif 

    @record_time
    def move(self):
        """ Assigns new locations to agents.
        """
        for index in range(self.number_agents):
            # get new location position
            new_location = self.locations[index] + self.move_step * self.random_vector()

            # check that the new location is within the space, otherwise use boundary values
            for i in range(3):
                if new_location[i] > self.size[i]:
                    self.locations[index][i] = self.size[i]
                elif new_location[i] < 0:
                    self.locations[index][i] = 0
                else:
                    self.locations[index][i] = new_location[i]

    @record_time
    def reproduce(self):
        """ Determine which agents will hatch a new agent during this step.
        """
        for index in range(self.number_agents):
            if r.random() < 0.1:
                self.mark_to_hatch(index)
                
    # pip install numpy, numba, opencv-python

    @stencil
    def laplacian(c):
        if c.size
        return (c[0, 1] + c[1, 0] + c[0, -1] + c[-1, 0]) - 4 * c[0, 0]


    @njit(cache=True)    # use @njit(cache=True, parallel=True) for large arrays
    def turing_numba(a, b, dt, dx, times, size_x, size_y, diff, num_agents, locations, colors, size_z=None, boundary_type="Neumann"):
        for i in range(times):
            #for i in range(len(species)):
            # mirror first and last rows (Neumann boundary condition)
            #species[i][0,:] = species[i][1,:]
            #species[i][-1,:] = species[i][-2,:]
            a[0] = a[1]
            a[-1] = a[-2]
            b[0] = b[1]
            b[-1] = b[-2]

            # mirror first and last columns (Neumann boundary condition)
            a[:, 0] = a[:, 1]
            a[:, -1] = a[:, -2]
            b[:, 0] = b[:, 1]
            b[:, -1] = b[:, -2]

            # integrate PDEs
            a += dt * diff[0] * laplacian(a) / dx**2
            b += dt * diff[1] * laplacian(b) / dx**2

            # make sure values are non-negative (you can remove this for simple diffusion)
            for j in range(size_x):
                for k in range(size_y):
                    if a[j][k] < 0:
                        a[j][k] = 0
                    if b[j][k] < 0:
                        b[j][k] = 0

        return a,b


    def turing(a, b, dx, diffusivities, times, num_agents, locations, colors):
        # pad array edges with zeros (ghost points)
            
        big_a = np.pad(a, 1)
        big_b = np.pad(b, 1)
        
        dt = dx**2/6/np.min(diffusivities)
        #print(dt)
        # perform integration calculation
        out_a, out_b = turing_numba(big_a, big_b, dt, dx, times, a.shape[0] + 2, a.shape[1] + 2,diffusivities, num_agents, locations, colors)

        # return array without edges
        return out_a[1:-1, 1:-1], out_b[1:-1, 1:-1]
    def diffusion(self):
        """ Let chemicals diffuse for length of one abm timestep
        """
        
        self.diff_species1, self.diff_species2 = diffusion.turing(self.diff_species1, self.diff_species2, 
                                                                    1, self.diffusivities, 200,
                                                                    self.number_agents, self.locations, self.colors)
        
    def membrane_diffuse(self):
        """ Lets chemicals diffuse between cell and surrounding patch
        """

if __name__ == "__main__":
    TestSimulation.start("~/Desktop/Repos/pythonabm/redoxabm/output")
