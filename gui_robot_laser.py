import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import platform
import sys
from itertools import product
from itertools import groupby, chain
from sklearn.base import RegressorMixin
from Source import VimbaCameraDriver as vcd
from tkinter import messagebox as mbox, IntVar
import tkinter as tk
from vimba import *
import numpy as np
import glob
import threading
import open3d as o3d
from Source.UTILS.pcd_numpy_utils import *
import Source.REGISTRATION.PC_registration as PC_registration
from scipy.spatial import ConvexHull
import skimage.morphology, skimage.data
import cv2
import random
import time
import copy
import json
import serial
import sklearn.cluster as cluster
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#from Source import STEPAnalyzer
from Source import helper
from Source.UTILS.pcd_numpy_utils import *
from Source.REGISTRATION.ICP_utils import *
from scipy.spatial.transform import Rotation as Rot
#import torch
import math
import multiprocessing as mp
#import sys
from pathlib import Path
from os.path import dirname, abspath
# include path python
from Source.UTILS.FrankUtilities import *
import alphashape
isMacOS = (platform.system() == "Darwin")
from scipy.spatial import ConvexHull
import numpy_indexed as npi
from Source.SerialJob2 import SJ2
from scipy.interpolate import Rbf
from scipy.spatial.transform import Rotation 
import robot

class Settings:
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"

    DEFAULT_PROFILE_NAME = "Bright day with sun at +Y [default]"
    POINT_CLOUD_PROFILE_NAME = "Cloudy day (no direct sun)"
    CUSTOM_PROFILE_NAME = "Custom"
    LIGHTING_PROFILES = {
        DEFAULT_PROFILE_NAME: {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at -Y": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at +Z": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at -Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Z": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        POINT_CLOUD_PROFILE_NAME: {
            "ibl_intensity": 60000,
            "sun_intensity": 50000,
            "use_ibl": True,
            "use_sun": False,
            # "ibl_rotation":
        },
    }

    DEFAULT_MATERIAL_NAME = "Polished ceramic [default]"
    PREFAB = {
        DEFAULT_MATERIAL_NAME: {
            "metallic": 1.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Metal (rougher)": {
            "metallic": 1.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Metal (smoother)": {
            "metallic": 1.0,
            "roughness": 0.3,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Plastic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.5,
            "clearcoat": 0.5,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Glazed ceramic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.1,
            "anisotropy": 0.0
        },
        "Clay": {
            "metallic": 0.0,
            "roughness": 1.0,
            "reflectance": 0.5,
            "clearcoat": 0.1,
            "clearcoat_roughness": 0.287,
            "anisotropy": 0.0
        },
    }

    def __init__(self):
        self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA
        self.bg_color = gui.Color(1, 1, 1)
        self.show_skybox = False
        self.show_axes = False
        self.use_ibl = True
        self.use_sun = True
        self.new_ibl_name = None  # clear to None after loading
        self.ibl_intensity = 45000
        self.sun_intensity = 45000
        self.sun_dir = [0.577, -0.577, -0.577]
        self.sun_color = gui.Color(1, 1, 1)

        self.apply_material = True  # clear to False after processing
        self._materials = {
            Settings.LIT: rendering.MaterialRecord(),
            Settings.UNLIT: rendering.MaterialRecord(),
            Settings.NORMALS: rendering.MaterialRecord(),
            Settings.DEPTH: rendering.MaterialRecord()
        }
        self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LIT].shader = Settings.LIT
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        # Conveniently, assigning from self._materials[...] assigns a reference,
        # not a copy, so if we change the property of a material, then switch
        # to another one, then come back, the old setting will still be there.
        self.material = self._materials[Settings.LIT]

    def set_material(self, name):
        self.material = self._materials[name]
        self.apply_material = True

    def apply_material_prefab(self, name):
        assert (self.material.shader == Settings.LIT)
        prefab = Settings.PREFAB[name]
        for key, val in prefab.items():
            setattr(self.material, "base_" + key, val)

    def apply_lighting_profile(self, name):
        profile = Settings.LIGHTING_PROFILES[name]
        for key, val in profile.items():
            setattr(self, key, val)


class AppWindow:

    
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_SHOW_ACTIONS = 12
    MENU_ABOUT = 21
    threshold1 = 0
    threshold2 = 0
    counter_images=70
    rand_id = random.randint(100,999)
    DEFAULT_IBL = "default"
    Cameras_ID = ["DEV_1AB22C010C0B","DEV_1AB22C010C0C", "DEV_1AB22C010C0A"]
    Cameras_connected = []
    acquire = None
    scene_points = []
    hh = helper.Helperhandler()
    match_cad = []
    match_cad_temp = []
    match_det_temp = []
    match_cad_jobs = []
    cad_views = []
    match_det = []
    cad_jobs = []
    STL_MESH = []
    chosen_bests=[]
    aligned_match_cad = []
    scene_jobs = []
    scene_mesh = []
    matched_indexes = []
    correction_plane=False
    correction_3d=False
    include_dir = str(Path(__file__).resolve().parent)
    sys.path.append(include_dir)


    MATERIAL_NAMES = ["Lit", "Unlit", "Normals", "Depth"]
    MATERIAL_SHADERS = [
        Settings.LIT, Settings.UNLIT, Settings.NORMALS, Settings.DEPTH
    ]

    def __init__(self, width, height):
        self.fx_err = None
        self.fy_err = None
        self.fx_err_3d = None
        self.fy_err_3d = None
        self.SJ = None
        self.tool_orientation = [0.5,0.5,0.5,-0.5]
        self.ip_laser = '127.0.0.1'
        self.port_laser = 5195
        self.settings = Settings()
        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + AppWindow.DEFAULT_IBL
        self.image = None
        self.thresh_image = None
        self.object_image = None
        self.z_calib = None
        self.x_max = None
        self.x_step  = None
        self.origin = None
        self.back_color = True
        self.rotation = None
        self.pool = mp.Pool()
        self.imgs = [0,1]
        self.all_folder_check = False
        self.laser_connected_check = False
        self._x_value = 0
        self._y_value = 0
        self._z_value = 0
        self.exposure_setpoint = 7000
        self.zero_height = 0
        self.scale_factor = 0.00105
        self.temp_img1 = []
        self.temp_img2 = []
        self.camera1_connected=False
        self.camera2_connected=False
        self.camera3_connected=False
        self.laser_homed = False
        self.job_loaded = False
        self.window = gui.Application.instance.create_window(
            "RME MARKER", width, height)
        w = self.window  # to make the code more concise
        self.LoadConfig(self.include_dir+"//DATA//CONFIG//config.json")
        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.set_on_sun_direction_changed(self._on_sun_dir)
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))
        #2D widget

                ## Tabs
    

        # ---- Settings panel ----
        # Rather than specifying sizes in pixels, which may vary in size based
        # on the monitor, especially on macOS which has 220 dpi monitors, use
        # the em-size. This way sizings will be proportional to the font size,
        # which will create a more visually consistent size across platforms.
        

        # Widgets are laid out in layouts: gui.Horiz, gui.Vert,
        # gui.CollapsableVert, and gui.VGrid. By nesting the layouts we can
        # achieve complex designs. Usually we use a vertical layout as the
        # topmost widget, since widgets tend to be organized from top to bottom.
        # Within that, we usually have a series of horizontal layouts for each
        # row. All layouts take a spacing parameter, which is the spacing
        # between items in the widget, and a margins parameter, which specifies
        # the spacing of the left, top, right, bottom margins. (This acts like
        # the 'padding' property in CSS.)
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self._settings_panel.visible = False
        self._actions_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self._actions_panel.visible = True

        laser_ctrl= gui.CollapsableVert("LASER CONTROL", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))
        #buttons to manually control laser

        self._connect_button_laser = gui.Button("CONNECT LASER")
        self._connect_button_laser.horizontal_padding_em = 0.5
        self._connect_button_laser.vertical_padding_em = 0
        self._connect_button_laser.set_on_clicked(self._connect_laser)
        self._connection_check = gui.Checkbox("")
        self._connection_check.set_on_checked(self._on_laser_conn_check)
        self._homing_button_laser = gui.Button("HOMING")
        self._homing_button_laser.horizontal_padding_em = 0.5
        self._homing_button_laser.vertical_padding_em = 0
        self._homing_button_laser.set_on_clicked(self._on_homing_laser)
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._connect_button_laser)
        h.add_stretch()
        h.add_child(self._connection_check)
        h.add_stretch()
        h.add_child(self._homing_button_laser)
        laser_ctrl.add_child(h)

        self._move_button = gui.Button("MOVE LASER")
        self._move_button.horizontal_padding_em = 0.5
        self._move_button.vertical_padding_em = 0
        self._move_button.set_on_clicked(self._on_move_laser)
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._move_button)
        h.add_stretch()
        label = gui.Label("X, Y, Z")
        h.add_child(label)
        h.add_stretch()
        self._x_box = gui.TextEdit()
        self._x_box.set_on_value_changed(self._on_x_box_value)
        h.add_child(self._x_box)
        h.add_stretch()
        laser_ctrl.add_child(h)  

        self._move_J_button = gui.Button("MOVE JOINTS")
        self._move_J_button.horizontal_padding_em = 0.5
        self._move_J_button.vertical_padding_em = 0
        self._move_J_button.set_on_clicked(self._on_move_joints)
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._move_J_button)
        h.add_stretch()
        label = gui.Label("J,")
        h.add_child(label)
        h.add_stretch()
        self._j_box = gui.TextEdit()
        self._j_box.set_on_value_changed(self._on_j_box_value)
        h.add_child(self._j_box)
        h.add_stretch()
        laser_ctrl.add_child(h)  

        self.start_laser_button = gui.Button("START LSR")
        self.start_laser_button.horizontal_padding_em = 0.5
        self.start_laser_button.vertical_padding_em = 0
        self.start_laser_button.set_on_clicked(self.on_start_laser)
        self.on_laser_button = gui.Button("ON LSR")
        self.on_laser_button.horizontal_padding_em = 0.5
        self.on_laser_button.vertical_padding_em = 0
        self.on_laser_button.set_on_clicked(self._on_turn_on_laser)
        self.stop_laser_button = gui.Button("STOP LSR")
        self.stop_laser_button.horizontal_padding_em = 0.5
        self.stop_laser_button.vertical_padding_em = 0
        self.stop_laser_button.set_on_clicked(self.on_stop_laser)
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self.start_laser_button)
        h.add_stretch()
        h.add_child(self.on_laser_button)
        h.add_stretch()
        h.add_child(self.stop_laser_button)
        h.add_stretch()

        laser_ctrl.add_child(h)  
        h = gui.Horiz(0.25 * em)
        label = gui.Label("[%]")
        h.add_child(label)
        h.add_stretch()
        self._thresh1 = gui.Slider(gui.Slider.INT)
        self._thresh1.set_limits(0, 100)
        self._thresh1.set_on_value_changed(self._on_duty_box_value)
        h.add_child(self._thresh1)
        h.add_stretch()
        laser_ctrl.add_child(h)  
       
        #self._duty_box = gui.TextEdit()
        #self._duty_box.set_on_value_changed(self._on_duty_box_value)
        #h.add_child(self._duty_box)





        laser_ctrl.add_child(h)
        self._load_job_button = gui.Button("load job")
        self._load_job_button.horizontal_padding_em = 0.5
        self._load_job_button.vertical_padding_em = 0
        self._load_job_button.set_on_clicked(self._load_job_file)
        h = gui.Vert(0.05 * em)  # row 1
        h.add_stretch()
        h.add_child(self._load_job_button)
        laser_ctrl.add_child(h)  
        #h = gui.Horiz(0.25 * em)  # row 1
        #h.add_stretch()
        #h.add_child(self._connect_button_laser)
        #h.add_stretch()
        #laser_ctrl.add_child(h)


        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self._settings_panel.visible = False
        self._actions_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self._actions_panel.visible = True

        action_list= gui.CollapsableVert("IMAGE CONTROL", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        self._connect_button = gui.Button("CONNECT CAMERAS")
        self._connect_button.horizontal_padding_em = 0.5
        self._connect_button.vertical_padding_em = 0
        self._connect_button.set_on_clicked(self._connect_cameras)

        self._connection_check1 = gui.Checkbox("")
        self._connection_check1.set_on_checked(self._on_conn1_check)        
        self._connection_check2 = gui.Checkbox("")
        self._connection_check2.set_on_checked(self._on_conn2_check)
        self._connection_check3 = gui.Checkbox("")
        self._connection_check3.set_on_checked(self._on_conn3_check)

        h = gui.Horiz(0.05 * em)  # row 1
        h.add_stretch()
        h.add_child(self._connect_button)
        h.add_stretch()
        h.add_child(self._connection_check1)
        h.add_stretch()
        h.add_child(self._connection_check2)
        h.add_stretch()
        h.add_child(self._connection_check3)
        action_list.add_child(h)

        self._exposure_button = gui.Button("CHANGE EXPOSURE")
        self._exposure_button.horizontal_padding_em = 0
        self._exposure_button.vertical_padding_em = 0
        self._exposure_button.set_on_clicked(self.change_exposure)

        self._exp_box = gui.TextEdit()
        self._exp_box.set_on_value_changed(self._on_exp_box_value)

        h = gui.Horiz(0.05 * em)  # row 1
        h.add_stretch()
        h.add_child(self._exposure_button)
        h.add_stretch()
        h.add_child(self._exp_box)
        h.add_stretch()
        action_list.add_child(h)


        self._capture_button = gui.Button("CAPTURE IMAGE")
        self._capture_button.horizontal_padding_em = 0
        self._capture_button.vertical_padding_em = 0
        self._capture_button.set_on_clicked(self._capture_image)
        self._capture_back_button = gui.Button("CAPTURE BACK")
        self._capture_back_button.horizontal_padding_em = 0
        self._capture_back_button.vertical_padding_em = 0
        self._capture_back_button.set_on_clicked(self._capture_back_image)
        h = gui.Horiz(0.01 * em)  # row 1
        h.add_stretch()
        h.add_child(self._capture_button)
        h.add_stretch()
        h.add_child(self._capture_back_button)
        action_list.add_child(h)

        self._filter_button = gui.Button("FILTER IMAGE")
        self._filter_button.horizontal_padding_em = 0.5
        self._filter_button.vertical_padding_em = 0
        self._filter_button.set_on_clicked(self._filter_image)
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._filter_button)
        h.add_stretch()
        self._black_check = gui.Checkbox("black")
        self._black_check.set_on_checked(self._on_back_color)
        h.add_child(self._black_check)
        h.add_stretch()
        action_list.add_child(h)

        self._thresh1 = gui.Slider(gui.Slider.INT)
        self._thresh1.set_limits(0, 255)
        self._thresh1.set_on_value_changed(self._on_thresh1)

        self._thresh2 = gui.Slider(gui.Slider.INT)
        self._thresh2.set_limits(0, 200)
        self._thresh2.set_on_value_changed(self._on_thresh2)

        #grid = gui.Horiz(0.25 * em)
        #grid.add_child(gui.Label("threshold1"))
        #grid.add_child(self._thresh1)
        #action_list.add_child(grid)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("threshold2"))
        grid.add_child(self._thresh2)
        action_list.add_child(grid)
        
        self._find_origin_button = gui.Button("find origin")
        self._find_origin_button.horizontal_padding_em = 0.5
        self._find_origin_button.vertical_padding_em = 0
        self._find_origin_button.set_on_clicked(self._find_origin)
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._find_origin_button)
        h.add_stretch()
        action_list.add_child(h)

        #self._load_image_button = gui.Button("load image")
        #self._load_image_button.horizontal_padding_em = 0.5
        #self._load_image_button.vertical_padding_em = 0
        #self._load_image_button.set_on_clicked(self._load_image)
        #h = gui.Horiz(0.25 * em)  # row 1
        #h.add_stretch()
        #h.add_child(self._load_image_button)
        #h.add_stretch()
        #action_list.add_child(h)



        step_ctrl= gui.CollapsableVert("STEP CONTROL", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))
        self._load_step_button = gui.Button("load step")
        self._load_step_button.horizontal_padding_em = 0.5
        self._load_step_button.vertical_padding_em = 0
        self._load_step_button.set_on_clicked(self._load_step)
        #self._arcball_button = gui.Button("Arcball")
        #self._arcball_button.horizontal_padding_em = 0.5
        #self._arcball_button.vertical_padding_em = 0
        #self._arcball_button.set_on_clicked(self._set_mouse_mode_rotate)
        #self._fly_button = gui.Button("Fly")
        #self._fly_button.horizontal_padding_em = 0.5
        #self._fly_button.vertical_padding_em = 0
        #self._fly_button.set_on_clicked(self._set_mouse_mode_fly)


        self._all_folder = gui.Checkbox("all folder")
        self._all_folder.set_on_checked(self._on_check_folder)
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._load_step_button)
        h.add_stretch()
        h.add_child(self._all_folder)

        step_ctrl.add_child(h)  

        self._match_button = gui.Button("find match")
        self._match_button.horizontal_padding_em = 0.5
        self._match_button.vertical_padding_em = 0
        self._match_button.set_on_clicked(self._on_find_match)
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._match_button)
        h.add_stretch()
        step_ctrl.add_child(h)  


        self._align_match_button = gui.Button("align match")
        self._align_match_button.horizontal_padding_em = 0.5
        self._align_match_button.vertical_padding_em = 0
        self._align_match_button.set_on_clicked(self._on_align_match)
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._align_match_button)
        h.add_stretch()
        step_ctrl.add_child(h)

        #self._display_scene_button = gui.Button("display scene")
        #self._display_scene_button.horizontal_padding_em = 0.5
        #self._display_scene_button.vertical_padding_em = 0
        #self._display_scene_button.set_on_clicked(self._on_display_scene)

        self._mark_button = gui.Button("MARK")
        self._mark_button.horizontal_padding_em = 0.5
        self._mark_button.vertical_padding_em = 0
        self._mark_button.set_on_clicked(self._on_mark)

        self._preview_button = gui.Button("PREVIEW")
        self._preview_button.horizontal_padding_em = 0.5
        self._preview_button.vertical_padding_em = 0
        self._preview_button.set_on_clicked(self._on_preview)


        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._mark_button)
        h.add_stretch()
        self._corr_plane = gui.Checkbox("cor")
        self._corr_plane.set_on_checked(self._on_corr_plane )
        self._correct_3d = gui.Checkbox("3d")
        self._correct_3d.set_on_checked(self._on_correct_3d )
        h.add_child(self._corr_plane )
        h.add_stretch()
        h.add_child(self._correct_3d )
        h.add_stretch()
        h.add_child(self._preview_button)
        step_ctrl.add_child(h)

        self._calibrate_button = gui.Button("CALIBRATE")
        self._calibrate_button.horizontal_padding_em = 0.5
        self._calibrate_button.vertical_padding_em = 0
        self._calibrate_button.set_on_clicked(self._on_calibrate_button)
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        label = gui.Label("Zh, Xm, Xstp")
        h.add_child(label)
        h.add_stretch()
        

        self._z_x_box = gui.TextEdit()
        self._z_x_box.set_on_value_changed(self._on_z_x_box_value)
        h.add_stretch()
        h.add_child(self._z_x_box)
        h.add_stretch()
        h.add_stretch()
        h.add_stretch()
        h.add_stretch()

        h.add_child(self._calibrate_button)
        step_ctrl.add_child(h)




        # Create a collapsable vertical widget, which takes up enough vertical
        # space for all its children when open, but only enough for text when
        # closed. This is useful for property pages, so the user can hide sets
        # of properties they rarely use.
        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        self._arcball_button = gui.Button("Arcball")
        self._arcball_button.horizontal_padding_em = 0.5
        self._arcball_button.vertical_padding_em = 0
        self._arcball_button.set_on_clicked(self._set_mouse_mode_rotate)
        self._fly_button = gui.Button("Fly")
        self._fly_button.horizontal_padding_em = 0.5
        self._fly_button.vertical_padding_em = 0
        self._fly_button.set_on_clicked(self._set_mouse_mode_fly)
        self._model_button = gui.Button("Model")
        self._model_button.horizontal_padding_em = 0.5
        self._model_button.vertical_padding_em = 0
        self._model_button.set_on_clicked(self._set_mouse_mode_model)
        self._sun_button = gui.Button("Sun")
        self._sun_button.horizontal_padding_em = 0.5
        self._sun_button.vertical_padding_em = 0
        self._sun_button.set_on_clicked(self._set_mouse_mode_sun)
        self._ibl_button = gui.Button("Environment")
        self._ibl_button.horizontal_padding_em = 0.5
        self._ibl_button.vertical_padding_em = 0
        self._ibl_button.set_on_clicked(self._set_mouse_mode_ibl)
        view_ctrls.add_child(gui.Label("Mouse controls"))
        # We want two rows of buttons, so make two horizontal layouts. We also
        # want the buttons centered, which we can do be putting a stretch item
        # as the first and last item. Stretch items take up as much space as
        # possible, and since there are two, they will each take half the extra
        # space, thus centering the buttons.
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._arcball_button)
        h.add_child(self._fly_button)
        h.add_child(self._model_button)
        h.add_stretch()
        view_ctrls.add_child(h)
        h = gui.Horiz(0.25 * em)  # row 2
        h.add_stretch()
        h.add_child(self._sun_button)
        h.add_child(self._ibl_button)
        h.add_stretch()
        view_ctrls.add_child(h)

        self._show_skybox = gui.Checkbox("Show skymap")
        self._show_skybox.set_on_checked(self._on_show_skybox)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_skybox)

        self._bg_color = gui.ColorEdit()
        self._bg_color.set_on_value_changed(self._on_bg_color)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("BG Color"))
        grid.add_child(self._bg_color)
        view_ctrls.add_child(grid)

        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_axes)

        self._profiles = gui.Combobox()
        for name in sorted(Settings.LIGHTING_PROFILES.keys()):
            self._profiles.add_item(name)
        self._profiles.add_item(Settings.CUSTOM_PROFILE_NAME)
        self._profiles.set_on_selection_changed(self._on_lighting_profile)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(gui.Label("Lighting profiles"))
        view_ctrls.add_child(self._profiles)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(view_ctrls)


        self._actions_panel.add_fixed(separation_height)
        self._actions_panel.add_child(laser_ctrl)
        self._actions_panel.add_child(action_list)
        self._actions_panel.add_child(step_ctrl)

        advanced = gui.CollapsableVert("Advanced lighting", 0,
                                       gui.Margins(em, 0, 0, 0))
        advanced.set_is_open(False)

        self._use_ibl = gui.Checkbox("HDR map")
        self._use_ibl.set_on_checked(self._on_use_ibl)
        self._use_sun = gui.Checkbox("Sun")
        self._use_sun.set_on_checked(self._on_use_sun)
        advanced.add_child(gui.Label("Light sources"))
        h = gui.Horiz(em)
        h.add_child(self._use_ibl)
        h.add_child(self._use_sun)
        advanced.add_child(h)

        self._ibl_map = gui.Combobox()
        for ibl in glob.glob(gui.Application.instance.resource_path +
                             "/*_ibl.ktx"):

            self._ibl_map.add_item(os.path.basename(ibl[:-8]))
        self._ibl_map.selected_text = AppWindow.DEFAULT_IBL
        self._ibl_map.set_on_selection_changed(self._on_new_ibl)
        self._ibl_intensity = gui.Slider(gui.Slider.INT)
        self._ibl_intensity.set_limits(0, 200000)
        self._ibl_intensity.set_on_value_changed(self._on_ibl_intensity)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("HDR map"))
        grid.add_child(self._ibl_map)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._ibl_intensity)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Environment"))
        advanced.add_child(grid)

        self._sun_intensity = gui.Slider(gui.Slider.INT)
        self._sun_intensity.set_limits(0, 200000)
        self._sun_intensity.set_on_value_changed(self._on_sun_intensity)
        self._sun_dir = gui.VectorEdit()
        self._sun_dir.set_on_value_changed(self._on_sun_dir)
        self._sun_color = gui.ColorEdit()
        self._sun_color.set_on_value_changed(self._on_sun_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._sun_intensity)
        grid.add_child(gui.Label("Direction"))
        grid.add_child(self._sun_dir)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._sun_color)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Sun (Directional light)"))
        advanced.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(advanced)

        material_settings = gui.CollapsableVert("Material settings", 0,
                                                gui.Margins(em, 0, 0, 0))

        self._shader = gui.Combobox()
        self._shader.add_item(AppWindow.MATERIAL_NAMES[0])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[1])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[2])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[3])
        self._shader.set_on_selection_changed(self._on_shader)
        self._material_prefab = gui.Combobox()
        for prefab_name in sorted(Settings.PREFAB.keys()):
            self._material_prefab.add_item(prefab_name)
        self._material_prefab.selected_text = Settings.DEFAULT_MATERIAL_NAME
        self._material_prefab.set_on_selection_changed(self._on_material_prefab)
        self._material_color = gui.ColorEdit()
        self._material_color.set_on_value_changed(self._on_material_color)
        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        self._point_size.set_on_value_changed(self._on_point_size)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Type"))
        grid.add_child(self._shader)
        grid.add_child(gui.Label("Material"))
        grid.add_child(self._material_prefab)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._material_color)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        material_settings.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(material_settings)
        # ----

        # Normally our user interface can be children of all one layout (usually
        # a vertical layout), which is then the only child of the window. In our
        # case we want the scene to take up all the space and the settings panel
        # to go above it. We can do this custom layout by providing an on_layout
        # callback. The on_layout callback should set the frame
        # (position + size) of every child correctly. After the callback is
        # done the window will layout the grandchildren.
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        w.add_child(self._actions_panel)

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("About", AppWindow.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", AppWindow.MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item("Open...", AppWindow.MENU_OPEN)
            file_menu.add_item("Export Current Image...", AppWindow.MENU_EXPORT)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.add_item("Lighting & Materials",
                                   AppWindow.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, False)

            settings_menu.add_item("Actions",
                                   AppWindow.MENU_SHOW_ACTIONS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_ACTIONS, True)
  

            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT,
                                     self._on_menu_export)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)

        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_ACTIONS,
                                     self._on_menu_toggle_actions_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ----

        self._apply_settings()

    def LoadConfig(self,filename):

            with open(filename,"r") as file:
                data = json.load(file)
            self.zero_height = data.get("ZeroPlane")
            self.save_photo = data.get("SavePhotos")
            self.scale_factor = data.get("ScaleFactor")
            print("config file found, loaded succesfully\n", data)
            print("zero height : ",self.zero_height )
            print("save photo : ",self.save_photo )

 

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_skybox(self.settings.show_skybox)
        self._scene.scene.show_axes(self.settings.show_axes)
        if self.settings.new_ibl_name is not None:
            self._scene.scene.scene.set_indirect_light(
                self.settings.new_ibl_name)
            # Clear new_ibl_name, so we don't keep reloading this image every
            # time the settings are applied.
            self.settings.new_ibl_name = None
        self._scene.scene.scene.enable_indirect_light(self.settings.use_ibl)
        self._scene.scene.scene.set_indirect_light_intensity(
            self.settings.ibl_intensity)
        sun_color = [
            self.settings.sun_color.red, self.settings.sun_color.green,
            self.settings.sun_color.blue
        ]
        self._scene.scene.scene.set_sun_light(self.settings.sun_dir, sun_color,
                                              self.settings.sun_intensity)
        self._scene.scene.scene.enable_sun_light(self.settings.use_sun)

        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False

        self._bg_color.color_value = self.settings.bg_color
        self._show_skybox.checked = self.settings.show_skybox
        self._show_axes.checked = self.settings.show_axes
        self._use_ibl.checked = self.settings.use_ibl
        self._use_sun.checked = self.settings.use_sun
        self._ibl_intensity.int_value = self.settings.ibl_intensity
        self._sun_intensity.int_value = self.settings.sun_intensity
        self._sun_dir.vector_value = self.settings.sun_dir
        self._sun_color.color_value = self.settings.sun_color
        self._material_prefab.enabled = (
            self.settings.material.shader == Settings.LIT)
        c = gui.Color(self.settings.material.base_color[0],
                      self.settings.material.base_color[1],
                      self.settings.material.base_color[2],
                      self.settings.material.base_color[3])
        self._material_color.color_value = c
        self._point_size.double_value = self.settings.material.point_size

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)
        height = min(
            r.height,
            self._actions_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._actions_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)

    def _set_mouse_mode_rotate(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _set_mouse_mode_fly(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)

    def _set_mouse_mode_sun(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_SUN)

    def _set_mouse_mode_ibl(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_IBL)

    def _set_mouse_mode_model(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

    def _on_bg_color(self, new_color):
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_show_skybox(self, show):
        self.settings.show_skybox = show
        self._apply_settings()

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_use_ibl(self, use):
        self.settings.use_ibl = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_use_sun(self, use):
        self.settings.use_sun = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_lighting_profile(self, name, index):
        if name != Settings.CUSTOM_PROFILE_NAME:
            self.settings.apply_lighting_profile(name)
            self._apply_settings()

    def _on_new_ibl(self, name, index):
        self.settings.new_ibl_name = gui.Application.instance.resource_path + "/" + name
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_ibl_intensity(self, intensity):
        self.settings.ibl_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_intensity(self, intensity):
        self.settings.sun_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_dir(self, sun_dir):
        self.settings.sun_dir = sun_dir
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_color(self, color):
        self.settings.sun_color = color
        self._apply_settings()

    def _on_shader(self, name, index):
        self.settings.set_material(AppWindow.MATERIAL_SHADERS[index])
        self._apply_settings()

    def _on_material_prefab(self, name, index):
        self.settings.apply_material_prefab(name)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_material_color(self, color):
        self.settings.material.base_color = [
            color.red, color.green, color.blue, color.alpha
        ]
        self.settings.apply_material = True
        self._apply_settings()

    def _on_point_size(self, size):
        self.settings.material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb",
                       "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)


    def _on_load_image_dialog_done(self, filename):
        self.window.close_dialog()
        print(filename)
        image = None
        image = o3d.io.read_image(filename)
        if image is not None:
            try:
                self._display_image(image)
            except Exception as e:
                print(e)

    def _on_load_step_dialog_done(self, filename):
        self.window.close_dialog()
        self._apply_settings()
        print(filename)
        self._scene.scene.clear_geometry()
        folder  = os.path.dirname(str(filename))
        #step_ls =  ["Paratacco SX SBK.stp", "leva freno RACE piegata.stp", "SUPP PEDANA DX kawz900.stp", "Tendi catena DX scr.stp", "manubri 52 DX.stp", "manubri 52 SX.stp"  ]
        #for kk in range(len(step_ls)):
        #stp = step_ls[kk]

        if self.all_folder_check:
            files1 = glob.glob(folder+"/*.stp")
            files2 = glob.glob(folder+"/*.step")
            files = files1 + files2
        else:
            files = [filename]
        self.load_step_list(files)

    def _on_load_job_dialog_done(self, filename):

            self.window.close_dialog()
            self._apply_settings()
            print(filename)
            #self._scene.scene.clear_geometry()
            folder  = os.path.dirname(str(filename))
            #step_ls =  ["Paratacco SX SBK.stp", "leva freno RACE piegata.stp", "SUPP PEDANA DX kawz900.stp", "Tendi catena DX scr.stp", "manubri 52 DX.stp", "manubri 52 SX.stp"  ]
            #for kk in range(len(step_ls)):
            #stp = step_ls[kk]
            if not self.laser_connected_check:
                print("NOT CONNECTED TO LASER")
            else:
                res = self._change_K1_K2(2)
                if res==0:
                    res = self.SJ.loadJobFile(filename)
                    res =  self.SJ.getFlyCadStatus()
                    print(res)
                    res =  self.SJ.getFlyCadStatus()
                    print(res)
                    res =  self.SJ.getDynTextNumber()
                    print(res)
                    res =  self.SJ.getFlyCadStatus()
                    print(res)
                    print("job loaded successfully")

                    self.job_loaded = True
                else:
                    print("error loading job")
                    self.job_loaded = False

        #self._display_scene()

    def load_step_list(self, files):
        self.m = 0      
        self.cad_views = []
        self.cad_jobs = []
        self.cad_mesh = []
        self.scene_jobs = []
        numbers=[]
        for i in  range(len(files)):
            numbers.append(i)
        res = self.pool.starmap(mp_load_step_views, zip(files,numbers))
        #res = mp_load_step_views(files[0],numbers[0])
        res = np.asarray(res)
        points_ls = res[:,2]
        self.cad_views = res[:,0]
        self.cad_jobs = res[:,1]
        for i in range(len(res[:,3])):
            r = res[:,3][i]
            temp = []
            for m in r:
                for n in m:
                    v = n[0]
                    t = n[1]
                    norm = n[2]
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(v)
                    mesh.triangles = o3d.utility.Vector3iVector(t)
                    mesh.vertex_normals = o3d.utility.Vector3dVector(norm)
                    temp.append(mesh)
            self.cad_mesh.append([temp])
        m=0
        self.window.close_dialog()
        self._scene.scene.clear_geometry()
        self._apply_settings()
        #frame=o3d.geometry.TriangleMesh.create_coordinate_frame(size=100) 
        #self._scene.scene.add_geometry(str(random.random()*1000), frame,self.settings.material )
        for i in range(len(points_ls)):
            p = points_ls[i]
            mesh = copy.deepcopy(self.cad_mesh[i][0][0])
            t = np.min(p,axis=  0)
            p = p-t
            t[0]+=m
            mesh = mesh.translate(-t)
            m+=10
            rot = Rotation.random().as_matrix()
            mesh = mesh.rotate(rot,center=[0,0,0])
            geometry=mesh
            if geometry is not None:
                try:
                    material = rendering.MaterialRecord()
                    material.base_color = [random.random(), random.random(), random.random(), random.random()]
                    self._scene.scene.add_geometry(str(random.random())*1000, geometry,material)
                    bounds = geometry.get_axis_aligned_bounding_box()
                    self._scene.setup_camera(60, bounds, bounds.get_center())
                    print("step loaded")
                except Exception as e:
                    print(e)


            

    
        
    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)
        if self._actions_panel.visible:
            self._actions_panel.visible = not self._actions_panel.visible
            gui.Application.instance.menubar.set_checked(
                AppWindow.MENU_SHOW_ACTIONS, self._actions_panel.visible)
        
    def _on_menu_toggle_actions_panel(self):
        self._actions_panel.visible = not self._actions_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_ACTIONS, self._actions_panel.visible)
        if self._settings_panel.visible:
            self._settings_panel.visible = not self._settings_panel.visible
            gui.Application.instance.menubar.set_checked(
                AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)
    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Open3D GUI Example"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()
    def _on_exp_box_value(self,value):
        print(value)
        self.exposure_setpoint = float(value)

    def load(self, path):
        self._scene.scene.clear_geometry()

        geometry = None
        geometry_type = o3d.io.read_file_geometry_type(path)

        mesh = None
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            mesh = o3d.io.read_triangle_mesh(path)
        if mesh is not None:
            if len(mesh.triangles) == 0:
                print(
                    "[WARNING] Contains 0 triangles, will read as point cloud")
                mesh = None
            else:
                mesh.compute_vertex_normals()
                if len(mesh.vertex_colors) == 0:
                    mesh.paint_uniform_color([1, 1, 1])
                geometry = mesh
            # Make sure the mesh has texture coordinates
            if not mesh.has_triangle_uvs():
                uv = np.array([[0.0, 0.0]] * (3 * len(mesh.triangles)))
                mesh.triangle_uvs = o3d.utility.Vector2dVector(uv)
        else:
            print("[Info]", path, "appears to be a point cloud")

        if geometry is None:
            cloud = None
            try:
                cloud = o3d.io.read_point_cloud(path)
            except Exception:
                pass
            if cloud is not None:
                print("[Info] Successfully read", path)
                if not cloud.has_normals():
                    cloud.estimate_normals()
                cloud.normalize_normals()
                geometry = cloud
            else:
                print("[WARNING] Failed to read points", path)

        if geometry is not None:
            try:
                self._scene.scene.add_geometry("__model__", geometry,
                                               self.settings.material)
                bounds = geometry.get_axis_aligned_bounding_box()
                self._scene.setup_camera(60, bounds, bounds.get_center())
            except Exception as e:
                print(e)

    def export_image(self, path, width, height):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)
    
    def _connect_cameras(self):
        print("connecting to cameras")
        all_available=False
        not_conn = []
        self.Cameras_connected=[]
        for i in range(len(self.Cameras_ID)):
            id = self.Cameras_ID[i]
            cam = vcd.Vimba_camera()
            res = cam.connect(camera_id = id)
            #res=1
            if res>0:
                print("connected to :", id)
                cam.load_config_file(self.include_dir + "\\DATA\\CONFIG\\"+id+"_intrinsic.json")   #load if available config file
                self.Cameras_connected.append(cam)
               # cam.ChangeExposureTime(10000) #set as default exposure to 12000
                if i==0:
                    self.camera1_connected = True
                    self._connection_check1.checked = True
                elif i==1:
                    self.camera2_connected = True
                    self._connection_check2.checked = True
                elif i==2:
                    self.camera3_connected = True
                    self._connection_check3.checked = True
            else:
                not_conn.append(id)
            
        if len(self.Cameras_connected)==len(self.Cameras_ID):
            print("connected to all listed cameras")

            return 1 
        else:
            print("not connected to camera id : ", not_conn)
            return -1


    def change_exposure(self):
        if len(self.Cameras_connected)==0:
            print("No camera Connected, Please Connect to cameras")
            return -1
        else:
            time = float(self.exposure_setpoint)
            for c in self.Cameras_connected:
                c.ChangeExposureTime(time)
            return 1 

    def _capture_image(self):
        if len(self.Cameras_connected)==0:
            print("no cameras connected")
            return
        try:
            real = True
            if real:
            #res = self.move_to_target(0,0,200)
                images = []
                i = 0
                t = time.time()
                for c in self.Cameras_connected:
                    if self.save_photo:
                        filename = self.include_dir+"//DATA//image1_"+str(i)+".png"
                    else:
                        filename = ""
                    image = c.grab_image(filename =filename)
                    images.append(np.squeeze(image))
                    i+=1
                print("tempo per scattare foto : ", time.time()-t)
                self.first_images = copy.deepcopy(images)
                pxmm = self.Cameras_connected[0].pxmm
                t = time.time()
                img1 = mp_stitch_images(images,self.Cameras_connected,sf = 1)
                if self.save_photo:
                    cv2.imwrite(self.include_dir+"//DATA//stitched1.png",img1)
                print("tempo per stitching foto : ", time.time()-t)
               
                cv2.imwrite("C:\\Users\\alberto.scolari\Pictures\\\digiCamControl\\Session1\\stitched"+"_"+str(self.rand_id)+"_"+str(self.counter_images)+".png",img1)
                self.counter_images+=1
                self.image = img1
                self._display_image(img1)
            else:
                    i=0
                    images=[]
                    for c in self.Cameras_connected:
                            img1 = cv2.imread("C:\\Users\\alberto.scolari\\source\\repos\\MARCATURA_ROBOT\\DATA\\image1_"+str(i)+".png")
                            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                            img1 = img1.astype(np.uint8)
                            images.append(img1)
                            i+=1
                    self.first_images = copy.deepcopy(images)
                    #print("scattato tutto")
                    #pxmm = self.Cameras_connected[0].pxmm
                    #t = time.time()
                    ##t1 = threading.Thread(target = self.helper_stitch_images, args = ([images,self.Cameras_connected,1,0]))
                    ##t1.start()
                    #t = time.time()
                    img1 = mp_stitch_images(images,self.Cameras_connected,sf = 1)
                    #print(time.time()-t)
                    #i=0
                    #images=[]
                    #for c in self.Cameras_connected:
                    #    img2 = cv2.imread("C:\\Users\\alberto.scolari\\output\\calib_fotos\\1//image2_"+str(i)+".png")
                    #    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    #    img2 = img2.astype(np.uint8)
                    #    images.append(img2)
                    #    i+=1
                    #self.second_images = copy.deepcopy(images)
                    #print("scattato tutto")
                    #t = time.time()
                    ##t2 = threading.Thread(target = helper_stitch_images, args = ([images,self.Cameras_connected,1,1]))
                    ##t2.start()
                    #img2 = mp_stitch_images(images,self.Cameras_connected,sf = 1)
                    #print(time.time()-t)
                    ##t1.join()
                    ##t2.join()
                    #img3 = self.combine_half_images(img1,img2)
                    cv2.imwrite(self.include_dir+"//DATA//stitched_new.png",img1)

                    #img1 = np.squeeze(cv2.imread(self.include_dir + "\\DATA\\stitched106.png"))
                    #img1 = img1.astype(np.uint8)
                    #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    #img2 = np.squeeze(cv2.imread(self.include_dir + "\\DATA\\stitched206.png"))
                    #img2 = img2.astype(np.uint8)
                    #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    #img3 = self.combine_half_images(img1,img2)
                    #cv2.imwrite(self.include_dir+"//DATA//stitched.png",img)
                    #img = np.squeeze(cv2.imread("C:\\Users\\alberto.scolari\\source\\repos\\humans-RME\\DATA\\stitched - Copia.png"))
                    ##self.stitched_image.value = img # np.squeeze(images[2])
                    #img  = cv2.cvtColor( img , cv2.COLOR_BGR2GRAY)
                    #img3  = img.astype(np.uint8)


                    self.image = img1

                    self._display_image(img1)
        except Exception as e:
                    print(e)
      
    def projected_contour(self):

        aligned_points = copy.deepcopy(self.match_cad)
        aligneed_jobs = copy.deepcopy(self.scene_jobs)
        camera_positions = []
        boxs = []
        reg_hl = PC_registration.PointCloudRegistration()
        c = self.Cameras_connected[0]
        x_image = int(c.x_length_px*3)
        y_image = int( c.y_length_px*1.2)

        x_offset = self.origin[1]
        y_offset = y_image/c.pxmm - self.origin[0]
        z_offset = 0
        offset = np.asarray([x_offset, y_offset, z_offset])

        I = np.asarray([[1,0,0],
        [0,1,0],
        [0,0,1]])
        #convert in world coordinates
        rot = [[0,1,0],[-1,0,0],[0,0,1]]
        rot1 = [[-1,0,0],[0,-1,0],[0,0,1]]
        for c in self.Cameras_connected:
            x = c.x_coord + c.x_length_px/2/c.pxmm
            y = c.y_coord + c.y_length_px/2/c.pxmm
            z = c.z_coord
            camera_positions.append(reg_hl.Transformation_with_list(copy.deepcopy(np.asarray([x,y,z])), [I, I],[[0,0,0],-offset]))
            b = o3d.geometry.TriangleMesh.create_box(c.x_length_px/c.pxmm,c.y_length_px/c.pxmm,1)
            b.translate([c.x_coord,c.y_coord,-1])
            b.translate(-offset)
            boxs.append(b)




        frame=o3d.geometry.TriangleMesh.create_coordinate_frame(size=100) 
        frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=camera_positions[0])
        frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=camera_positions[1]) 
        frame3 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=camera_positions[2]) 
        
        rot = [[0,1,0],[-1,0,0],[0,0,1]] #rotation -90° asse z
        scene = [frame, frame1,frame2,frame3]
                #temp_scene_points =  reg_hl.Transformation_with_list(copy.deepcopy(self.scene_points[i]), [self.rotation.T,rot],[-self.origin,[0,0,0]])
        for i in range(len(aligned_points)):
            aligned_points[i]=reg_hl.Transformation_with_list(copy.deepcopy(aligned_points[i]), [I,rot],[-self.origin,[0,0,0]])
            self.scene_jobs[i] = reg_hl.Transformation_with_list(copy.deepcopy(self.scene_jobs[i]), [I,rot],[-self.origin,[0,0,0]])
            scene.append(NumpyToPCD(aligned_points[i]))
        for b in boxs:
            scene.append(b.paint_uniform_color([random.random(), random.random(), random.random()]))
        #
        #o3d.visualization.draw_geometries(scene)
        
        new_contours = []

        #create new contours
        for i in range(len(aligned_points)):
            a = copy.deepcopy(aligned_points[i])
            m = np.mean(a,axis=0)
            #choose closest camera
            dist = []
            for c in camera_positions:
                dist.append(np.linalg.norm(m-c))
            index = np.argmin(dist)
            c = camera_positions[index]
            nc = []
            #crate lines 
            for j in range(len(a)):
                p = a[j]
                v = c-p
                t = -p[2]/v[2]
                z=0
                x= p[0]+v[0]*t
                y= p[1] + v[1]*t
                nc.append(np.asarray([x,y,z]))
            new_contours.append(nc)
        for i in range(len(new_contours)):
            scene.append(NumpyToPCD(new_contours[i]).paint_uniform_color([1,0,0]))
       # o3d.visualization.draw_geometries(scene)
        for i in range(len(self.match_cad)):
            mc = np.squeeze(copy.deepcopy(new_contours[i]))
            md = np.squeeze(copy.deepcopy(self.match_det[i]))
            md = reg_hl.Transformation_with_list(copy.deepcopy(md), [I,rot],[-self.origin,[0,0,0]])
            #mm = self.match_mesh[i]
            Source =  copy.deepcopy(mc)
            Target = copy.deepcopy(md)
            Source_transf, RT_ls, Tls, error = reg_hl.SparseAlignment(Source,Target,1000,alpha = False,VISUALIZE = False)
            scene.append(NumpyToPCD(md).paint_uniform_color([0,0,0]))
            scene.append(NumpyToPCD(Source_transf).paint_uniform_color([0,0,1]))
            self.scene_jobs[i] = reg_hl.Transformation_with_list(copy.deepcopy(self.scene_jobs[i]), RT_ls,Tls)
       # o3d.visualization.draw_geometries(scene)


        pass
        #self.scene_points.append(copy.deepcopy(self.match_cad[k]))
        #self.scene_jobs.append(copy.deepcopy(self.match_cad_jobs[k]))
        #self.scene_mesh.append(copy.deepcopy(self.match_mesh[k]))


      

    def recompute_scaled_contours(self):
        #######
        ## rescale image based on z height of detected cad
        #######
        #store z height
        try:
            scale_f = self.scale_factor#0.00105
       
            z_ls = []
            z_old = 0
            count = []
            index_rep = []
            z_all = []
            print("changing height")
            for mc in self.match_cad:
                temp = copy.deepcopy(mc)
                temp[:,2] = temp[:,2]-np.min(temp[:,2])
                z = int(np.max(temp[:,2]))
                if z not in z_ls:
                    z_ls.append(z)
                z_all.append(z)
            print("z list : ", z_ls)
            img_ls =  []
            cam_ls = []
            sf_ls = []
            print("restitching with scaling")
            for i in z_ls:
                print("appending img_ls")
                img_ls.append(self.first_images)
                print(len(img_ls), " ", type(img_ls))
                print("appending cam_ls")
                cam_ls.append(self.Cameras_connected)
                print(len(cam_ls), " ", type(cam_ls))
                print("appending sf_ls")
                sf_ls.append((1+(self.zero_height+i)*scale_f))
                print(len(sf_ls), " ", type(sf_ls))

            print("mp stitching")
            tread_ls=[]
            self.temp_img1 = list(np.zeros(len(img_ls)))
            for i in range(len(img_ls)):
                t1 = threading.Thread(target=self.helper_stitch_images,args = (img_ls[i],cam_ls[i],sf_ls[i],i))
                t1.start()
                tread_ls.append(t1)
            for t in tread_ls:
                t.join()
            img1 = copy.deepcopy(self.temp_img1)
            print("mp stitching finished")

            img_ls =  []
            cam_ls = []
            sf_ls = []
            for i in z_ls:
                print("appending img_ls")
                img_ls.append(self.second_images)
                print(len(img_ls), " ", type(img_ls))
                print("appending cam_ls")
                cam_ls.append(self.Cameras_connected)
                print(len(cam_ls), " ", type(cam_ls))
                print("appending sf_ls")
                sf_ls.append((1+(self.zero_height+i)*scale_f))
                print(len(sf_ls), " ", type(sf_ls))
            self.temp_img1 = list(np.zeros(len(img_ls)))
            tread_ls=[]
            for i in range(len(img_ls)):
                t1 = threading.Thread(target=self.helper_stitch_images,args = (img_ls[i],cam_ls[i],sf_ls[i],i))
                t1.start()
                tread_ls.append(t1)
            for t in tread_ls:
                t.join()
            img2 = copy.deepcopy(self.temp_img1)
            print("mp stitching finished")
            #img2 = self.pool.starmap(mp_stitch_images,zip(img_ls,cam_ls,sf_ls))
            img = []
            print("combining half images")

            for i in range(len(img1)):
                n_img = self.combine_half_images(img1[i],img2[i])
                if self.save_photo:
                    cv2.imwrite(self.include_dir+"//DATA//scaled_image_"+str(i)+".png",n_img)
                img.append(n_img)

            # filter and extrect contour images
            thrs = int(self.threshold2)
            t_ls = []
            print("filtering")
            for i in range(len(img)):
                t_ls.append(thrs)
            thresh = self.pool.starmap(mp_filter,zip(img,t_ls))
            if self.back_color:
                for i in range(len(thresh)):
                    thresh[i] = cv2.bitwise_not(thresh[i])
            # extract new contorus
            print("extracting new contours")
            cam = vcd.Vimba_camera()
            t = time.time()
            points_ls = self.pool.starmap(mp_image_to_points,zip(thresh))
            print(time.time()-t)
            res = self.pool.starmap(mp_clusterization,zip(points_ls))
            det_objects = []
            contours1 = []
            for r in res:
                det_objects.append(r[0])
                contours1.append(r[1])


            complete_det_objects = []
            print("substituting new contours")

            for i in range(len(z_all)):
                for j in range(len(z_ls)):
                    if z_all[i]==z_ls[j]:   
                        do = det_objects[j]
                        complete_det_objects.append(do)
            # now len(complete_det_object) = len(self.matchdet) 
            ## change self.match_det contours

            for i in range(len(self.match_det)):
                cdo = complete_det_objects[i]
                md = np.mean(self.match_det[i],axis=0) #mid point
                md_ls = []
                for c in cdo:
                    md_ls.append(np.mean(c,axis=0))
                dist = np.linalg.norm(md - np.asarray(md_ls), axis=1) #list of all distances
                indx = np.argmin(dist)

                self.match_det[i] = cdo[indx]
                #find the closest in list of all 
            print("finished substituting the correct contours")
        except Exception as ex:
            print(ex)
            trace = []
            tb = ex.__traceback__
            while tb is not None:
                trace.append({
                    "filename": tb.tb_frame.f_code.co_filename,
                    "name": tb.tb_frame.f_code.co_name,
                    "lineno": tb.tb_lineno
                })
                tb = tb.tb_next
            print(str({
                'type': type(ex).__name__,
                'message': str(ex),
                'trace': trace
            }))


    def _capture_back_image(self):
            if len(self.Cameras_connected)==0:
                print("no cameras connected")

                return
            else:
                if self.laser_connected_check==False:
                    print("no laser connected")
                    return
                else:
                
                    if self.laser_homed==False:
                        print("laser is not been homed")
                        return
                    else:
                        #moving head out of field
                    
                        res = self.move_to_target(0,0,200)

                        #res = str(input("acquire first "))
                        images = []
                        all_points = []
                        i = 0
                        for c in self.Cameras_connected:
                            image = c.grab_image()
                            images.append(np.squeeze(image))
                            #all_points.append(self.hh.ImageToPoints(images[i],c))
                            i+=1
                        #images = self.pool.starmap(mp_grab_image,zip(self.Cameras_connected))
                        #all_points = np.concatenate(all_points,axis=0)
                        print("scattato tutto")
                        pxmm = self.Cameras_connected[0].pxmm
                        img1 = self.hh.stitch_images(images, self.Cameras_connected)

                        res = self.move_to_target(500,0,200)
                        #res = str(input("acquire second "))
                        images = []
                        all_points = []
                        i = 0
                        for c in self.Cameras_connected:
                            image = c.grab_image()
                            images.append(np.squeeze(image))
                            #all_points.append(self.hh.ImageToPoints(images[i],c))
                            i+=1
                       # images = self.pool.starmap(mp_grab_image,zip(self.Cameras_connected)) 
                        #all_points = np.concatenate(all_points,axis=0)
                        print("scattato tutto")
                        pxmm = self.Cameras_connected[0].pxmm
                        img2 = self.hh.stitch_images(images, self.Cameras_connected)
                    #    print("capturing image")
                    #    img1 = np.squeeze(cv2.imread(self.include_dir + "\\DATA\\test_rme\\data0_0.png"))
                    #    img1 = img1.astype(np.uint8)
                    #    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    #    img2 = np.squeeze(cv2.imread(self.include_dir + "\\DATA\\test_rme\\data0_1.png"))
                    #    img2 = img2.astype(np.uint8)
                    #    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

                        img = self.combine_half_images(img1,img2)
                        cv2.imwrite(self.include_dir+"//DATA//background.png",img)
                        
                        #self.stitched_image.value = img # np.squeeze(images[2])
                        self.image = img

                        self._display_image(img)

    def combine_half_images(self,img1,img2):
        temp = copy.deepcopy(img2)
        temp[:,int(temp.shape[0]/3*2):-1] = img1[:,int(temp.shape[0]/3*2):-1]
        return temp


    def _filter_image(self):
        self.scene_points = []
        print("filtering image")
        t = time.time()
      
        thrs = int(self.threshold2)
        thresh = self.image #.astype(np.uint8)
        t_ls = []
        thresh = mp_filter(self.image,thrs)
        if self.back_color:
            thresh = cv2.bitwise_not(thresh)
        self.thresh_image = thresh

        #cv2.imwrite(self.include_dir + "/thresholded_image"+str(random.randint(100,999))+".png", thresh1)
        print(time.time()-t)
        self._display_image(thresh)
        self.button_find_object()

    def button_find_object(self):
        #thresh = self.thresh_image
        #thresh = thresh.astype(np.uint8)
        cam = vcd.Vimba_camera()
        #n_process = 8
        #img_ls =  []
        #shifts = []
        #for i in range(n_process):
        #    img_ls.append(thresh[int(thresh.shape[0]/n_process*i):int(thresh.shape[0]/n_process*(i+1)),:])
        #    shifts.append(int(thresh.shape[0]/n_process*i))
        t = time.time()
        points_ls = mp_image_to_points(self.thresh_image)
        print(time.time()-t)

        self.scene_points.append(points_ls)
        temp = points_ls.copy()

        det_objects, contours1 = mp_clusterization(points_ls)
        

        imgcont = self.thresh_image.copy()
        imgcont = imgcont.astype(np.uint8)
        detected_box = []
        for cnt in contours1:
            #cnt = cnt.astype(np.uint8)
            #cnt = cv2.cvtColor(cnt, cv2.COLOR_BGR2GRAY)
            cnt = np.array(cnt).reshape((-1,1,2)).astype(np.int32)
            box = cv2.minAreaRect(cnt)
            points = cv2.boxPoints(box)
            points = np.int0(points)
            cv2.drawContours(imgcont,[points],0,(0,0,255),20)

        for cnt in det_objects:
            if len(det_objects)>1:
                cnt = cnt[:,0:2]
            else:
                cnt = cnt[0][:,0:2]
            cnt = np.array(cnt).reshape((-1,1,2)).astype(np.int32)
            box = cv2.minAreaRect(cnt)
            box = list(box)
            width = box[1][0]
            height = box[1][1]

            if width>height:
                temp = copy.deepcopy(height)
                height = copy.deepcopy(width)
                width = copy.deepcopy(temp)
            detected_box.append([width,height])
        self.det_objects = det_objects
        self.detected_box = detected_box
        self._display_image(imgcont)


        


            

    def _on_thresh1(self,thresh):
        #if not int(thresh)%2:
        #    self.threshold1 = int(thresh)+1
        #else:
        self.threshold1 = int(thresh)
        print(self.threshold1)
    def _on_thresh2(self,thresh):
        self.threshold2 = int(thresh)
        print(self.threshold2)
    def _load_step(self):
        print("loading step file")
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(".stp .step",
            "step files (.stp, .step)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_step_dialog_done)
        self.window.show_dialog(dlg)

    def _load_job_file(self):
            print("loading job file")
            dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                                 self.window.theme)
            dlg.add_filter(".job .JOB .lmf .LMF .Lmf .stl",
                "JOB files (.job, .JOB, .lmf, .LMF, .Lmf, .stl)")
            dlg.set_on_cancel(self._on_file_dialog_cancel)
            dlg.set_on_done(self._on_load_job_dialog_done)
            self.window.show_dialog(dlg)
        

    def _load_folder(self):
        print("loading folder step")
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_folder_step_dialog_done)
        self.window.show_dialog(dlg)
    def _load_image(self):
        print("loading step file")
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(".jpg .png .jpeg",
            "step files (.jpg, .png, .jpeg)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_image_dialog_done)
        self.window.show_dialog(dlg)
    def _on_corr_plane(self,check):
        self.correction_plane = check
    def _on_correct_3d(self,check):
        self.correction_3d = check
    def _on_back_color(self,check):
        self.back_color = check
         # Add the text
        

    def _on_find_match(self):
        """
        matching object in image based on bounding box
        """ 
        self.match_cad = []
        self.match_det = []
        self.matched_indexes = []
        self.match_cad_jobs = []
        self.chosen_bests = []
        self.match_mesh = []
        print("finding match")
        tim = time.time()
        if len(self.cad_views)==0:
            print("no cad loaded")
            return
        t_ls = []
        #self.det_objects_remained =copy.deepcopy(self.det_objects)
        #self.detected_box_remained = copy.deepcopy(self.detected_box)
        #if not self.force_match:
        for i in range(len(self.cad_views)):
            views = self.cad_views[i]
            jobs = self.cad_jobs[i]
            mesh = self.cad_mesh[i]
            t1  = threading.Thread(target = self.match_box_thread, args = ([views,jobs,mesh]))
            t1.start()
            t_ls.append(t1)
        
        for t in t_ls:
            t.join()

            
        #self.match_cad = np.squeeze(self.match_cad)
        #self.match_det = np.squeeze(self.match_c)

        # I have all the matches, now I choose the best for each detected shape based on a shape factor
        self.match_cad = [inner for outer in self.match_cad for inner in outer]
        self.match_det = [inner for outer in self.match_det for inner in outer]
        self.matched_indexes = [inner for outer in self.matched_indexes for inner in outer]
        self.match_cad_jobs = [inner for outer in self.match_cad_jobs for inner in outer]
        self.match_mesh =[inner for outer in self.match_mesh for inner in outer]

        if not self.all_folder_check:   
            # just one selected case : 
            # user tell the machine that there is at least 1 of the selected CAD
            # hipotesis : all views of selected object are the same

            self.match_single_cad()


            pass
        if self.all_folder_check:
            # multiple object present
            # need a fast way to decrease the number of matches 
            # use other shape factors
            print("numero totale di match : ", len(self.match_cad))
            shapes, temp_cad, temp_det = self.match_shape_indicator()
            temp_cad = list(temp_cad)
            temp_det = list(temp_det)
            # regroup based on the same index of the detected
            # choose the best 3 for each detected
            struct = np.asarray([self.match_cad, self.match_det, self.match_cad_jobs, shapes, self.matched_indexes,temp_cad,temp_det,self.match_mesh]).T
            groups = npi.group_by(self.matched_indexes, struct)[1]
            bests = []
            for i in range(len(groups)):
                g = groups[i]
                #sort based on shapes
                ind = np.argsort(g[:,3])
                best3 = g[ind[0:15]]
                bests.append(best3)
                #best1 = np.asarray(g[ind][3])
        #bests  = np.asarray([inner for outer in bests for inner in outer]).T
        
        
            for i in range(len(bests)):
                #self.choose_best_detected(bests[i])
                t1  = threading.Thread(target = self.choose_best_detected, args = ([bests[i]]))
                t1.start()
                t_ls.append(t1)
            for t in t_ls:
                t.join()

            new_bests = np.asarray(self.chosen_bests).T
            self.match_cad = new_bests[0]
            self.match_det = new_bests[1]
            self.match_cad_jobs = new_bests[2] 
            self.matched_indexes = new_bests[4] 
            self.match_mesh = new_bests[7]
 
        print("matched object indexes : ", np.shape(self.matched_indexes))

        print("final match cad len : ", np.shape(self.match_cad))
        print("total matching  time : ", time.time()-tim)
        pcds_cad = []
        pcds_det = []
        for i in range(len(self.match_cad)):
            mc = self.match_cad[i]
            md = self.match_det[i]
            
            cad_pts = copy.deepcopy(mc)
            cad_pts = np.squeeze(cad_pts)

            cad_pts[:,2] = 0
            cad_pts[:,0] = cad_pts[:,0] - np.mean(cad_pts[:,0])
            cad_pts[:,1] = cad_pts[:,1] - np.mean(cad_pts[:,1])
            det_pts = copy.deepcopy(md)
            det_pts = np.squeeze(det_pts)

            det_pts[:,2] = 0
            #translate in 0
            det_pts[:,0] = det_pts[:,0] - np.mean(det_pts[:,0])
            det_pts[:,1] = det_pts[:,1] - np.mean(det_pts[:,1])
            pcds_det.append(NumpyToPCD(det_pts))
            pcds_cad.append(NumpyToPCD(cad_pts))

        self._display_matches(pcds_det,pcds_cad)
                #o3d.visualization.draw_geometries([NumpyToPCD(det_pts).paint_uniform_color([0,0,1]), NumpyToPCD(cad_pts).paint_uniform_color([1,0,0])])
    
    def choose_best_detected(self, bests):
        
        
        match_cad_temp = copy.deepcopy(bests[:,5])
        match_det_temp = copy.deepcopy(bests[:,6])
       
        points1  = []
        points2 = []
        false_ls= []
        for i in range(len(match_cad_temp)):
            points1.append(match_det_temp[i])
            points2.append(match_cad_temp[i])
            false_ls.append(True)
        #parallel computation
        shape = self.pool.starmap(mp_shape_match,zip(points1,points2,false_ls))
        #shape = mp_shape_match(points1[0],points2[0],false_ls[0])
        ind = np.argmin(np.asarray(shape))
        self.chosen_bests.append(bests[ind])
        return



    def match_shape_indicator(self):
    
        cam = vcd.Vimba_camera()
        shapes = []
        n = mp.cpu_count()
        match_cad_temp = np.array_split(self.match_cad, n)
        match_det_temp = np.array_split(self.match_det, n)
        test_order = np.array_split(range(len(self.match_cad)),n)
        res = np.asarray(self.pool.starmap(mp_shape_indicator_computation,zip(match_cad_temp,match_det_temp, test_order)))
        shapes = res[:,0]
        temp_cad = res[:,1]
        temp_det = res[:,2]

         #shapes, temp_cad, temp_det
        #shapes = mp_shape_indicator_computation(match_cad_temp[0],match_det_temp[0], test_order[0] )
        shapes = np.concatenate(shapes,axis=0)
        temp_cad = np.asarray([inner for outer in temp_cad for inner in outer])
        temp_det = np.asarray([inner for outer in temp_det for inner in outer])
        return shapes, temp_cad, temp_det


    def invert_x_y(points):
        temp = copy.deepcopy(points[:,0])
        points[:,0] = copy.deepcopy(points[:,1])
        points[:,1] = copy.deepcopy(temp)
        return points


    def _on_mark(self):
      try:
            print("START MARKING")
            I = np.asarray([[1,0,0],
                    [0,1,0],
                    [0,0,1]])
            reg_hl = PC_registration.PointCloudRegistration()
            for i in range(len(self.scene_jobs)):
                rot = [[0,1,0],[-1,0,0],[0,0,1]]
                #temp_jobs =  reg_hl.Transformation_with_list(copy.deepcopy(self.scene_jobs[i]), [self.rotation.T, rot],[-self.origin,[0,0,0]])
                if  self.correction_3d is not True:
                  temp_jobs =  reg_hl.Transformation_with_list(copy.deepcopy(self.scene_jobs[i]), [I, rot],[-self.origin,[0,0,0]])
                else:
                    temp_jobs = self.scene_jobs[i]
                p = temp_jobs[0] #position
                #orientation
                V_x = temp_jobs[1] - temp_jobs[0]
                norm = np.linalg.norm(V_x)
                V_x = V_x/norm
                V_y = temp_jobs[3] - temp_jobs[0]
                norm = np.linalg.norm(V_y)
                V_y = V_y/norm
                V_z = temp_jobs[5] - temp_jobs[0]
                norm = np.linalg.norm(V_z)
                V_z = V_z/norm
                #find rotation angle in deegrees

                if V_z[2] >0.8:
                    theta1 = math.degrees(math.atan2(V_x[1],V_x[0]))
                    theta2 = math.degrees(-math.atan2(V_y[0],V_y[1]))
                    theta = (theta1+theta2)/2
                else: 
                    print("zeta is not upword")
                    theta1 = 0
                    theta2 = 0
                print(p, theta1, theta2)
                x = p[0]
                y = p[1]
                z = p[2] + self.zero_height

                if self.correction_plane:
                    x_err = self.fx_err(x,y)
                    print("x_error : ",x_err)
                    y_err = self.fy_err(x,y)
                    print("y_error : ", y_err)
                    x = x+x_err
                    y = y+y_err
               
                #if self.correction_3d:

                #    x_err = self.fx_err_3d(x,y,z)
                #    print(x_err)
                #    y_err = self.fy_err_3d(x,y,z)
                #    print(y_err)

                #    x = x+x_err
                #    y = y+y_err
                p = [x,y,z]
                px = p+3*V_x
                py = p+3*V_y
                self._on_homing_laser()
                self.robot.robot.buffer_set([[[x, y, z],self.tool_orientation], [[px[0], px[1], px[2]],self.tool_orientation],[[py[0], py[1], py[2]],self.tool_orientation]])
                self.robot.robot.go_mark_cross()
                self.robot.robot.clear_buffer()
                time.sleep(1)
                self._on_homing_laser()
      except Exception as e:
            print(e)
    
    #def on_mark_calibration_table(self):

               
    #               # t = time.time()
    #               ## res = self.robot.robot.buffer_set(data)
    #               # print("buffer set, time taken : ", time.time()-t)
    #               # for i in range(190,200,10):  
                 
    #               #         for j in range(-30,180,10):
    #               #             print("going to position : ", [i , j])
    #               #             self.robot.robot.go_circle([[i, j, 0],self.tool_orientation])
    #                        #self.robot.turnOnLaser()
    #                        #self.robot.robot.buffer_execute()
    #                        #self.robot.turnOffLaser()
    #    data = []
    #    with open(self.include_dir+"\\targets.txt") as f:
    #        lines = f.readlines()
    #    for l in lines:
    #        data.append(np.asarray(l.split(";")).astype(float)



    #    res = self.robot.robot.buffer_set(data)
                #res = self.robot.StartMark()
                #print(res)
                            # res = str(input("GO TO THE NEXT POSITION?"))
                    #    else:
                    #        print("error during movement")
                    #        return
                    #else:
                    #    print("y position out of range, going to next")
                    #    pass
    def _on_preview(self):
        print("PREVIEW")
        if len(self.scene_jobs) == 0:
            print("no target detected")
            return
        else:
            if self.laser_connected_check == False:
                print("laser not connected")
                return
            else:
                if self.job_loaded ==False:
                    print("NO JOB LOADED")
                    return
                else:
                    print("START PREVIEW")
                    I = np.asarray([[1,0,0],
                            [0,1,0],
                            [0,0,1]])
                    reg_hl = PC_registration.PointCloudRegistration()
                    for i in range(len(self.scene_jobs)):
                        rot = [[0,1,0],[-1,0,0],[0,0,1]]
                        #temp_jobs =  reg_hl.Transformation_with_list(copy.deepcopy(self.scene_jobs[i]), [self.rotation.T, rot],[-self.origin,[0,0,0]])
                        temp_jobs =  reg_hl.Transformation_with_list(copy.deepcopy(self.scene_jobs[i]), [I, rot],[-self.origin,[0,0,0]])
                        
                        p = temp_jobs[0] #position
                        
                        #orientation
                        V_x = temp_jobs[1] - temp_jobs[0]
                        norm = np.linalg.norm(V_x)
                        V_x = V_x/norm
                        V_y = temp_jobs[3] - temp_jobs[0]
                        norm = np.linalg.norm(V_y)
                        V_y = V_y/norm
                        V_z = temp_jobs[5] - temp_jobs[0]
                        norm = np.linalg.norm(V_z)
                        V_z = V_z/norm
                        #find rotation angle in deegrees

                        if V_z[2] >0.8:
                            theta1 = math.degrees(math.atan2(V_x[1],V_x[0]))
                            theta2 = math.degrees(-math.atan2(V_y[0],V_y[1]))
                            theta = (theta1+theta2)/2
                        else: 
                            print("zeta is not upword")
                            theta1 = 0
                            theta2 = 0
                        print(p, theta1, theta2)
                        x = p[0]
                        y = p[1]
                        z = p[2]
                        res = self.move_to_target(x,y,z,theta=theta)
        
                        if res ==0: #target reached
                            print(p)
                            res = self.SJ.runPreview()
                            print(res)
                            res = str(input("GO TO THE NEXT POSITION?"))
                            res = self.SJ.halt()
                        else:
                            print("error during movement")
                            return
    def _on_align_match(self):
        print("aligning mathces")
        print("changing matched shape")
        t = time.time()
        #self.recompute_scaled_contours()
        #print(time.time()-t)
        t = time.time()
        reg_hl = PC_registration.PointCloudRegistration()
        #self.matched_indexes = []
        errors = []
        RT = []
        T = []
        Source = []
        Target = []
        pools = []
        self.scene_jobs  = []
        self.scene_mesh = []
        self.scene_points= [self.scene_points[0]]
        for i in range(len(self.match_cad)):
   
            mc = np.squeeze(copy.deepcopy(self.match_cad[i]))
            md = np.squeeze(copy.deepcopy(self.match_det[i]))
            mi = self.matched_indexes[i]
            mj = self.match_cad_jobs[i]
            #mm = self.match_mesh[i]
            Source.append( copy.deepcopy(mc))
            Target.append( copy.deepcopy(md))

                #pools.append(self.pool)
        res = self.pool.starmap(mp_align_cad,zip(Source,Target))
        for r in res:
            errors.append(r[3])
            print("error : ", r[3])
            RT.append(r[1])
            T.append(r[2])
        for k in range(len(RT)):
            try:
                self.match_cad[k] = reg_hl.Transformation_with_list(np.squeeze(copy.deepcopy(self.match_cad[k])),RT[k],T[k])
            except:
                self.match_cad[k][0] = reg_hl.Transformation_with_list(np.squeeze(copy.deepcopy(self.match_cad[k])),RT[k],T[k])
            for j in range(len(RT[k])):
                    r = RT[k][j]
                    t = T[k][j]
                    self.match_mesh[k].rotate(r,center=(0, 0, 0))
                    self.match_mesh[k].translate(t)
            
            #o3d.visualization.draw_geometries([self.match_mesh[k], NumpyToPCD(self.match_cad[k])])
            I = np.asarray([[1,0,0],
                            [0,1,0],
                            [0,0,1]])
            T_tmp = [0,0,-min(self.match_cad[k][:,2])]

            self.match_mesh[k].translate(T_tmp)
            self.match_cad[k] = reg_hl.Transformation_with_list(self.match_cad[k],[I],[T_tmp])
            #allineo i jobs
            self.match_cad_jobs[k] = reg_hl.Transformation_with_list(self.match_cad_jobs[k],RT[k],T[k])
            
            I = np.asarray([[1,0,0],
                            [0,1,0],
                            [0,0,1]])
            self.match_cad_jobs[k] = reg_hl.Transformation_with_list(self.match_cad_jobs[k],[I],[T_tmp])
            
            #o3d.visualization.draw_geometries([self.match_mesh[k], NumpyToPCD(self.match_cad[k])])


        #now i have all aligned mathces, need to choose the best one for the same detected points
        #fo
        
            self.scene_points.append(copy.deepcopy(self.match_cad[k]))
            self.scene_jobs.append(copy.deepcopy(self.match_cad_jobs[k]))
            self.scene_mesh.append(copy.deepcopy(self.match_mesh[k]))

                
        
        #same_detected = []
        #for i in range(len(self.aligned_cad_matched)):

        if self.correction_3d:
            print("applying projection correction")
            self.projected_contour()


        self.match_det = []
        self.match_boxes = []
        self.match_cad = []
        self.matched_indexes = []
        self.aligned_match_cad = []
        print("finished alignement")
        print(time.time()-t)
        self._display_scene()

    def _on_display_scene(self):
        self._display_scene()

    def _on_check_folder(self,check):
        self.all_folder_check = check

    def _on_laser_conn_check(self,check):
        if check:
            self._connection_check.checked = False
        else:
            self._connection_check.checked = True

    def _on_conn1_check(self,check):
        if check:
            self._connection_check1.checked = False
        else:
            self._connection_check1.checked = True
    def _on_conn2_check(self,check):
        if check:
            self._connection_check2.checked = False
        else:
            self._connection_check2.checked = True
    def _on_conn3_check(self,check):
        if check:
            self._connection_check3.checked = False
        else:
            self._connection_check3.checked = True

    def _on_x_box_value(self,value):
        try:
            print(value)
            xyz = value.split(", ")
            self._x_value = float(xyz[0])
            self._y_value = float(xyz[1])
            self._z_value = float(xyz[2])
        except:
            pass

    def _on_j_box_value(self,value):
        try:
            print(value)
            joints = value.split(", ")
            self.j1 = float(joints[0])
            self.j2 = float(joints[1])
            self.j3 = float(joints[2])
            self.j4 = float(joints[3])
            self.j5 = float(joints[4])
            self.j6 = float(joints[5])
        except:
            pass

    def on_start_laser(self):
        try:
            print("starting laser")

            self.robot.StartLaser()
        except:
            print("error starting laser")
            pass
    def on_stop_laser(self):
        try:
            print("stopping laser")
            self.robot.StopLaser()
        except:
            print("error stopping laser")
            pass


    def _on_turn_on_laser(self):
        try:
            
            self.robot.turnOnLaser()
        except:
            print("error setting frequency")
            pass


    def _on_duty_box_value(self,value):
        try:
            value = float(value)/100*255
            self.arduino.write(bytes(str(int(value)),"utf-8"))
            
        except Exception as e:
            print("error setting duty cycle", e)
            pass


    def _on_z_x_box_value(self,value):
        zx = value.split(", ")
        self.x_max = float(zx[1])
        print("x_max set to : ", self.x_max)
        self.z_calib = float(zx[0])
        print("z_height set to : ",self.z_calib)
        self.x_step = float(zx[2])
        print("x_step set to : ",self.x_step)

    def match_single_cad(self):
        #if np.asarray(copy.deepcopy(self.match_cad)).ndim<3:
        #    self.match_cad = [self.match_cad]
        match_cad_temp = copy.deepcopy(self.match_cad)
        match_det_temp = copy.deepcopy(self.match_det)
        match_temp_index = copy.deepcopy(self.matched_indexes)
        match_cad_job = copy.deepcopy(self.match_cad_jobs)
        match_mesh_temp = copy.deepcopy(self.match_mesh)

        points1  = []
        points2 = []
           
        for i in range(len(match_cad_temp)):
            points1.append(match_det_temp[i])
            points2.append(match_cad_temp[i])
        #parallel computation
        shape = self.pool.starmap(mp_shape_match,zip(points1,points2))
        
        print("shape all :  _ _ ", shape)

        try:
            ind = np.argmin(shape)
            s = shape[ind]
            delta = [abs(shape[i]-s)/s for i in range(len(shape))]
            print(delta)
            #print(lenght)
            bol = np.asarray(delta)<0.3    #maybe add as parameter
            match_cad_temp = list(np.asarray(match_cad_temp)[bol])
            match_det_temp = list(np.asarray(match_det_temp)[bol])
            match_temp_index = list(np.asarray(match_temp_index)[bol])
            match_cad_job = list(np.asarray(match_cad_job)[bol])
            match_mesh_temp = list(np.asarray(match_mesh_temp)[bol])

            self.match_cad = match_cad_temp
            self.match_det = match_det_temp
            self.matched_indexes = match_temp_index
            self.match_cad_jobs = match_cad_job
            self.match_mesh = match_mesh_temp
        except:
            pass




    def _display_matches(self,pcds_det,pcds_cad):
        self.window.close_dialog()
        self._scene.scene.clear_geometry()
        self._apply_settings()
        m=0
        frame=o3d.geometry.TriangleMesh.create_coordinate_frame(size=100) 
        for i in range(len(pcds_det)):
            pcd1 = pcds_det[i]
            pcd1 = PCDToNumpy(pcd1)
            pcd1 = pcd1-np.min(pcd1,axis=0)
            pcd1[:,0] = pcd1[:,0]+m +10
            pcd1 = NumpyToPCD(pcd1)
            pcd2 = pcds_cad[i]
            pcd2 = PCDToNumpy(pcd2)
            pcd2 = pcd2-np.min(pcd2,axis=0)
            pcd2[:,0] = pcd2[:,0]+m+10
            m = max(pcd2[:,0])
            pcd2 = NumpyToPCD(pcd2)
            material = rendering.MaterialRecord()
            material.base_color = [1, 0, 0, 0]
            self._scene.scene.add_geometry(str(random.random()*1000), pcd1,material )

            material = rendering.MaterialRecord()
            material.base_color = [0, 0, 1, 0]
            self._scene.scene.add_geometry(str(random.random()*1000), pcd2,material )
        #self._scene.scene.add_geometry(str(random.random()*1000), frame,material )



    def _display_pointcloud(self,pointcloud):
        self.window.close_dialog()
        self._scene.scene.clear_geometry()
        self._apply_settings()
        material = rendering.MaterialRecord()
        material.base_color = [random.random(), random.random(), random.random(), random.random()]
        self._scene.scene.add_geometry(str(random.random()*1000), pointcloud,material )

    def _display_scene(self):
        self.window.close_dialog()
        self._scene.scene.clear_geometry()
        self._apply_settings()
        reg_hl = PC_registration.PointCloudRegistration()
        m=0
        frame=o3d.geometry.TriangleMesh.create_coordinate_frame(size=100) 
        I = np.asarray([[1,0,0],
                        [0,1,0],
                        [0,0,1]])
        for i in range(len(self.scene_points)):
            if i==0:
                material = rendering.MaterialRecord()
                material.base_color = [0, 0, 0, 0]
            else:
                material = rendering.MaterialRecord()
                prefab = Settings.PREFAB["Metal (smoother)"]
                for key, val in prefab.items():
                    setattr(material, "base_" + key, val)
                material.base_color = self.settings.material.base_color #[random.random(), random.random(), random.random(), random.random()]
                material.shader = self.settings.material.shader
            # transform all points relative to origin marked by lase
            if self.origin is not None:
                rot = [[0,1,0],[-1,0,0],[0,0,1]] #rotation -90° asse z
                #temp_scene_points =  reg_hl.Transformation_with_list(copy.deepcopy(self.scene_points[i]), [self.rotation.T,rot],[-self.origin,[0,0,0]])
                temp_scene_points =  reg_hl.Transformation_with_list(copy.deepcopy(self.scene_points[i]), [I,rot],[-self.origin,[0,0,0]])

                if i>0:
                    temp_mesh = copy.deepcopy(self.scene_mesh[i-1]) #.rotate(self.rotation.T,center=[0,0,0])
                    temp_mesh = copy.deepcopy(temp_mesh).translate(-self.origin)
                    temp_mesh = copy.deepcopy(temp_mesh).rotate(rot,center=[0,0,0])

            #need to be converted in laser coordinates
            else: 
                temp_scene_points = copy.deepcopy(self.scene_points[i])
                if i>0:
                    temp_mesh = copy.deepcopy(self.scene_mesh[i-1])
            if i==0:
                pcd = NumpyToPCD(temp_scene_points)
            else:
                pcd = temp_mesh
                #pcd.estimate_normals()
            self._scene.scene.add_geometry(str(random.random()*1000), pcd,material )
        for i in range(len(self.scene_jobs)):
           
            
            material = rendering.MaterialRecord()

            material.base_color = [random.random(), random.random(), random.random(), random.random()]
            material.point_size  = 20
            if self.origin is not None:
                #temp_jobs =  reg_hl.Transformation_with_list(copy.deepcopy(self.scene_jobs[i]), [self.rotation.T,rot],[-self.origin,[0,0,0]])
                temp_jobs =  reg_hl.Transformation_with_list(copy.deepcopy(self.scene_jobs[i]), [I ,rot],[-self.origin,[0,0,0]])
                
                V_x = temp_jobs[1] - temp_jobs[0]
                norm = np.linalg.norm(V_x)
                V_x = V_x/norm
                V_y = temp_jobs[3] - temp_jobs[0]
                norm = np.linalg.norm(V_y)
                V_y = V_y/norm
                V_z = temp_jobs[5] - temp_jobs[0]
                norm = np.linalg.norm(V_z)
                V_z = V_z/norm
                #find rotation angle in deegrees

                theta1 = math.degrees(math.acos(V_x[0]))
                theta2 = math.degrees(math.asin(V_y[0]))
                theta = (theta1+theta2)/2
                        
            else: 
                temp_jobs =copy.deepcopy(self.scene_jobs[i]) 
            pcd = NumpyToPCD(temp_jobs)

            self._scene.scene.add_geometry(str(random.random()*1000), pcd,material )
        bounds = frame.get_axis_aligned_bounding_box()
        self._scene.setup_camera(60, bounds, bounds.get_center())
        self._scene.scene.add_geometry(str(random.random()*1000), frame,material )

    def _on_calibrate_button(self):
        """
        mark at a specified z height the job load reapeatadly equally spaced of 10mmm
        until a given limit x_max
        """
        try:
            if self.job_loaded==False:
                print("No job file loaded")
                return
            if self.laser_connected_check == False:
                    print("laser not connected")
                    return
            if self.z_calib == None:
                    print("no calib Z set")
                    return
            print("STARTING CALIBRATE MARKING")

            x_max = self.x_max
            x_step = self.x_step
            N = math.ceil(x_max/x_step)
            for i in range(N):
                x = x_step*i
                z = self.z_calib
                y = 0
                res = self.move_to_target(x,y,z)
                if res ==0: #target reached
                    res = self.SJ.startMark()
                    print(res)
                    # res = str(input("GO TO THE NEXT POSITION?"))
                else:
                    print("error during movement")
                    return
            print("finished")
        except Exception as  e:
            print(e)




    def _on_load_render_done(self,file):
        self.render_file = file
        print(file)
    def _on_load_render_back_done(self,file):
        self.render_back= file
        print(file)

    def _display_image(self,image):
        self.window.close_dialog()
        self._scene.scene.clear_geometry()
        try:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        except:
            pass


            #convert to Open3d image
        image = o3d.geometry.Image(image.astype(np.uint8))
        if image is not None:
            try:
                print(type(image))
                self._scene.scene.set_background(np.asarray([0,0,0,0]),image)
                print("image loaded")
            except Exception as e:
                print(e)







    def match_box_thread(self,views,jobs,mesh):
        views = np.squeeze(views)
        jobs = np.squeeze(jobs)
        mesh=np.squeeze(mesh)
        cad_box = []
        match_cad_temp = []
        match_det_temp = []
        match_cad_job = []
        match_mesh_temp =[]
        for v in views:
                v = v[:,0:2]
                v = np.array(v).reshape((-1,1,2)).astype(np.int32)
                box = cv2.minAreaRect(v)
                box = list(box)
                width = box[1][0]
                height = box[1][1]
                #points = cv2.boxPoints(box)
                #points = np.int0(points)
                #area = cv2.contourArea(points)
                #print(area)
                #perimeter = cv2.arcLength(points,True)
                #print(perimeter)
                if width>height:
                    temp = copy.deepcopy(height)
                    height = copy.deepcopy(width)
                    width = copy.deepcopy(temp)
                cad_box.append([width,height])
                #dx = max(v[:,0])-min(v[:,0])
                ##dy =  max(v[:,0])-min(v[:,0])
                #cad_box.append([dx*dy,dx+dy])
    
        det_box = self.detected_box

        ## loop on all detected
        thrsh = 0.3
        i=0
        k = 0
        match = []
        for d in det_box:
            k=0
            for c in cad_box:
                #print(str(c[0]) + " " + str(d[0]) + " " + str(c[1]) + " " + str(d[1]) + " " )
                print(str(abs(c[0]-d[0])/c[0]*100) + " " + str(abs(c[1]-d[1])/c[1]*100) )
                if abs(c[0]-d[0])/c[0] < thrsh and abs(c[1]-d[1])/c[1] <thrsh:
                    #we have a match
                    match.append([i,k])
                    #self.match_boxes.append(np.mean([abs(c[0]-d[0])/c[0],abs(c[1]-d[1])/c[1]]))
                    print("we have a match!")
                k+=1
                #print("i : "+ str(i) + " k : " + str(k))
                #print()
            i+=1
        match_temp_index = []
        for m in match:
            cad_pts = copy.deepcopy(views[m[1]])
            cad_pts[:,2] = 0
            #translate in 0
            cad_pts[:,0] = cad_pts[:,0] - np.mean(cad_pts[:,0])
            cad_pts[:,1] = cad_pts[:,1] - np.mean(cad_pts[:,1])
            det_pts = copy.deepcopy(self.det_objects[m[0]])
            det_pts[:,2] = 0
            #translate in 0
            det_pts[:,0] = det_pts[:,0] - np.mean(det_pts[:,0])
            det_pts[:,1] = det_pts[:,1] - np.mean(det_pts[:,1])
            match_cad_temp.append(copy.deepcopy(views[m[1]]))
            match_det_temp.append(copy.deepcopy(self.det_objects[m[0]]))
            match_temp_index.append(m[0])
            match_cad_job.append(copy.deepcopy(jobs[m[1]]))
            match_mesh_temp.append(copy.deepcopy(mesh[m[1]]))
            #o3d.visualization.draw_geometries([NumpyToPCD(det_pts).paint_uniform_color([0,0,1]), NumpyToPCD(cad_pts).paint_uniform_color([1,0,0])])

        #among this matching select based on other features, shapeindicators

        points1  = []
        #points2 = []
           
        #for i in range(len(match_cad_temp)):
        #    points1.append(match_det_temp[i])
        #    points2.append(match_cad_temp[i])
        ##parallel computation
        ##res = self.pool.starmap(mp_shape_match,zip(points1,points2))
        #for i in range(len(points1)):
        #    res = self.match_shape(points1[i],  points2[i])


        #    shape.append(s)
        #    lenght.append(l)
        #    area.append(a)
        #shape = np.asarray(res)#[:,0]
        #f1 = np.asarray(res)[:,1]
        #print("shape list is : _  __  ", shape)
        #print("f1 list is : _  __  ", f1)

        try:

            #ind = np.argmin(shape)
            #s = shape[ind]
            #delta = [abs(shape[i]-s)/s for i in range(len(shape))]
            #print(delta)
            ##print(lenght)
            #bol = np.asarray(delta)<0.3
            #match_cad_temp = list(np.asarray(match_cad_temp)[bol])
            #match_det_temp = list(np.asarray(match_det_temp)[bol])
            #match_temp_index = list(np.asarray(match_temp_index)[bol])
            #match_cad_job = list(np.asarray(match_cad_job)[bol])loadin
        
            self.match_cad.append(match_cad_temp)
            self.match_det.append(match_det_temp)
            #store each matched cad with self.det_objects indexes
            self.matched_indexes.append(match_temp_index)
            self.match_cad_jobs.append(match_cad_job)
            self.match_mesh.append(match_mesh_temp)
            print("indexes :" , self.matched_indexes)
            print("first cad match len :" , np.shape(self.match_cad))
        except:
           pass


    def _connect_laser(self):
        try:
            self.arduino = serial.Serial(port="COM6", baudrate=57600,timeout=.1)
        except:
            print("not connected to arduino")
        self.robot  = robot.newFrank()

        

        res = self.robot.Connect()

        if res==1:
            self.laser_connected_check = True
            self._connection_check.checked = True
            print("connected successfully to laser")


        else:
            self.laser_connected_check = False
            self._connection_check.checked = False
            print("error connecting to laser")
        try:

            res = np.loadtxt(self.include_dir + "\\DATA\\CONFIG\\calibrazione_piano.txt")
            res = res.T
            x = res[0]
            y = res[1]
            x_err = res[2]
            y_err = res[3]

            self.fx_err = Rbf(x,y,x_err, function = "cubic" )
            self.fy_err = Rbf(x,y,y_err, function = "cubic" )
            print("calibrazione piano base caricata")

            #3d compensation
            ### test 3d correction
            x = np.linspace(0,480, 33)
            y = np.linspace(-75,75,11)
            z = np.asarray([3,20,40,60])

            xyz = np.array(list(product(z,x,y)))
            z = xyz[:,0]
            x = xyz[:,1]
            y = xyz[:,2]
            x_err = []
            y_err = []
            for i in [0,20,40,60]:
                i = int(i)
                res = np.loadtxt(self.include_dir +"\\DATA\\CONFIG\\calibrazione_piano_"+str(i)+".txt").T
                x_err.append(res[2])
                y_err.append(res[3])
            x_err = np.concatenate(x_err, axis= 0)
            y_err = np.concatenate(y_err, axis= 0)


            self.fx_err_3d = Rbf(x,y,z,x_err, function = "cubic" )
            self.fy_err_3d = Rbf(x,y,z,y_err, function = "cubic" )
            print("calibrazione piano 3d caricata")

        except Exception as e:
            print(e)

    def _on_homing_laser(self):
       
        try:
            print("homing")
            self.robot.GoToJointsCoord(-90,20,30,0,-50,0)
        except:
            print("error")

    def _on_move_joints(self):
        print("going to position : ",[self.j1,self.j2,self.j3,self.j4,self.j5,self.j6])
        self.robot.GoToJointsCoord(self.j1,self.j2,self.j3,self.j4,self.j5,self.j6)
        pass




    def _on_move_laser(self):
        if self.laser_connected_check == False:
            print("connect to laser")
        else:
            x = float(self._x_value)
            y = float(self._y_value)
            z = float(self._z_value)
            print("going to position : ", [[x, y, z],self.tool_orientation])

            self.robot.robot.set_cartesian([[x, y, z],self.tool_orientation])

    def move_to_target(self, x,y,z, theta=0):
        if self.laser_connected_check == False:
            print("connect to laser")
            return -1
        else:
            #setting status to K2
            try:
                print("going to position : ", [[x, y, z], [0.5, 0.5, 0.5, -0.5]])
                self.robot.robot.set_cartesian([[x, y, z], [0.5, 0.5, 0.5, -0.5]])
                return 1
            except:
                return -1




    def _change_K1_K2(self,target):
        # allow to witch from status K1 to status K2 and viceversa
        if self.laser_connected_check == False:
            print("connect to laser")
            return -1
        else:
            res  =  self.SJ.getFlyCadStatus()
            print("Stauts : ", res)
            res = self.SJ.getFlyCadStatus()
            print("Stauts : ", res)
            K = res[1]
            if K=="K-1":
                print("Stauts : ", res)
                time.sleep(0.5)
                self.SJ.getDynTextNumber()
                time.sleep(0.5)
                res  = self.SJ.getFlyCadStatus()
                print("Stauts : ", res)
            if K=="K54":
                time.sleep(0.5)
                self.SJ.getFlyCadStatus()
                print("Stauts : ", res)
            print("Stauts : ", res)
            K = res[1]
            if K=="K1" and target==2:
                print("changing status from K1 to K2")
                self.SJ.setRunMode()
                time.sleep(0.5)
                self.SJ.getFlyCadStatus()
                print("Stauts : ", res)
                time.sleep(0.5)
                self.SJ.getDynTextNumber()
                time.sleep(0.5)
                res  = self.SJ.getFlyCadStatus()
                print("Stauts : ", res)
                K_N = res[1]
                if K_N =="K2":
                    print("status changed successfully")
                    self.laser_status = K_N
                    return 0
                else:
                    print("error changing status")
                    return -1
            if K=="K2" and target==1:
                print("changing status from K2 to K1")
                self.SJ.homing()
                time.sleep(0.5)
                res  =  self.SJ.getFlyCadStatus()
                print("Stauts : ", res)
                K_N = res[1]
                if K_N =="K1":
                    print("status changed successfully")
                    self.laser_status = K_N
                    return 0
                else:
                    print("error changing status")
                    return -1
            if K=="K1" and target==1:
                print("no need to change status")
                self.laser_status = K
                return 0
            if K=="K2" and target==2:
                print("no need to change status")
                self.laser_status = K
                return 0

    def _find_origin(self):
        image = self.image#cv2.imread("C:\\Users\\alberto.scolari\\source\\repos\\MARCATURA_ROBOT\\DATA\\stitched_filt.png")
        cam = vcd.Vimba_camera()
        #arUco dictionary selection
        arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters = params)
        print(corners)

        if corners!=():
            corners_mm = np.squeeze(np.asarray(copy.deepcopy(corners)))/cam.pxmm


            

            v1 = list(corners_mm[1]-corners_mm[0])
            v1.append(0)
            n = np.linalg.norm(v1)
            v1 = v1/n
            v2 = list(corners_mm[3]-corners_mm[0])
            v2.append(0)
            n = np.linalg.norm(v2)
            v2 = v2/n
            v3 = [0,0,1]

            vectorx  = np.asarray([v1*k for k in range(1000)])
            vectory = np.asarray([v2*k for k in range(1000)])
            frame=o3d.geometry.TriangleMesh.create_coordinate_frame(size=100) 
            o3d.visualization.draw_geometries([NumpyToPCD(vectorx).paint_uniform_color([1,0,0]),NumpyToPCD(vectory).paint_uniform_color([0,0,1]),frame]) #,
                                   #NumpyToPCD(pointsx_mm).paint_uniform_color([1,0,0]),NumpyToPCD(pointsy_mm).paint_uniform_color([0,0,1])])


            RT =  [[v1[0], v2[0], v3[0]], [v1[1], v2[1], v3[1]], [v1[2], v2[2], v3[2]]]

            self.rotation = np.asarray(RT)



            corners = np.array(corners).reshape((-1,1,2)).astype(np.int32)
            try:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            except:
                pass
            #image = cv2.aruco.drawDetectedMarkers(image, corners)
            try:
                cv2.drawContours(image,[corners],0,(0,255,0),20)
            except:
                pass

            self._display_image(image)
            try:
                self.origin = np.asarray([np.mean(corners_mm[:,1]),np.mean(corners_mm[:,0]),0])
            except:
                pass
            print(self.origin)

            f = open(self.include_dir+"/DATA/CONFIG/machine_origin.txt","w" )
            f.write(str(self.origin[0]) + ";"+str(self.origin[1]) + ";"+str(self.origin[2]))
            f.write("\n")
            f.write(str(RT[0][0]) + ";"+str(RT[0][1]) + ";"+str(RT[0][2]) + "\n"+
                    str(RT[1][0]) + ";"+str(RT[1][1]) + ";"+str(RT[1][2]) + "\n"+
                    str(RT[2][0]) + ";"+str(RT[2][1]) + ";"+str(RT[2][2]) )
            f.close()
        else:
            with open(self.include_dir+"/DATA/CONFIG/machine_origin.txt") as f:
                lines = f.readlines()
            self.origin = np.asarray(lines[0].split(";")).astype(float)
            self.rotation = np.asarray([lines[1].split(";"),lines[2].split(";"),lines[3].split(";")]).astype(float)
            print(self.rotation)

    def match_shape(self,points1, points2):
        points2 = Edge_Detection(points2, 5, 0.001)
        points2[:,2] = points2[:,2]*0
        cam = vcd.Vimba_camera()
        img1, points1  = self.hh.PointCloudToImage(points1, cam.pxmm)
        img2, points2  = self.hh.PointCloudToImage(points2, cam.pxmm)
        #img1 = cv2.resize(img1, (int(img1.shape[1]/10), int(img1.shape[0]/10)))
        #bol = img1>0
        #img1[bol] = 255
        #img2 = cv2.resize(img2, (int(img2.shape[1]/10), int(img2.shape[0]/10)))
        #bol = img2>0
        #img2[bol] = 255
        v1 = points1[:,0:2]
        v2 = points2[:,0:2]

    def helper_stitch_images(self,images,cams,sf,ind):
            img = mp_stitch_images(images,cams,sf)
            self.temp_img1[ind] = img



        #cnt1, hierarchy1 = cv2.findContours(img1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cnt2, hierarchy2 = cv2.findContours(img2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #for i in range(len(cnt1)):
        #    color = (random.random()*255, random.random()*255,random.random()*255)
        #    cv2.drawContours(img1, cnt1, i, color, 2, cv2.LINE_8, hierarchy1, 0)
        #for i in range(len(cnt2)):
        #    color = (random.random()*255, random.random()*255,random.random()*255)
        #    cv2.drawContours(img2, cnt2, i, color, 2, cv2.LINE_8, hierarchy2, 0)

        #shape1 = cv2.matchShapes(img1,img2,cv2.CONTOURS_MATCH_I1,0.0)
        #shape2 = cv2.matchShapes(img1, cv2.flip(img2,0),cv2.CONTOURS_MATCH_I1,0.0)
        #if shape1<shape2:
        #    I1 = shape1
        #else:
        #    I1 = shape2
        #    img2 = cv2.flip(img2,0)
        
        #shape1 = cv2.matchShapes(img1,img2,cv2.CONTOURS_MATCH_I2,0.0)
        #shape2 = cv2.matchShapes(img1, cv2.flip(img2,0),cv2.CONTOURS_MATCH_I2,0.0)
        #if shape1<shape2:
        #    I2 = shape1
        #else:
        #    I2 = shape2
        #    img2 = cv2.flip(img2,0)

        #shape1 = cv2.matchShapes(img1,img2,cv2.CONTOURS_MATCH_I3,0.0)
        #shape2 = cv2.matchShapes(img1, cv2.flip(img2,0),cv2.CONTOURS_MATCH_I3,0.0)
        #if shape1<shape2:
        #    I3 = shape1
        #else:
        #    I3 = shape2
        #    img2 = cv2.flip(img2,0)
        #print("I1 = "+str(I1)+" I2 = "+str(I2)+" I3 = "+str(I3))







        #cv2.imwrite(self.include_dir+"/cad_"+str(random.randint(100,999))+".png", img2)
        #cv2.imwrite(self.include_dir+"/det_"+str(random.randint(100,999))+".png", img1)

def mp_filter( img,thrs):
        """
        input: portion of the image that needs to be filtered
        output: filtered portion
        """
   
        #thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        #thresh = back.astype(np.uint8)
        #back = cv2.adaptiveThreshold(back,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,thrs,thrs1)
    
        #if thrs!=0:
        #    t, back = cv2.threshold(back,thrs,255,cv2.THRESH_BINARY)
        #else:
        #    t, back = cv2.threshold(back,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #back = cv2.medianBlur(back,5)
        #back = cv2.medianBlur(back,5)

        #bol = back<0.1
        
        #thresh = cv2.medianBlur(thresh,5)
        #thresh = cv2.bilateralFilter(thresh ,9,75,75)
        ##thresh = cv2.adaptiveThreshold(thresh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,thrs,thrs1)
        #filter_case=2
        #if filter_case==1:
        #    thresh = cv2.GaussianBlur(thresh,(5,5),2)
        #    thresh = cv2.GaussianBlur(thresh,(5,5),2)
        #    thresh = cv2.GaussianBlur(thresh,(5,5),2)
        #    thresh = thresh - back
        #    thresh = cv2.adaptiveThreshold(thresh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,5)
        #    thresh = cv2.medianBlur(thresh,5)
        #    thresh = cv2.bitwise_not(thresh)
        #    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        #    thresh = cv2.erode(thresh, element, iterations = 1)
        #    thresh = cv2.medianBlur(thresh,3)
        #if filter_case ==2:
        img = img.astype(np.uint8)
        if thrs!=0:
            t, img = cv2.threshold(img,thrs,255,cv2.THRESH_BINARY)
        else:
            t, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #back = cv2.medianBlur(back,5)
        #back = cv2.medianBlur(back,5)

        ##bol = back<0.1
        #back = cv2.GaussianBlur(back,(5,5),2)
        #thresh = cv2.GaussianBlur(thresh,(5,5),2)

        ##thresh = thresh - back

        #if thrs!=0:
        #    t, thresh = cv2.threshold(thresh,thrs,255,cv2.THRESH_BINARY)
        #else:
        #    t, thresh = cv2.threshold(thresh,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #thresh[bol] = 255
        #for i in range(5):
        # thresh = cv2.medianBlur(thresh,5)
        #img = cv2.bitwise_not(img)
            #t, thresh = cv2.threshold(thresh,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
        return img

def mp_image_to_points(image):
        cam = vcd.Vimba_camera()
        hh =  helper.Helperhandler()
        pts = hh.ImageToPoints(image,cam) 
        #print(len(pts))
        bol = pts[:,2]<0.5
        pts = pts[bol]
        index = np.random.choice(pts.shape[0],50000, replace = False)
        pts = pts[index]
        #index = np.random.choice(pts.shape[0],int(200000/8), replace = False)
        #pts = pts[index]
        return pts
def mp_stitch_images(images,cams,sf):
        hh =  helper.Helperhandler()
        img = hh.stitch_images(images,cams,scale_f=sf)
        return img

def mp_align_cad(Source,Target):
        #estraggo edges
        #Source = Edge_Detection(Source, 5, 0.001)
        #Target = Edge_Detection(Target, 5, 0.001)
        # o3d.visualization.draw_geometries([NumpyToPCD(Source).paint_uniform_color([1,0,0]), NumpyToPCD(Target).paint_uniform_color([0,0,1])])
        Source[:,2] = 0
        Target[:,2] = 0
        reg_hl = PC_registration.PointCloudRegistration()
        Source_transf, RT_ls, Tls, error = reg_hl.SparseAlignment(Source,Target,100,alpha = False)
        #o3d.visualization.draw_geometries([NumpyToPCD(Source_transf).paint_uniform_color([1,0,0]), NumpyToPCD(Target).paint_uniform_color([0,0,1])])
        return Source_transf, RT_ls, Tls, error

def invert_x_y(points):
    temp = copy.deepcopy(points[:,0])
    points[:,0] = copy.deepcopy(points[:,1])
    points[:,1] = copy.deepcopy(temp)
    return points

def mp_shape_indicator_computation(cad, det, number):
    shapes = []
    numbers = []
    cam = vcd.Vimba_camera()
    hh =  helper.Helperhandler()
    temp_cad = []
    temp_det = []
    for i in range(len(cad)):
            points1 = copy.deepcopy(cad[i])
            points2 = copy.deepcopy(det[i])
            n = number[i]
            img1, points1  = hh.PointCloudToImage(points1, cam.pxmm)
            img2, points2  = hh.PointCloudToImage(points2, cam.pxmm)
            scale_factor = 4
            img1 = cv2.resize(img1, (int(img1.shape[1]/scale_factor), int(img1.shape[0]/scale_factor)))
            bol = img1>0
            img1[bol] = 255
            img2 = cv2.resize(img2, (int(img2.shape[1]/scale_factor), int(img2.shape[0]/scale_factor)))
            bol = img2>0
            img2[bol] = 255

            #cv2.imwrite("C:/Users/alberto.scolari/Desktop/foto_prova_shape/img1_"+str(i)+".png",img1)
            #cv2.imwrite("C:/Users/alberto.scolari/Desktop/foto_prova_shape/img2_"+str(i)+".png", img2 )


            v1 = points1[:,0:2]/scale_factor
            v1 = invert_x_y(v1)
            
            v2 = points2[:,0:2]/scale_factor
            v2 = invert_x_y(v2)
            alpha = 0.5
            #npoints = 1000
            #v1 = np.asarray(random.sample(list(v1), npoints))
            #v2 = np.asarray(random.sample(list(v2), npoints))
            i = 0
            while True:
                try:
                    v1 = np.asarray(alphashape.alphashape(v1, 0.2+i).exterior.coords.xy).T
                    v2 = np.asarray(alphashape.alphashape(v2, 0.2+i).exterior.coords.xy).T
                    break
                except:
                    i-=0.01

                    #print("*")
                    pass
            cd = np.zeros((len(v1),3))
            dt = np.zeros((len(v2),3))
            #print("ok1")
            cd[:,0:2] = v1
            dt[:,0:2]= v2
            #print("ok2")

            cnt_a = np.array(v1).reshape((-1,1,2)).astype(np.int32)
            cnt_b = np.array(v2).reshape((-1,1,2)).astype(np.int32)
            #cnt_a = cv2.convexHull(cnt_a)
            #cnt_b = cv2.convexHull(cnt_b)


            img1 = np.zeros((img1.shape[0], img1.shape[1])) 
            img2 = np.zeros((img2.shape[0], img2.shape[1])) 

            cv2.fillPoly(img1, pts = [cnt_a], color =(255,255,255))
            cv2.fillPoly(img2, pts = [cnt_b], color =(255,255,255))
            n =int(random.random()*100)
            #cv2.imwrite("C:/Users/alberto.scolari/Desktop/foto_prova_shape/img_"+str(n)+str(i)+"_1.png",img1)
            #cv2.imwrite("C:/Users/alberto.scolari/Desktop/foto_prova_shape/img_"+str(n)+str(i)+"_2.png", img2 )
            shape1 = cv2.matchShapes(img1,img2,cv2.CONTOURS_MATCH_I1,0.0)
            shape2 = cv2.matchShapes(img1, cv2.flip(img2,0),cv2.CONTOURS_MATCH_I1,0.0)
            if shape1<shape2:
                I1 = shape1
            else:
                I1 = shape2
                img2 = cv2.flip(img2,0)
            shapes.append(I1)
            numbers.append(n)
            temp_cad.append(cd)
            temp_det.append(dt)
    return [shapes, temp_cad, temp_det]

def mp_shape_match(points1,points2,multiple=False):
           #### use different indicator to match the two pointcloud
    points1 = np.squeeze(points1)
    points2 = np.squeeze(points2)
    if multiple==False:
        points2 = Edge_Detection(points2, 5, 0.001)
    points2[:,2] = points2[:,2]*0
    #npoints =3000
    #if len(points1)>npoints:
    #        points1 = np.asarray(random.sample(list(points1), npoints))
    #if len(points2)>npoints:
    #        points2 = np.asarray(random.sample(list(points2), npoints))
    reg_hl = PC_registration.PointCloudRegistration()
    Source_transf, RT_ls, Tls, error = reg_hl.SparseAlignment(points1,points2,100, alpha = False)
    
    print(error)

    return  error

def calc_precision_recall(contours_a, contours_b, threshold):
    x = contours_a
    y = contours_b
    print("len y : ", len(y))
    xx = np.array(x)
    hits = []
    for yrec in y:
        d = np.square(xx[:,0] - yrec[0]) + np.square(xx[:,1] - yrec[1])
        #print("D = ", d)
        hits.append(np.any(d < threshold*threshold))
    top_count = np.sum(hits)
    print("top_count : ", top_count)
    try:
        precision_recall = top_count / len(y)
    except ZeroDivisionError:
        precision_recall = 0
        print("exception occurred")
    return precision_recall, top_count, len(y)

def mp_load_step_views(f,n):
        temp_cad_views = []
        temp_cad_jobs = []

        temp_cad_mesh = []
        hh= helper.Helperhandler()
        reg_hl = PC_registration.PointCloudRegistration()
        points, j1, mesh1 = hh.LoadSTEP(f,n)
        #o3d.visualization.draw_geometries([NumpyToPCD(points), mesh1])
        cad_points = points
        view1 = copy.deepcopy(np.asarray(points))
        print(len(view1))
        #rotation around x by 90°
        rot = [[1,0,0],[0,0,1],[0,-1,0]] #rotation -90° asse x
        view2 =  reg_hl.Transformation_with_list(copy.deepcopy(points), [rot],[[0,0,0]])
        mesh2 = copy.deepcopy(mesh1).rotate(rot,center= [0,0,0])
        j2 = reg_hl.Transformation_with_list(copy.deepcopy(j1), [rot],[[0,0,0]])
        # o3d.visualization.draw_geometries([NumpyToPCD(view2), mesh2])
        print(len(view2))
            
        #rotation around y by 90°
        rot = [[0,0,-1],[0,1,0],[1,0,0]] #rotation -90° asse x
        view3 =  reg_hl.Transformation_with_list(copy.deepcopy(points), [rot],[[0,0,0]])
        mesh3 = copy.deepcopy(mesh1).rotate(rot,center= [0,0,0])
        j3 = reg_hl.Transformation_with_list(copy.deepcopy(j1), [rot],[[0,0,0]])
        print(len(view3))

        #o3d.visualization.draw_geometries([NumpyToPCD(view3), mesh3])

        # img3,_ = self.hh.PointCloudToImage(temp2)
        temp_cad_views.append([view1,view2,view3])
        temp_cad_jobs.append([j1,j2,j3])
        temp_points = points

        temp_cad_mesh.append([[np.asarray(mesh1.vertices),np.asarray(mesh1.triangles),np.asarray(mesh1.vertex_normals)],
                              [np.asarray(mesh2.vertices),np.asarray(mesh2.triangles),np.asarray(mesh2.vertex_normals)],
                              [np.asarray(mesh3.vertices),np.asarray(mesh3.triangles),np.asarray(mesh3.vertex_normals)]])
        #print(np.asarray(mesh1.triangle_normals).shape)
        #print(np.asarray(mesh1.vertex_normals).shape)

        #mesh = o3d.geometry.TriangleMesh()
        #mesh.vertices = mesh1.vertices
        #mesh.triangles = mesh1.triangles
        #o3d.visualization.draw_geometries([mesh])
        #print(len(temp_cad_views))
        #print(len(temp_cad_jobs))
        #print(len(temp_cad_mesh))
        #print(len( [temp_cad_views, temp_cad_jobs, temp_points, temp_cad_mesh]))
        
        return [temp_cad_views, temp_cad_jobs, temp_points, temp_cad_mesh]
            
def mp_grab_image(camera):
    image = camera.grab_image()
    return image

def mp_clusterization(temp):
            cam = vcd.Vimba_camera()
            model = cluster.DBSCAN(eps= 10, min_samples = 30)#, n_jobs= -1)
            model.fit_predict(temp)
            clusters = [[]for n in range(max(model.labels_)+1)]
            contours1 = [[]for n in range(max(model.labels_)+1)]
            for i in range(max(model.labels_)+1):
                bol = np.equal(model.labels_,np.ones((len(temp),))*i,)
                p = temp[bol]
                clusters[i].append(p)
                contours1[i].append(np.asarray([np.asarray(p[:,1]*cam.pxmm).astype(np.uint32),np.asarray(p[:,0]*cam.pxmm).astype(np.uint32)]).T)
            if len(clusters)>1:
                clusters = np.squeeze(clusters)
                contours1 = np.squeeze(np.asarray(contours1))
            det_objects = copy.deepcopy(clusters)
            if not len(clusters)>1:
                lens = len(det_objects[0][0])
                bol = [lens>1000]
            else:
                lens = np.asarray([len(det_objects[i]) for i in range(len(det_objects))])
                bol = lens>1000

            det_objects = list(np.asarray(det_objects)[bol])
            clusters = list(np.asarray(clusters)[bol])
            contours1 = list(np.asarray(contours1)[bol])
            return det_objects, contours1


#def main():
    # We need to initalize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.


    #if len(sys.argv) > 1:
    #    path = sys.argv[1]
    #    if os.path.exists(path):
    #        w.load(path)
    #    else:
    #        w.window.show_message_box("Error",
    #                                  "Could not open file '" + path + "'")

    # Run the event loop. This will not return until the last window is closed.


if __name__ == "__main__":
    #o3d.visualization.webrtc_server.enable_webrtc()
    mp.freeze_support()
    #mp.set_start_method('fork')
    gui.Application.instance.initialize()
    w = AppWindow(1500, 900)
    gui.Application.instance.run()


   # main()