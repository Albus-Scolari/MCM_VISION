#import FrankCamera
#import FrankLight
#import FrankUtilities
#import FrankStitcher
#import FrankCommons
#import FrankProfilometer

import abb
import threading
import socket
import time
import math

import struct

#import cv2
from pathlib import Path
#from pyntcloud import PyntCloud
#import pandas as pd
#import numpy as np
#import open3d as o3d
import serial
#import events

class newFrank():

    #myevents = events.Events(('CameraCalib_NewImage', 'WorkingArea_NewImage', 'NewRobotCoord', 'PointCloudProgress'))

    #Dichiarazione variabili ****************************************************
    robot = None                    #Variabile globale per accedere al Robot
    light = None                    #Variabile globale per accedere alla luce
    stitcher = None                 #Variable to access Stitcher
    #arduino = serial.Serial(port="COM6", baudrate=57600,timeout=.1)

    #robotIP = "127.0.0.1"       #IP del robot
    robotIP = "192.168.125.1"   #IP del robot
    toolLen = 0    #lunghezza del tool (Z tool coordinate)

    LoggerPort = 5001
    LoggerSampleTime = 0.01
    CoordinateLog = [[], [], []]
    CoordinateLogStarted = False
    CoordinateLogFileName = str(Path(__file__).resolve().parent)+"//Coords.txt"
    CoordinateLogSaveToFile = False

    Connesso = False #Non cambiare
    #exitFlag = False #Non cambiare

    ConnessoLuce = False #Non cambiare
    lightIP = "192.168.125.2"   #IP della luce
    lightIntensity = 12
    lightChannel = 1
    lightPort = 40001

    ConnessoCamera = False

    ConnessoProfilometer = False
    frequency = 0.02
    duty_cycle = 50
    #Dichiarazione variabili ****************************************************

    # Da finire
    def __init__(self, IProbot = "192.168.125.1"):
        #self.robotIP = IProbot
        #self.lightIP = IPlight
        pass
    def LogCoord(self):
        
        fCoord = None

        while self.Connesso:
            data = self.logsock.recv(300)
            N = round(len(data) / 30)
            nums = []
            coords = []
            for i in range(0, N):
                id = i * 30
                intVal = struct.unpack('h',data[id:id+2])[0]
                id = id + 2
                floatVals = struct.unpack('fffffff',data[id:4*7+id])
                nums.append(intVal)
                coords.append(floatVals)
                pass

            for i in range(0, len(nums)):
                if nums[i] != -1:
                    if not self.CoordinateLogStarted:
                        self.CoordinateLog = [[], [], []]
                        self.CoordinateLogStarted = True
                    if self.CoordinateLogSaveToFile:
                        if fCoord == None:
                            fCoord = open(self.CoordinateLogFileName, "w") #TO BE REMOVED
                            fCoord.write("Number\tX [mm]\tY [mm]\tZ [mm]\tqw\tqx\tqy\tqz\n") #TO BE REMOVED

                        fCoord.write(str(nums[i]) + "\t" + str(coords[i][0]) + "\t" + str(coords[i][1]) + "\t"
                                      + str(coords[i][2]) + "\t" + str(coords[i][3]) + "\t" + str(coords[i][4]) + "\t"
                                       + str(coords[i][5]) + "\t" + str(coords[i][6]) + "\n") #TO BE REMOVED
                    self.CoordinateLog[0].append(nums[i])
                    self.CoordinateLog[1].append([coords[i][0], coords[i][1], coords[i][2]])
                    self.CoordinateLog[2].append([coords[i][3], coords[i][4], coords[i][5], coords[i][6]])
                else:
                    if not fCoord is None:
                        fCoord.close
                        fCoord = None
                        pass
                    self.CoordinateLogStarted = False


            #self.myevents.NewRobotCoord(nums, coords)

        self.logsock.shutdown(socket.SHUT_RDWR)

    def Connect(self):
        '''
        Connect to robot and light.
        Returns 0 when robot and light are connected
        Returns -1 if robot is not connected
        Returns -2 if light is not connected
        Returns -4 if camera is not connected
        Returns -8 if profilomer is not connected
        '''
        if self.Connesso: return 0

        try:
            self.robot = abb.Robot(ip = self.robotIP)
           # self.robot.set_tool([[0, 0, 0], [1, 0, 0, 0]]) #La lunghezza del tool � fino al sensore della camera
            #self.SetToolLength(self.toolLen)
            print("Robot connesso")
            self.Connesso = True
            #self.robot.start_stop_profile(0, self.LoggerSampleTime) #New Motion server required
        
            pass
        except :
            print("Errore nella connessione al robot, indirizzo IP: " + self.robotIP)
            self.Connesso = False
            pass


        if self.Connesso:
            try:
                self.logsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.logsock.settimeout(5)
                self.logsock.connect((self.robotIP, self.LoggerPort))
                #t3 = threading.Thread(target = self.LogCoord)
                #t3.start()
                print("connected to Logger")       
                pass
            except :
                print("Logger not connected")
                pass
           
            return 1
        else:
            return -1

    def Disconnect(self):
        "Si disconnette dal robot con l'IP impostato nella variabile 'robotIP'"
        if self.Connesso:
            self.robot.close()
            #if not self.useLogger: self.robotLogger.close() #not working anymore (RAPID has changed)
            self.Connesso = False
       

        return 0

    def SetToolLength(self, NewLen : float):
        if not self.Connesso:
            return -1

        self.robot.set_tool([[0, 0, NewLen], [1, 0, 0, 0]]) #La lunghezza del tool � fino al sensore della camera
        self.toolLen = NewLen

        print("Tool length changed to " + str(NewLen))

        return 0

    def SetSingularityInterpolation(self, value):
        '''
        value
            0 -> Off
            1 -> LockAxis4
            2 -> Wrist
        '''
        if not self.Connesso:
            return -1
        if not (value == 0 or value == 1 or value == 2):
            return -2

        self.robot.change_singularity_interp(value)
        print("Singularity changed to " + str(value))

        return 0

    def SetConfigurationLinearMonitoring(self, value):
        '''
        value
            0 -> Off
            1 -> On
        '''
        if not self.Connesso:
            return -1
        if not (value == 0 or value == 1):
            return -2

        self.robot.change_singularity_interp(value)
        print("ConfL changed to " + str(value))

        return 0

    

    def SetSpeed(self, linear: int, orientation: int, external_linear: int, external_orientation: int):
        '''
        speed: [robot TCP linear speed (mm/s), TCP orientation speed (deg/s),
                external axis linear, external axis orientation]
        '''
        if not self.Connesso: return -1
        self.robot.set_speed([linear, orientation, external_linear, external_orientation])

        print("Speed changed to " + str(linear) + " (linear) , " + str(orientation) + " (orientation) , " 
              + str(external_linear) + " (external_linear) , " + str(external_orientation) + " (external_orientation)")

        return 0

    def GetSpeed(self):
        '''
        speed: [robot TCP linear speed (mm/s), TCP orientation speed (deg/s),
                external axis linear, external axis orientation]
        '''
        if not self.Connesso: return -1

        return self.robot.get_speed()

    def GoToJointsCoord(self, J1, J2, J3, J4, J5, J6):
        '''
        Joints coordinate are in degree
        '''
        if not self.Connesso:
            return -1
        self.robot.set_joints([J1, J2, J3, J4, J5, J6])


    def GoToCartesianEuler(self, newPosition, newEZ, newEY, newEX):
        '''
        Cartesian coordinate are in mm, angles in degrees
        '''
        if not self.Connesso:
            return -1
        EZ = math.radians(newEZ)
        EY = math.radians(newEY)
        EX = math.radians(newEX)
        quat = FrankUtilities.Euler2Quaternion([EZ,EY,EX])
        if quat == -1: return -1
        self.robot.set_cartesian([newPosition, quat])

    def GoToCartesianRT(self, newPosition, RTMatrix):
        '''
        Cartesian coordinate are in mm
        '''
        if not self.Connesso:
            return -1
        quat = FrankUtilities.RotMatrix2Quaternion(RTMatrix)
        if quat == -1: return -1
        self.robot.set_cartesian([newPosition,quat])

    def StartMark(self):
        self.robot.start_mark()
        return 1
    def StartLaser(self):
        self.robot.start_laser()
        #self.arduino.write(bytes(int(self.duty_cycle/100*255),"utf-8"))
        return 1
    def StopLaser(self):
        self.robot.stop_laser()
        #self.arduino.write(bytes(int(0),"utf-8"))
        return 1
    def turnOffLaser(self):
        self.robot.turn_off_laser()
        return 1
    def turnOnLaser(self):
        self.robot.turn_on_laser()
        return 1
    def setDutyCycle(self,value):
        if value<100and value >0:
            self.duty_cycle=value
            self.arduino.write(bytes(int(self.duty_cycle/100*255),"utf-8"))
        else:
            print("value not good")
        return 1