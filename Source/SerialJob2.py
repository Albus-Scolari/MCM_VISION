import socket

class SJ2():
    def __init__(self, timeout=20000):
        self.__soh = bytearray([1])
        self.__eot = bytearray([4])
        self.__ack = bytearray([6])
        self.__nak = bytearray([21])
        self.__stx = bytearray([2])
        self.__etx = bytearray([3])

        self.__bufsize = 1024

        self.__s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__s.settimeout(timeout)

        self.__connected = False

    def connect(self, ipaddress: str, port: int):
        try:
            self.__s.connect((ipaddress, port))
            self.__connected = True
            err = 0
            data = ''
        except socket.timeout:
            err = 1
            data = 'Timeout error'
        except Exception as e:
            err = 1
            data = str(e)
        return err, data

    def disconnect(self):
        try:
            self.__s.close()
            self.__connected = False
            err = 0
            data = ''
        except socket.timeout:
            err = 1
            data = 'Timeout error'
        except Exception as e:
            err = 1
            data = str(e)
        return err, data

    def isConnected(self):
        return self.__connected

    def __receive(self):
        buff = self.__s.recv(self.__bufsize)
        if buff[1] == self.__ack[0]:
            err = 0
        else:
            err = 1
        return err, buff[2:-1]
            
    def __sendAndReceive(self, message):
        try:
            self.__s.sendall(message)
            err, data = self.__receive()
        except socket.timeout:
            err = 1
            data = b'SendAndReceive timeout error'
        except Exception as e:
            err = 1
            data = str(e).encode('utf-8')
        return err, data
    
    def getFlyCadStatus(self):
        # <SOH> K <EOT>
        msg = self.__soh + b'K' + self.__eot
        err, data = self.__sendAndReceive(msg)
        return err, data.decode('utf-8')

    def getDynTextNumber(self):
        # <SOH> #D <EOT>
        msg = self.__soh + b'#D' + self.__eot
        err, data = self.__sendAndReceive(msg)
        return err, data.decode('utf-8')

    def setRunMode(self, XN=False):
        if XN:
            # <SOH> XN <EOT>
            msg = self.__soh + b'XN' + self.__eot
        else:
            # <SOH> XR <EOT>
            msg = self.__soh + b'XR' + self.__eot
        err, data = self.__sendAndReceive(msg)
        if err == 0:
            checksum = sum(msg)
            if str(checksum) != data.decode('utf-8'):
                err == 1
                data = b'Checksum error'
        return err, data.decode('utf-8')

    def startMark(self):
        # <SOH> S <EOT>
        msg = self.__soh + b'S' + self.__eot
        err, data = self.__sendAndReceive(msg)
        if err == 0:
            checksum = sum(msg)
            if str(checksum) != data.decode('utf-8'):
                err == 1
                data = b'Checksum error'
        if err == 0:
            err, data = self.__receive()
        return err, data.decode('utf-8')

    def halt(self):
        # <SOH> H <EOT>
        msg = self.__soh + b'HLT' + self.__eot
        err, data = self.__sendAndReceive(msg)
        if err == 0:
            checksum = sum(msg)
            if str(checksum) != data.decode('utf-8'):
                err == 1
                data = b'Checksum error'
        return err, data.decode('utf-8')

    def reset(self):
        # <SOH> RST <EOT>
        msg = self.__soh + b'RST' + self.__eot
        err, data = self.__sendAndReceive(msg)
        if err == 0:
            checksum = sum(msg)
            if str(checksum) != data.decode('utf-8'):
                err == 1
                data = b'Checksum error'
        return err, data.decode('utf-8')

    def runPreview(self):
        # <SOH> PRW <EOT>
        msg = self.__soh + b'PRW' + self.__eot
        err, data = self.__sendAndReceive(msg)
        if err == 0:
            checksum = sum(msg)
            if str(checksum) != data.decode('utf-8'):
                err == 1
                data = b'Checksum error'
        return err, data.decode('utf-8')

    def runFocusSearch(self):
        # <SOH> FCS <EOT>
        msg = self.__soh + b'FCS' + self.__eot
        err, data = self.__sendAndReceive(msg)
        if err == 0:
            checksum = sum(msg)
            if str(checksum) != data.decode('utf-8'):
                err == 1
                data = b'Checksum error'
        return err, data.decode('utf-8')

    def setDynamicText(self, ID, text, position, height, font):
        # <SOH><STX> Tnnsss <ETX><EOT>
        if 1 <= id <= 99:
            msg = self.__stx + (f'T{id:02}' + text).encode('utf-8') + self.__etx

        pass

    def loadJobFile(self, path):
        # <SOH><STX> Jsss <ETX><EOT>
        msg = self.__soh + self.__stx + b'J' + path.encode('utf-8') + self.__etx + self.__eot
        err, data = self.__sendAndReceive(msg)
        if err == 0:
            checksum = sum(msg)
            if str(checksum) != data.decode('utf-8'):
                err == 1
                data = b'Checksum error'
        return err, data.decode('utf-8')

    def getMotorStatus(self):
        # <SOH><STX> @10K <ETX><EOT>
        msg = self.__soh + self.__stx + b'@10K' + self.__etx + self.__eot
        err, data = self.__sendAndReceive(msg)
        return err, data.decode('utf-8')

    def getMotorPosition(self):
        # <SOH><STX> @13ssn <ETX><EOT>
        msg = self.__soh + self.__stx + b'@13000,100,200' + self.__etx + self.__eot
        err, data = self.__sendAndReceive(msg)
        return err, data.decode('utf-8')

    def homing(self):
        # <SOH><STX> @10H <ETX><EOT>
        msg = self.__soh + self.__stx + b'@10H' + self.__etx + self.__eot
        err, data = self.__sendAndReceive(msg)
        if err == 0:
            checksum = sum(msg)
            if str(checksum) != data.decode('utf-8'):
                err == 1
                data = b'Checksum error'
        return err, data.decode('utf-8')

    def moveHead(self, x, z, absolute=False):
        # <SOH><STX> @11ssn <ETX><EOT>
        if absolute:
            XX = 'A'
        else:
            XX = 'R'
        ss0 = '0' + XX + f'{x:.2f}'
        ss2 = '2' + XX + f'{z:.2f}'
        ss = '@11' + ss0 + ',' + ss2
        msg = self.__soh + self.__stx + ss.encode('utf-8') + self.__etx + self.__eot
        err, data = self.__sendAndReceive(msg)
        if err == 0:
            checksum = sum(msg)
            if str(checksum) != data.decode('utf-8'):
                err == 1
                data = b'Checksum error'
        return err, data.decode('utf-8')

        # Potrei renderla io bloccante guardando lo status se è di motori in movimento
        
    def setMarkParameters(self, reps=-1, vel=-1, freq=-1, pow=-1, wtype=-1, wfreq=-1, wampx=-1, wampy=-1, delayON=-1, delayOFF=-1, delaypos=-1, delayang=-1, boundang=-1, fpkt=-1, fpkhl=-1, fpkll=-1):
        # <SOH><STX> @16id,val <ETX><EOT>
        # Capire se posso concatenare le cose in unico STX oppure se devo farne uno per ogni parametro
        bytemsg = b''
        if reps > 0:
            bytemsg = bytemsg + self.__stx + f'@1601,{reps:.0f}'.encode('utf-8') + self.__etx
        if vel > 0:
            bytemsg = bytemsg + self.__stx + f'@1602,{vel:.0f}'.encode('utf-8') + self.__etx
        if freq > 0:
            bytemsg = bytemsg + self.__stx + f'@1603,{freq:.0f}'.encode('utf-8') + self.__etx
        if pow > 0:
            bytemsg = bytemsg + self.__stx + f'@1604,{pow:.0f}'.encode('utf-8') + self.__etx
        msg = self.__soh + bytemsg + self.__eot
        if len(bytemsg) > 0:
            err, data = self.__sendAndReceive(msg)
        else:
            err = 1
            data = b'Empty message'
        return err, data.decode('utf-8')

    def moveMarkField(self, x=0, y=0):
        # <SOH><STX> @01xx.xx,yy.yy <ETX><EOT>
        if -75 < x < 75 and -75 < y < 75:
            msg = self.__soh + self.__stx + f'@01{x:.2f},{y:.2f}'.encode('utf-8') + self.__etx + self.__eot
            err, data = self.__sendAndReceive(msg)
        else:
            err = 1
            data = b'One or more parameters are out of range'
        if err == 0:
            checksum = sum(msg)
            if str(checksum) != data.decode('utf-8'):
                err == 1
                data = b'Checksum error'
        return err, data.decode('utf-8')

    def scaleMarkField(self, x=0, y=0, p=0):
        # <SOH><STX> @02xx.x,yy.y,p <ETX><EOT>
        if 0.1 < x < 10 and 0.1 < y < 10 and 0 <= p <= 4:
            msg = self.__soh + self.__stx + f'@02{x:.1f},{y:.1f},{p:.0f}'.encode('utf-8') + self.__etx + self.__eot
            err, data = self.__sendAndReceive(msg)
        else:
            err = 1
            data = b'One or more parameters are out of range'
        if err == 0:
            checksum = sum(msg)
            if str(checksum) != data.decode('utf-8'):
                err == 1
                data = b'Checksum error'
        return err, data.decode('utf-8')

    def rotateMarkField(self, r=0, p=0):
        # <SOH><STX> @03rr.rr,p,G <ETX><EOT>
        # Setto sempre G=0 per applicare la rotazione a tutti i gruppi nel disegno
        if -180 < r < 180 and 0 <= p <= 4:
            msg = self.__soh + self.__stx + f'@03{r:.2f},{p:.0f},0'.encode('utf-8') + self.__etx + self.__eot
            err, data = self.__sendAndReceive(msg)
        else:
            err = 1
            data = b'One or more parameters are out of range'
        if err == 0:
            checksum = sum(msg)
            if str(checksum) != data.decode('utf-8'):
                err == 1
                data = b'Checksum error'
        return err, data.decode('utf-8')