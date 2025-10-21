import socket
import threading
import struct
import time
import queue
import numpy as np
import csv
from scipy.spatial.transform import Rotation as R

def global_to_local_velocity(global_velocity, rotation_quaternion):
    rotation_quaternion_xyzw = [rotation_quaternion[1], rotation_quaternion[2], 
                                rotation_quaternion[3], rotation_quaternion[0]]
    r = R.from_quat(rotation_quaternion_xyzw)
    rotation_matrix = r.as_matrix()
    return rotation_matrix.T @ np.array(global_velocity)

def bin2int(int_bin):
    return struct.unpack('>I', int_bin)[0]

def nparray2bin(a):
    return a.tobytes()

def bin2nparray(b):
    return np.frombuffer(b, dtype=float).reshape(16)

def int2bin(integer):
    return struct.pack(">I", integer)

class TcpServer:
    def __init__(self, host, port):
        self.host_ = host
        self.port_ = port
        self.sock_ = None
        self.q_ = queue.Queue()
        self.quit_event_ = threading.Event()

    def launch(self):
        self.sock_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_.bind((self.host_, self.port_))
        self.sock_.listen(1)
        self.quit_event_.clear()
        self.start_server_()

    def stop(self):
        self.quit_event_.set()
        if self.sock_:
            try:
                self.sock_.shutdown(socket.SHUT_RDWR)
                self.sock_.close()
            except:
                pass
            self.sock_ = None

    def get(self):
        try:
            return self.q_.get(timeout=0.1)
        except queue.Empty:
            return None

    def start_server_(self):
        while not self.quit_event_.is_set():
            try:
                conn, addr = self.sock_.accept()
                threading.Thread(target=self.handle_connection_, args=(conn, addr)).start()
            except Exception as e:
                if not self.quit_event_.is_set():
                    print(f"Accept error: {e}")

    def handle_connection_(self, conn, addr):
        conn.settimeout(1)
        while not self.quit_event_.is_set():
            try:
                pack_size = conn.recv(4)
                if not pack_size:
                    break
                pack_size = bin2int(pack_size)
                data = self.recv_all_(conn, pack_size)
                self.q_.put(data)
            except (socket.timeout, ConnectionResetError):
                continue
            except:
                break
        conn.close()

    def recv_all_(self, sock, msg_length):
        data = b""
        while len(data) < msg_length and not self.quit_event_.is_set():
            try:
                recv_data = sock.recv(msg_length - len(data))
                if not recv_data:
                    break
                data += recv_data
            except:
                break
        return data

class Vicon:
    def __init__(self):
        self.server = TcpServer("0.0.0.0", 8801)
        self.position = np.zeros(3)
        self.rotation = np.zeros(4)
        self.rpy = np.zeros(3)
        self.velocity = np.zeros(3)
        self.velocity2 = np.zeros(3)
        self.rotation_rate =  np.zeros(3)
        self.quit_event = threading.Event()
        self.server_thread = threading.Thread(target=self.server.launch)
        self.data_thread = None
        self.server_thread.start()
        self.start_data_thread()

    def start_data_thread(self):
        self.data_thread = threading.Thread(target=self.process_data)
        self.data_thread.start()

    def process_data(self):
        while not self.quit_event.is_set():
            data = self.server.get()
            if data is None:
                continue
            data = bin2nparray(data)
            self.update_states(data)

    def update_states(self, data):
        self.position = data[0:3]
        self.velocity = global_to_local_velocity(data[3:6], data[9:13])
        self.velocity2 = data[3:6]

        self.rpy = data[6:9]
        self.rotation = data[9:13]
        self.rotation_rate = data[13:16]
    def stop(self):
        self.quit_event.set()
        self.data_thread.join()
        self.server.stop()
        self.server_thread.join()

if __name__ == "__main__":
    vicon = Vicon()
    input("Press Enter to start recording...")
    
    data = []
    start_time = time.time()

    while time.time() - start_time < 25:  # 记录5秒数据
        position = vicon.position
        data.append([
            time.time(), 
            *vicon.position, 
            *vicon.velocity, 
            *vicon.rpy, 
            *vicon.rotation
        ])
        time.sleep(0.01)
        print("vicon.position", vicon.position)

        if  position[2]==0.0:
            break
    
    with open('vicon_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time','PosX','PosY','PosZ','VelX','VelY','VelZ',
                        'Roll','Pitch','Yaw','QW','QX','QY','QZ'])
        writer.writerows(data)
    
    vicon.stop()
    print("Data saved and server stopped.")