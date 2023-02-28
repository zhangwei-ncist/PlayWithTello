import logging
import sys
import threading
import socket
import time
from queue import Queue

import cv2
import keyboard
import numpy as np
from mediapipe.framework.formats import landmark_pb2

from PoseRecognizer import PoseRecognizer


class TelloController:
    """直接使用Tello SDK控制Tello"""

    # def command_recv(self):#控制命令不用线程来做，采用阻塞式
    #     count = 0
    #     while True:
    #         try:
    #             data, server = self.sock_command.recvfrom(1518)
    #             print(data.decode(encoding="utf-8"))
    #         except Exception as e:
    #             self.logger.error(e)
    #             break

    TELLO_CAMERA_V_DEGREE=44.4#TELLO摄像头垂直角度
    def _state_recv(self):  # 状态线程方法
        while True:
            if self.flag_myself_running:
                try:
                    data, server = self.sock_state.recvfrom(1618)
                    self.tello_state_str = data.decode(encoding="utf-8")
                    # print(self.tello_state_str)
                    # pitch:0;roll:0;yaw:0;vgx:0;vgy:0;vgz:0;templ:65;temph:67;tof:10;h:0;bat:100;baro:-99.69;time:0;agx:-2.00;agy:-2.00;agz:-1000.00;
                    self._parse_state()  # 解析state_str
                except Exception as e:
                    self.logger.error(e)
            else:
                break

    def _display_text(self, img, txt, left=10, bottom=30):  # 内部方法，在指定位置显示字符串
        cv2.putText(img, txt, org=(left, bottom), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0),
                    thickness=2)

    def _write_info(self, img):
        # bat = self._current_drone_sate.get("bat")
        key_info = ["bat", "h", "templ", "temph","tof"]
        if self.tello_state_dict:
            for n in range(len(key_info)):
                self._display_text(img, f"{key_info[n]}={self.tello_state_dict.get(key_info[n])}", 10,
                                   (n + 1) * 30)  # 在图像上写关键信息

        if not self.flag_video_on:  # 显示video信息
            s = "Video OFF!"
        else:
            s = "Video ON!"
        self._display_text(img, s, self.video_w - 200, self.video_h - 50)

    def _video_recv(self):
        address = "udp://@0.0.0.0:11111"
        self.cap: cv2.VideoCapture = cv2.VideoCapture(address)
        while True:
            if self.flag_myself_running:
                if self.flag_video_on:  # 视频流打开了
                    # print("videon=",self.flag_video_on)
                    ret, frame = self.cap.read()
                    if not ret:  # 没有成功获得图像
                        frame_plus = np.zeros((self.video_h, self.video_w, 3), np.uint8)  # 应该创建一个空的图像
                        # print("Can't receive frame ,sleep in 5s!")
                        # time.sleep(5)
                        # continue
                    else:
                        frame_plus = frame.copy()
                        if self.flag_pose_recognize:  # 加入身体识别命令
                            # 开一个线程，防止阻塞图像线程
                            if self.queue_frame_from_video.empty():  # 只有队列为空时才识别，也就是一次只识别一个动作，一个个处理
                                self.queue_frame_from_video.put(frame.copy())  # 深拷贝
                            pass
                else:
                    frame_plus = np.zeros((self.video_h, self.video_w, 3), np.uint8)  # 应该创建一个空的图像
                self._write_info(frame_plus)  ## 在图像上写关键信息
                cv2.imshow('tello_frame', frame_plus)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    self.flag_video_on = False
                    self.command_streamoff()  # 关闭视频流
                    continue  # 不停止
            else:
                break
        self.cap.release()

    def __init__(self):
        # 初始化日志记录器
        logger: logging = logging.getLogger("Tello Controller Log")
        logger.setLevel(logging.INFO)  # 设置默认的级别
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(fmt='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
                                          datefmt="%H:%M:%S"))
        logger.addHandler(ch)
        self.logger = logger

        # 定义一些变量
        self.flag_myself_running = True  # TC运行总开关
        self.flag_tello_connected = False
        self.flag_video_on: bool = False  # 是否正在接受无人机的视频
        self.flag_tello_in_the_air = False  # tello还在天上
        self.flag_pose_recognize = False  # 是否开始识别拍摄者姿势

        self.tello_state_str = None
        self.tello_state_dict = {}  # tello状态字典

        #定义一些特殊的变量
        self.searched_cw_degree=0#搜索人时的旋转角度

        # FIFO队列
        self.queue_frame_from_video: Queue = Queue(1)  # 存储需要识别的图像副本

        self.video_h = 720
        self.video_w = 960

        #
        self.keyboard_to_func_name: {str: (str, {})} = {
            'esc': (self.shutdown, {}),
            'c': (self.command_command, {}),
            'v': (self.command_streamon, {}),
            'o': (self.command_streamoff, {}),
            't': (self.command_takeoff, {}),
            'y': (self.command_land, {}),
            'w': (self.command_up, {}),
            's': (self.command_down, {}),
            'a': (self.command_ccw, {'x': 20}),
            'd': (self.command_cw, {'x': 20}),
            'left': (self.command_left, {'x': 20}),
            'right': (self.command_right, {'x': 20}),
            'up': (self.command_forward, {'x': 20}),
            'down': (self.command_back, {'x': 20}),
            'e': (self.command_emergency, {}),  # 紧急停止
            'i': (self.command_flip, {'x': 'f'}),  # 前翻
            'k': (self.command_flip, {'x': 'b'}),  # 后翻
            'j': (self.command_flip, {'x': 'l'}),  # 左翻
            'l': (self.command_flip, {'x': 'r'}),  # 右翻
            'p': (self.switch_pose_recognize, {})
        }  # 将键盘按键映射到方法调用

        # Create a Command UDP socket
        self.sock_command = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_command.settimeout(5)  # 设置5s超时
        self.address_tello_command = ('192.168.10.1', 8889)  # 发送命令到tello的8889端口，并从此接受状态信息
        self.address_local_command = ('', 9999)  #
        self.sock_command.bind(self.address_local_command)  # 在本地的9999端口

        # 定义state UDP socket
        self.sock_state = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_command.settimeout(2)  # 设置2s超时
        self.address_local_state = ('0.0.0.0', 8890)  #
        self.sock_state.bind(self.address_local_state)  # 在本地的8890端口

        # 定义三个线程，分别用于：状态、视频
        self.thread_state: threading.Thread = threading.Thread(target=self._state_recv, daemon=True)  # 监听tello的状态，并更新
        self.thread_video: threading.Thread = threading.Thread(target=self._video_recv, daemon=True)  # 视频流接受线程
        self.thread_pose: threading.Thread = threading.Thread(target=self._pose_recognize, daemon=True)  # 姿态识别线程
        self.thread_keyboard: threading.Thread = threading.Thread(target=self._read_command_from_keyboard,
                                                                  daemon=True)  # 读取键盘按键线程

    def _pose_recognize(self):  # 姿态识别线程方法
        while True:
            if self.flag_myself_running:  # 系统在运行
                if self.flag_pose_recognize:  # 识别姿势开关
                    image = self.queue_frame_from_video.get()
                    # todo#识别姿势
                    with PoseRecognizer.mp_pose.Pose(
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as pose:
                        # To improve performance, optionally mark the image as not writeable to
                        # pass by reference.
                        image.flags.writeable = False
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = pose.process(image)
                    #test
                    pose_landmarks: landmark_pb2.NormalizedLandmarkList = results.pose_landmarks

                    if not pose_landmarks:
                        # 没有找到人，按照顺时针旋转10°
                        if self.searched_cw_degree == 360:
                            # 已经转过一圈了，还没找到，放弃姿态识别
                            self.logger.warning("Search a round,but have not found people! Stop Pose Recognize!")
                            self.flag_pose_recognize = False
                            continue
                        else:
                            # 进入顺时针搜索
                            self.searched_cw_degree += 10
                            self.command_cw(10)
                            continue
                    else:
                        # 找到了目标或者身体的某一部分
                        # todo

                        pass

                    if pose_landmarks:  # 判断一下是否得到了pose
                        # print(pose_landmarks.landmark[0])  #
                        print(PoseRecognizer.isCrossedWrists(pose_landmarks))#支持什么？
                    # Draw the pose annotation on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    PoseRecognizer.mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        PoseRecognizer.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=PoseRecognizer.mp_drawing_styles.get_default_pose_landmarks_style())
                    # Flip the image horizontally for a selfie-view display.
                    # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
                    cv2.imshow('MediaPipePose', image)
                    if cv2.waitKey(3) & 0xFF == 27:
                        continue
                    #test
                    self.queue_frame_from_video.task_done()
                    pass
                else:
                    image=np.zeros((self.video_h, self.video_w, 3), np.uint8)  # 应该创建一个空的图像
                    cv2.imshow('MediaPipePose', image)
                    if cv2.waitKey(1) & 0xFF == 27:
                        continue
                    pass

            else:  # 结束线程方法
                cv2.destroyWindow('MediaPipePose')  # 关闭窗口
                break

    def _has_needed_marks(self,pose_landmarks: landmark_pb2.NormalizedLandmarkList):#拿到识别结果了，怎么处理
        """判断是否包含了必要的节点"""
        #todo
        pass
    def _read_command_from_keyboard(self):  # 循环读取键盘指令的线程
        while True:
            if self.flag_myself_running:
                # print(self.keyboard_to_func_name)
                event: keyboard.KeyboardEvent = keyboard.read_event()
                if event.event_type == keyboard.KEY_DOWN:  # 确定是press事件，则读入字符。这样一直按住就会发送多条指令！
                    # if event.name == "esc":
                    #     self.shutdown()
                    print(event.name)
                    item = self.keyboard_to_func_name.get(event.name)  # 从按键字符找到函数名和参数
                    if item:
                        func_name, params = item
                        x = func_name(**params)  # 双星号（**）将参数以字典的形式导入:
            else:
                break

    def startup(self):  # 启动Controller
        while True:
            result = tc.command_command()
            self.logger.info("connect=" + str(result))
            if result:
                tc.flag_tello_connected = True
                break
            else:
                time.sleep(2)  # 2s 后重试
                continue

        tc.thread_state.start()  # 启动状态监听线程
        tc.thread_keyboard.start()  # 启动键盘监听
        tc.thread_video.start()  # 启动视频监控
        tc.thread_pose.start()  # 启动姿态识别线程
        self.logger.info("TelloController is ready!")
        pass

    def shutdown(self):
        # 清理系统的工作
        if self.flag_video_on:
            self.command_streamoff()
        if self.flag_tello_in_the_air:
            self.command_land()
        self.flag_myself_running = False

        self.logger.info("App is shutdown!")

    def _send_controll_or_set_command(self, command: str):
        """发送控制命令或设置参数，等待tello返回ok，否则函数返回False"""
        self.logger.info(f"向Tello发送控制命令或设置参数：{command}")
        msg = command.encode(encoding="utf-8")
        sent = self.sock_command.sendto(msg, self.address_tello_command)  # 初始化时完成飞机的command命令
        try:
            data, server = self.sock_command.recvfrom(1518)
            result = data.decode(encoding="utf-8").lower()
            result = result.replace('\r', '').replace('\n', '')
            self.logger.info(command + "-->" + result)
            if result == "ok":
                return True
            else:
                return False
        except Exception as e:
            self.logger.error(e)
            return False
        pass

    def _send_read_command(self, command):
        """发送读取命令，等待tello返回具体信息的值，出错了返回None"""
        self.logger.info(f"向Tello发送读取命令：{command}")
        msg = command.encode(encoding="utf-8")
        sent = self.sock_command.sendto(msg, self.address_tello_command)  # 初始化时完成飞机的command命令
        try:
            data, server = self.sock_command.recvfrom(1518)
            result = data.decode(encoding="utf-8").lower()
            result = result.replace('\r', '').replace('\n', '')
            return result
        except Exception as e:
            self.logger.error(e)
            return None
        pass

    def _parse_func_to_command(self, fun_name: [str]):  # 分析函数名，并发送控制指令
        if fun_name[0] == "command":  # 函数是以 command开头
            if len(fun_name) == 3:  # 说明带参数
                command = str(fun_name[1]) + ' ' + str(fun_name[2])  # 字符串连接
            else:
                command = fun_name[1]  # 获得_后的面的命令

            if self._send_controll_or_set_command(command):
                self.logger.info(f"Success!{fun_name[0]}={fun_name[1:]}")
                return True
            else:
                self.logger.error(f"Error!{fun_name[0]}={fun_name[1:]}")
                return False
        else:
            return False

    def switch_pose_recognize(self):  # 切换是否开启姿势识别
        self.flag_pose_recognize = not self.flag_pose_recognize
        self.logger.info("Pose Recognize=" + str(self.flag_pose_recognize))
        pass

    def command_command(self):
        """c"""
        # 向Tello发送command，启动sdk模式
        fun_name = sys._getframe().f_code.co_name.split('_')  # 获得当前函数名，并用_拆分成字符串数组  [0]=command [1]=command
        return self._parse_func_to_command(fun_name)

    def command_takeoff(self):
        """t"""
        # 向Tell发送takeoff命令
        fun_name = sys._getframe().f_code.co_name.split('_')  # 获得当前函数名，并用_拆分,[0]=command [1]=takeoff
        result = self._parse_func_to_command(fun_name)
        if result:
            self.flag_tello_in_the_air = True
        return result

    def command_land(self):
        """t"""
        # 向Tell发送takeoff命令
        fun_name = sys._getframe().f_code.co_name.split('_')  # 获得当前函数名，并用_拆分,[0]=command [1]=land
        result = self._parse_func_to_command(fun_name)
        if result:
            self.flag_tello_in_the_air = False
        return result

    def command_streamon(self):
        """v"""
        # 向Tell发送streamon命令
        fun_name = sys._getframe().f_code.co_name.split('_')  # 获得当前函数名，并用_拆分,[0]=command
        result = self._parse_func_to_command(fun_name)
        if result:
            self.flag_video_on = True
        else:
            self.flag_video_on = False
        return result

    def command_streamoff(self):
        """b"""
        # 向Tell发送streamoff命令
        fun_name = sys._getframe().f_code.co_name.split('_')  # 获得当前函数名，并用_拆分,[0]=command
        result = self._parse_func_to_command(fun_name)
        if result:
            self.flag_video_on = False
        else:
            self.flag_video_on = True
        return result

    def command_emergency(self):
        """e"""
        # 向Tell发送emergency命令
        fun_name = sys._getframe().f_code.co_name.split('_')  # 获得当前函数名，并用_拆分,[0]=command
        return self._parse_func_to_command(fun_name)

    def command_up(self, x=20):
        """w"""
        # 向Tell发送up x命令,x单位为厘米
        if 500 >= x >= 20:
            fun_name = sys._getframe().f_code.co_name.split('_')  # 获得当前函数名，并用_拆分,[0]=command [1]=up
            fun_name += [x]  # 传入参数
            return self._parse_func_to_command(fun_name)
        else:
            self.logger.error("x should be in [20,500] cm!")

    def command_down(self, x=20):
        """s"""
        # 向Tell发送down x命令,x单位为厘米
        if 500 >= x >= 20:
            fun_name = sys._getframe().f_code.co_name.split('_')  # 获得当前函数名，并用_拆分,[0]=command [1]=up
            fun_name += [x]  # 传入参数
            return self._parse_func_to_command(fun_name)
        else:
            self.logger.error("x should be in [20,500] cm!")

    def command_left(self, x=20):
        """left"""
        # 向Tell发送left x命令
        if 500 >= x >= 20:
            fun_name = sys._getframe().f_code.co_name.split('_')  # 获得当前函数名，并用_拆分,[0]=command [1]=up
            fun_name += [x]  # 传入参数
            return self._parse_func_to_command(fun_name)
        else:
            self.logger.error("x should be in [20,500] cm!")

    def command_right(self, x=20):
        """right"""
        # 向Tell发送right x命令
        if 500 >= x >= 20:
            fun_name = sys._getframe().f_code.co_name.split('_')  # 获得当前函数名，并用_拆分,[0]=command [1]=up
            fun_name += [x]  # 传入参数
            return self._parse_func_to_command(fun_name)
        else:
            self.logger.error("x should be in [20,500] cm!")

    def command_forward(self, x=20):
        """up"""
        # 向Tell发送forward x命令
        if 500 >= x >= 20:
            fun_name = sys._getframe().f_code.co_name.split('_')  # 获得当前函数名，并用_拆分,[0]=command [1]=up
            fun_name += [x]  # 传入参数
            return self._parse_func_to_command(fun_name)
        else:
            self.logger.error("x should be in [20,500] cm!")

    def command_back(self, x=20):
        """down"""
        # 向Tell发送back x命令
        if 500 >= x >= 20:
            fun_name = sys._getframe().f_code.co_name.split('_')  # 获得当前函数名，并用_拆分,[0]=command [1]=up
            fun_name += [x]  # 传入参数
            return self._parse_func_to_command(fun_name)
        else:
            self.logger.error("x should be in [20,500] cm!")

    def command_cw(self, x=10):
        """d"""
        # 向Tell发送cw x命令
        if 3600 >= x >= 1:
            fun_name = sys._getframe().f_code.co_name.split('_')  # 获得当前函数名，并用_拆分,[0]=command [1]=up
            fun_name += [x]  # 传入参数
            return self._parse_func_to_command(fun_name)
        else:
            self.logger.error("x should be in [1,3600] Degree!")

    def command_ccw(self, x=10):
        """a"""
        # 向Tell发送ccw x命令
        if 3600 >= x >= 1:
            fun_name = sys._getframe().f_code.co_name.split('_')  # 获得当前函数名，并用_拆分,[0]=command [1]=up
            fun_name += [x]  # 传入参数
            return self._parse_func_to_command(fun_name)
        else:
            self.logger.error("x should be in [1,3600] Degree!")

    def command_flip(self, x='f'):

        # 向Tell发送flip x命令
        if x in ['l', 'r', 'f', 'b']:
            fun_name = sys._getframe().f_code.co_name.split('_')  # 获得当前函数名，并用_拆分,[0]=command [1]=up
            fun_name += [x]  # 传入参数
            return self._parse_func_to_command(fun_name)
        else:
            self.logger.error("x should be in [l,r,f,b] 4 deriection!")

    def command_go(self, x, y, z, speed):
        # 以speed飞向坐标x,y,z
        if 500 >= x >= 20 and 500 >= y >= 20 and 500 >= z >= 20 and 100 >= speed >= 10:
            command = 'go ' + str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(speed)
            if self._send_controll_or_set_command(command):
                self.logger.info(f"Success!{command}")
                return True
            else:
                self.logger.error(f"Error!{command}")
                return False
        else:
            self.logger.error("go:x y z should be in [20,500] cm!,speed in [10,100] cm/s!")
            return False

    def command_curve(self, x1, y1, z1, x2, y2, z2, speed):
        # 以speed飞过弧线，弧线由x1y1z1,x2y2z2定义
        if 500 >= x1 >= -500 and 500 >= y1 >= -500 and 500 >= z1 >= -500 and 500 >= x2 >= -500 and 500 >= y2 >= -500 and 500 >= z2 >= -500 and 60 >= speed >= 10:
            if -20 <= x1 <= 20 and -20 <= y1 <= 20 and -20 <= z1 <= 20:  # 不可执行
                self.logger.error("Curve: x1 y1 z1 should not be in [-20,20] at the same time!")
                return False
            if -20 <= x2 <= 20 and -20 <= y2 <= 20 and -20 <= z2 <= 20:  # 不可执行
                self.logger.error("Curve: x2 y2 z2 should not be in [-20,20] at the same time!")
                return False
            command = 'curve ' + str(x1) + ' ' + str(y1) + ' ' + str(z1) + ' ' + str(x2) + ' ' + str(y2) + ' ' + str(
                z2) + ' ' + str(speed)
            if self._send_controll_or_set_command(command):
                self.logger.info(f"Success!{command}")
                return True
            else:
                self.logger.error(f"Error!{command}")
                return False
            pass
        else:
            self.logger.error("Curve: x y z should be in [-500,500] cm!,speed in [10,60] cm/s!")
            pass

    def command_set_speed(self, x):
        # 设置当前速度为x cm/s [10,100]
        if 100 >= x >= 10:
            command = 'speed ' + str(x)
            if self._send_controll_or_set_command(command):
                self.logger.info(f"Success!{command}")
                return True
            else:
                self.logger.error(f"Error!{command}")
                return False
        else:
            self.logger.error("speed:x should be in [10,100] cm/s!")
            return False
        pass

    def command_set_rc(self, roll, pitch, throttle, yaw):
        # 设置遥控器的4个通道量
        if 100 >= roll >= -100 and 100 >= pitch >= -100 and 100 >= throttle >= -100 and 100 >= yaw >= -100:
            command = 'rc ' + str(roll) + ' ' + str(pitch) + ' ' + str(throttle) + ' ' + str(yaw)
            if self._send_controll_or_set_command(command):
                self.logger.info(f"Success!{command}")
                return True
            else:
                self.logger.error(f"Error!{command}")
                return False
        else:
            self.logger.error("rc:roll,pitch,throttle,yaw should be in [-100,100]!")
            return False

    def command_set_wifi(self, ssid, password):
        # 设置wifi的ssid和密码
        command = 'wifi ' + str(ssid) + ' ' + str(password)
        if self._send_controll_or_set_command(command):
            self.logger.info(f"Success!{command}")
            return True
        else:
            self.logger.error(f"Error!{command}")
            return False

    # query_**方法利用 **? 命令来查询参数值
    def _parse_func_to_query(self, func_name) -> str:
        result = (self._send_read_command(func_name[1] + "?"))  # 构建命令
        result = result.replace('\r', '').replace('\n', '').replace(' ', '')  # 清除多余字符
        if result == 'ok':
            return None
        else:
            return result  # 原样返回

    def query_speed(self) -> float:
        """向Tello发送speed?，等待返回具体速度数字  10-100 cm/s"""
        # result = (self._send_read_command("speed?"))
        # result = result.replace('\r', '').replace('\n', '').replace(' ', '')
        fun_name = sys._getframe().f_code.co_name.split('_')  # 获得函数名
        result = self._parse_func_to_query(fun_name)
        if result:
            x = float(result)
            if 100 >= x >= 10:
                return x
            else:
                return None
        else:
            return None

    def query_wifi(self) -> float:
        """向Tello发送wifi?，等待返回具体snr """
        fun_name = sys._getframe().f_code.co_name.split('_')  # 获得函数名
        result = self._parse_func_to_query(fun_name)
        if result:
            return float(result)
        else:
            return None

    def query_battery(self) -> float:
        """向Tello发送battery?，等待返回具体数字  0-100 %"""
        fun_name = sys._getframe().f_code.co_name.split('_')  # 获得函数名
        result = self._parse_func_to_query(fun_name)
        if result:
            x = float(result)
            if 100 >= x >= 0:
                return x
            else:
                return None
        else:
            return None

    def query_time(self) -> str:
        """向Tello发送time?，等待返回具体数字 ，电机运转时间s"""
        fun_name = sys._getframe().f_code.co_name.split('_')  # 获得函数名
        result = self._parse_func_to_query(fun_name)
        if result:
            return result
        else:
            return None

    def query_height(self) -> str:
        """向Tello发送height?，等待返回具体数字 ，相对高度dm"""
        fun_name = sys._getframe().f_code.co_name.split('_')  # 获得函数名
        result = self._parse_func_to_query(fun_name)
        if result:
            return result
        else:
            return None

    def query_temp(self) -> str:
        """向Tello发送temp?，等待返回具体数字 ，主板温度,92~95c"""
        fun_name = sys._getframe().f_code.co_name.split('_')  # 获得函数名
        result = self._parse_func_to_query(fun_name)
        if result:
            return result
        else:
            return None

    def query_attitude(self) -> (float, float, float):
        """向Tello发送attitude?，等待返回具体数字 ，获取IMU 三轴姿态数据,pitch roll yaw
pitch=（-89°- 89°）
roll=（-179°- 179°）
yaw=（-179°- 179°）"""
        fun_name = sys._getframe().f_code.co_name.split('_')  # 获得函数名
        result = self._parse_func_to_query(fun_name)
        if result:
            # pitch,roll,yaw=result
            # return float(pitch), float(roll), float(yaw)#返回一个元组3
            return result
        else:
            return None

    def query_baro(self) -> float:
        """向Tello发送baro?，等待返回具体数字 ，获取气压计高度(m)"""
        fun_name = sys._getframe().f_code.co_name.split('_')  # 获得函数名
        result = self._parse_func_to_query(fun_name)
        if result:
            return float(result)

        else:
            return None

    def query_acceleration(self) -> (float, float, float):
        """向Tello发送acceleration?，等待返回具体数字 ，获取获取IMU 三轴加速度数据(0.001g)"""
        fun_name = sys._getframe().f_code.co_name.split('_')  # 获得函数名
        result = self._parse_func_to_query(fun_name)
        if result:
            # x, y, z = result
            # return float(x), float(y), float(z)  # 返回一个元组
            return result

        else:
            return None

    def query_tof(self) -> str:
        """向Tello发送tof?，等待返回具体数字 ，获取ToF 的高度值(mm)"""
        fun_name = sys._getframe().f_code.co_name.split('_')  # 获得函数名
        result = self._parse_func_to_query(fun_name)
        if result:
            return result
        else:
            return None

    # query_state_**方法从tello的状态信息获取各种参数值
    def _parse_state(self):
        if self.tello_state_str:
            l = self.tello_state_str.split(';')
            del l[-1]  # 最后一个是空串，删除他
            for item in l:
                t = item.split(':')
                self.tello_state_dict[t[0]] = float(t[1])
        else:
            pass

    def query_state_pitch(self):  # 俯仰角度，度数
        fun_name = sys._getframe().f_code.co_name.split('_')  # 获得函数名
        command_index = 2
        return self.tello_state_dict.get(fun_name[command_index])

    def query_state_roll(self):  # 横滚角度，度数
        fun_name = sys._getframe().f_code.co_name.split('_')
        command_index = 2
        return self.tello_state_dict.get(fun_name[command_index])

    def query_state_yaw(self):  # 偏航偏航，度数
        fun_name = sys._getframe().f_code.co_name.split('_')
        command_index = 2
        return self.tello_state_dict.get(fun_name[command_index])

    def query_state_vgx(self):  # x 轴速度，
        fun_name = sys._getframe().f_code.co_name.split('_')
        command_index = 2
        return self.tello_state_dict.get(fun_name[command_index])

    def query_state_vgy(self):
        fun_name = sys._getframe().f_code.co_name.split('_')
        command_index = 2
        return self.tello_state_dict.get(fun_name[command_index])

    def query_state_vgz(self):
        fun_name = sys._getframe().f_code.co_name.split('_')
        command_index = 2
        return self.tello_state_dict.get(fun_name[command_index])

    def query_state_templ(self):  # 主板最低温度，摄氏度
        fun_name = sys._getframe().f_code.co_name.split('_')
        command_index = 2
        return self.tello_state_dict.get(fun_name[command_index])

    def query_state_temph(self):  # 主板最高温度，摄氏度
        fun_name = sys._getframe().f_code.co_name.split('_')
        command_index = 2
        return self.tello_state_dict.get(fun_name[command_index])

    def query_state_tof(self):  # 获取ToF 的高度值(cm)
        fun_name = sys._getframe().f_code.co_name.split('_')
        command_index = 2
        return self.tello_state_dict.get(fun_name[command_index])

    def query_state_h(self):  # 相对起飞点高度，厘米
        fun_name = sys._getframe().f_code.co_name.split('_')
        command_index = 2
        return self.tello_state_dict.get(fun_name[command_index])

    def query_state_bat(self):  # ：当前电量百分比，％
        fun_name = sys._getframe().f_code.co_name.split('_')
        command_index = 2
        return self.tello_state_dict.get(fun_name[command_index])

    def query_state_baro(self):  # 获取气压计高度(m)
        fun_name = sys._getframe().f_code.co_name.split('_')
        command_index = 2
        return self.tello_state_dict.get(fun_name[command_index])

    def query_state_time(self):  # 电机运转时间，秒
        fun_name = sys._getframe().f_code.co_name.split('_')
        command_index = 2
        return self.tello_state_dict.get(fun_name[command_index])

    def query_state_agx(self):  # x 轴加速度
        fun_name = sys._getframe().f_code.co_name.split('_')
        command_index = 2
        return self.tello_state_dict.get(fun_name[command_index])

    def query_state_agy(self):
        fun_name = sys._getframe().f_code.co_name.split('_')
        command_index = 2
        return self.tello_state_dict.get(fun_name[command_index])

    def query_state_agz(self):
        fun_name = sys._getframe().f_code.co_name.split('_')
        command_index = 2
        return self.tello_state_dict.get(fun_name[command_index])

    def testQueryFunc(self):
        for func in dir(TelloController):
            if func.startswith("query_"):  #
                # func就是函数名
                # getattr()获得真正的函数
                result = getattr(self, func)()  # 调用函数！！
                print(func, "=", result)


if __name__ == '__main__':
    tc: TelloController = TelloController()
    tc.startup()
    # tc.testQueryFunc()
    #主进程不能结束！
    while True:
        if tc.flag_myself_running:
            # print(tc.query_state_bat())
            # tc.logger.info("listeing tello state!")
            # tc.logger.info("capturing tello video!")
            # tc.logger.info("listeing key board!")
            # dc.logger.info("Waiting for instruction...Or Press ESC to Exit！")
            time.sleep(10)
        else:
            break
    tc.logger.info("Main Process Exit！")
