import sys
import threading
import time
from queue import Queue

import cv2
import djitellopy as tello
import logging
import djitellopy as djitello
import keyboard
import numpy as np
from mediapipe.framework.formats import landmark_pb2

import DroneState
from Instruction import Instruction
from DroneState import DroneState
import mediapipe as mp

from PoseRecognizer import PoseRecognizer


class DroneController:

    def __init__(self):  # 初始化所有变量
        logger: logging = logging.getLogger("Drone Controller Log")
        logger.setLevel(logging.INFO)  # 设置默认的级别
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(fmt='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
                                          datefmt="%H:%M:%S"))
        logger.addHandler(ch)

        self.logger = logger
        self.drone = djitello.Tello()

        Instruction.setLogger(self.logger)  # 设置Instruction的logger

        self.support_instructions: {str, Instruction} = {}  # DC中支持的指令，{key=字符串:value=Instruction对象}
        self.keyboard_to_instruction_name: {str, str} = {}  # 将键盘按键映射到指令名称
        for func in dir(DroneController):
            if func.startswith("drone_"):  # 初始化控制无人机的指令集合
                # func就是函数名
                # getattr()获得真正的函数
                if func.startswith("drone_state_"):  # 初始化控制无人 机状 态指令集合
                    self.support_instructions[func] = Instruction(func, getattr(self, func),
                                                                  Instruction.StateIns)  # func就是指令名称
                    self.keyboard_to_instruction_name[getattr(self, func).__doc__] = func  # 获取函数的特殊注释，作为键盘的映射字符

                if func.startswith("drone_move_"):  # 初始化控制无人机 移动 指令集合
                    self.support_instructions[func] = Instruction(func, getattr(self, func),
                                                                  Instruction.MoveIns)  # func就是指令名称
                    self.keyboard_to_instruction_name[getattr(self, func).__doc__] = func  # 获取函数的特殊注释，作为键盘的映射字符

        # self.support_instructions = {
        #     "drone_connect": Instruction("drone_connect", self.drone_connect)
        # }
        # self.keyboard_to_instrucion_name={"c":"drone_connect"}

        self.dc_running = True  # 系统是否可用
        self.flag_drone_position = DroneState.OnGround  # 无人机是否起飞
        self.flag_video_on: bool = False  # 是否正在接受无人机的视频
        self.video_h = 720
        self.video_w = 960
        self.flag_drone_online: bool = False  # 无人机是否在线
        self.flag_listen_keyboard: bool = True  # 是否监听键盘
        self.flag_pose_recognize: bool = False  # 是否进行姿态识别
        self.drone_moving_lock = threading.Lock()  # 用于操纵无人机的动作锁，必须获得这个锁，才能执行具体的飞行动作

        # FIFO，用户的指令队列
        self.queue_user_instruction: Queue = Queue(2)  # 存储用户指令
        self.queue_frame_from_video: Queue = Queue(1)  # 存储需要识别的图像副本

        # 用户指令相关线程
        self.t_consumer: threading.Thread = threading.Thread(target=self._act_instruction,
                                                             daemon=True)  # 消费者线程，用于读取指令并执行
        self.t_producer_keyboard: threading.Thread = threading.Thread(target=self._read_instruction_from_keyboard,
                                                                      daemon=True)  # 生产者线程，从不同渠道获得指令并填充队列

        # 心跳线程
        self.t_heartbeat: threading.Thread = threading.Thread(target=self._heartbeat, daemon=True)

        # 视频线程
        self.t_video: threading.Thread = threading.Thread(target=self._show_video, daemon=True)  # 获取视频的线程，并显示窗口

        # 视频识别功能单开线程
        self.t_pose_recognize: threading.Thread = threading.Thread(target=self._pose_recognize, daemon=True)  # 姿势识别线程

        self._current_drone_sate: dict = None
        # {'pitch': 0, 'roll': 1, 'yaw': 0, 'vgx': 0, 'vgy': 0, 'vgz': 0,
        # 'templ': 75, 'temph': 77, 'tof': 10, 'h': 0, 'bat': 81, 'baro': -237.05,
        # 'time': 0, 'agx': -3.0, 'agy': -31.0, 'agz': -998.0}

        self.t_producer_keyboard.start()  # 启动指令队列相关的线程

        self.t_consumer.start()  # 启动指令队列相关的线程
        # 连接时自动self.t_heartbeat.start()#启动心跳

    def shutdown(self):
        self.dc_running = False  # 关闭系统运行标志

        if self.flag_drone_position != DroneState.OnGround:  # 飞机没在地面上，需要先降落
            self.logger.info("Landing First! The Drone will be landing after 5 seconds!")
            self.flag_drone_position = DroneState.OnGround
            self.drone_land(after_seconds=5)

        if self.flag_video_on:
            self.flag_video_on = False
            self.drone.streamoff()
        cv2.destroyAllWindows()
        self.logger.info("DroneController exit！")
        # 飞机状态

    def _heartbeat(self):  # 心跳线程的方法
        while True:
            time.sleep(1)  # 每隔1s
            if self.dc_running:  # 系统运行时才判断
                try:
                    self._current_drone_sate: dict = self.drone.get_current_state()
                    self.flag_drone_online = True
                except Exception as e:
                    self.logger.error(e)
                    self.flag_drone_online = False
            else:
                break

    def _read_instruction_from_keyboard(self):  # 循环读取键盘指令的线程
        while True:
            if self.dc_running:
                if self.flag_listen_keyboard:  # 启动了监听键盘的情况下
                    event: keyboard.KeyboardEvent = keyboard.read_event()
                    if event.event_type == keyboard.KEY_DOWN:  # 确定是press事件，则读入字符。这样一直按住就会发送多条指令！
                        if event.name == "esc":
                            self.shutdown()
                        instruction_name = self.keyboard_to_instruction_name.get(event.name)  # 从按键字符找到指令名
                        if instruction_name:
                            self.queue_user_instruction.put(
                                self.support_instructions.get(instruction_name))  # 找到指令，并存入队列
            else:
                self.logger.info("App is shutdown,Thread for reading keyboard is finished!")
                break  # 终止循环，线程方法也会退出

            # self.user_instruction.put(self.support_instructions.get("drone_connect"))

    def _act_instruction(self):  # 执行指令的线程方法
        # 循环获取队列并执行todo
        while True:
            if self.dc_running:
                ins: Instruction = self.queue_user_instruction.get()
                # print(ins)
                if ins.ins_type == Instruction.MoveIns:  # 动作类指令，需要移动无人机！
                    with self.drone_moving_lock:  # 获得移动无人机的锁才能进一步执行.自动枷锁和解锁
                        ins.act()  # 实际的动作
                else:  # 执行状态类指令
                    ins.act()  # 实际的动作
                self.queue_user_instruction.task_done()  # 完成一个
                self.logger.info(self._current_drone_sate)  # 记录一下无人机状态
            else:
                break  # 结束线程方法

    def _display_text(self, img, txt, left=10, bottom=30):  # 内部方法，在指定位置显示字符串
        cv2.putText(img, txt, org=(left, bottom), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0),
                    thickness=2)

    def _write_info(self, img):
        # bat = self._current_drone_sate.get("bat")
        key_info = ["bat", "h", "templ", "temph"]
        if self._current_drone_sate:
            for n in range(len(key_info)):
                self._display_text(img, f"{key_info[n]}={self._current_drone_sate.get(key_info[n])}", 10,
                                   (n + 1) * 30)  # 在图像上写关键信息

        if not self.flag_video_on:  # 显示video信息
            self._display_text(img, "Video Off!", self.video_w // 2 - 10, self.video_h // 2 + 10)

    def _pose_recognize(self):  # 姿态识别线程方法
        while True:
            if self.dc_running:  # 系统在运行
                if self.flag_pose_recognize:
                    image = self.queue_frame_from_video.get()
                    # todo#识别姿势
                    with PoseRecognizer.mp_pose.Pose(
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as pose:
                        # To improve performance, optionally mark the image as not writeable to
                        # pass by reference.
                        image.flags.writeable = False
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = pose.process(image)#test
                        pose_landmarks: landmark_pb2.NormalizedLandmarkList = results.pose_landmarks
                        if pose_landmarks:  # 判断一下是否得到了pose
                            # print(pose_landmarks.landmark[0])  #
                            print(PoseRecognizer.isCrossedWrists(pose_landmarks))
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
                        cv2.imshow('MediaPipe Pose',image)
                        if cv2.waitKey(5) & 0xFF == 27:
                            break
                    self.queue_frame_from_video.task_done()
                    pass
            else:  # 结束线程方法
                break

    def _show_video(self):  # 显示无人机图像的线程方法
        last_frame = None
        while True:
            if self.dc_running:  # 系统可用时
                if self.flag_video_on:  # and self.frame_read:  # 打开了视频开关且获得了frame读取对象
                    try:
                        frame = self.frame_read.frame
                        frame_plus=frame.copy()
                        if self.flag_pose_recognize:  # todo加入身体识别命令
                            # todo我觉得应该开一个线程，防止阻塞图像线程
                            if self.queue_frame_from_video.empty():  # 只有队列为空时才识别，也就是一次只识别一个动作，一个个处理
                                self.queue_frame_from_video.put(frame.copy())  # 深拷贝
                            pass

                    except Exception as e:
                        self.logger.error(e)
                        time.sleep(1)  # 遇到异常,暂停1s试试

                else:
                    frame_plus = np.zeros((self.video_w, self.video_w, 3), np.uint8)  # todo，应该创建一个空的图像
                    pass
                self._write_info(frame_plus)  ## 在图像上写关键信息
                cv2.imshow('drone_frame', frame_plus)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
            else:
                break

    # 往下就是支持的无人机的各种操作了.对于各种飞行动作必须首先获得无人机的动作锁！！！
    #
    # 函数名就是指令的索引名！！,函数的注释就是键盘的按键
    def drone_state_connect(self):  # drone_connect
        """c"""
        self.logger.info("Tello Connect....")
        self.drone.connect()
        self.logger.info("Tello Connected!")

        self.t_heartbeat.start()  # 连接后应该启动心跳线程！
        self.logger.info("Begin HeartBeating!")
        self.t_video.start()  # 视频线程启动

    def drone_state_video_switch(self):
        """v"""
        if self.flag_video_on:  # 是否需要打开或关闭视频流，切换动作
            self.drone.streamoff()
            self.frame_read = None

            pass
        else:
            self.drone.streamon()
            self.frame_read = self.drone.get_frame_read()  # 获得frame读取对象
            pass
        self.flag_video_on = not self.flag_video_on
        if not self.flag_video_on:
            self.flag_pose_recognize = False  # 如果关闭了video，也应该关闭姿势识别！
        self.logger.info(f"Video {'On' if self.flag_video_on else 'Off'}!")

    def drone_state_pose_recognize_switch(self):
        """p"""
        self.flag_pose_recognize = not self.flag_pose_recognize
        if self.flag_pose_recognize and not self.flag_video_on:  # 开启了姿势识别，但是video没开
            self.drone_state_video_switch()  # 需要打开video
        if self.flag_pose_recognize:
            self.t_pose_recognize.start()  # 启动线程
        self.logger.info(f"Pose Recognize {'On' if self.flag_pose_recognize else 'Off'}!")

    def drone_move_takeoff(self, after_seconds=0):
        """t"""
        if self.flag_drone_position != DroneState.OnGround:
            self.logger.error(f"Done is not on the ground! DroneState={self.flag_drone_position}")
        else:
            time.sleep(after_seconds)
            self.logger.info("taking off")
            # self.drone.takeoff()
            self.flag_drone_position = DroneState.InAir

    def drone_move_land(self, after_seconds=0):
        """l"""
        if self.flag_drone_position != DroneState.InAir:
            self.logger.error(f"Done is not in the Air! DroneState={self.flag_drone_position}")
        else:
            self.logger.info("Landing！")
            # self.drone.land()
            self.flag_drone_position = DroneState.OnGround

    def drone_move_up(self, distance=1):
        """w"""
        if self.flag_drone_position == DroneState.OnGround:
            self.logger.error(f"Done is on the ground! DroneState={self.flag_drone_position}")
        else:
            self.logger.info(f"Up! {distance}")
            # self.drone.move_up(x=distance)

    def drone_move_down(self, distance=1):
        """s"""
        if self.flag_drone_position == DroneState.OnGround:
            self.logger.error(f"Done is on the ground! DroneState={self.flag_drone_position}")
        else:
            self.logger.info(f"Down! {distance}")
            # self.drone.move_down(x=distance)

    def drone_move_left(self, distance=1):
        """left"""
        if self.flag_drone_position == DroneState.OnGround:
            self.logger.error(f"Done is on the ground! DroneState={self.flag_drone_position}")
        else:
            self.logger.info(f"Left! {distance}")
            # self.drone.move_left(x=distance)

    def drone_move_right(self, distance=1):
        """right"""
        if self.flag_drone_position == DroneState.OnGround:
            self.logger.error(f"Done is on the ground! DroneState={self.flag_drone_position}")
        else:
            self.logger.info(f"Right! {distance}")
            # self.drone.move_right(x=distance)

    def drone_move_forward(self, distance=1):
        """up"""
        if self.flag_drone_position == DroneState.OnGround:
            self.logger.error(f"Done is on the ground! DroneState={self.flag_drone_position}")
        else:
            self.logger.info(f"Forward! {distance}")
            # self.drone.move_forward(x=distance)

    def drone_move_backward(self, distance=1):
        """down"""
        if self.flag_drone_position == DroneState.OnGround:
            self.logger.error(f"Done is on the ground! DroneState={self.flag_drone_position}")
        else:
            self.logger.info(f"Backward! {distance}")
            # self.drone.move_back(x=distance)

    def drone_move_yawl(self, digree=1):
        """a"""
        if self.flag_drone_position == DroneState.OnGround:
            self.logger.error(f"Done is on the ground! DroneState={self.flag_drone_position}")
        else:
            self.logger.info(f"Yaw Left! {digree}")
            # self.drone.rotate_counter_clockwise(x=digree)

    def drone_move_yawr(self, digree=1):
        """d"""
        if self.flag_drone_position == DroneState.OnGround:
            self.logger.error(f"Done is on the ground! DroneState={self.flag_drone_position}")
        else:
            self.logger.info(f"Yaw Right! {digree}")
            # self.drone.rotate_clockwise(x=digree)

    def pose_action(self,poseName:str):
        # avaiablePoses={"crossedWrists":}
        """
        面对Drone：
        左大小臂水平=yawr，但不能让人物离开画面
        右大小臂水平=yawl，但不能让人物离开画面
        双手紧握，偏向身体左侧=right，但不能让人物离开画面
        双手紧握，偏向身体右侧=left，但不能让人物离开画面
        双手紧握，举过头顶=up，但不能让人物离开画面
        双手紧握，在胸部以下=down，但不能让人物离开画面
        左右大臂水平，左右小臂垂直向上=forward，但不能让人物离开画面
        左右大臂水平，左右小臂垂直向下=backward，但不能让人物离开画面
         
        """
        if poseName:
            pass
        else:
            pass



if __name__ == '__main__':
    dc = DroneController()
    print(dc.keyboard_to_instruction_name)
    print(dc.support_instructions)

    # 是否真正结束？
    while dc.dc_running:
        dc.queue_user_instruction.join()  # 队列为空后，主进程会来到这里
        # dc.logger.info("Waiting for instruction...Or Press ESC to Exit！")
        time.sleep(1)
    dc.logger.info("Main Process Exit！")

    # dc.drone_connect()
