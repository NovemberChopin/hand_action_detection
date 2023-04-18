# -*- coding:utf-8 -*-

import cv2
import numpy as np
import os, sys
import math
import PySimpleGUI as sg
import datetime
import tensorflow as tf
from subprocess import call

from hand import Detector

if sys.platform == 'linux':
    FLAG = 2
else:               # win
    FLAG = 1        
    import pyaudio
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100


MAX_DB = 200
    

class SGCV:
    def __init__(self) -> None:
        self.video_path = os.path.join(os.getcwd(), 'videos')
        print(self.video_path)
        if not os.path.isdir(self.video_path):
            os.mkdir(self.video_path)

        self.cameras = [i for i in range(self.get_cam_num())]
        self.camera_index = 0
        self.mirror = True      # 默认显示镜像
        self.right_layout = [[sg.Image(filename='', key='image')]]
        self.webcamera_layout = [[sg.Text('镜像显示：'), 
                                  sg.Radio(text='是', group_id='mirror', enable_events=True, default=True, key='is_mirror'),
                                  sg.Radio(text='否', group_id='mirror', enable_events=True, default=False, key='not_mirror')]]
        for cam in self.cameras:
            default = False
            text ='camera {}'.format(cam)
            key = 'camera_{}'.format(cam)
            if cam == self.camera_index:
                default = True
            self.webcamera_layout.append([sg.Radio(text=text, group_id='radio', enable_events=True ,default=default, key=key)])
        box_height = 10
        if len(self.cameras) >= 2:  box_height = 6
        self.record_layout = [[sg.Button('开始录像', key='recording'), sg.Button('保存路径', key='btn_show')],
                              [sg.Listbox(values=self.get_video_names(), size=(37, box_height), key='list_video', expand_x=True)]]
        self.voice_layout = [[sg.Text('扬声器'), sg.Button('开始测试', key='test_sound'), sg.Button('标定', key='calib')],
                             [sg.ProgressBar(100, orientation='h', size=(25, 20), key='progressbar1', expand_x=True)],
                             [sg.Text('麦克风'), sg.Button('开始测试', key='test_voice')],
                             [sg.ProgressBar(100, orientation='h', size=(20, 20), key='progressbar2', expand_x=True)]]
        # self.oper_layout = [[sg.Button("手势识别", key='gesHand'), sg.Button("人脸识别", key='gesFace'), sg.Button('退出', key='exit')]]
        self.oper_layout = [[sg.Text('开启检测：'),
                             sg.Checkbox('手势', default=False, enable_events=True, key='check_hand'),
                             sg.Checkbox('面部', default=False, enable_events=True, key='check_face'),
                             sg.Button('退出', key='exit')]]

        self.deviceFrame = [sg.Frame(title='选择摄像头',layout=self.webcamera_layout, expand_x=True)]
        self.recordFrame = [sg.Frame(title='录像', layout=self.record_layout, expand_x=True)]
        self.voiceFrame = [sg.Frame(title='音量测试', layout=self.voice_layout, expand_x=True)]
        self.op_frame = [sg.Frame(title='操作', layout=self.oper_layout, expand_x=True)]

        self.layout = [[sg.Column(layout=[self.op_frame, self.deviceFrame, self.voiceFrame, self.recordFrame], size=(300, 500)), 
                        sg.VSeparator(), sg.Column(self.right_layout)]]
        
        self.window = sg.Window(title='智能座舱测试软件', layout=self.layout)

        ###################
        self.video_out = None       # 视频录制对象
        self.is_record = False      # 是否在录制视频
        self.is_calib = False       # 是否在标定模式
        self.is_test_sound = False  # 是否使用手势控制音量
        self.is_test_voice = False  # 测试麦克风

        self.detec_hand = False      # 是否要检测手势
        self.detec_face = False      # 是否要检测人脸
        # self.window['image'].set_size((800, 600))
        # self.actions = np.array(['one', 'yeah', 'three', 'four', 'good', 'not good', 'ok', 'palm', 'fist'])
        self.actions = np.array(['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine' , 'ten', 'good', 'not good', 'ok'])
        if len(self.cameras) > 0:
            self.cap = cv2.VideoCapture(self.cameras[self.camera_index] * FLAG)
        self.detector = Detector(is_face=self.detec_face, is_hand=self.detec_hand)
        self.model = tf.keras.models.load_model(self.resource_path('./Model/action_13.h5'))

        if sys.platform != 'linux':
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume = cast(interface, POINTER(IAudioEndpointVolume))
            global CHUNK,FORMAT,CHANNELS,RATE
            p = pyaudio.PyAudio()
            self.stream = p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK)

        # 标定相关变量
        self.stop_time = None
        self.len_max = 0


    def get_video_names(self):
        """ 获取视频列表 """
        if not os.path.isdir(self.video_path):
            os.mkdir(self.video_path)
        try:
            file_list = os.listdir(self.video_path)
        except:
            file_list = []
        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(self.video_path, f))
            and f.lower().endswith((".avi"))
        ]
        return fnames
    

    def get_cam_num(self):
        num = 0
        max_num = 10
        for device in range(0, max_num, FLAG):
            stream = cv2.VideoCapture(device)
            grabbed = stream.grab()
            stream.release()
            if not grabbed:
                break
            num += 1
        return num


    def resource_path(self, relative_path):
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)


    def predictAction(self, frame):
        # 使用模型预测动作
        if self.detec_hand:
            if self.detector.results.multi_hand_landmarks:
                x_train = np.array([[res.x, res.y, res.z] for res in self.detector.results.multi_hand_landmarks[0].landmark]).flatten()
                x_train = np.reshape(x_train, (1, -1))
                y_pre = self.model.predict(x_train)
                act = self.actions[np.argmax(y_pre[0])]
                cv2.putText(frame, 'Action is: {}'.format(act), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
        return frame
    

    def calib(self, frame):
        """ 标定食指和拇指指尖的距离 """
        if self.detector.results.multi_hand_landmarks:
            land_marks = self.detector.results.multi_hand_landmarks[0]
            normalized_landmarks = self.detector.Normalize_landmarks(image=frame, hand_landmarks=land_marks)
            frame, length = self.detector.Draw_hand_points(frame, normalized_landmarks)
            strRate = 'Start calibration'
            cv2.putText(frame, strRate, (10, 410), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 0), 2)
            strRate1 = 'max length = %d'%self.len_max
            cv2.putText(frame, strRate1, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 0), 2)

            if length > self.len_max:
                self.len_max = length
        return frame


    def gesCtrlSound(self, frame):
        """ 根据检测到拇指和食指距离调整系统音量 """
        if self.detector.results.multi_hand_landmarks:
            land_marks = self.detector.results.multi_hand_landmarks[0]
            normalized_landmarks = self.detector.Normalize_landmarks(image=frame, hand_landmarks=land_marks)
            try:
                frame, length = self.detector.Draw_hand_points(frame, normalized_landmarks)
                cv2.rectangle(frame, (50, 150), (85, 350), (255, 0, 0), 1)
                if length >self.len_max:
                    length = self.len_max

                vol = int((length) / self.len_max * 100)
                if sys.platform != 'linux':      # win
                    self.volume.SetMasterVolumeLevel(self.detector.vol_tansfer(vol), None)
                else:                   # linux
                    call(["amixer", "-D", "pulse", "sset", "Master", "{}%".format(vol)])
                
                self.window['progressbar1'].update(vol)
 
                cv2.rectangle(frame, (50, 150+200-2*vol), (85, 350), (255, 0, 0), cv2.FILLED)
                percent = int(length / self.len_max * 100)
 
                strRate = str(percent) + '%'
                cv2.putText(frame, strRate, (40, 410), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 0), 2)
                cv2.putText(frame, vol, (10, 470), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 0), 2)
            except:
                pass
    

    def testVoice(self):
        if sys.platform != 'linux':      # win
            data = self.stream.read(CHUNK)
            audio_data = np.fromstring(data, dtype=np.short)
            max_dB = np.max(audio_data)
            vol = int((max_dB) / MAX_DB * 100)
            self.window['progressbar2'].update(vol)
        else:
            pass


    def run(self):
        # 获取系统扬声器、麦克风的值
        while True:
            if len(self.cameras) == 0:
                sg.popup('没有检测到摄像头，请检查连接', title='错误')
                break

            event, value = self.window.read(timeout=40)
            if event == sg.WIN_CLOSED or event == 'exit':
                break

            # if event == 'gesHand':          # 手势界面
            #     sg.popup('目前可以识别包括1-10以及good、not good、ok在内的13种手势', title='手势说明')

            if event == 'check_hand':           # 开启/关闭 检测手势
                self.detec_hand = value['check_hand']
                self.detector.setModel(face=self.detec_face, hand=self.detec_hand)

            if event == 'check_face':           # 开启/关闭 检测人脸
                self.detec_face = value['check_face']
                self.detector.setModel(face=self.detec_face, hand=self.detec_hand)

            # 镜像切换
            if event == 'is_mirror':
                self.mirror = True
            if event == 'not_mirror':
                self.mirror = False
            
            # 切换相机事件
            if event in ['camera_{}'.format(i) for i in self.cameras]:
                for i in self.cameras:
                    if value['camera_{}'.format(i)] == True and i != self.camera_index:
                        self.cap.release()
                        self.cap = cv2.VideoCapture(i * FLAG)
                        self.camera_index = i

            # 录像事件
            if event == 'recording':
                if self.is_record:
                    self.video_out.release()
                    self.video_out = None
                    self.is_record = False
                    self.window['recording'].update('开始录制')
                    # 更新列表
                    self.window['list_video'].update(self.get_video_names())
                else:                   # 否则，开始录制
                    filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    self.video_out = cv2.VideoWriter('videos/{}.avi'.format(filename), cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))
                    self.is_record = True
                    self.window['recording'].update('停止录制')
            
            if event == 'btn_show':
                sg.popup(self.video_path, title='视频保存路径')

            if event == 'test_sound':           # 测试扬声器
                if not self.detec_hand:
                    sg.popup("请先开启手势检测后操作")
                elif self.len_max == 0:
                    sg.popup('请先点击右侧标定按钮进行标定')
                elif self.is_test_sound:        # 使用手势控制扬声器音量大小
                    self.is_test_sound = False
                    self.window['test_sound'].update('开始测试')
                else:
                    self.is_test_sound = True
                    self.window['test_sound'].update('停止测试')
            

            if event == 'test_voice':           # 测试麦克风
                if self.is_test_voice:
                    self.is_test_voice = False
                    self.window['test_voice'].update('开始测试')
                else:
                    self.is_test_voice = True
                    self.window['test_voice'].update('停止测试')


            if event == 'calib':       # 调试扬声器
                if self.detec_hand:
                    sg.popup("点击OK开始标定，请将拇指与食指张开到最大距离")
                    self.is_calib = True
                    self.len_max = 0
                    self.stop_time = datetime.datetime.now() + datetime.timedelta(seconds=5)
                else:
                    sg.popup("请先开启手势检测后操作")

            
            if self.cap is not None:
                ret, frame = self.cap.read()
                if self.mirror:
                    frame = cv2.flip(frame, 1)

                frame = self.detector.runDetec(frame)
                if self.is_calib:           # 如果当前在标定模式
                    if datetime.datetime.now() < self.stop_time:
                        frame = self.calib(frame=frame)
                    else:   # 标定结束
                        sg.popup('标定结束')
                        self.is_calib = False
                        self.window['calib'].update('重新标定')
                        # print("len_max is {}".format(self.len_max))
                elif self.is_test_sound:       # 手势控制音量
                    self.gesCtrlSound(frame=frame)
                else:
                    # 检测人脸和手势
                    frame = self.predictAction(frame=frame)
                
                if self.is_test_voice:
                    self.testVoice()

                if self.is_record:
                    self.video_out.write(frame)
                    cv2.putText(frame,'Recording',  (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
                
                flag, frame = cv2.imencode(ext='.png', img=frame)
                self.window['image'].update(data=frame.tobytes())

sg_demo = SGCV()
sg_demo.run()
