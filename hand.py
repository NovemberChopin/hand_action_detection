import cv2
import math
import mediapipe as mp


class Detector:
    def __init__(self, static_image_mode=False,
                        max_num_hands=2,
                        model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                        is_face=True,
                        is_hand=True) -> None:
        # 检测手势
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=static_image_mode,
                                        max_num_hands=max_num_hands,
                                        model_complexity=model_complexity,
                                        min_detection_confidence=min_detection_confidence,
                                        min_tracking_confidence=min_tracking_confidence)
        # 面部检测
        self.mp_face_detection = mp.solutions.face_detection
        self.face = self.mp_face_detection.FaceDetection(
                                model_selection=0,
                                min_detection_confidence=min_detection_confidence)
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.tipIds = [4, 8, 12, 16, 20]			# 指尖列表
        self.lm_x_point = []
        self.lm_y_point = []
        # self.handTypes = []
        self.frame_index = 0

        self.results = None
        self.face_results = None

        self.is_face = is_face
        self.is_hand = is_hand


    def runDetec(self, image):
        """ 运行检测程序 """
        if self.is_face or self.is_hand:
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.is_face:    self.face_results = self.face.process(image)
            if self.is_hand:    self.results = self.hands.process(image)
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # draw hand
        if self.is_hand:
            if self.results.multi_hand_landmarks:
                for hand_landmarks in self.results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
        # draw face
        if self.is_face:
            if self.face_results.detections:
                for detection in self.face_results.detections:
                    self.mp_drawing.draw_detection(image, detection)
        return image
    

    def setModel(self, face, hand):
        """ 设置是否开启人脸和手势检测 """
        self.is_face = face
        self.is_hand = hand


    def getBox(self, img) -> list:
        """ 计算手部周围的边界框 """
        bboxs = []
        image_height, image_width, _ = img.shape
        if self.results.multi_hand_landmarks:
            # 遍历检测到的所有的手
            self.lm_x_point.clear()
            self.lm_y_point.clear()
            # self.handTypes.clear()
            index = 0
            self.frame_index = (self.frame_index + 1) % 20
            for hand_landmarks in self.results.multi_hand_landmarks:
                xList, yList = [], []
                for _, lm in enumerate(hand_landmarks.landmark):
                    xList.append(int(lm.x * image_width))
                    yList.append(int(lm.y * image_height))
                # 判断左右手,也可以直接用 multi_handedness 来判断，见方法 fingersIsUp
                # if xList[17] < xList[5]:    
                #     self.handTypes.append("Right")
                # else:
                #     self.handTypes.append("Left")
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                              (0, 255, 0), 2)
                hand_type = self.results.multi_handedness[index].classification[0].label
                cv2.putText(img, "id: {}, hand: {}".format(index, hand_type), 
                            (xmin - 25, ymin - 20), 
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                cv2.putText(img, "frame_index: {}".format(self.frame_index), (20, 20), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                index += 1
                self.lm_x_point.append(xList)
                self.lm_y_point.append(yList)
                bboxs.append(bbox)
        return bboxs


    def fingersIsUp(self) -> list:
        """
        返回竖起和弯曲的手指列表,1表示竖起,0表示弯曲
        """
        index = 0
        multi_fingers = []
        for landhands in self.results.multi_hand_landmarks:
            hand_type = self.results.multi_handedness[index].classification[0].label
            fingers = []
            # 判断大拇指
            if hand_type == "Right":
                # 如果大拇指尖x坐标小于
                if self.lm_x_point[index][self.tipIds[0]] > self.lm_x_point[index][self.tipIds[0] - 1]:
                    fingers.append(0)
                else:   fingers.append(1)
            else:
                if self.lm_x_point[index][self.tipIds[0]] < self.lm_x_point[index][self.tipIds[0] - 1]:
                    fingers.append(0)
                else:   fingers.append(1)

            # 4 Fingers，4个手指在握拳状态，使用 y 轴坐标来判断
            for i in range(1, 5):
                # 如果指尖坐标小于下面两个关节的 y 坐标，说明手是伸直的
                if self.lm_y_point[index][self.tipIds[i]] < self.lm_y_point[index][self.tipIds[i] - 2]:
                    fingers.append(1)
                else: fingers.append(0)
            multi_fingers.append(fingers)
            index += 1
        return multi_fingers
        
    def getLeftOrRight(self):
        if self.results.multi_handedness:
            print("------------------")
            for handedness in self.results.multi_handedness:
                print(handedness.classification[0].label)
    

    def Normalize_landmarks(self, image, hand_landmarks):
        new_landmarks = []
        for i in range(0, len(hand_landmarks.landmark)):
            float_x = hand_landmarks.landmark[i].x
            float_y = hand_landmarks.landmark[i].y
            width = image.shape[1]
            height = image.shape[0]
            pt = self.mp_drawing._normalized_to_pixel_coordinates(float_x, float_y, width, height)
            new_landmarks.append(pt)
        return new_landmarks


    def Draw_hand_points(self, image, normalized_hand_landmarks):
        cv2.circle(image, normalized_hand_landmarks[4], 12, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(image, normalized_hand_landmarks[8], 12, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.line(image, normalized_hand_landmarks[4], normalized_hand_landmarks[8], (255, 0, 255), 3)
        x1, y1 = normalized_hand_landmarks[4][0], normalized_hand_landmarks[4][1]
        x2, y2 = normalized_hand_landmarks[8][0], normalized_hand_landmarks[8][1]
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.sqrt((x2 - x1)**2+(y2 - y1)**2) #得到大拇指到食指的距离
        if length < 100:
            cv2.circle(image, (mid_x, mid_y), 12, (0, 255, 0), cv2.FILLED)
        else:
            cv2.circle(image, (mid_x, mid_y), 12, (255, 0, 255), cv2.FILLED)
        return image, length
    

    def vol_tansfer(self, x):
        dict = {0: -65.25, 1: -56.99, 2: -51.67, 3: -47.74, 4: -44.62, 5: -42.03, 6: -39.82, 7: -37.89, 8: -36.17,
                9: -34.63, 10: -33.24,
                11: -31.96, 12: -30.78, 13: -29.68, 14: -28.66, 15: -27.7, 16: -26.8, 17: -25.95, 18: -25.15, 19: -24.38,
                20: -23.65,
                21: -22.96, 22: -22.3, 23: -21.66, 24: -21.05, 25: -20.46, 26: -19.9, 27: -19.35, 28: -18.82, 29: -18.32,
                30: -17.82,
                31: -17.35, 32: -16.88, 33: -16.44, 34: -16.0, 35: -15.58, 36: -15.16, 37: -14.76, 38: -14.37, 39: -13.99,
                40: -13.62,
                41: -13.26, 42: -12.9, 43: -12.56, 44: -12.22, 45: -11.89, 46: -11.56, 47: -11.24, 48: -10.93, 49: -10.63,
                50: -10.33,
                51: -10.04, 52: -9.75, 53: -9.47, 54: -9.19, 55: -8.92, 56: -8.65, 57: -8.39, 58: -8.13, 59: -7.88,
                60: -7.63,
                61: -7.38, 62: -7.14, 63: -6.9, 64: -6.67, 65: -6.44, 66: -6.21, 67: -5.99, 68: -5.76, 69: -5.55, 70: -5.33,
                71: -5.12, 72: -4.91, 73: -4.71, 74: -4.5, 75: -4.3, 76: -4.11, 77: -3.91, 78: -3.72, 79: -3.53, 80: -3.34,
                81: -3.15, 82: -2.97, 83: -2.79, 84: -2.61, 85: -2.43, 86: -2.26, 87: -2.09, 88: -1.91, 89: -1.75,
                90: -1.58,
                91: -1.41, 92: -1.25, 93: -1.09, 94: -0.93, 95: -0.77, 96: -0.61, 97: -0.46, 98: -0.3, 99: -0.15, 100: 0.0}
        return dict[x]


    def vol_tansfer_reverse(self, x):
        error = []
        dict = {0: -65.25, 1: -56.99, 2: -51.67, 3: -47.74, 4: -44.62, 5: -42.03, 6: -39.82, 7: -37.89, 8: -36.17,
                9: -34.63, 10: -33.24,
                11: -31.96, 12: -30.78, 13: -29.68, 14: -28.66, 15: -27.7, 16: -26.8, 17: -25.95, 18: -25.15, 19: -24.38,
                20: -23.65,
                21: -22.96, 22: -22.3, 23: -21.66, 24: -21.05, 25: -20.46, 26: -19.9, 27: -19.35, 28: -18.82, 29: -18.32,
                30: -17.82,
                31: -17.35, 32: -16.88, 33: -16.44, 34: -16.0, 35: -15.58, 36: -15.16, 37: -14.76, 38: -14.37, 39: -13.99,
                40: -13.62,
                41: -13.26, 42: -12.9, 43: -12.56, 44: -12.22, 45: -11.89, 46: -11.56, 47: -11.24, 48: -10.93, 49: -10.63,
                50: -10.33,
                51: -10.04, 52: -9.75, 53: -9.47, 54: -9.19, 55: -8.92, 56: -8.65, 57: -8.39, 58: -8.13, 59: -7.88,
                60: -7.63,
                61: -7.38, 62: -7.14, 63: -6.9, 64: -6.67, 65: -6.44, 66: -6.21, 67: -5.99, 68: -5.76, 69: -5.55, 70: -5.33,
                71: -5.12, 72: -4.91, 73: -4.71, 74: -4.5, 75: -4.3, 76: -4.11, 77: -3.91, 78: -3.72, 79: -3.53, 80: -3.34,
                81: -3.15, 82: -2.97, 83: -2.79, 84: -2.61, 85: -2.43, 86: -2.26, 87: -2.09, 88: -1.91, 89: -1.75,
                90: -1.58,
                91: -1.41, 92: -1.25, 93: -1.09, 94: -0.93, 95: -0.77, 96: -0.61, 97: -0.46, 98: -0.3, 99: -0.15, 100: 0.0}
        for i in range (100):
            error.append(abs(dict[i]-x))
        return error.index(min(error))


if __name__ == '__main__':
    detector = Detector(is_face=True, is_hand=True)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        flag, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not flag:
            print("Ignoring empty camera frame.")
            continue
        frame = detector.runDetec(frame)
        bboxs = detector.getBox(frame)
        # detector.getLeftOrRight()
        if bboxs:
            print(detector.fingersIsUp())
            # print(detector.handTypes)

        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(50) & 0xFF == 27:
            break