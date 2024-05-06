import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import win32api

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(30 * 6 * 3, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.fc1(x.view(-1, 30 * 6 * 3)))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

class Eye_Pos_Time:
    #눈 부분 추출
    def Eye(self, image, keypoint, w_size, h_size):
        if image is None:
            print("이미지가 없습니다.")
            return None

        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((w_size, h_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        w_patch_half_size = w_size // 2
        h_patch_half_size = h_size // 2
        keypoint_x, keypoint_y = keypoint

        image_width, image_height = image.size

        if keypoint_x - w_patch_half_size < 0 or keypoint_x + w_patch_half_size >= image_width or keypoint_y - h_patch_half_size < 0 or keypoint_y + h_patch_half_size >= image_height:
            return None

        patch = image.crop((keypoint_x - w_patch_half_size, keypoint_y - h_patch_half_size, keypoint_x + w_patch_half_size, keypoint_y + h_patch_half_size))
        patch = transform(patch)

        return patch.to(device)

    #눈 부분에 사각형 그리기
    def Eye_Rec(self, EYE, num, w_size, h_size, frame, eye_state):
        w_size_half = w_size // 2
        h_size_half = h_size // 2
        if EYE is not None:
            Eye_patch = EYE.cpu().numpy()
            Eye_patch = np.transpose(Eye_patch, (1, 2, 0))
            Eye_patch = cv2.cvtColor(Eye_patch, cv2.COLOR_GRAY2BGR) * 255
            Eye_patch = cv2.resize(Eye_patch, (w_size, h_size))
            Eye_patch = Eye_patch.astype(np.uint8)

            Eye_x = keypoints[num][0] - w_size_half
            Eye_y = keypoints[num][1] - h_size_half

            frame[Eye_y:Eye_y + h_size, Eye_x:Eye_x + w_size] = Eye_patch

            cv2.rectangle(frame, (Eye_x, Eye_y), (Eye_x + w_size, Eye_y + h_size), (0, 255, 0), 1)

            text = "OPEN" if eye_state == "Open" else "CLOSED"
            text_color = (0, 255, 0) if eye_state == "Open" else (255, 0, 0)
            cv2.putText(frame, text, (Eye_x, Eye_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1, cv2.LINE_AA)

    #눈 감긴건지 확인
    def Eye_state(self, L_Eye, R_Eye, frame):
        Left_Eye_Message = ""
        Right_Eye_Message = ""
        if L_Eye == None:
            Left_Eye_Message = "fail"
        if R_Eye == None:
            Right_Eye_Message = "fail"
        if L_Eye != None and R_Eye != None:
            Left_Eye_result = blink_model(L_Eye)
            Right_Eye_result = blink_model(R_Eye)
            if Left_Eye_result[0][0] > Left_Eye_result[0][1]:
                Left_Eye_Message = "Open"
            else:
                Left_Eye_Message = "Close"
            if Right_Eye_result[0][0] > Right_Eye_result[0][1]:
                Right_Eye_Message = "Open"
            else:
                Right_Eye_Message = "Close"

        # cv2.putText(frame,
        #             f'Left eye : {Left_Eye_Message}, Right eye : {Right_Eye_Message}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return Left_Eye_Message, Right_Eye_Message

    #어깨 길이, 기울기 측정
    def shoulder_state(self, L_shoulder, R_shoulder, FRAME):
        cv2.line(FRAME, L_shoulder, R_shoulder, (0, 255, 0), 2)
        length = np.linalg.norm(np.array(L_shoulder) - np.array(R_shoulder))
        inclination = (R_shoulder[1] - L_shoulder[1]) / (R_shoulder[0] - L_shoulder[0])
        # cv2.putText(frame,
        #             f'Shoulder Length : {length:.2f}, Shoulder inclination: {inclination:.2f}',
        #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 0, 0), 1, cv2.LINE_AA)
        return length, inclination

    #시간 업데이트 및 화면 출력
    def Time_show(self, SAVE_DATA, TIME, SHOULDER_LENGTH, SHOULDER_INCLINATION, SHOULDER_X, SHOULDER_Y, EYE_CLOSE_TIME, AVERAGE_Y_EYE):

        message = ""
        shoulder_length_condition = 0
        shoulder_inclination_condition = 0
        shoulder_x_condition = 0
        shoulder_y_condition = 0
        eye_close_condition = 0
        average_y_eye_condtion = 0
        if SAVE_DATA[0] - 10 < SHOULDER_LENGTH and SAVE_DATA[0] + 10 > SHOULDER_LENGTH:
            shoulder_length_condition = 0
        elif SAVE_DATA[0] - 10 > SHOULDER_LENGTH:
            shoulder_length_condition = 1
            message += "몸이 앞으로 기움. "
        elif SAVE_DATA[0] + 10 < SHOULDER_LENGTH:
            shoulder_length_condition = 2
            message += "몸이 뒤로 기움. "

        if SAVE_DATA[1] - 0.2 < SHOULDER_INCLINATION and SAVE_DATA[1] + 0.2 > SHOULDER_INCLINATION:
            shoulder_inclination_condition = 0
        elif SAVE_DATA[1] - 0.2 > SHOULDER_INCLINATION:
            shoulder_inclination_condition = 1
            message += "어깨가 왼쪽으로 기움. "
        elif SAVE_DATA[1] + 0.2 < SHOULDER_INCLINATION:
            shoulder_inclination_condition = 2
            message += "어깨가 오른쪽으로 기움. "

        if SAVE_DATA[2] - 10.0 < SHOULDER_X and SAVE_DATA[2] + 10.0 > SHOULDER_X:
            shoulder_x_condition = 0
        elif SAVE_DATA[2] - 10.0 > SHOULDER_X:
            shoulder_x_condition = 1
            message += "몸이 왼쪽으로 움직임. "

        elif SAVE_DATA[2] + 10.0 < SHOULDER_X:
            shoulder_x_condition = 2
            message += "몸이 오른쪽으로 움직임. "

        if SAVE_DATA[3] - 10.0 < SHOULDER_Y and SAVE_DATA[3] + 10.0 > SHOULDER_Y:
            shoulder_y_condition = 0
        elif SAVE_DATA[3] - 10.0 > SHOULDER_Y:
            shoulder_y_condition = 1
            message += "몸이 뒤로 기움. "
        elif SAVE_DATA[3] + 10.0 < SHOULDER_Y:
            shoulder_y_condition = 2
            message += "몸이 앞으로 기움. "

        if EYE_CLOSE_TIME < 10:
            eye_close_condition = 0
        else:
            eye_close_condition = 1
            message += "눈이 감김. "

        if SAVE_DATA[4] - 10.0 < AVERAGE_Y_EYE and SAVE_DATA[4] + 10.0 > AVERAGE_Y_EYE:
            average_y_eye_condtion = 0
        elif SAVE_DATA[4] - 10.0 < AVERAGE_Y_EYE:
            average_y_eye_condtion = 1
            message += "목이 앞으로 기우러짐. "
        elif SAVE_DATA[4] + 10.0 < AVERAGE_Y_EYE:
            average_y_eye_condtion = 2
            message += "목이 뒤로 기우러짐. "

        if shoulder_length_condition == 0 and shoulder_inclination_condition == 0  and shoulder_x_condition == 0 and shoulder_y_condition == 0 and eye_close_condition == 0 and average_y_eye_condtion == 0:
            TIME += 1
        print(message)
        cv2.putText(frame,
                    f'Time : {TIME:d}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
        return TIME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()

EPT = Eye_Pos_Time()

blink_model = CNN()
blink_model = torch.load('C:/Users/SeoJun/PycharmProjects/capstone/model.pt')

Time_Count = 0
Eye_Close_Time = 0
Eye_Detect_W = 50
Eye_Detect_H = 28
cap = cv2.VideoCapture(0)
cnt = False
save_data = [0.0, 0.0, 0.0, 0.0, 0.0]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.line(frame, (frame.shape[1] // 7 * 3, frame.shape[0] // 3 + 15), (frame.shape[1] // 7 * 3, frame.shape[0] // 2), (255, 0, 0), 2)
    cv2.line(frame, (frame.shape[1] // 7 * 4, frame.shape[0] // 3 + 15), (frame.shape[1] // 7 * 4, frame.shape[0] // 2), (255, 0, 0), 2)
    cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 4), 70,(255,0,0), 2)
    cv2.line(frame, (frame.shape[1] // 4, frame.shape[0] // 2), (frame.shape[1] // 4 * 3, frame.shape[0] // 2),(255, 0, 0), 2)
    cv2.line(frame, (frame.shape[1] // 4, frame.shape[0] // 2), (frame.shape[1] // 7 * 1, frame.shape[0]),(255, 0, 0), 2)
    cv2.line(frame, (frame.shape[1] // 4 * 3, frame.shape[0] // 2), (frame.shape[1] // 7 * 6, frame.shape[0]), (255, 0, 0), 2)
    cv2.imshow('Pose Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    image_size = img.size
    trf = T.Compose([T.ToTensor()])
    input_img = trf(img).to(device)

    with torch.no_grad():
        out = model([input_img])[0]

    THRESHOLD = 0.95
    shoulder_length = 0
    shoulder_inclination = 0
    average_X_shouler = 0
    average_Y_shouler = 0
    average_Y_Eye = 0
    for score, keypoints in zip(out['scores'], out['keypoints']):

        score = score.detach().cpu().numpy()
        if score < THRESHOLD:
            continue

        keypoints = keypoints.detach().cpu().numpy().astype(int)[:, :2]

        left_shoulder = tuple(keypoints[5])
        right_shoulder = tuple(keypoints[6])

        average_Y_Eye = (keypoints[1][1] + keypoints[2][1]) / 2

        average_X_shouler = (left_shoulder[0] + right_shoulder[0]) / 2
        average_Y_shouler = (left_shoulder[1] + right_shoulder[1]) / 2

        shoulder_length, shoulder_inclination = EPT.shoulder_state(left_shoulder, right_shoulder, frame)

        Left_Eye = EPT.Eye(img, keypoints[1], Eye_Detect_W, Eye_Detect_H)
        Right_Eye = EPT.Eye(img, keypoints[2], Eye_Detect_W,  Eye_Detect_H)

        Left_Eye_State, Right_Eye_State = EPT.Eye_state(Left_Eye, Right_Eye, frame)

        EPT.Eye_Rec(Left_Eye, 1, Eye_Detect_W, Eye_Detect_H, frame, Left_Eye_State)
        EPT.Eye_Rec(Right_Eye, 2, Eye_Detect_W, Eye_Detect_H, frame, Right_Eye_State)

        if Left_Eye_State == 'Close' or Right_Eye_State == 'Close':
            Eye_Close_Time += 1
        else:
            Eye_Close_Time = 0

        if cnt == False:
            cnt = True
            save_data[0] = shoulder_length
            save_data[1] = shoulder_inclination
            save_data[2] = average_X_shouler
            save_data[3] = average_Y_shouler
            save_data[4] = average_Y_Eye

        Time_Count = EPT.Time_show(save_data, Time_Count, shoulder_length, shoulder_inclination, average_X_shouler, average_Y_shouler, Eye_Close_Time, average_Y_Eye)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Pose Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('r'):
        cnt = True
        save_data[0] = shoulder_length
        save_data[1] = shoulder_inclination
        save_data[2] = average_X_shouler
        save_data[3] = average_Y_shouler
        save_data[4] = average_Y_Eye
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
