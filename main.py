import subprocess
import time
import re

import cv2
import numpy as np

import DobotDllType as dType


# =========================
# Camera + Calibration
# =========================

MATRIX = np.load("matrix.npy")

cap = cv2.VideoCapture(1)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

if not cap.isOpened():

    print("Camera接続失敗")
    exit()

print("Camera接続成功")


IMG_W = 640
IMG_H = 480


# =========================
# 箱子的固定位置
# =========================
# 请根据实际箱子位置校准下面的坐标。
BOX_X = 280
BOX_Y = 0
BOX_Z = 80
BOX_L = 500


# =========================
# 手动输入指令
# =========================

def input_command():

    print("\n指令を入力してください：")

    text = input("入力：").strip()

    print("入力:")
    print(text)

    return text


# =========================
# OpenCV 红色检测
# =========================

def detect_red():

    ret, frame = cap.read()

    if not ret:

        print("Camera读取失败")

        return None

    frame = cv2.resize(
        frame,
        (IMG_W, IMG_H)
    )

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    # =========================
    # 红色范围1
    # =========================

    red1 = cv2.inRange(
        hsv,
        np.array([0, 120, 70]),
        np.array([10, 255, 255])
    )

    # =========================
    # 红色范围2
    # =========================

    red2 = cv2.inRange(
        hsv,
        np.array([170, 120, 70]),
        np.array([180, 255, 255])
    )

    mask = red1 + red2

    # =========================
    # 去除噪声
    # =========================

    kernel = np.ones(
        (7, 7),
        np.uint8
    )

    mask = cv2.erode(
        mask,
        kernel,
        iterations=1
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=2
    )

    # =========================
    # 查找轮廓
    # =========================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:

        cv2.imshow(
            "Camera",
            frame
        )

        cv2.waitKey(1)

        return None

    # =========================
    # 最大轮廓
    # =========================

    largest = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(
        largest
    )

    if area < 150:

        return None

    # =========================
    # 外接矩形
    # =========================

    x, y, w, h = cv2.boundingRect(
        largest
    )

    # =========================
    # 中心点
    # =========================

    cx = x + w // 2
    cy = y + h // 2

    # =========================
    # 显示检测结果
    # =========================

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.circle(
        frame,
        (cx, cy),
        5,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        frame,
        f"Red: ({cx},{cy})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Camera",
        frame
    )

    cv2.waitKey(1)

    print(
        "Pixel:",
        cx,
        cy
    )

    # =========================
    # Pixel → Robot
    # =========================

    point = np.array(
        [[[cx, cy]]],
        dtype=np.float32
    )

    robot = cv2.perspectiveTransform(
        point,
        MATRIX
    )

    X = float(
        robot[0][0][0]
    )

    Y = float(
        robot[0][0][1]
    )

    print(
        "Robot:",
        round(X, 1),
        round(Y, 1)
    )

    return X, Y


# =========================
# LLM
# =========================

# =========================
# OpenCV 箱子检测
# =========================

# =========================
# LLM
# =========================

def llm_command(text):

    prompt = f"""
日本語のロボット操作指令を解析してください。

あなたはロボット制御コマンドを出力するだけです。

入力:

{text}

使用可能なコマンド:

FIND RED
FIND BOX
MOVE_RIGHT
MOVE_LEFT
MOVE_FORWARD
MOVE_BACKWARD
MOVE_UP
MOVE_DOWN
HOME
STOP
CLEAR_ALARM

方向:

右 = MOVE_RIGHT
左 = MOVE_LEFT
前 = MOVE_FORWARD
後ろ = MOVE_BACKWARD
上 = MOVE_UP
下 = MOVE_DOWN

重要:

入力に存在する動作だけを出力してください。

箱のルール:
箱は固定された位置にあります。
「箱を探して」「箱を見つけて」など、箱の固定位置へ移動する指示の場合は FIND BOX を出力してください。
FIND BOX はカメラで箱を検出するのではなく、あらかじめ設定した固定座標へ移動します。
「箱に入れて」「箱に置いて」「箱へ移動して」などの場合も、固定された箱の位置へ移動する目的なら FIND BOX を使用してください。

入力に存在しない方向を絶対に追加しないでください。

例えば、

入力:
赤い物を探して、前に5センチ動いて

正しい:
FIND RED
MOVE_FORWARD

間違い:
FIND RED
MOVE_FORWARD
MOVE_RIGHT

入力:
赤い物を探して、右に10センチ動いて

正しい:
FIND RED
MOVE_RIGHT

間違い:
FIND RED
MOVE_RIGHT
MOVE_FORWARD

入力:
赤い物を探して

正しい:
FIND RED

間違い:
FIND RED
MOVE_FORWARD

入力の動作順序を維持してください。

絶対に説明禁止
絶対に文章禁止
絶対に質問禁止
絶対に挨拶禁止
絶対に入力の繰り返し禁止
絶対に余計な動作禁止

1コマンドにつき1行

出力:
"""

    result = subprocess.run(
        [
            "ollama",
            "run",
            "llama3.2"
        ],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    output = result.stdout.strip()

    print("\nLLM:")
    print(output)

    # =========================
    # LLMから動作種類だけ取得
    # =========================

    llm_actions = []

    valid_actions = {
        "MOVE_RIGHT",
        "MOVE_LEFT",
        "MOVE_FORWARD",
        "MOVE_BACKWARD",
        "MOVE_UP",
        "MOVE_DOWN",
        "FIND RED",
        "FIND BOX",
        "HOME",
        "STOP",
        "CLEAR_ALARM",
        "PLACE BOX_A"
    }

    for raw_line in output.splitlines():

        line = raw_line.strip().upper()

        line = line.replace(
            "**",
            ""
        )

        line = line.replace(
            "`",
            ""
        )

        line = line.strip()

        if line in valid_actions:

            llm_actions.append(
                line
            )

            continue

        # 例: 「箱を探して → FIND BOX」から
        # 許可されたコマンドを取得
        for valid_action in valid_actions:
            if valid_action in line:
                llm_actions.append(valid_action)
                break

    # =========================
    # Pythonで元の日本語から
    # 実際の数値・単位を取得
    # =========================

    # =========================
    # LLM誤認識対策
    # =========================
    red_words = ["赤", "赤い", "レッド", "red"]
    if not any(word in text.lower() for word in red_words):
        llm_actions = [a for a in llm_actions if a != "FIND RED"]

    commands = []

    # =========================
    # FIND RED
    # =========================

    if (
        "赤い物" in text
        or "赤いもの" in text
    ):

        if "探して" in text or "見つけて" in text:

            commands.append(
                "FIND RED"
            )

        # =========================
    # 方向と数値
    # =========================

    # =========================
    # FIND BOX
    # =========================
    if "箱" in text and ("探して" in text or "見つけて" in text):
        if "FIND BOX" in llm_actions:
            commands.append("FIND BOX")

    # =========================
    # 方向と数値
    # =========================

    direction_patterns = [
        (
            "MOVE_RIGHT",
            r"右[^0-9０-９]*([0-9０-９]+(?:\.[0-9]+)?)\s*(センチメートル|センチ|cm|CM|ミリメートル|ミリ|mm|MM)"
        ),
        (
            "MOVE_LEFT",
            r"左[^0-9０-９]*([0-9０-９]+(?:\.[0-9]+)?)\s*(センチメートル|センチ|cm|CM|ミリメートル|ミリ|mm|MM)"
        ),
        (
            "MOVE_FORWARD",
            r"前[^0-9０-９]*([0-9０-９]+(?:\.[0-9]+)?)\s*(センチメートル|センチ|cm|CM|ミリメートル|ミリ|mm|MM)"
        ),
        (
            "MOVE_BACKWARD",
            r"後ろ[^0-9０-９]*([0-9０-９]+(?:\.[0-9]+)?)\s*(センチメートル|センチ|cm|CM|ミリメートル|ミリ|mm|MM)"
        ),
        (
            "MOVE_UP",
            r"上[^0-9０-９]*([0-9０-９]+(?:\.[0-9]+)?)\s*(センチメートル|センチ|cm|CM|ミリメートル|ミリ|mm|MM)"
        ),
        (
            "MOVE_DOWN",
            r"下[^0-9０-９]*([0-9０-９]+(?:\.[0-9]+)?)\s*(センチメートル|センチ|cm|CM|ミリメートル|ミリ|mm|MM)"
        )
    ]

    detected_moves = []

    for action, pattern in direction_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match is not None:

            value = match.group(1)
            unit_text = match.group(2).lower()

            # =========================
            # 全角数字 → 半角数字
            # =========================
            trans_table = str.maketrans(
                "０１２３４５６７８９",
                "0123456789"
            )

            value = value.translate(
                trans_table
            )

            # =========================
            # 单位转换
            # =========================
            if unit_text in [
                "センチ",
                "センチメートル",
                "cm"
            ]:
                unit = "CM"

            elif unit_text in [
                "ミリ",
                "ミリメートル",
                "mm"
            ]:
                unit = "MM"

            else:
                continue

            detected_moves.append(
                (
                    match.start(),
                    action,
                    value,
                    unit
                )
            )

    # =========================
    # 按照原文位置排序
    # =========================

    detected_moves.sort(
        key=lambda x: x[0]
    )

    # =========================
    # 加入MOVE命令
    # =========================

    for _, action, value, unit in detected_moves:

        commands.append(
            f"{action} {value} {unit}"
        )

    # =========================
    # HOME
    # =========================

    if "ホーム" in text:

        commands.append(
            "HOME"
        )

    # =========================
    # STOP
    # =========================

    if "停止" in text:

        commands.append(
            "STOP"
        )

    # =========================
    # CLEAR ALARM
    # =========================

    if "アラーム解除" in text:

        commands.append(
            "CLEAR_ALARM"
        )

    # =========================
    # Commands
    # =========================

    print("\nCommands:")

    for cmd in commands:

        print(cmd)

    return commands

# =========================
# 单位转换
# =========================

def convert_distance(value, unit):

    value = float(value)

    unit = unit.upper()

    if unit == "MM":

        return value

    elif unit == "CM":

        return value * 10

    else:

        return 0


# =========================
# Dobot连接
# =========================

api = dType.load()

state = dType.ConnectDobot(
    api,
    "",
    115200
)[0]

if state != dType.DobotConnect.DobotConnect_NoError:

    print("接続失敗")

    exit()

print("Dobot接続成功")


dType.SetQueuedCmdClear(
    api
)

dType.SetQueuedCmdStartExec(
    api
)


dType.SetPTPJointParams(
    api,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
    100,
    isQueued=0
)


dType.SetPTPCommonParams(
    api,
    50,
    50,
    isQueued=0
)


# =========================
# 滑轨开启
# =========================

dType.SetDeviceWithL(
    api,
    1
)


# =========================
# 初始位置
# =========================

X = 250

arm_Y = 0

MOVE_Z = 60

L = 500


# =========================
# Main
# =========================

while True:

    # =========================
    # 输入指令
    # =========================

    text = input_command()


    if text == "":

        continue


    if text.lower() == "exit":

        break


    # =========================
    # ③ LLM理解
    # =========================

    commands = llm_command(
        text
    )


    if len(commands) == 0:

        print(
            "命令解析失败"
        )

        continue


    # =========================
    # ④ 执行指令
    # =========================

    stop_requested = False


    for command in commands:

        print(
            "\n実行:",
            command
        )

        parts = command.split()

        action = parts[0]


        # =====================
        # FIND RED
        # =====================

        if action == "FIND":

            if len(parts) != 2:

                print(
                    "FIND命令格式错误"
                )

                continue

            target = parts[1]

            # =====================
            # FIND RED
            # =====================

            if target == "RED":

                result = detect_red()

                if result is None:

                    print(
                        "赤い物なし"
                    )

                    continue

                X, Y = result

                MOVE_Z = 100

                OFFSET = 20

                L_CENTER = 500

                if Y >= 0:

                    arm_Y = OFFSET

                    L = (
                        L_CENTER
                        - (Y - OFFSET)
                    )

                else:

                    arm_Y = -OFFSET

                    L = (
                        L_CENTER
                        - (Y + OFFSET)
                    )

                print(
                    "Target:",
                    X,
                    arm_Y,
                    MOVE_Z,
                    L
                )

            # =====================
            # FIND BOX
            # =====================

            elif target == "BOX":

                X = BOX_X
                arm_Y = BOX_Y
                MOVE_Z = BOX_Z
                L = BOX_L

                print(
                    "Box Target:",
                    "X=", X,
                    "Y=", arm_Y,
                    "Z=", MOVE_Z,
                    "L=", L
                )

            else:

                print(
                    "未知のFIND対象:",
                    target
                )

                continue


        # =====================
        # MOVE
        # =====================

        elif action.startswith("MOVE_"):

            # =====================
            # 安全检查
            # =====================

            if len(parts) != 3:

                print(
                    "MOVE命令格式错误:",
                    command
                )

                continue

            value_text = parts[1]

            unit = parts[2]


            # =====================
            # 数字检查
            # =====================

            try:

                value = float(
                    value_text
                )

            except ValueError:

                print(
                    "MOVE数値エラー:",
                    command
                )

                continue


            if unit not in [
                "CM",
                "MM"
            ]:

                print(
                    "単位エラー:",
                    command
                )

                continue


            distance = convert_distance(
                value,
                unit
            )


            # =====================
            # 方向
            # =====================

            if action == "MOVE_RIGHT":

                L += distance


            elif action == "MOVE_LEFT":

                L -= distance


            elif action == "MOVE_FORWARD":

                X += distance


            elif action == "MOVE_BACKWARD":

                X -= distance


            elif action == "MOVE_UP":

                MOVE_Z += distance


            elif action == "MOVE_DOWN":

                MOVE_Z -= distance


            else:

                print(
                    "未知のMOVE命令:",
                    command
                )

                continue


        # =====================
        # HOME
        # =====================

        elif action == "HOME":

            X = 250

            arm_Y = 0

            MOVE_Z = 60

            L = 500


        # =====================
        # STOP
        # =====================

        elif action == "STOP":

            print(
                "STOP"
            )

            dType.SetQueuedCmdStopExec(
                api
            )

            stop_requested = True

            break


        # =====================
        # CLEAR ALARM
        # =====================

        elif action == "CLEAR_ALARM":

            print(
                "Clear Alarm"
            )

            dType.ClearAllAlarmsState(
                api
            )

            time.sleep(
                1
            )

            continue


        # =====================
        # PLACE BOX_A
        # =====================

        elif action == "PLACE":

            if len(parts) != 2 or parts[1] != "BOX_A":

                print(
                    "PLACE命令格式错误:",
                    command
                )

                continue

            # 箱子的固定位置
            X = BOX_X
            arm_Y = BOX_Y
            MOVE_Z = BOX_Z
            L = BOX_L

            print(
                "Box Target:",
                "X=", X,
                "Y=", arm_Y,
                "Z=", MOVE_Z,
                "L=", L
            )


        # =====================
        # 其他
        # =====================

        else:

            print(
                "未知の命令:",
                command
            )

            continue


        # =========================
        # 安全限制
        # =========================

        X = max(
            150,
            min(
                320,
                X
            )
        )

        L = max(
            200,
            min(
                800,
                L
            )
        )

        MOVE_Z = max(
            20,
            min(
                200,
                MOVE_Z
            )
        )


        print(
            "Current:",
            "X=",
            round(X, 1),
            "Y=",
            round(arm_Y, 1),
            "Z=",
            round(MOVE_Z, 1),
            "L=",
            round(L, 1)
        )


        # =========================
        # 执行机械臂移动
        # =========================

        dType.ClearAllAlarmsState(
            api
        )


        result = dType.SetPTPWithLCmd(
            api,
            dType.PTPMode.PTPMOVLXYZMode,
            X,
            arm_Y,
            MOVE_Z,
            0,
            L,
            isQueued=0
        )


        print(
            "Dobot:",
            result
        )


        time.sleep(
            3
        )


    if stop_requested:

        print(
            "停止当前指令执行"
        )

        continue


# =========================
# 结束
# =========================

cap.release()

cv2.destroyAllWindows()

dType.DisconnectDobot(
    api
)

print(
    "終了"
)