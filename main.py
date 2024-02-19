import os
import time
import json
import cv2
import gradio as gr

from servo_utils import servoToAngle, servoState, uart, changeServeID


def get_angle_value(id):
    id,target_angle = servoState(id)
    print(f"id为：{id} 的角度{target_angle}")
    return target_angle

def complex_analysis(joint1,joint2,joint3,joint4,joint5,frame):
    uart.open()
    servoToAngle(1, joint1, 10)
    servoToAngle(2, joint2, 10)
    servoToAngle(3, joint3, 10)
    servoToAngle(4, joint4, 10)
    servoToAngle(5, joint5, 10)

    uart.close()

    return "好",frame

def function2(path,task,pattern,action_name,name,frame):
    uart.open()

    a1 = get_angle_value(1)
    a2 = get_angle_value(2)
    a3 = get_angle_value(3)
    a4 = get_angle_value(4)
    a5 = get_angle_value(5)

    uart.close()

    if not os.path.exists(path):
        os.makedirs(path)
    tip = ""
    if pattern != None:
        path = os.path.join(path, task)
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, pattern)
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, action_name)
        if not os.path.exists(path):
            os.makedirs(path)
        data = {
            "img":name+'.png',
            "action":[a1,a2,a3,a4,a5]
        }
        json_data = json.dumps(data)
        with open(os.path.join(path,name+'.json'), 'w') as f:
            f.write(json_data)

        cv2.imwrite(os.path.join(path,name+'.png'),cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        tip = "数据保存成功！！！"
        print("数据保存成功！！！")


    return f"第一个关节的角度为{a1},第二个关节的角度为{a2},第三个关节的角度为{a3},第四个关节的角度为{a4},第五个关节的角度为{a5}",tip,frame

def function3(num):
    changeServeID(num)
    return "yes"


iface1 = gr.Interface(
    fn=complex_analysis,
    inputs=[
        gr.Slider(minimum=0, maximum=255,value=120, label="关节1"),
        gr.Slider(minimum=0, maximum=255,value=120, label="关节2"),
        gr.Slider(minimum=0, maximum=255,value=120, label="关节3"),
        gr.Slider(minimum=0, maximum=255,value=120, label="关节4"),
        gr.Slider(minimum=0, maximum=255,value=120, label="关节5"),
        gr.Image(sources="webcam", streaming=True),
    ],
    outputs=[
        "text",
        "image"
    ],
    title="机械臂控制",
    # live=True
)

iface2 = gr.Interface(
    function2,
    inputs=[
        gr.Textbox(value="./datasets",placeholder="请在这输入保存数据的位置...."),
        gr.Textbox(value="push",placeholder="请在这输入任务名称...."),
        gr.Radio(["train", "val", "test",None]),
        gr.Textbox(value="1",placeholder="请在这输入一组动作的名称...."),
        gr.Textbox(value="1",placeholder="请在这输入保存数据的名称...."),
        gr.Image(sources="webcam", streaming=True),
    ],
    outputs=[
        gr.Textbox(label="机械臂角度信息："),
        gr.Textbox(label="数据保存提示："),
        "image"
    ],
    # live=True
)

iface3 = gr.Interface(
    function3,
    inputs=[
        gr.Number(value=1,placeholder="请在这输入id...."),
    ],
    outputs=[
        gr.Textbox(label="修改信息："),
    ],
)

tabbed_interface = gr.TabbedInterface([iface1, iface2,iface3], ["控制机械臂", "读取数据并制作","修改舵机id"])

if __name__ == '__main__':
    tabbed_interface.launch()