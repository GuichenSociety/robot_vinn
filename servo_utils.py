import time
import serial


uart = serial.Serial('COM3', 1000000)
uart.close()

def changeServeID(target_id):
    # 广播修改舵机id，因为是广播，需要一个一个编改舵机，多个一起会形成统一的ID
    mess = bytes([250, 175, 0, 205, 0, target_id, 0, 0, target_id+205, 237])
    uart.write(mess)

def servoState(id):
    mess = bytes([250, 175, id, 2, 0, 0, 0, 0, 2+id, 237])
    # mess = b'\xfa\xaf\x00\x02\x00\x00\x00\x02\xed'
    uart.write(mess)   # 广播查询舵机角度
    time.sleep(0.01)
    rev_data = uart.read(20)
    integer_array = [int(byte) for byte in rev_data]
    # print(rev_data,integer_array)
    id = integer_array[-8]
    target_angle = integer_array[-3]
    return id,target_angle

def servoToAngle(id,angle,t):
    """
    :param id:    设备id，0为广播所有
    :param angle: 取值0-240  , 120是在中间
    :param t:  运动时间，最大时间为255x20=5100ms，取值0-255，单位20ms，如1000ms 为 1000/20= 50
    :return:
    """
    # 0 是常锁
    mess = bytes([250, 175, id, 1, angle, int(t/20), 0, 0,id+1+angle+int(t/20), 237])
    uart.write(mess)   # 广播查询舵机角度
    time.sleep(0.2)
    rev_data = uart.read(10)
    print(rev_data)

