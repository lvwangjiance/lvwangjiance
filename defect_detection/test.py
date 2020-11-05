# -*- coding: utf-8 -*-
import os
import shutil
import tensorflow as tf
import numpy as np
import time
import pyttsx3
import threading

import socket
import sys
import struct

from load_data import *
from model import *
import matplotlib.pyplot as plt
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import chardet
import codecs

lock = threading.Lock()
start = time.time()


class MyDirEventHandler(FileSystemEventHandler):
    global global_iii

    def on_moved(self, event):
        print(event)
        eval()

    def on_created(self, event):
        print(event)

    def on_deleted(self, event):
        print(event)

    def on_modified(self, event):
        print("modified:", event)
        eval()


def socket_service_image():
    global event1, event2, answer
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 设置成非阻塞
        # s.setblocking(False)
        # s.bind(('192.168.43.180', 1902))
        #        s.bind(('192.168.226.1', 1900))
        s.bind(('192.168.43.180', 1904))
        s.listen(10)
    except socket.error as msg:
        print(msg)
        sys.exit(1)

    print("Wait for Connection.....................")

    while True:
        sock, addr = s.accept()  # addr是一个元组(ip,port)
        print("已建立连接")
        deal_image(sock, addr)


'''
def deal_image(sock, addr):
    global event1,event2,answer
    print("Accept connection from {0}".format(addr))  # 查看发送端的ip和端口
    filename = "D:\\360Download\\guoshushibie\\data\\receive\\corn.1.jpg" #接收到的图片写入的路径

#    filename= "D:\\360Download\\guoshushibie\\data\\receive\\"
#    filename0="cuke.1"
#    filename += filename0 + '.jpg'

    while True:
        data = sock.recv(1024)
        if data:
            try:
                myfile = open(filename,'wb')
                print("%s 文件打开成功" % filename)
            except IOError:
                print("%s 文件打开失败，该文件不存在" % filename)
            myfile.write(data)



        while True:
            data=sock.recv(1024)
            if not data:
                myfile.close()
                break
            myfile.write(data)

        #myfile.close()    

            ###识别结果
            #event1.set()

        event2.set()#唤醒图像识别
        print("5",event1.isSet())
        print("6",event2.isSet())
        event1.wait()#睡眠自己
        #time.sleep(1)
        print("7",event1.isSet())
        print("8",event2.isSet())
        #print("test########################")
        print("输出结果为:",answer)
        send_data = answer

        sock.send(send_data.encode("gbk"))              ##############这边是接收到图片，后发出数据到电子秤
       # sock.shutdown()
        event1.clear()#变成False
        print("9",event1.isSet())
        print("10",event2.isSet())
        '''


def deal_image(sock, addr):
    global event1, event2, answer
    print("Accept connection from {0}".format(addr))  # 查看发送端的ip和端口
    filename = "D:\\360Download\\guoshushibie\\data\\receive\\corn.1.jpg"  # 接收到的图片写入的路径

    #    filename= "D:\\360Download\\guoshushibie\\data\\receive\\"
    #    filename0="cuke.1"
    #    filename += filename0 + '.jpg'

    while True:
        # try:
        # data = sock.recv(4096)
        datahead = sock.recv(5)

        # codeType = chardet.detect(datahead)["encoding"]  #检测编码方式
        # print(u"编码是 ", codeType)
        # size=datahead.decode('utf-8','replace')
        print(datahead)
        size = datahead.decode()
        if size == '':
            break
        size_int = int(size)
        print(size_int)
        # size = size[:5]

        # size_int=int(size)

        # size=datahead.decode()
        # size_int=int(size)
        # datahead = int(datahead.decode())
        # print(datahead)
        # datahead.decode()
        # print(datahead.type())
        # datahead1=str(datahead)
        # datahead2=int(datahead1)
        # print(datahead2)

        # txt = str(data)
        # print(txt)
        inital = 0
        myfile = open(filename, 'wb')
        print("%s 文件打开成功" % filename)
        while (inital != size_int):
            data = sock.recv(1024)
            myfile.write(data)
            inital = inital + len(data)
            # print(inital)
        myfile.close()

        event2.set()  # 唤醒图像识别
        print("5", event1.isSet())
        print("6", event2.isSet())
        event1.wait()  # 睡眠自己
        # time.sleep(1)
        print("7", event1.isSet())
        print("8", event2.isSet())
        # print("test########################")
        print("输出结果为:", answer)
        send_data = answer

        sock.send(send_data.encode("gbk"))  ##############这边是接收到图片，后发出数据到电子秤
        # sock.shutdown()
        event1.clear()  # 变成False
        print("9", event1.isSet())
        print("10", event2.isSet())
        # except:
        # sock.close()
        # continue


# 测试检查点
def eval():
    global event1, event2, answer
    print("waiting for socket")
    #    print(socket.gethostbyname(socket.gethostname()))
    while True:
        # print("waiting for socket222")
        event2.wait()  # 睡眠自己
        # time.sleep(1)
        print("开始调用")

        tf.reset_default_graph()
        N_CLASSES = 3
        IMG_SIZE = 208
        BATCH_SIZE = 1
        CAPACITY = 200
        MAX_STEP = 1

        test_dir = '/Users/heyiyuan/Desktop/defect_detection/test'
        logs_dir = 'logs_1'  # 检查点目录
        path = test_dir
        sess = tf.Session()

        i = 1
        # 对目录下的文件进行遍历
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)) == True:
                # 设置新文件名
                new_name = file.replace(file, "corn.%d.jpg" % i)
            # 重命名
            os.rename(os.path.join(path, file), os.path.join(path, new_name))
            i += 1
        # 结束

        train_list = get_all_files(test_dir, is_random=True)
        image_train_batch, label_train_batch = get_batch(train_list, IMG_SIZE, BATCH_SIZE, CAPACITY, True)
        train_logits = inference(image_train_batch, N_CLASSES)
        train_logits = tf.nn.softmax(train_logits)  # 用softmax转化为百分比数值

        # 载入检查点
        saver = tf.train.Saver()
        print('\n载入检查点...')
        ckpt = tf.train.get_checkpoint_state(logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('载入成功，global_step = %s\n' % global_step)
        else:
            print('没有找到检查点')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for step in range(MAX_STEP):
                if coord.should_stop():
                    break

                image, prediction = sess.run([image_train_batch, train_logits])
                max_index = np.argmax(prediction)

                #            data=open("D:\\360Download\\guoshushibie\\data\\data.txt",'a')

                if max_index == 0:
                    #                print ('%.2f%% is a cuke.' % (prediction[0][0] * 100))
                    #                data=open("D:\\360Download\\guoshushibie\\data\\data.txt",'w+')
                    #                print('cuke',file=data)
                    answer = "corn"
                    print(answer)
                    plt.imshow(image[0])
                    plt.show()
                #                time.sleep(3)
                #                break
                #                engine=pyttsx3.init()
                #                voice=engine.getProperty('voice')
                #                voices=engine.getProperty('voices')
                #                for item in voices:
                #                    print(item.id,item.languages)
                #                engine.setProperty('voice','zh')
                #               engine.say('黄瓜 单价是 三块五一斤。The unit price of cucumber is three pieces per catty.')
                #              engine.runAndWait()

                elif max_index == 1:
                    #                print ( '%.2f%% is a bittergourd.' % (prediction[0][1] * 100))
                    #               print('grape')
                    # data=open("D:\\360安全浏览器下载\\果蔬识别\\data\\data.txt",'w+')
                    #                print('bittergourd',file=data)
                    answer = "cucumber"
                    print(answer)

                    #                engine=pyttsx3.init()
                    #                voice=engine.getProperty('voice')
                    #                voices=engine.getProperty('voices')
                    #                for item in voices:
                    #                    print(item.id,item.languages)
                    #                engine.setProperty('voice','zh')
                    #                engine.say('我的天哪！苦 瓜 今日 特价 打八折 单价是 十三块五一斤。  Oh my god! Bitter melon today special price, hit twenty per cent off, the unit price is thirteen yuan per catty')
                    #                engine.runAndWait()
                    plt.imshow(image[0])
                    plt.show()

                elif max_index == 2:
                    #                print ('%.2f%% is a tomato.' % (prediction[0][2] * 100))
                    #               print('tomato')
                    # data=open("D:\\360安全浏览器下载\\果蔬识别\\data\\data.txt",'w+')
                    #                print('tomato',file=data)
                    answer = "orange"
                    print(answer)
                    # data.close()
                    #                engine=pyttsx3.init()
                    #                voice=engine.getProperty('voice')
                    #                voices=engine.getProperty('voices')
                    #                for item in voices:
                    #                    print(item.id,item.languages)
                    #                engine.setProperty('voice','zh')
                    #                engine.say('我的天哪！番 茄 今日 特价 打九折 单价是 六块五一斤 Oh my god!Tomato today special price, ten per cent off, the unit price is Six five per catty')
                    #                engine.runAndWait()
                    plt.imshow(image[0])
                    plt.show()

        except tf.errors.OutOfRangeError:
            print('Done.')
        finally:
            coord.request_stop()

        coord.join(threads=threads)
        # 删除文件
        filelist = []  # 选取删除文件夹的路径,最终结果删除img文件夹
        filelist = os.listdir(test_dir)  # 列出该目录下的所有文件名
        for f in filelist:
            filepath = os.path.join(test_dir, f)  # 将文件名映射成绝对路劲
            if os.path.isfile(filepath):  # 判断该文件是否为文件或者文件夹
                os.remove(filepath)  # 若为文件，则直接删除
                print(str(filepath) + " removed!")
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath, True)  # 若为文件夹，则删除该文件夹及文件夹内所有文件
                print("dir " + str(filepath) + " removed!")
        tf.reset_default_graph()

        sess.close()
        print("结束eval函数")
        print("answer:", answer)
        print("11", event1.isSet())
        print("12", event2.isSet())
        event2.clear()
        print("1", event1.isSet())
        print("2", event2.isSet())
        print("**********************************")
        while True:
            time.sleep(0.1)
            if event1.isSet() == False:
                event1.set()
                break
        # event1.set()
        print("3", event1.isSet())
        print("4", event2.isSet())


#        print("clear event")


if __name__ == '__main__':
    # for i1 in range(0,200):
    # while(1):
    # eval()
    # time.sleep(1)
    event1 = threading.Event()
    event2 = threading.Event()

    answer = "none"

    test_dir = '/Users/heyiyuan/Desktop/defect_detection/test'
    logs_dir = 'logs_1'  # 检查点目录
    path = test_dir
    #    eval()
    #    print(answer)

    t1 = threading.Thread(target=socket_service_image, args=())
    t2 = threading.Thread(target=eval, args=())
    t2.start()
    t1.start()
'''    
    for mmm in range(1000):
        break_flag=0
        for i in range(1000):      
            #监听from文件
            work_path = 'D:\\360Download\\guoshushibie\\data\\from'
            if os.listdir(work_path):
                print( '目录为有')
                time.sleep(1)
                f=open('D:\\360Download\\guoshushibie\\data\\data.txt', "r+")
                f.truncate()
                os.remove(r'D:\\360Download\\guoshushibie\\data\\from\\from.txt')
            for file in os.listdir(path):
                if os.path.isfile(os.path.join(path,file))==False:                
                    time.sleep(1)
                if os.path.isfile(os.path.join(path,file))==True:                
                    break_flag=1
                    break
            if(break_flag==1):
                break
            time.sleep(1)
'''



