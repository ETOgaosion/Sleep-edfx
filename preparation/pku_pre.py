from mne.io import read_raw_edf
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py


# 1.解析xml
def parse_xml(xml_file):
    labels = []
    try:
        root = ET.parse(xml_file).getroot()
        for type_tag in root.iter('SleepStage'):
            label = type_tag.text
            labels.append(label)
    except Exception as e:
        print("error:",e)
    return labels


# 2.读取test文件下被试,并分割成30s的片段,采样频率需要指定;
def segment_eeg(data_file, label_len, time_interval=30, resample_sfreq=100):
    datasets = []
    raw_data = read_raw_edf(data_file,preload=True)# 定义要抽取的通道名称
    if int(raw_data.info['sfreq']) != resample_sfreq:
        print("Resample to {} Hz...".format(resample_sfreq))
        raw_data = raw_data.copy().resample(resample_sfreq, npad='auto')
    pick_chans = raw_data.ch_names[0:9]
    extract_raw_data = raw_data.pick_channels(pick_chans)
    print(extract_raw_data)
    extract_data, _ = extract_raw_data[:, :]
    print(extract_data.shape)
    print(pick_chans)
    i = 0
    start_time = 0
    while i < label_len:
        end_time = start_time + time_interval
        start_index, end_index = extract_raw_data.time_as_index([start_time, end_time])
        print("slice index:", start_index, end_index)
        slice_data, _ = extract_raw_data[:, start_index:end_index]
        print("slice_shape:", slice_data.shape)
        print("start-end-index", start_time, end_time, i)
        datasets.append(slice_data)
        start_time = end_time
        i += 1
        if end_time == start_time:
            assert("Error: the start index equal to end index...")
    return datasets


# 读取h5数据
def read_h5(file_name):
    with h5py.File(file_name, 'r') as f:
        data = f["data"][:]
        label = f["label"][:]
    return (data, label)


def save_h5(save_file_name, data, label):
    # 保存为h5格式
    with h5py.File(save_file_name, 'w') as f:
        f["data"] = data
        f["label"] = label


# 3.分割每个被试,并保存对应标签
# files = sorted(os.listdir("test"))
files = sorted(os.listdir("./pku_database"))
print(files)
for i in range(0, len(files), 2):

    sub = os.path.join("./pku_database", files[i].split(".")[0] + ".edf")
    sub_label = os.path.join("./pku_database", files[i].split(".")[0] + ".edf.XML")

    print("sub path:", sub, "sub label:", sub_label)
    save_file_name = os.path.basename(sub).split(".")[0] + ".h5"
    print("save file name:", save_file_name)

    ## 　处理对应xml标签
    label = parse_xml(sub_label)
    label = np.array(label).astype(int)
    label_len = len(label)
    print("label_len:", label_len)

    ## 　分割eeg数据
    data = segment_eeg(sub, label_len, resample_sfreq=100)
    data = np.array(data)

    ## 检查数据分割片段是否和对应标签一致
    if label_len != data.shape[1]:
        assert ("Error!!")

    # 保存为h5格式
    with h5py.File(save_file_name, 'w') as f:
        f["data"] = data
        f["label"] = label

    print("{} save successfully...".format(save_file_name))


def draw_eeg(datas, labels, index):
    channel_names = [u'E1-M2', u'E2-M2', u'F3-M2', u'F4-M1', u'C3-M2', u'C4-M1', u'O1-M2', u'O2-M1',
                     u'Chin 1-Chin 2']
    data = datas[index, ...]
    label = labels[index]
    fig = plt.figure(figsize=(30, 8 * data.shape[0]))
    for i, chan in enumerate(data):
        ax = plt.subplot(len(data), 1, 1 + i)
        ax.set_title("label:" + label)
        ax.set_xlabel(channel_names[i])
        ax.plot(chan.T)


def draw_sleep_stage(labels):
    fig = plt.figure(figsize=(30, 8))
    plt.plot(labels)




