import pandas as pd
import numpy as np
from datasets import load_dataset, load_from_disk, Dataset
from sympy.strategies.core import switch
from tqdm import tqdm
import os
from loguru import logger
from config import settingDownload, DownloadSettings
import sys


#使用MedQuad数据集
def load_download_dataset_huggingface(key_route, save_route):
    global dir
    global param

    if type(key_route) == str:
        dir = key_route.split("/")[-1]
        param = ''
    elif type(key_route) == tuple and len(key_route) >= 2:
        dir = key_route[0].split("/")[-1]
        param = key_route[1]
        key_route = key_route[0]

    save_route = "-".join([os.path.join(save_route,dir), param]) if param else os.path.join(save_route,dir)

    for format in settingDownload.FORMAT_PARAM:
        try:
            f_route = os.path.join(save_route, format) # type: ignore[misc]
            logger.info(f"start loading dataset--{dir} from disk route:{f_route}!")
            ds = check_load(format, f_route, dir) # type: ignore[misc]

        except FileNotFoundError as e:
            logger.warning(f"dataset--{dir} with formate:{format} not saved: {e}")
            try:
                #从huggingface拉取数据集
                if not param:
                    ds = load_dataset(key_route)
                else:
                    # 取第一个param
                    ds = load_dataset(key_route, param)
            except Exception as e:
                logger.error(f"dataset--{dir} load error: {e}")
            else:
                try:
                    if not os.path.exists(f_route):
                        os.makedirs(f_route, exist_ok=True)
                    if settingDownload.IF_ARROW and format == "arrow":
                        ds.save_to_disk(f_route) # type: ignore[misc]

                    if settingDownload.IF_CSV and format == "csv":
                        logger.info(f"start saving csv format dataset--{dir}")
                        arrow_to_csv(ds['train'], f_route, dir)

                    if settingDownload.IF_JSON and format == "json":
                        logger.info(f"start saving json format dataset--{dir}")
                        arrow_to_json(ds['train'], f_route, dir)

                except Exception as e:
                    logger.error(f"dataset--{dir} save error: {e}")
                else:
                    logger.info(f"dataset--{dir} load and save success!")
        else:
            logger.info(f"dataset--{dir} loading from disk success!")

    return ds


def arrow_to_csv(ds, f_route, file_name):
    file_name = file_name + ".csv"
    try:
        if not os.path.exists(f_route):
            os.makedirs(f_route, exist_ok=True)
        file_path = os.path.join(f_route, file_name)
        logger.info(f"checking file path existence {file_path}")
        logger.info(f"dataset--{dir} csv length ==== {len(ds.to_pandas())}")
        logger.info(f"dataset--{dir} arrow length ==== {len(ds)}")
        if not os.path.exists(file_path):
            ds.to_csv(file_path,
                      header=True,
                      sep=',',
                      encoding='utf-8',
                      num_proc=os.cpu_count()
                      )
        else:
            logger.info(f"dataset--{dir} csv format already exist route:{f_route}")
            return
    except Exception as e:
        logger.error(f"dataset--{dir} saving csv format route:{f_route} error: {e}")
        return
    else:
        logger.info(f"dataset--{dir} saving csv format route:{f_route} success!")
        return

def arrow_to_json(ds, f_route, file_name):
    file_name = file_name + ".jsonl"
    try:
        if not os.path.exists(f_route):
            os.makedirs(f_route, exist_ok=True)
        file_path = os.path.join(f_route, file_name)
        logger.info(f"checking file path existence {file_path}")
        if not os.path.exists(file_path):
            ds.to_json(file_path,
                       lines=True,
                       num_proc=os.cpu_count(),
                       force_ascii=False
                       )
        else:
            logger.info(f"dataset--{dir} json format already exist route:{f_route}")
            return
    except Exception as e:
        logger.error(f"dataset--{dir} saving json format route:{f_route} error: {e}")
        return
    else:
        logger.info(f"dataset--{dir} saving json format route:{f_route} success!")
        return

def check_load(format, f_route, file_name) -> Dataset:
    if format == "csv":
        f_route = os.path.join(f_route, file_name + ".csv")
        ds = load_dataset(format, data_files=f_route)
    elif format == "json":
        f_route = os.path.join(f_route, file_name + ".jsonl")
        ds = load_dataset(format, data_files=f_route)
    elif format == "arrow":
        ds = load_from_disk(f_route)
    return ds

if __name__ == '__main__':
    # logger.add(sys.stdout, level="INFO", serialize=True, colorize=True)
    # 加载数据
    for route in tqdm(settingDownload.ROUTE_LIST):
        ds = load_download_dataset_huggingface(route, settingDownload.SAVE_DIR)
    # 下载MedDialog数据集（CN）
    # import openxlab
    # openxlab.login(ak='0ymrgj9qwklnemzjlzvn',sk='gyzexlwyajxa72noypzy5e1nwo6d30mpk4lbq5n9')  # 进行登录，输入对应的AK/SK，可在个人中心添加AK/SK
    # from openxlab.dataset import info, get, download
    # info(dataset_repo='OpenDataLab/MedDialog')  # 数据集信息查看
    # get(dataset_repo='OpenDataLab/MedDialog', target_path='../datasets')  # 数据集下载


