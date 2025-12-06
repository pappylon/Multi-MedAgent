import pandas as pd
import numpy as np
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import os
import dataset
from twisted.scripts.htmlizer import header
from loguru import logger
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

    save_route = os.path.join(save_route,dir)
    try:
        logger.info("start loading dataset from disk")
        ds = load_from_disk(os.path.join(save_route,"origin")) # type: ignore[misc]
    except FileNotFoundError as e:
        logger.warning(f"dataset--{dir} not saved: {e}")
        try:
            #从huggingface拉取数据集
            if not param:
                ds = load_dataset(key_route)
            else:
                # 取第一个param
                ds = load_dataset(key_route, param)
                save_route = "-".join([save_route, param])
        except Exception as e:
            logger.error(f"dataset--{dir} load error: {e}")
        else:
            try:
                f_route = os.path.join(save_route,"origin")
                if not os.path.exists(f_route):
                    os.makedirs(save_route, exist_ok=True)

                ds.save_to_disk(f_route) # type: ignore[misc]
                logger.info(f"start saving csv format dataset--{dir}")
                arrow_to_csv(ds['train'], save_route, dir)
                logger.info(f"start saving json format dataset--{dir}")
                arrow_to_json(ds['train'], save_route, dir)
            except Exception as e:
                logger.error(f"dataset--{dir} save error: {e}")
            else:
                logger.info(f"dataset--{dir} load and save success!")
                return ds
    else:
        logger.info(f"dataset--{dir} loading from disk success!")
        return ds

def arrow_to_csv(ds, save_route, file_name):
    f_route = os.path.join(save_route, "csv")
    file_name = file_name + ".csv"
    try:
        if not os.path.exists(f_route):
            os.makedirs(f_route, exist_ok=True)
        if not os.path.exists(os.path.join(f_route, file_name)):
            ds.to_csv(os.path.join(f_route, file_name),
                      header=True,
                      sep=',',
                      encoding='utf-8',
                      num_proc=os.cpu_count()
                      )
        else:
            logger.info(f"dataset--{dir} csv format already exist")
            return
    except Exception as e:
        logger.error(f"dataset--{dir} saving csv format error: {e}")
    else:
        logger.info(f"dataset--{dir} saving csv format success!")

def arrow_to_json(ds, save_route, file_name):
    f_route = os.path.join(save_route, "json")
    file_name = file_name + ".json"
    try:
        if not os.path.exists(f_route):
            os.makedirs(f_route, exist_ok=True)
        if not os.path.exists(os.path.join(f_route, file_name)):
            ds.to_json(os.path.join(f_route, file_name),
                       lines=True,
                       num_proc=os.cpu_count()
                       )
        else:
            logger.info(f"dataset--{dir} json format already exist")
            return
    except Exception as e:
        logger.error(f"dataset--{dir} saving json format error: {e}")
    else:
        logger.info(f"dataset--{dir} saving json format success!")

# def data_preprocess():


if __name__ == '__main__':
    save_route = "..\\datasets"

    # 元组为route和对应param
    route_list = [
        'keivalya/MedQuad-MedicalQnADataset',
        ('FreedomIntelligence/medical-o1-reasoning-SFT','en'), #问答数据集
        'GBaker/MedQA-USMLE-4-options', #QA选择题数据集
        ('qiaojin/PubMedQA', 'pqa_artificial', 'pqa_labled', 'pqa_unlabeld'), #QA问答数据集
        'Flmc/DISC-Med-SFT',
        'openlifescienceai/medmcqa',
        ('lavita/medical-qa-datasets', 'all-processed', 'chatdoctor-icliniq', 'chatdoctor_healthcaremagic') # 全套问答集并附有test数据集
    ]

    # logger.add(sys.stdout, level="INFO", serialize=True, colorize=True)

    # 加载数据
    for route in tqdm(route_list):
        ds = load_download_dataset_huggingface(route, save_route)

    # 下载MedDialog数据集（CN）
    # import openxlab
    # openxlab.login(ak='0ymrgj9qwklnemzjlzvn',sk='gyzexlwyajxa72noypzy5e1nwo6d30mpk4lbq5n9')  # 进行登录，输入对应的AK/SK，可在个人中心添加AK/SK
    # from openxlab.dataset import info, get, download
    # info(dataset_repo='OpenDataLab/MedDialog')  # 数据集信息查看
    # get(dataset_repo='OpenDataLab/MedDialog', target_path='../datasets')  # 数据集下载

    # 对数据进行评估

    # 对fine-tune数据进行预处理

