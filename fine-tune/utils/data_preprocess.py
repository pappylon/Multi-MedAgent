from __future__ import annotations

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from loguru import logger
import pandas as pd
import re


class MedDataProcessor:
    """
    暂时只对文字数据做清理，之后再完善
    """
    def __init__(self,
                 data_dir:str = "",
                 ) -> None:
        self.data_dir = data_dir


    def basic_clean(self, _string: str) -> str:
        """
        clean symbols
        :param _string:
        :return:
        """
        flag = 1
        _string = _string.strip()
        if re.search(r'[\t\n]',_string):
            print("old-----" + _string)
            flag = 0

        re.sub(r'[\t\n]', '', _string, flags=re.I)
        if flag == 0:
            print("new------" + _string)
        re.sub(r'[\#\^\&\*\=\+\~\{\}\/]', '', _string, flags=re.I)
        return _string


if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.unicode.ambiguous_as_wide', True)
    # pd.set_option('display.unicode.east_asian_width', True)
    pd.set_option('display.max_colwidth', 200)
    # pd.set_option('expand_frame_repr', True)
    # pd.set_option('display.width', 20)
    SAVE_DIR = '../../datasets/MedQuad-MedicalQnADataset/json/MedQuad-MedicalQnADataset.jsonl'
    df = pd.read_json(SAVE_DIR, lines=True, orient='records')
    # data = df.to_dict(orient='records')
    illegal_char_pattern = r'[^a-zA-Z0-9\s.,?!]'
    processor = MedDataProcessor()

    count = 0
    df.dropna(axis=0,how='any',inplace=True)
    for i in tqdm(range(len(df))):
        if df.iloc[i].isnull().values.any():
            continue

        for j in range(len(df.iloc[i])):
            if type(df.iloc[i, j])==str:
                df.iloc[i, j] = processor.basic_clean(df.iloc[i, j])
            else:
                pass

    df.to_json(SAVE_DIR,
               lines=True,
               orient='records',
               force_ascii=False,
               indent=4
               )

