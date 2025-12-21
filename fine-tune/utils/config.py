from symtable import Class
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class DownloadSettings(BaseSettings):
    SAVE_DIR: str = Field(default="..\\..\\datasets", description="数据集保存位置")
    ROUTE_LIST: list[str] = Field(default=[], description="需要下载的模型ID列表（huggingface的id）")
    IF_JSON: bool = Field(default=True, description="是否需要下载json格式")
    IF_CSV: bool = Field(default=True, description="是否需要下载CSV格式")
    IF_ARROW: bool = Field(default=True, description="是否需要下载ARROW格式")
    FORMAT_PARAM: list[str] = Field(default=["arrow","json","csv"], description="数据集加载参数")

    model_config = ConfigDict(
        extra='allow',
        case_sensitive=False,
        env_prefix='',
        env_file='.env'
    )


class DataProcessor(BaseSettings):
    TMP: str = Field(default="")

class TrainingSettings(BaseSettings):
    TMP: str = Field(default="")


settingDownload = DownloadSettings()
settingDownload.ROUTE_LIST = [
        'keivalya/MedQuad-MedicalQnADataset',
        ('FreedomIntelligence/medical-o1-reasoning-SFT','en'), #问答数据集
        'GBaker/MedQA-USMLE-4-options', #QA选择题数据集
        ('qiaojin/PubMedQA', 'pqa_artificial', 'pqa_labled', 'pqa_unlabeld'), #QA问答数据集
        'Flmc/DISC-Med-SFT',
        'openlifescienceai/medmcqa',
        ('lavita/medical-qa-datasets', 'all-processed', 'chatdoctor-icliniq', 'chatdoctor_healthcaremagic') # 全套问答集并附有test数据集
    ]
settingProcessor = DataProcessor()