# -*- coding:utf-8 -*-

'''
 Author       : Xuexin
 Date         : 2024-05-07 10:32:43
 LastEditTime : 2024-05-07 18:15:43
 FilePath     : \\self_llm\\sft\\sft.py
 Description  : 
'''

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

"""
Optional
Optional，可选类型， Optional[X] 等价于 X | None （或 Union[X, None] ）。 意思是说这个参数可以为空或已经声明的类型。

但值得注意的是，这个并不等价于可选参数，当它作为参数类型注解的时候，不代表这个参数可以不传递了，而是说这个参数可以传为 None。
"""

import datasets
import evaluate
import torch
import transformers
from datasets import load_dataset
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
                          default_data_collator, set_seed)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


@dataclass
class DataTrainingArguments:
    """
    定义用于训练的和评估的数据参数
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "使用数据集的名称"}
        )
    dataset_config_name: Optional[str] = field(
        default="None", metadata={"help": "使用数据集的配置名称"}
    )
    train_file: Optional[str] = field(default=None, metadata = {"help": "训练数据集的路径(文本文件)"})

    validation_file: Optional[str] = field(
        default=None,
        metadata = {"help": "用于评估的数据集，评估困惑度。（文本文件）"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "调试使用的最大训练数据量。会在训练数据中取出指定数量的数据",
                "例如：训练集有1000条数据，max_train_samples=100，那么只取100条数据进行训练"
            )
        }
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": ("调试使用的最大评估数据量。会在评估数据中取出指定数量的数据",
                     "例如：评估集有1000条数据，max_eval_samples=100，那么只取100条数据进行评估")
        }
    )
    streaming: bool = field(default=False, metadata={"help": "是否使用流式输出"})
    block_size: Optional[int] = field(
        default=None,
        metadata={help: (
            "标记化后的可选输入序列长度。",
            "训练数据集将被截断为这个大小的块以进行训练。",
            "默认为单句输入的模型最大输入长度（考虑特殊标记）。"
        )}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "覆盖缓存的训练集和评估集，一个命令行参数，指示在缓存存在的情况下覆盖缓存。"}
    )
    validation_split_percentage:Optional[int] = field(
        default=5,
        metadata={"help": "如果不存在评估使用的验证集，从数据集中划分的比例"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "用于预处理的进程数。"},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "使用TXT文件时是否保留换行符."}
    )
