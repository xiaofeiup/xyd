"""
模型分数数据库模块

提供本地 SQLite 数据库的初始化与模型分数写入功能。

表结构参考建模样本 CSV 的表头，去掉了 pric_type_code 和 pric_type_desc
两列。写入采用按 test_id 去重的 upsert 方式：已存在的 test_id 更新其分数列，
不存在则插入新行。
"""

import os
import math
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# 默认数据库文件路径
DEFAULT_DB_PATH = os.path.join(
    os.path.expanduser('~'),
    'Job', 'Job_xyd', '数据', '总', 'model_scores.db'
)

# 默认表名
DEFAULT_TABLE_NAME = 'model_scores'

# 主键列
PRIMARY_KEY = 'test_id'

# 已从表头中剔除的列（定价类型相关）
EXCLUDED_COLUMNS = ['pric_type_code', 'pric_type_desc']

# 默认建立索引的列（test_id 已是主键自带索引，无需重复）
DEFAULT_INDEX_COLUMNS = ['high_sample_type', 'mid_sample_type', 'partner_code']

# 大文件导入时默认的分块行数
DEFAULT_CHUNKSIZE = 50000

# 表结构定义：列名 -> SQLite 类型
# 参考 CSV 表头顺序，去掉 pric_type_code / pric_type_desc
SCORE_TABLE_SCHEMA: Dict[str, str] = {
    'test_id': 'INTEGER PRIMARY KEY',
    'id_number': 'TEXT',
    'mobile': 'TEXT',
    'apply_date': 'INTEGER',
    'partner_code': 'TEXT',
    'flag': 'REAL',
    'enc_method': 'TEXT',
    'partner_name': 'TEXT',
    '蜜蜂分_中利率版2_2_proba': 'REAL',
    '蜜蜂分_中利率版2_2': 'REAL',
    '蜜蜂分_中利率版6_2_proba': 'REAL',
    '蜜蜂分_中利率版6_2': 'REAL',
    '蜜蜂分_中利率版7_2_proba': 'REAL',
    '蜜蜂分_中利率版7_2': 'REAL',
    '蜜蜂分_高利率版5_2_proba': 'REAL',
    '蜜蜂分_高利率版5_2': 'REAL',
    '蜜蜂分_高利率版6_2_proba': 'REAL',
    '蜜蜂分_高利率版6_2': 'REAL',
    '蜜蜂分_高利率版7_2_proba': 'REAL',
    '蜜蜂分_高利率版7_2': 'REAL',
    'high_sample_type': 'TEXT',
    'mid_sample_type': 'TEXT',
    '1-阿里反欺诈v2标品': 'REAL',
    '蜜蜂分_中利率版6_3_proba': 'REAL',
    'b058_bh_prob': 'REAL',
    'c031_bh_prob': 'REAL',
    'hl_yuxin_score_v3_a': 'REAL',
    'hl_yuxin_score_v3_b': 'REAL',
    'hl_yuxin_score_v3_c': 'REAL',
    '蜜蜂分_中利率版8_1_proba_a073': 'REAL',
    '蜜蜂分_中利率版8_2_proba_a074': 'REAL',
    '蜜蜂分_中利率版8_3_proba_a075': 'REAL',
    'hl_jianpu_score_v1': 'REAL',
}


def init_score_db(db_path: str = DEFAULT_DB_PATH,
                  table_name: str = DEFAULT_TABLE_NAME) -> str:
    """
    初始化模型分数数据库

    创建数据库文件（如不存在）及分数表，表结构由 SCORE_TABLE_SCHEMA 定义。

    Parameters:
    -----------
    db_path : str, default=DEFAULT_DB_PATH
        SQLite 数据库文件路径
    table_name : str, default=DEFAULT_TABLE_NAME
        分数表名

    Returns:
    --------
    db_path : str
        数据库文件路径
    """
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    columns_sql = ',\n    '.join(
        f'"{col}" {col_type}' for col, col_type in SCORE_TABLE_SCHEMA.items()
    )
    create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" (\n    {columns_sql}\n)'

    with sqlite3.connect(db_path) as conn:
        conn.execute(create_sql)
        conn.commit()

    print(f"模型分数数据库已就绪: {db_path} (表: {table_name})")
    return db_path


def ensure_indexes(db_path: str = DEFAULT_DB_PATH,
                  table_name: str = DEFAULT_TABLE_NAME,
                  columns: Optional[List[str]] = None) -> List[str]:
    """
    为常用查询列建立索引

    test_id 已是主键自带索引，无需重复。默认对 DEFAULT_INDEX_COLUMNS 中的
    列建索引以加快常用维度筛选的查询。

    Parameters:
    -----------
    db_path : str, default=DEFAULT_DB_PATH
        SQLite 数据库文件路径
    table_name : str, default=DEFAULT_TABLE_NAME
        分数表名
    columns : list, optional
        要建索引的列，为 None 时使用 DEFAULT_INDEX_COLUMNS

    Returns:
    --------
    created : list
        实际建立索引的列名
    """
    if columns is None:
        columns = DEFAULT_INDEX_COLUMNS

    created = []
    with sqlite3.connect(db_path) as conn:
        for col in columns:
            if col not in SCORE_TABLE_SCHEMA:
                print(f"列 {col} 不在表结构中，跳过建索引")
                continue
            index_name = f'idx_{table_name}_{col}'
            conn.execute(
                f'CREATE INDEX IF NOT EXISTS "{index_name}" '
                f'ON "{table_name}" ("{col}")'
            )
            created.append(col)
        conn.commit()

    if created:
        print(f"已为以下列建立索引: {created}")
    return created


def _to_sqlite_value(value: object) -> object:
    """
    将单个值转换为 SQLite 可绑定的原生类型

    处理 numpy 标量与缺失值（NaN/NaT/None 统一转为 None）。

    Parameters:
    -----------
    value : object
        原始值

    Returns:
    --------
    converted : object
        转换后的值
    """
    if value is None:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and math.isnan(value):
        return None

    if value is pd.NaT:
        return None

    return value


def _get_table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    """
    读取表中实际存在的列名（按建表顺序）

    Parameters:
    -----------
    conn : sqlite3.Connection
        数据库连接
    table_name : str
        表名

    Returns:
    --------
    columns : list
        列名列表
    """
    rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    return [row[1] for row in rows]


def write_model_scores(data: pd.DataFrame,
                      db_path: str = DEFAULT_DB_PATH,
                      table_name: str = DEFAULT_TABLE_NAME,
                      init_if_missing: bool = True,
                      add_new_columns: bool = True) -> int:
    """
    写入模型分数（按 test_id 去重更新）

    自动剔除 pric_type_code / pric_type_desc 列。按表中实际存在的列进行匹配，
    已存在的 test_id 会更新其分数列，不存在则插入新行。当数据中出现表里没有
    的新分数列时，默认自动 ALTER TABLE 加列（REAL 类型）后再写入。

    Parameters:
    -----------
    data : pd.DataFrame
        待写入的分数数据，必须包含主键列 test_id
    db_path : str, default=DEFAULT_DB_PATH
        SQLite 数据库文件路径
    table_name : str, default=DEFAULT_TABLE_NAME
        分数表名
    init_if_missing : bool, default=True
        表不存在时是否自动初始化
    add_new_columns : bool, default=True
        遇到表中不存在的新列时是否自动加列；为 False 时忽略并提示

    Returns:
    --------
    n_rows : int
        写入（插入或更新）的行数
    """
    if PRIMARY_KEY not in data.columns:
        raise ValueError(f"数据中缺少主键列: {PRIMARY_KEY}")

    if init_if_missing:
        init_score_db(db_path, table_name)

    # 剔除定价类型列
    write_data = data.drop(columns=EXCLUDED_COLUMNS, errors='ignore')

    with sqlite3.connect(db_path) as conn:
        existing_columns = set(_get_table_columns(conn, table_name))

        # 数据中表里没有的列
        new_columns = [
            col for col in write_data.columns if col not in existing_columns
        ]

        if add_new_columns and new_columns:
            for col in new_columns:
                conn.execute(
                    f'ALTER TABLE "{table_name}" ADD COLUMN "{col}" REAL'
                )
            conn.commit()
            existing_columns.update(new_columns)
            print(f"已新增分数列: {new_columns}")
        elif new_columns:
            print(f"以下列不在表结构中，已忽略: {new_columns}")

        # 仅保留表中实际存在的列
        columns = [col for col in write_data.columns if col in existing_columns]

        if PRIMARY_KEY not in columns:
            raise ValueError(f"主键列 {PRIMARY_KEY} 不在表结构中")

        write_data = write_data[columns]

        # 组装 upsert 语句
        quoted_cols = ', '.join(f'"{col}"' for col in columns)
        placeholders = ', '.join('?' for _ in columns)

        update_cols = [col for col in columns if col != PRIMARY_KEY]
        if update_cols:
            update_clause = ', '.join(
                f'"{col}" = excluded."{col}"' for col in update_cols
            )
            conflict_sql = f'DO UPDATE SET {update_clause}'
        else:
            conflict_sql = 'DO NOTHING'

        insert_sql = (
            f'INSERT INTO "{table_name}" ({quoted_cols}) VALUES ({placeholders}) '
            f'ON CONFLICT("{PRIMARY_KEY}") {conflict_sql}'
        )

        records = [
            tuple(_to_sqlite_value(v) for v in row)
            for row in write_data.itertuples(index=False, name=None)
        ]

        conn.executemany(insert_sql, records)
        conn.commit()

    print(f"已写入 {len(records)} 行模型分数到 {table_name}")
    return len(records)


def import_csv_to_db(csv_path: str,
                    db_path: str = DEFAULT_DB_PATH,
                    table_name: str = DEFAULT_TABLE_NAME,
                    encoding: str = 'utf-8',
                    chunksize: Optional[int] = DEFAULT_CHUNKSIZE,
                    build_indexes: bool = True) -> int:
    """
    将建模样本 CSV 导入数据库（初始回填）

    读取 CSV 后按 test_id 去重写入，自动剔除 pric_type_code / pric_type_desc。
    适合首次用原始 CSV 做全量回填；重复导入同一 CSV 不会产生重复行。

    Parameters:
    -----------
    csv_path : str
        建模样本 CSV 文件路径
    db_path : str, default=DEFAULT_DB_PATH
        SQLite 数据库文件路径
    table_name : str, default=DEFAULT_TABLE_NAME
        分数表名
    encoding : str, default='utf-8'
        CSV 文件编码
    chunksize : int, optional, default=DEFAULT_CHUNKSIZE
        分块读取的行数，默认分块以降低大文件的内存占用；传 None 可一次性读入
    build_indexes : bool, default=True
        回填后是否对常用查询列建立索引

    Returns:
    --------
    n_rows : int
        写入（插入或更新）的总行数
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    init_score_db(db_path, table_name)

    total = 0
    if chunksize:
        for chunk in pd.read_csv(csv_path, encoding=encoding, chunksize=chunksize):
            total += write_model_scores(
                chunk, db_path=db_path, table_name=table_name, init_if_missing=False
            )
    else:
        data = pd.read_csv(csv_path, encoding=encoding)
        total = write_model_scores(
            data, db_path=db_path, table_name=table_name, init_if_missing=False
        )

    if build_indexes:
        ensure_indexes(db_path, table_name)

    print(f"CSV 回填完成: {csv_path} -> {table_name}，共 {total} 行")
    return total


def read_model_scores(db_path: str = DEFAULT_DB_PATH,
                     table_name: str = DEFAULT_TABLE_NAME,
                     test_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """
    读取模型分数

    Parameters:
    -----------
    db_path : str, default=DEFAULT_DB_PATH
        SQLite 数据库文件路径
    table_name : str, default=DEFAULT_TABLE_NAME
        分数表名
    test_ids : list, optional
        指定要读取的 test_id 列表，为 None 时读取全部

    Returns:
    --------
    data : pd.DataFrame
        分数数据
    """
    with sqlite3.connect(db_path) as conn:
        if test_ids:
            placeholders = ', '.join('?' for _ in test_ids)
            query = f'SELECT * FROM "{table_name}" WHERE "{PRIMARY_KEY}" IN ({placeholders})'
            data = pd.read_sql_query(query, conn, params=list(test_ids))
        else:
            data = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)

    return data
