"""列类型 -> (size, precision, scale) 的共享解析实现

三个元数据加载器的类型信息形态不同：

- MySQL ``information_schema.columns.COLUMN_TYPE`` 与 Trino ``SHOW COLUMNS``
  给出类型串（如 ``decimal(10,2)`` / ``varchar(64)``），走 :func:`parse_type_string`；
- PostgreSQL ``information_schema.columns`` 把长度与精度拆成独立属性列
  （``character_maximum_length`` / ``numeric_precision`` / ``numeric_scale``），
  走 :func:`parse_type_attributes`。

两个入口共用同一套三元组语义：

- ``size``：字符/位串类型的最大长度，其余类型为 0
- ``precision``：精确小数类型的总位数，其余为 0
- ``scale``：精确小数类型的小数位数，其余为 0

整型不计入 precision：MySQL ``int(11)`` 的括号参数是显示宽度、PostgreSQL 整型的
``numeric_precision`` 是位宽语义，都与 decimal 的精度语义不同，统一返回 (0, 0, 0)，
与 DuckDBLoader 的产出保持一致。
"""

import re
from typing import Any

import pandas as pd

# 匹配类型串：base 或 base(params)，如 decimal(10,2)、varchar(19)、bigint
_TYPE_RE = re.compile(r"^(\w+)\s*(?:\(([^)]*)\))?$")
# 数值类型的 unsigned / zerofill 后缀（如 decimal(10,2) unsigned、int(10) unsigned zerofill），
# 解析参数前剥离，否则整串不匹配会导致 precision/scale 静默丢失
_MODIFIER_RE = re.compile(r"(?:\s+(?:unsigned|zerofill))+$", re.IGNORECASE)

# 串式语法下带长度参数的字符类型（MySQL COLUMN_TYPE / Trino SHOW COLUMNS）
STRING_ARG_TYPES = frozenset({"char", "varchar"})
# 串式语法下带 (p[,s]) 参数的精确小数类型
DECIMAL_ARG_TYPES = frozenset({"decimal"})
# 属性式语法下长度取 character_maximum_length 的类型（PostgreSQL data_type）
LENGTH_ATTRIBUTE_TYPES = frozenset(
    {"bit", "bit varying", "character", "character varying"}
)
# 属性式语法下精度取 numeric_precision / numeric_scale 的类型
DECIMAL_ATTRIBUTE_TYPES = frozenset({"decimal", "numeric"})


def _int_or_zero(value: Any) -> int:
    """把 information_schema 属性值安全转为 int

    判空统一用 ``pd.notna``：``_fetch`` 经 pd.DataFrame 承接整型 + NULL 混合列时
    会被推断为 float64，SQL NULL 变成 NaN（既非 None 也非 falsy），
    用 ``is not None`` 或真值判断会在 int(NaN) 上抛 ValueError。

    Args:
        value: 属性列取值（int / float / NaN / None 均可能出现）

    Returns:
        int: 取值本身；缺失（None / NaN）时为 0
    """
    return int(value) if pd.notna(value) else 0


def parse_type_string(
    type_str: str, *, strip_numeric_modifiers: bool = False
) -> tuple[int, int, int]:
    """从类型字符串解析 (size, precision, scale)

    ``decimal(p,s)`` -> (0, p, s)；``decimal(p)`` -> (0, p, 0)
    ``varchar(n)`` / ``char(n)`` -> (n, 0, 0)；无参 ``varchar`` -> (0, 0, 0)
    其余类型与不匹配的类型串 -> (0, 0, 0)

    Args:
        type_str: 类型字符串，如 ``decimal(10,2)``
        strip_numeric_modifiers: 是否先剥离 ``unsigned`` / ``zerofill`` 后缀
            （MySQL COLUMN_TYPE 需要；Trino 类型串不出现这类后缀）

    Returns:
        tuple[int, int, int]: (size, precision, scale)
    """
    candidate = type_str.strip()
    if strip_numeric_modifiers:
        candidate = _MODIFIER_RE.sub("", candidate)

    match = _TYPE_RE.match(candidate)
    if not match:
        return 0, 0, 0
    base = match.group(1).lower()
    params = match.group(2)
    if params is None:
        return 0, 0, 0

    nums = [int(part.strip()) for part in params.split(",") if part.strip().isdigit()]
    if base in DECIMAL_ARG_TYPES:
        if len(nums) >= 2:
            return 0, nums[0], nums[1]
        if len(nums) == 1:
            return 0, nums[0], 0
        return 0, 0, 0
    if base in STRING_ARG_TYPES:
        return (nums[0] if nums else 0), 0, 0
    return 0, 0, 0


def parse_type_attributes(
    data_type: str,
    character_maximum_length: float | None,
    numeric_precision: float | None,
    numeric_scale: float | None,
) -> tuple[int, int, int]:
    """从 information_schema 类型属性解析 (size, precision, scale)

    ``character varying`` / ``character`` / ``bit`` / ``bit varying`` ->
    (character_maximum_length, 0, 0)；``numeric`` / ``decimal`` ->
    (0, numeric_precision, numeric_scale)；其余类型 -> (0, 0, 0)。

    Args:
        data_type: information_schema.columns.data_type 类型名
        character_maximum_length: 字符/位串最大长度（无则为 None/NaN）
        numeric_precision: 数值精度（无则为 None/NaN）
        numeric_scale: 数值小数位（无则为 None/NaN）

    Returns:
        tuple[int, int, int]: (size, precision, scale)
    """
    base = data_type.strip().lower()
    if base in LENGTH_ATTRIBUTE_TYPES:
        return _int_or_zero(character_maximum_length), 0, 0
    if base in DECIMAL_ATTRIBUTE_TYPES:
        return 0, _int_or_zero(numeric_precision), _int_or_zero(numeric_scale)
    return 0, 0, 0
