"""数据源密码加密工具

使用 Fernet 对称加密保护配置文件中的数据库密码。
密钥存储在 ~/.govio/.key，仅当前用户可读。
"""

import logging
import os
import re
import stat
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

# 默认密钥路径
DEFAULT_KEY_PATH = Path.home() / ".govio" / ".key"

# URL 中密码的正则：匹配 scheme://user:password@host 部分
# 密码可能包含 @，因此使用贪婪匹配到最后一个 @
_URL_PASSWORD_RE = re.compile(
    r"^([^:]+://[^:]+:)(.*)(@[^@]+)$"
)


def get_or_create_key(key_path: Path | None = None) -> bytes:
    """获取或生成 Fernet 密钥

    Args:
        key_path: 密钥文件路径，默认 ~/.govio/.key

    Returns:
        bytes: Fernet 密钥
    """
    key_path = key_path or DEFAULT_KEY_PATH

    if key_path.exists():
        return key_path.read_bytes().strip()

    # 生成新密钥
    key = Fernet.generate_key()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(key)

    # 仅当前用户可读写 (0o600)
    os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)

    logger.info("已生成加密密钥: %s", key_path)
    return key


def encrypt_value(plaintext: str, key_path: Path | None = None) -> str:
    """加密字符串

    Args:
        plaintext: 明文
        key_path: 密钥文件路径

    Returns:
        str: Fernet 加密后的字符串
    """
    key = get_or_create_key(key_path)
    f = Fernet(key)
    return f.encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_value(ciphertext: str, key_path: Path | None = None) -> str:
    """解密字符串

    Args:
        ciphertext: Fernet 密文
        key_path: 密钥文件路径

    Returns:
        str: 解密后的明文

    Raises:
        ValueError: 密文无效或密钥不匹配
    """
    key = get_or_create_key(key_path)
    f = Fernet(key)
    try:
        return f.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        raise ValueError("解密失败：密文无效或密钥不匹配")


def parse_password_from_url(url: str) -> tuple[str, str | None]:
    """从 URL 中解析出密码

    Args:
        url: 数据库连接 URL，如 mysql+pymysql://user:pass@host/db

    Returns:
        tuple: (脱敏URL, 密码)。无密码时返回 (原URL, None)
    """
    match = _URL_PASSWORD_RE.match(url)
    if not match:
        return url, None

    prefix, password, suffix = match.group(1), match.group(2), match.group(3)
    masked_url = f"{prefix}***{suffix}"
    return masked_url, password


def reconstruct_url(masked_url: str, password: str) -> str:
    """将密码还原回脱敏 URL

    Args:
        masked_url: 脱敏后的 URL（密码位置为 ***）
        password: 原始密码

    Returns:
        str: 完整 URL
    """
    match = _URL_PASSWORD_RE.match(masked_url)
    if not match:
        return masked_url

    prefix, _, suffix = match.group(1), match.group(2), match.group(3)
    return f"{prefix}{password}{suffix}"


def is_encrypted(value: str) -> bool:
    """判断字符串是否为 Fernet 密文

    Args:
        value: 待检测字符串

    Returns:
        bool: 是否为合法的 Fernet 密文格式
    """
    if not value:
        return False
    try:
        return value.startswith("gAAAAA") and len(value) > 50
    except Exception:
        return False
