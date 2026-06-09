# 待修复测试用例

## test_assets_generator_falkordb_names

**文件:** `tests/test_assets_generator.py:118`

**状态:** 预存失败，与 config 重构无关

**现象:** 测试期望生成 `names/应用1_APP1.md`，但该文件未被创建。mock 的 `query` 返回数据格式与 `AssetsGenerator` 实际处理逻辑不匹配。

**复现:**
```bash
uv run pytest tests/test_assets_generator.py::test_assets_generator_falkordb_names -v
```
