# docs/tasks/ — 任务进展目录约定

本目录存放"新工作模式"下各任务的过程进展，配合 [`docs/task-plan.json`](../task-plan.json) 使用。

- 每个任务一个目录，目录名与计划 JSON 中任务的 `dir` 字段一致（如 `T4-mysql-loader/`）。
- 每个任务目录至少包含两个文件：
  - `progress.md` — 过程记录：目标、改动文件清单、验证命令与结果、遗留问题。
  - `status.json` — 完成状态，格式：

    ```json
    {
      "task": "T4",
      "status": "done",
      "summary": "一句话结论",
      "files": ["src/govio/metadata/mysql_loader.py", "tests/test_mysql_loader.py"]
    }
    ```

    `status` 取值：`done` / `partial`（未完成但有产出）/ `failed`（被阻塞）。
- 计划 JSON（`docs/task-plan.json`）由编排侧的 recorder 步骤根据各任务 `status.json` 统一更新，任务执行者不要直接改计划 JSON。
