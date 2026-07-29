# afmkit 开发团队

afmkit 由一个 agent 团队协作完成。所有 agent 共享同一个 git 仓库，按下表分工。

## 角色

| 角色 | 身份 | 责任 |
|---|---|---|
| **Lead / Orchestrator** | mavis（root session） | 规划 v0.1 scope、拆分任务、spawn 子 agent、合并成果、commit、监控 CI、汇报用户 |
| **Coder** | sub-agent `coder` | 按规格实现具体模块；写单元测试；通过本地的 ruff/mypy/pytest 验证后交付 |
| **Verifier** | sub-agent `verifier` | 独立 review 改动、跑全套测试、生成回归报告、给 Lead 红绿信号 |

CI 通过 GitHub Actions 跑（不在 agent 角色里），由 Lead 用 cron self-reminder 监控。

## 沟通契约

- **Lead → Coder**：给一段 task prompt，含：
  - 任务一句话目标
  - 必读文件（如 `src/afmkit/core/curve.py`）
  - 必须满足的接口（签名、Protocol）
  - 必须通过的本地检查（`ruff check`、`mypy`、`pytest tests/unit/...`）
  - 不允许触碰的文件清单
  - 期望产出（commit message 草案 + 改动文件列表）
- **Coder → Lead**：返回 commit 哈希 + 改动文件列表 + 本地测试结果
- **Lead → Verifier**：在 Coder 交付后，把 commit hash 给 Verifier
- **Verifier → Lead**：返回红绿信号 + 任何问题清单
- **Lead → 用户**：每个 phase 完成时同步一次；任何 scope 变更立刻同步

## Commit 规范

Conventional Commits，scope 字段标模块路径：

```
feat(io): add JPKTxtLoader for 4-column Nanowizzard exports
feat(models): add WLCModel (Marko-Siggia)
feat(fitting): add LmfitEngine + FitResult
test(io): add unit tests for JPKTxtLoader
docs(migration): expand Igor → afmkit mapping
chore(release): tag v0.1.0
```

每个模块一个 commit，**先写测试再写实现**（TDD），commit 里同时包含 tests 和 impl。
Verifier 通过后才由 Lead commit + push。**Coder 不直接 push**。

## 状态

- 当前 phase：**Phase 1 — Core & IO**
- 下一个 phase：Phase 2 — Models & Fitting
- 完整路线图：见 README.md 与用户沟通过的 5-phase plan
