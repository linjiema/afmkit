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

- **当前 release：v0.5.0**（2026-07-31）— `to_mat` / `to_parquet` 接 fits+reviewers，`.ibw` note 全 rehydrate + `roundtrip_ibw` helper
- **当前 develop HEAD：v0.5.0.dev0**（v0.5 切完后的下个 cycle 起点）
- 下一个 phase：v0.5.1（修 + 增量），或者直接进 v0.6 候选（`.ibw` v5 stdlib-only reader / matplotlib TUI native image / FJC 2-col round-trip helper / 等）
- 完整路线图：见 [`CHANGELOG.md`][changelog] 的 `## [Unreleased]` 段（含 v0.5+ 候选）和 [`docs/v0.3-roadmap.md`](v0.3-roadmap.md)（已 SHIPPED，作为历史记录保留）

[changelog]: https://github.com/linjiema/afmkit/blob/main/CHANGELOG.md

## Phase history

| Version | Date | Theme |
|---|---|---|
| v0.1.0 | 2026-07-29 | Core data model, JPK loader, WLC, lmfit, exporters (CSV / `.mat` / Parquet / Markdown) |
| v0.2.0 | 2026-07-30 | Sawtooth peak detection, Igor `.ibw` v2 round-trip, eWLC model, Textual TUI, pluggy |
| v0.3.0 | 2026-07-30 | FJC model, peak review data model, TUI integration, mkdocs site |
| v0.4.0 | 2026-07-30 | v0.3 retrospective (workflow / release checklist), peak review in CSV+Markdown, real matplotlib in TUI, `.ibw` v5, pre-commit in CI, pluggy fix |
| v0.5.0 | 2026-07-31 | Peak review in `.mat`+Parquet (multi-file layout), `.ibw` note full rehydration + `roundtrip_ibw` |
