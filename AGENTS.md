# Global instructions

Limit the amount of comments you put in the code to a strict minimum. You should almost never add comments, except sometimes on non-trivial code, function definitions if the arguments aren't self-explanatory, and class definitions and their members.

## Agent skills

### Tickets

Local markdown: tickets live as files under `.scratch/<feature>/` in this repo. See `docs/agents/local-markdown-tickets.md`.

### Domain docs

Single-context: one `GLOSSARY.md` at the repo root plus `docs/adr/`. See `docs/agents/domain.md`.

添加或删改 `GLOSSARY.md` 中的条目，必须逐条请求人类审核，审核通过后才允许写入。

## 测试规范

仓库内只留持久运行的测试，并在 CI 流水线上运行。一次性测试应在测试完成后当场删除。
