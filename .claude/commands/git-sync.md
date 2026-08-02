---
description: Push or pull changes for the causal-eval-engine repo
model: claude-haiku-4-5-20251001
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git fetch:*), Bash(git log:*), Bash(git add:*), Bash(git commit:*), Bash(git pull:*), Bash(git push:*)
---

You are helping the user sync the causal-eval-engine repository (git@github.com:mugenzi93/causal-eval-engine.git) with the remote.

First, run `git status` to understand the current state of the repo.

**If the user wants to push (or has uncommitted changes):**
1. Run `git diff` to review the changes.
2. Ask for a commit message if one was not provided.
3. Stage only relevant files — never `git add -A` unless explicitly asked.
4. Commit using a HEREDOC to pass the message cleanly.
5. Run `git push origin main`.
6. Confirm with `git log --oneline -5`.

**If the user wants to pull:**
1. Warn if there are uncommitted local changes — a pull may cause conflicts.
2. Run `git fetch origin` then show incoming commits with `git log HEAD..origin/main --oneline`.
3. Run `git pull origin main` once the user confirms (or their prompt already implies it).
4. Confirm with `git log --oneline -5`.

Always follow git safety rules: never force push to main, never skip hooks, never amend published commits.
