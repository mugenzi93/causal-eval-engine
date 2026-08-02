---
description: Stage, commit, and push changes to the remote causal-eval-engine repo
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git log:*), Bash(git add:*), Bash(git commit:*), Bash(git push:*)
---

You are helping the user commit and push changes to the causal-eval-engine repository (git@github.com:mugenzi93/causal-eval-engine.git).

Steps:
1. Run On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean to show the user what has changed.
2. Run  to review the changes.
3. Ask the user for a commit message if they have not provided one in their prompt.
4. Stage only the relevant files (avoid  unless the user explicitly asks — do not accidentally include .env files, large binaries, or unrelated changes).
5. Create the commit using a HEREDOC to pass the message cleanly.
6. Run  to push to the remote.
7. Confirm success and show the latest 989fbc9 included a .md file describing the project structure
3c019d7 included a .md file describing the project structure
413cbbe included a .md file describing the project structure
ad253fb included an additional drdid method
fa1c2a6 included an additional drdid method.

Always follow the git safety rules: never force push to main, never skip hooks, never amend published commits.
