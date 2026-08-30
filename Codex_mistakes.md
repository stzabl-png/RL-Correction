# Durable lessons

## Bootstrap must account for client filesystem and rsync versions

- Context/symptom: The first transfer from the authorized macOS clone did not produce a clean Linux worktree.
- Incorrect assumption or action: The transfer command assumed `rsync --protect-args` was available and that a case-insensitive macOS checkout could contain both `tasks/Pour` and `tasks/pour`.
- Root cause: The bundled macOS rsync is older and the repository contains paths that differ only by case; the Linux target correctly reported the missing lowercase files.
- Correction: Use portable rsync arguments, reconstruct case-colliding tracked paths from the Git object database on Linux, and verify `git status` on the destination before editing.
- Prevention/check: Treat the destination Linux `git status --short --branch` as a mandatory bootstrap gate and never use a macOS working tree as the sole completeness check for a case-sensitive repository.
