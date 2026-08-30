# Durable lessons

## Bootstrap must account for client filesystem and rsync versions

- Context/symptom: The first transfer from the authorized macOS clone did not produce a clean Linux worktree.
- Incorrect assumption or action: The transfer command assumed `rsync --protect-args` was available and that a case-insensitive macOS checkout could contain both `tasks/Pour` and `tasks/pour`.
- Root cause: The bundled macOS rsync is older and the repository contains paths that differ only by case; the Linux target correctly reported the missing lowercase files.
- Correction: Use portable rsync arguments, reconstruct case-colliding tracked paths from the Git object database on Linux, and verify `git status` on the destination before editing.
- Prevention/check: Treat the destination Linux `git status --short --branch` as a mandatory bootstrap gate and never use a macOS working tree as the sole completeness check for a case-sensitive repository.

## Interrupting SSH did not terminate the remote CPU diagnostic

- Context/symptom: A long Sweep reference audit was interrupted from the local SSH
  client, then a revised audit was launched. Process inspection showed both exact
  `build_reference.py --world_yaw_deg -80 --audit_side right` jobs still running.
- Incorrect assumption or action: I assumed sending Ctrl-C to the local non-PTY SSH
  session had also terminated the remote Python child.
- Root cause: The SSH frontend exited without propagating termination to the remote
  process, leaving the first CPU-only diagnostic orphaned.
- Correction: Resolved the two task-owned PIDs and terminated only those exact PIDs;
  verified both were gone and did not touch any foreign process.
- Prevention/check: After interrupting a remote long-running command, always inspect
  the exact command line and ownership before relaunching. Use task-owned tmux for
  persistent work and exact PID targeting for cleanup; never use broad process kills.
