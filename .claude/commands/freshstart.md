I'm working on the neurospatial project.

Start now by reading the files and telling me which task you'll work on first.

Your workflow MUST be:

    First, choose the relevant topic under docs/plans/, then read these files IN ORDER:
        docs/plans/<topic>/SCRATCHPAD.md (notes and current status, if present)
        docs/plans/<topic>/PLAN.md (overall plan)
        docs/plans/<topic>/TASKS.md (current tasks, if present)

    Find the FIRST unchecked [ ] task in the relevant TASKS.md

    For EVERY feature, follow TDD:
      a. Create the TEST file first  (or identify/modify existing tests)
      b. Run the test and verify it FAILS
      c. Only then create the implementation
      d. Run test until it PASSES
      e. Apply review agents (code-reviewer)
      f. Refactor for clarity and efficiency based on feedback
      g. Add/Update docstrings and types.
      h. Run ruff and mypy and fix any issues

    Update the relevant TASKS.md checkboxes as you complete items.

    Update the relevant SCRATCHPAD.md with notes when one exists.

    Commit frequently with messages like "feat(F24): implement error handling"

## Remember

- **Read before you code** - Use Read tool to understand context (funcion signatures, existing tests, etc)
- **Test before you implement** - TDD is mandatory
- **Ask when uncertain** - Better to ask than assume
- **Document as you go** - Update the relevant SCRATCHPAD.md with decisions/blockers when one exists
- **Test thoroughly** - Ensure all tests pass before marking tasks complete. There are no exceptions (no flaky tests or pre-existing failures). You MUST use systematic-debugging skill for failures.

---

## When Blocked

If you encounter any of these, STOP and document in the relevant SCRATCHPAD.md when one exists:

1. **Unclear requirements** - Ask for clarification
2. **Unexpected test failures** - Use systematic-debugging skill
3. **Conflicting requirements** - Ask for guidance
4. **Need to change baselines** - Request approval
5. **Missing dependencies** - Document and ask for help

**Never proceed with assumptions** - this is critical scientific infrastructure.

---

Now tell me: **What task are you working on next?**
