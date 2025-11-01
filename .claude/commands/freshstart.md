I'm working on the neurospatial project.

Start now by reading the files and telling me which task you'll work on first.

Your workflow MUST be:

    First, read these files IN ORDER:
        CLAUDE.md (implementation guide)
        docs/SCRATCHPAD.md (notes and current status)
        docs/TASKS.md (current tasks)

    THEN if you need more detail:
        docs/UX_IMPLEMENTATION_PLAN.md (overall project plan)

    Find the FIRST unchecked [ ] task in TASKS.md

    For EVERY feature, follow TDD:
      a. Create the TEST file first
      b. Run the test and verify it FAILS
      c. Only then create the implementation
      d. Run test until it PASSES
      e. Apply review agents (code-reviewer, other relevant agents)
      f. Refactor for clarity and efficiency based on feedback
      g. Add/Update docstrings and types.

    Update TASKS.md checkboxes as you complete items.

    Update SCRATCHPAD.md with notes

    Commit frequently with messages like "feat(F24): implement error handling"

Do not change tests or skip tests to match broken code. Ask permission to change requirements if needed.

## Remember

- **Read before you code** - Use Read tool to understand context
- **Test before you implement** - TDD is mandatory
- **Verify before you claim completion** - Use verification-before-completion skill
- **Ask when uncertain** - Better to ask than assume
- **Document as you go** - Update SCRATCHPAD.md with decisions/blockers

---

## When Blocked

If you encounter any of these, STOP and document in SCRATCHPAD.md:

1. **Unclear requirements** - Ask for clarification
2. **Unexpected test failures** - Use systematic-debugging skill
3. **Conflicting requirements** - Ask for guidance
4. **Need to change baselines** - Request approval
5. **Missing dependencies** - Document and ask for help

**Never proceed with assumptions** - this is critical scientific infrastructure.

You MUST investigate any test failures using the systematic debugging skill. Flaky tests are not an acceptable reason to skip or ignore tests.

---

Now tell me: **What task are you working on next?**
