# Contributing Guide

Thank you for contributing! This guide focuses on how to submit issues and pull requests effectively.

We are currently accepting contribution to the following part of the codebase:
- `chart`: Charting and visualisation tools
- `ui`: Chat interface with visualisation built-in
- `docs`: Documentation
- `refactor`: Code refactoring
- `chore`: Maintenance

**Currently, we do not accept pull request to the sampler, torch and mlx implementations.**


## Submitting Issues

### Before Creating an Issue
1. Search existing issues to avoid duplicates
2. Check if the issue is reproduced in the latest version
3. Check the documentation

### Issue Templates

#### Bug Reports
```markdown
Title: [Bug] Brief description of the problem

**Current Behavior**
Clear description of what's happening

**Expected Behavior**
What should happen instead

**Steps to Reproduce**
1. Step one
2. Step two
3. Step three

**Environment**
- OS: [e.g., Windows 10]
- Version: [e.g., v1.2.3]
- Browser/Runtime: [if applicable]

**Additional Context**
Screenshots, error messages, or other relevant information
```

#### Feature Requests
```markdown
Title: [Feature] Brief description of the feature

**Problem Statement**
What problem does this feature solve?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other solutions you've thought about

**Additional Context**
Any other relevant information
```

## Pull Requests

### Before Submitting
1. Create an issue first for non-trivial changes
2. Update your fork to the latest main branch
3. Test your changes thoroughly
4. Ensure code works on your local environment 

### PR Description Template
```markdown
Title: [Type] Brief description

Fixes #[issue number]

**Changes Made**
- Clear bullet points of changes
- One change per line
- Include context where needed

**Testing**
- [ ] Manual testing performed
  - Describe what you tested

**Screenshots**
If applicable, add screenshots

**Additional Notes**
Any extra information reviewers should know
```

### Commit Messages
Format: `type(scope): brief description`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `style`: Formatting
- `chore`: Maintenance

Examples:
```
feat(ui): add chat ui
fix(chart): correct dimensions for smaller screens
docs(readme): update setup guide steps
```

### Quick PR Checklist
- [ ] Issue created (for non-trivial changes)
- [ ] Code tested
- [ ] Commit messages follow format
- [ ] PR template filled completely
- [ ] Related issues linked